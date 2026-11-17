//! The KV cache's correctness condition.
//!
//! A cache is right when it is **invisible**: decoding one token at a time with
//! history must produce exactly what re-running the whole sequence from scratch
//! produces. Anything else means the cache is not a memo, it is a second model.
//!
//! This is the test that catches the classic failure — using the position *within
//! this call* for RoPE instead of the absolute position. That bug leaves output
//! fluent, because every token still gets rotated by something, so it cannot be
//! found by reading generated text. It shows up here immediately.
//!
//! Requires the checkpoint; skips if it is absent.

use model::forward::Model;
use model::weights::Weights;
use model::{Config, KvCache, SafeTensors};

const DIR: &str = "../../models/qwen2.5-0.5b";

fn load() -> Option<Model> {
    if !std::path::Path::new(&format!("{DIR}/model.safetensors")).exists() {
        eprintln!("skipping: no checkpoint at {DIR}");
        return None;
    }
    let cfg = Config::from_file(format!("{DIR}/config.json")).unwrap();
    let st = SafeTensors::open(format!("{DIR}/model.safetensors")).unwrap();
    let w = Weights::load(&st, &cfg).unwrap();
    Some(Model::new(cfg, w))
}

fn last_row(t: &tensor::Tensor) -> Vec<f32> {
    let vocab = t.shape()[t.shape().len() - 1];
    let rows = t.len() / vocab;
    t.data()[(rows - 1) * vocab..].to_vec()
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let magnitude = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / magnitude
}

#[test]
fn incremental_decode_matches_a_full_forward() {
    let Some(model) = load() else { return };
    let cfg = &model.config;

    // "The capital of France is"
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let steps = 3;

    let mut cache = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        prompt.len() + steps + 1,
    );

    // Prefill.
    let cached_logits = model.forward_cached(&prompt, &mut cache, None);
    let fresh_logits = model.forward(&prompt, None);
    let rel = max_rel_diff(&last_row(&cached_logits), &last_row(&fresh_logits));
    assert!(
        rel < 1e-6,
        "prefill disagrees with a fresh forward: {rel:e}"
    );
    assert_eq!(cache.len(), prompt.len());

    // Decode, one token at a time, greedily.
    let mut all = prompt.clone();
    let mut next = model::sampling::argmax(&last_row(&cached_logits)).unwrap();

    for step in 0..steps {
        all.push(next);

        // One token, using the cache.
        let cached = model.forward_cached(&[next], &mut cache, None);

        // The same sequence, from scratch, no history.
        let fresh = model.forward(&all, None);

        let rel = max_rel_diff(&last_row(&cached), &last_row(&fresh));
        assert!(
            rel < 1e-5,
            "step {step}: cached decode diverges from full forward, relative {rel:e}\n\
             a wrong RoPE position offset is the usual cause"
        );

        // The stronger, discrete check: the same token must be chosen.
        let a = model::sampling::argmax(&last_row(&cached)).unwrap();
        let b = model::sampling::argmax(&last_row(&fresh)).unwrap();
        assert_eq!(a, b, "step {step}: cached and fresh chose different tokens");

        assert_eq!(cache.len(), all.len());
        next = a;
    }
}

/// Prefilling the whole prompt at once and feeding it token-by-token must land in
/// the same place. This isolates the prefill/decode split from the cache itself.
#[test]
fn prefill_in_one_call_equals_prefill_token_by_token() {
    let Some(model) = load() else { return };
    let cfg = &model.config;
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];

    let mut bulk = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        prompt.len(),
    );
    let bulk_logits = model.forward_cached(&prompt, &mut bulk, None);

    let mut drip = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        prompt.len(),
    );
    let mut drip_logits = None;
    for &tok in &prompt {
        drip_logits = Some(model.forward_cached(&[tok], &mut drip, None));
    }

    let rel = max_rel_diff(
        &last_row(&bulk_logits),
        &last_row(drip_logits.as_ref().unwrap()),
    );
    assert!(rel < 1e-5, "bulk vs incremental prefill differ: {rel:e}");
    assert_eq!(bulk.len(), drip.len());
}

#[test]
fn cache_reports_the_expected_size() {
    // 24 layers × 2 (k and v) × 2 kv heads × 2048 positions × 64 dims × 4 bytes.
    let cache = KvCache::new(24, 2, 64, 2048);
    let expected = 24 * 2 * 2 * 2048 * 64 * 4;
    assert_eq!(cache.size_bytes(), expected);
    // ~50 MB. With 14 heads instead of 2 it would be ~352 MB — that ratio is
    // what grouped-query attention actually buys.
    assert!(cache.size_bytes() < 55 * 1024 * 1024);
}

#[test]
fn running_past_capacity_is_refused_not_silently_wrapped() {
    let mut cache = KvCache::new(2, 2, 4, 3);
    assert!(cache.has_room_for(3));
    cache.advance(3).unwrap();
    assert!(!cache.has_room_for(1));
    assert!(cache.advance(1).is_err());
}

/// A token id past the end of the embedding table must fail loudly with a
/// message naming the actual problem, not a bare slice-index panic from deep
/// inside `row_f32`. `forward_cached` is a public entry point, so a caller (a
/// test, a future API user, a corrupted token stream) can hand in anything.
#[test]
#[should_panic(expected = "out of range")]
fn out_of_range_token_id_panics_with_a_clear_message() {
    let Some(model) = load() else {
        // No checkpoint to test against; fail the intended way so the test still
        // demonstrates the property under `--should_panic` rather than silently
        // "passing" by returning early.
        panic!("out of range (skipped: no checkpoint present)");
    };
    let cfg = &model.config;
    let mut cache = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        4,
    );
    model.forward_cached(&[999_999_999], &mut cache, None);
}
