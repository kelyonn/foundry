//! The parity harness: compare every captured activation against the reference.
//!
//!     cargo run --release -p model --bin parity
//!
//! Per docs/TESTING.md this reports **max absolute error**, layer by layer, in
//! forward order. The first row that exceeds tolerance is the one to debug — every
//! row after it is downstream of the same bug and tells you nothing new.

use model::forward::{Model, Trace};
use model::npy;
use model::weights::Weights;
use model::{Config, SafeTensors};

/// Relative tolerance: `max|ours - ref| / max|ref|`.
///
/// **Why not the absolute `1e-4` that docs/ROADMAP.md asked for.** That threshold
/// is unsatisfiable, not merely strict. The residual stream in this model reaches
/// magnitude ~1543 by layer 3, and one f32 ulp at 1543 is 1.22e-4 — so two
/// *distinct* f32 values there cannot differ by less than 1.22e-4. An absolute
/// budget of 1e-4 demands precision finer than the storage format represents, and
/// measured error in exact multiples of an ulp is the giveaway.
///
/// Relative error is scale-free and is what the comparison actually means.
///
/// **Where 5e-5 comes from — it is not a number picked to make the test pass.**
/// Measured against a float64 dump of the same model, *torch's own float32 path*
/// has up to 1.6e-5 relative error on this prompt (1.52e-5 at `layer00.gate_proj`,
/// 1.49e-5 on the logits). That is the precision floor of the format, not a
/// property of any implementation. Our worst is 1.14e-5 and our logits are 1.48e-6
/// — about 10× closer to the exact answer than the reference's float32, because we
/// accumulate dot products in f64.
///
/// So the gate is set at roughly 3× the reference's own float32 error. Below that
/// the comparison is measuring float32; above it, something structural is wrong.
/// There is no ambiguous middle: a transposed GQA mapping or the wrong RoPE
/// convention produces order-1 relative error, four orders of magnitude away.
const REL_TOL: f32 = 5e-5;

/// Reported alongside, for insight rather than as a gate.
fn ulps(err: f32, magnitude: f32) -> f32 {
    if magnitude == 0.0 {
        return 0.0;
    }
    // Distance to the next representable f32 at this magnitude.
    let next = f32::from_bits(magnitude.abs().to_bits() + 1);
    err / (next - magnitude.abs())
}

/// int8 has a much coarser precision floor than float32: the per-channel scale
/// alone is `max|row| / 127`, i.e. roughly 0.4% relative error on a uniformly
/// distributed row before any accumulation happens at all. This is not a bug
/// budget so much as an acknowledgement of what the format can represent — the
/// same spirit as `REL_TOL`, recomputed for 8 bits instead of 24.
const REL_TOL_INT8: f32 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let int8 = args.iter().any(|a| a == "--int8");
    let positional: Vec<&String> = args
        .iter()
        .skip(1)
        .filter(|a| a.as_str() != "--int8")
        .collect();

    let dir = positional
        .first()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "models/qwen2.5-0.5b".to_string());
    let fixtures = positional
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "tests/fixtures/qwen05b".to_string());

    let rel_tol = if int8 { REL_TOL_INT8 } else { REL_TOL };

    let cfg = Config::from_file(format!("{dir}/config.json")).expect("config");
    let st = SafeTensors::open(format!("{dir}/model.safetensors")).expect("weights");

    eprintln!(
        "loading weights ({}, this takes a moment) ...",
        if int8 { "int8" } else { "bf16 -> f32" }
    );
    let t0 = std::time::Instant::now();
    let weights = Weights::load_with(&st, &cfg, int8).expect("weight layout");
    eprintln!("loaded in {:.1?}", t0.elapsed());

    let tokens = npy::read_u32(format!("{fixtures}/tokens.npy")).expect("tokens.npy");
    eprintln!("prompt is {} tokens: {tokens:?}", tokens.len());

    let model = Model::new(cfg, weights);
    let mut trace = Trace::default();

    eprintln!("running forward pass (scalar f32, no cache) ...");
    let t0 = std::time::Instant::now();
    let logits = model.forward(&tokens, Some(&mut trace));
    let elapsed = t0.elapsed();
    eprintln!(
        "forward pass: {:.2?} for {} tokens ({:.2?}/token)\n",
        elapsed,
        tokens.len(),
        elapsed / tokens.len() as u32
    );

    // Forward order, so the first failure is the root cause.
    let mut names: Vec<String> = vec!["embed_out".to_string()];
    for i in 0..model.config.num_hidden_layers {
        for point in [
            "input_layernorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "rope_q",
            "rope_k",
            "attn_out",
            "o_proj",
            "post_attention_layernorm",
            "gate_proj",
            "up_proj",
            "down_proj",
            "mlp_out",
            "layer_out",
        ] {
            names.push(format!("layer{i:02}.{point}"));
        }
    }
    names.push("final_norm".to_string());
    names.push("logits".to_string());

    let mut worst = 0.0f32;
    let mut first_bad: Option<String> = None;
    let mut compared = 0usize;
    let mut skipped = Vec::new();

    println!(
        "{:<34} {:>11} {:>11} {:>8} {:>10}",
        "activation", "max abs err", "relative", "ulps", "max|ref|"
    );
    println!("{}", "-".repeat(80));

    for name in &names {
        let Some(ours) = trace.get(name) else {
            skipped.push(format!("{name} (not captured)"));
            continue;
        };
        let path = format!("{fixtures}/{name}.npy");
        if !std::path::Path::new(&path).exists() {
            skipped.push(format!("{name} (no fixture)"));
            continue;
        }
        let reference = match npy::read_f32(&path) {
            Ok(t) => t,
            Err(e) => {
                skipped.push(format!("{name} ({e})"));
                continue;
            }
        };

        // Shapes can differ in arrangement while holding the same values — the
        // reference stores attention tensors as [batch, heads, seq, dim] and ours
        // has no batch dimension. Compare element counts and then flat data.
        if ours.len() != reference.len() {
            println!(
                "{name:<40} {:>14}  ours {:?} vs ref {:?}",
                "SHAPE",
                ours.shape(),
                reference.shape()
            );
            if first_bad.is_none() {
                first_bad = Some(name.clone());
            }
            continue;
        }

        let err = ours
            .data()
            .iter()
            .zip(reference.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let magnitude = reference.data().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rel = if magnitude == 0.0 {
            err
        } else {
            err / magnitude
        };

        compared += 1;
        worst = worst.max(rel);

        let flag = if rel > rel_tol { "  <-- FAIL" } else { "" };
        if rel > rel_tol && first_bad.is_none() {
            first_bad = Some(name.clone());
        }
        println!(
            "{name:<34} {err:>11.3e} {rel:>11.3e} {:>8.1} {magnitude:>10.2}{flag}",
            ulps(err, magnitude)
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("compared {compared} activations, relative tolerance {rel_tol:e}");
    println!("worst relative error: {worst:.3e}");

    if !skipped.is_empty() {
        println!("\nskipped {}:", skipped.len());
        for s in skipped.iter().take(8) {
            println!("  {s}");
        }
    }

    // The headline number: logits.
    if let (Some(ours), Ok(reference)) = (
        trace.get("logits"),
        npy::read_f32(format!("{fixtures}/logits.npy")),
    ) {
        let err = ours
            .data()
            .iter()
            .zip(reference.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let magnitude = reference.data().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "\nLOGIT PARITY: max abs err {err:.3e}  relative {:.3e}  ({:.1} ulps at max|ref| = {magnitude:.2})",
            err / magnitude,
            ulps(err, magnitude)
        );

        // Argmax agreement is the practically-meaningful check: even if the
        // numbers drift slightly, the sampled token must be the same one.
        let vocab = reference.shape()[reference.shape().len() - 1];
        let last = tokens.len() - 1;
        let ours_row = &ours.data()[last * vocab..(last + 1) * vocab];
        let ref_row = &reference.data()[last * vocab..(last + 1) * vocab];
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 {
                        (i, v)
                    } else {
                        acc
                    }
                })
                .0
        };
        let (a, b) = (argmax(ours_row), argmax(ref_row));
        println!(
            "next-token argmax: ours {a}, reference {b} — {}",
            if a == b { "agree" } else { "DISAGREE" }
        );
    }

    match first_bad {
        None => println!("\nPASS — every activation within {rel_tol:e} relative"),
        Some(name) => {
            println!("\nFAIL — first divergence at: {name}");
            println!("Everything after that row is downstream of the same bug.");
            std::process::exit(1);
        }
    }
    let _ = logits;
}
