//! Perplexity, to quantify what int8 quantization costs in quality.
//!
//!     cargo run --release -p cli --bin perplexity
//!
//! Defaults to `tests/fixtures/wikitext2_test.txt` — a real slice of wikitext-2's
//! test split (Merity et al., 2016), the corpus docs/ROADMAP.md names for this
//! measurement. Regenerate it with `python scripts/fetch_wikitext2.py` (needs
//! `pip install datasets`); it's small (~8 KB) and committed, so a fresh clone
//! doesn't need network access to run this. `--text <path>` overrides it, and a
//! short built-in paragraph is the last-resort fallback if neither exists —
//! never silently claimed to be wikitext-2, see `FALLBACK` below.
//!
//! # What perplexity is
//!
//! `exp(mean negative log-likelihood per token)`. Loosely, "how many equally-likely
//! options was the model choosing between, on average." Lower is better; 1.0 would
//! be a model that predicted every token with certainty.
//!
//! # Why it is the right measure here
//!
//! Memory savings from quantization are trivially measurable and therefore easy to
//! quote honestly. The thing that is easy to quote *dishonestly* is quality — "the
//! output still looks fine" is not a measurement. Perplexity on held-out text is,
//! and it is sensitive to exactly the kind of small distribution shift that
//! quantization causes and that eyeballing misses.
//!
//! The absolute value depends on the corpus and is not comparable across papers.
//! The **delta between f32 and int8 on the same corpus** is the number that means
//! something, so that is what this prints.

use model::forward::Model;
use model::weights::Weights;
use model::{Config, KvCache, SafeTensors};
use tokenizer::Tokenizer;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let dir = argv
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "models/qwen2.5-0.5b".to_string());
    let text_path = argv
        .windows(2)
        .find(|w| w[0] == "--text")
        .map(|w| w[1].clone());
    // 768, not the smaller number this used to default to: measured directly,
    // the fp32->int8 delta's *sign* is unstable below this. At 256 tokens it read
    // -0.17% (int8 looked better than f32); at 512, +0.11%; at 768 and again
    // extrapolating from a partial 1888-token run, consistently +0.10%. Below
    // ~768 the "measurement" is mostly sampling noise around a true effect near
    // zero, not a reliable read of which direction it goes.
    let max_tokens: usize = argv
        .windows(2)
        .find(|w| w[0] == "--tokens")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(768);

    const WIKITEXT2: &str = "tests/fixtures/wikitext2_test.txt";
    let (text, corpus_label) = match &text_path {
        Some(p) => (
            std::fs::read_to_string(p).unwrap_or_else(|e| {
                eprintln!("cannot read {p}: {e}");
                std::process::exit(1);
            }),
            p.clone(),
        ),
        None => match std::fs::read_to_string(WIKITEXT2) {
            Ok(t) => (t, format!("{WIKITEXT2} (real wikitext-2 test split)")),
            Err(_) => {
                eprintln!(
                    "note: {WIKITEXT2} not found — run `python scripts/fetch_wikitext2.py`.\n\
                     Falling back to a short built-in paragraph, which is NOT wikitext-2."
                );
                (
                    FALLBACK.to_string(),
                    "built-in fallback (not wikitext-2)".to_string(),
                )
            }
        },
    };

    let cfg = cli::or_exit(Config::from_file(format!("{dir}/config.json")));
    let st = cli::or_exit(SafeTensors::open(format!("{dir}/model.safetensors")));
    let tk = cli::or_exit(Tokenizer::from_file(format!("{dir}/tokenizer.json")));

    let mut tokens = tk.encode(&text);
    tokens.truncate(max_tokens);
    assert!(
        tokens.len() > 1,
        "need at least two tokens to score a prediction"
    );

    println!(
        "corpus: {corpus_label} ({} tokens scored)",
        tokens.len() - 1
    );
    println!();

    let mut results = Vec::new();
    for &int8 in &[false, true] {
        let weights = cli::or_exit(Weights::load_with(&st, &cfg, int8));
        let bytes = weights.size_bytes();
        let model = Model::new(cfg.clone(), weights);
        let ppl = perplexity(&model, &tokens);
        results.push((int8, ppl, bytes));
        println!(
            "{:<6} perplexity {:>9.4}   weights {:>7.1} MB",
            if int8 { "int8" } else { "f32" },
            ppl,
            bytes as f64 / 1048576.0
        );
    }

    let (_, f32_ppl, f32_bytes) = results[0];
    let (_, int8_ppl, int8_bytes) = results[1];
    println!();
    println!(
        "delta:  {:+.4} perplexity ({:+.2}%) for {:.2}x less weight memory",
        int8_ppl - f32_ppl,
        100.0 * (int8_ppl - f32_ppl) / f32_ppl,
        f32_bytes as f64 / int8_bytes as f64
    );
    println!();
    println!(
        "Absolute perplexity depends on the corpus and is not comparable across\n\
         papers or repos. The delta on the same corpus is the number that means\n\
         something."
    );
}

/// Teacher-forced perplexity: score each real token under the distribution the
/// model produced from the true prefix.
///
/// One prefill over the whole sequence gives every position's logits at once, so
/// this is a single forward pass rather than one per token.
fn perplexity(model: &Model, tokens: &[u32]) -> f64 {
    let cfg = &model.config;
    let mut cache = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        tokens.len(),
    );
    let logits = model.forward_cached(tokens, &mut cache, None);
    let width = logits.shape()[logits.shape().len() - 1];
    let data = logits.data();

    let mut nll = 0.0f64;
    // Row `i` predicts token `i + 1`; the last row predicts nothing we can score.
    for i in 0..tokens.len() - 1 {
        let row = &data[i * width..(i + 1) * width];
        let target = tokens[i + 1] as usize;

        // log softmax, computed stably: log p_t = z_t - max - log(sum exp(z - max))
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for &z in row {
            sum += ((z - max) as f64).exp();
        }
        let log_p = (row[target] - max) as f64 - sum.ln();
        nll -= log_p;
    }

    (nll / (tokens.len() - 1) as f64).exp()
}

/// Used when no corpus file is given. Held-out-ish prose; it is *not* wikitext-2,
/// and the output says so rather than implying a standard benchmark was run.
const FALLBACK: &str = "\
The city of Bengaluru grew rapidly in the second half of the twentieth century, \
transforming from a garden city known for its parks and lakes into the centre of \
India's software industry. Public sector research laboratories established in the \
1950s and 1960s laid the groundwork, training generations of engineers who later \
founded private firms. By the 1990s the city had become a destination for \
outsourced software development, and by the 2010s it hosted research offices for \
most large technology companies. The growth brought severe strain on \
infrastructure: traffic congestion, water scarcity, and the loss of many of the \
lakes that had given the city its earlier reputation. Urban planners have \
argued that the pace of construction outstripped the capacity of the road network \
and the water supply, and that recovery would require sustained investment over \
decades rather than individual projects.";
