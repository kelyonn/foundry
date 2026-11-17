//! Constrained vs. prompted-only JSON, on the real model.
//!
//!     cargo run --release -p cli --bin json_compare -- [--n 100] [--int8]
//!
//! Answers two questions for the README / writeup, with numbers from this harness
//! only (docs/RULES.md hard rule 3):
//!
//! 1. **Validity.** Given the *same* prompt (schema included in the text), how often
//!    does the model alone produce output that parses, and that conforms to the
//!    schema, versus the same model with the grammar mask applied?
//! 2. **Cost.** What does masking cost in decode tok/s, and how much of a step is the
//!    mask itself?
//!
//! "Valid" is judged by an independent parser (`serde_json`) and a small schema
//! checker written here — never by the grammar crate's own DFA, which would be
//! grading its own homework.

use grammar::{ConstrainedDecoder, Grammar, Vocab, VocabIndex};
use model::forward::Model;
use model::sampling::{Sampler, SamplingConfig};
use model::weights::Weights;
use model::{Config, KvCache, SafeTensors};
use serde_json::Value;
use std::time::{Duration, Instant};
use tokenizer::Tokenizer;

const MAX_TOKENS: usize = 128;

/// (task sentence, schema). Chosen to cover objects, enums, bounded strings, bounded
/// arrays, integers, floats and booleans.
const CASES: &[(&str, &str)] = &[
    (
        "Describe a fictional person.",
        r#"{"type":"object","properties":{"name":{"type":"string","maxLength":20},"age":{"type":"integer"},"active":{"type":"boolean"}}}"#,
    ),
    (
        "Give facts about a large city.",
        r#"{"type":"object","properties":{"city":{"enum":["London","Tokyo","Delhi"]},"population_millions":{"type":"integer"}}}"#,
    ),
    (
        "Suggest tags for a blog post about cooking.",
        r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string","maxLength":10},"minItems":2,"maxItems":4}}}"#,
    ),
    (
        "Describe a product sold in a shop.",
        r#"{"type":"object","properties":{"title":{"type":"string","maxLength":30},"price":{"type":"number"},"in_stock":{"type":"boolean"}}}"#,
    ),
];

/// A deliberately small schema checker for the subset used above. Independent of
/// the grammar crate. Unknown keys in an object are a failure (the grammar never
/// emits them, and "additional properties" is the usual way prompted output drifts).
fn conforms(v: &Value, schema: &Value) -> bool {
    if let Some(options) = schema.get("enum").and_then(|e| e.as_array()) {
        return options.contains(v);
    }
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => {
            let Some(obj) = v.as_object() else {
                return false;
            };
            let props = schema.get("properties").and_then(|p| p.as_object());
            let empty = serde_json::Map::new();
            let props = props.unwrap_or(&empty);
            if obj.keys().any(|k| !props.contains_key(k)) {
                return false;
            }
            // Foundry's grammar emits every declared property; hold the model to
            // the same standard rather than a softer "required" reading.
            props
                .iter()
                .all(|(k, s)| obj.get(k).is_some_and(|x| conforms(x, s)))
        }
        Some("array") => {
            let Some(arr) = v.as_array() else {
                return false;
            };
            let min = schema.get("minItems").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let max = schema
                .get("maxItems")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(usize::MAX);
            let item = schema.get("items");
            arr.len() >= min
                && arr.len() <= max
                && item.is_none_or(|s| arr.iter().all(|x| conforms(x, s)))
        }
        Some("string") => {
            let Some(s) = v.as_str() else {
                return false;
            };
            let n = s.chars().count();
            let min = schema
                .get("minLength")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as usize;
            let max = schema
                .get("maxLength")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(usize::MAX);
            n >= min && n <= max
        }
        Some("integer") => v.is_i64() || v.is_u64(),
        Some("number") => v.is_number(),
        Some("boolean") => v.is_boolean(),
        Some("null") => v.is_null(),
        _ => true,
    }
}

/// Strip a leading/trailing markdown code fence, the single most common way a
/// chat model "fails" strict parsing while being obviously trying to comply.
fn strip_fence(s: &str) -> &str {
    let t = s.trim();
    let t = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```"))
        .unwrap_or(t);
    t.strip_suffix("```").unwrap_or(t).trim()
}

#[derive(Default)]
struct Tally {
    runs: usize,
    parses_strict: usize,
    parses_unfenced: usize,
    conforms_strict: usize,
    conforms_unfenced: usize,
    truncated: usize,
    decode_tps: Vec<f64>,
    mask_ms_per_token: Vec<f64>,
}

impl Tally {
    fn record(&mut self, text: &str, schema: &Value, truncated: bool) {
        self.runs += 1;
        if truncated {
            self.truncated += 1;
        }
        if let Ok(v) = serde_json::from_str::<Value>(text) {
            self.parses_strict += 1;
            if conforms(&v, schema) {
                self.conforms_strict += 1;
            }
        }
        if let Ok(v) = serde_json::from_str::<Value>(strip_fence(text)) {
            self.parses_unfenced += 1;
            if conforms(&v, schema) {
                self.conforms_unfenced += 1;
            }
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn chat(prompt: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn main() {
    let mut n = 100usize;
    let mut int8 = false;
    let mut model_dir = "models/qwen2.5-0.5b".to_string();
    let mut temperature = 0.7f32;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--n" => {
                i += 1;
                n = argv[i].parse().expect("--n takes an integer");
            }
            "--model" => {
                i += 1;
                model_dir = argv[i].clone();
            }
            "--temperature" => {
                i += 1;
                temperature = argv[i].parse().expect("--temperature takes a float");
            }
            "--int8" => int8 = true,
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let cfg = cli::or_exit(Config::from_file(format!("{model_dir}/config.json")));
    let st = cli::or_exit(SafeTensors::open(format!("{model_dir}/model.safetensors")));
    let weights = cli::or_exit(Weights::load_with(&st, &cfg, int8));
    let tk = cli::or_exit(Tokenizer::from_file(format!("{model_dir}/tokenizer.json")));
    let model = Model::new(cfg, weights);

    let vocab = {
        let v = tk.vocab_size();
        let tokens: Vec<Vec<u8>> = (0..v as u32).map(|i| tk.bytes(i).to_vec()).collect();
        let special: Vec<bool> = (0..v as u32).map(|i| tk.is_special(i)).collect();
        Vocab::from_tokens(tokens, special, tk.eos())
    };

    println!("model: {model_dir} ({})", if int8 { "int8" } else { "f32" });
    println!(
        "generations: {n} per mode, {} schemas round-robin, temperature {temperature}, max {MAX_TOKENS} tokens",
        CASES.len()
    );
    println!("the same chat prompt (schema included in the text) is used in both modes\n");

    let mut free = Tally::default();
    let mut cons = Tally::default();
    let mut indexes: Vec<VocabIndex> = CASES.iter().map(|_| VocabIndex::new()).collect();
    let grammars: Vec<Grammar> = CASES
        .iter()
        .map(|(_, s)| Grammar::from_schema_str(s).expect("schema compiles"))
        .collect();

    for g in 0..n {
        let case = g % CASES.len();
        let (task, schema_str) = CASES[case];
        let schema: Value = serde_json::from_str(schema_str).unwrap();
        let prompt = chat(&format!(
            "{task} Respond with only a JSON object that matches this JSON Schema, and nothing else:\n{schema_str}"
        ));
        let ids = tk.encode_with_specials(&prompt);
        let seed = 1000 + g as u64;

        for constrained in [false, true] {
            let mut cache = KvCache::new(
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.head_dim(),
                ids.len() + MAX_TOKENS + 1,
            );
            let mut sampler = Sampler::new(
                SamplingConfig {
                    temperature,
                    top_k: None,
                    top_p: None,
                },
                seed,
            );
            let mut dec = constrained.then(|| {
                // SAFETY of borrows: index is borrowed mutably only for this scope.
                let (gr, ix) = (&grammars[case], &mut indexes[case]);
                // Extend lifetimes by constructing the decoder inline below.
                ConstrainedDecoder::new(&gr.dfa, &vocab, ix)
            });

            let mut logits = model.forward_cached(&ids, &mut cache, None);
            let width = logits.shape()[logits.shape().len() - 1];
            let mut out: Vec<u32> = Vec::new();
            let mut truncated = true;
            let mut mask_time = Duration::ZERO;
            let t0 = Instant::now();
            let mut steps = 0usize;

            for _ in 0..MAX_TOKENS {
                let rows = logits.len() / width;
                let row = &mut logits.data_mut()[(rows - 1) * width..];
                let tm = Instant::now();
                if let Some(d) = dec.as_mut() {
                    d.apply_mask(row).expect("mask");
                } else {
                    cli::mask_padding_tail(row, tk.vocab_size());
                }
                mask_time += tm.elapsed();
                let Some(next) = sampler.sample(row) else {
                    break;
                };
                if let Some(d) = dec.as_mut() {
                    let ta = Instant::now();
                    d.accept_token(next).expect("accept");
                    mask_time += ta.elapsed();
                    if d.is_finished() {
                        truncated = false;
                        break;
                    }
                } else if next == tk.eos() {
                    truncated = false;
                    break;
                }
                out.push(next);
                logits = model.forward_cached(&[next], &mut cache, None);
                steps += 1;
            }
            let elapsed = t0.elapsed();
            let text = tk.decode(&out);
            let tally = if constrained { &mut cons } else { &mut free };
            tally.record(&text, &schema, truncated);
            if steps > 0 {
                tally.decode_tps.push(steps as f64 / elapsed.as_secs_f64());
                tally
                    .mask_ms_per_token
                    .push(mask_time.as_secs_f64() * 1000.0 / (steps + 1) as f64);
            }
        }
        if (g + 1) % 10 == 0 {
            eprintln!(
                "  {}/{n}  prompted-only conforms {}  constrained conforms {}",
                g + 1,
                free.conforms_strict,
                cons.conforms_strict
            );
        }
    }

    println!("                                   prompted-only    constrained");
    let pct = |a: usize, t: &Tally| {
        format!(
            "{a:>4}/{:<4} {:>6.1}%",
            t.runs,
            100.0 * a as f64 / t.runs as f64
        )
    };
    println!(
        "parses as JSON (strict)          {}  {}",
        pct(free.parses_strict, &free),
        pct(cons.parses_strict, &cons)
    );
    println!(
        "parses after stripping ``` fence {}  {}",
        pct(free.parses_unfenced, &free),
        pct(cons.parses_unfenced, &cons)
    );
    println!(
        "conforms to schema (strict)      {}  {}",
        pct(free.conforms_strict, &free),
        pct(cons.conforms_strict, &cons)
    );
    println!(
        "conforms after stripping fence   {}  {}",
        pct(free.conforms_unfenced, &free),
        pct(cons.conforms_unfenced, &cons)
    );
    println!(
        "hit the {MAX_TOKENS}-token limit           {:>4}/{:<4}        {:>4}/{:<4}",
        free.truncated, free.runs, cons.truncated, cons.runs
    );
    let (mut a, mut b) = (free.decode_tps.clone(), cons.decode_tps.clone());
    println!(
        "\ndecode tok/s (median per generation): prompted-only {:.2} | constrained {:.2}",
        median(&mut a),
        median(&mut b)
    );
    let mut m = cons.mask_ms_per_token.clone();
    let mut mf = free.mask_ms_per_token.clone();
    println!(
        "logit-processing ms/token (median):   prompted-only {:.3} (padding mask only) | constrained {:.3} (DFA mask + advance)",
        median(&mut mf),
        median(&mut m)
    );
    let builds: u32 = indexes.iter().map(|i| i.stats.builds).sum();
    let hits: u32 = indexes.iter().map(|i| i.stats.hits).sum();
    println!("vocab-index cache: {builds} mask builds, {hits} hits across all schemas");
    println!("\nnote: truncated constrained runs count as invalid; they are reported, not hidden.");
}
