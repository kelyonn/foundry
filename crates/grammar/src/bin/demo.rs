//! The demo: prove that invalid output is unreachable, with no model involved.
//!
//! Run with:
//!     cargo run -p grammar --bin demo
//!     cargo run -p grammar --bin demo -- --tokenizer models/qwen2.5-0.5b/tokenizer.json
//!
//! # What this does and does not show
//!
//! The "model" here is a random number generator, optionally a *hostile* one that
//! actively tries to break the JSON. That is a deliberate choice: it isolates the
//! guarantee. If output stays valid when the token distribution is adversarial,
//! the validity does not depend on the model's cooperation at all — which is the
//! entire claim.
//!
//! It is NOT the unconstrained-baseline measurement described in docs/GRAMMAR.md.
//! That number requires a real model and real prompts, and it is not in this
//! build. The unconstrained rate printed below is the rate for *random sampling*,
//! which is near zero for uninteresting reasons. Do not quote it as "LLMs produce
//! invalid JSON N% of the time" — it does not measure that.

use grammar::{ConstrainedDecoder, Grammar, Vocab, VocabIndex};
use std::time::Instant;

/// xorshift64*, so runs are seeded and reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Stand-in for a model. Produces a logit vector with no idea what JSON is.
///
/// `hostile` biases mass toward bytes that break JSON structure — quotes,
/// braces, backslashes in the wrong places — so the constrained run is being
/// actively fought, not merely unassisted.
fn fake_logits(rng: &mut Rng, vocab: &Vocab, hostile: bool, out: &mut [f32]) {
    for (id, slot) in out.iter_mut().enumerate() {
        let mut v = rng.next_f32() * 4.0 - 2.0;
        if hostile {
            let bytes = vocab.bytes(id as u32);
            let breaks_json = bytes
                .iter()
                .any(|b| matches!(b, b'}' | b']' | b'"' | b'\\' | b',' | b':'));
            if breaks_json {
                v += 8.0;
            }
        }
        *slot = v;
    }
}

/// Sample from a logit vector. `-inf` entries have probability exactly zero.
fn sample(rng: &mut Rng, logits: &[f32]) -> Option<u32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None; // everything masked
    }
    let mut total = 0.0f64;
    for &l in logits {
        total += ((l - max) as f64).exp();
    }
    let mut target = rng.next_f32() as f64 * total;
    for (id, &l) in logits.iter().enumerate() {
        target -= ((l - max) as f64).exp();
        if target <= 0.0 {
            return Some(id as u32);
        }
    }
    logits
        .iter()
        .rposition(|l| l.is_finite())
        .map(|id| id as u32)
}

const MAX_TOKENS: usize = 256;

fn generate_constrained(
    grammar: &Grammar,
    vocab: &Vocab,
    index: &mut VocabIndex,
    rng: &mut Rng,
    hostile: bool,
) -> Result<String, String> {
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, vocab, index);
    let mut logits = vec![0.0f32; vocab.len()];

    for _ in 0..MAX_TOKENS {
        fake_logits(rng, vocab, hostile, &mut logits);
        decoder.apply_mask(&mut logits).map_err(|e| e.to_string())?;
        let id = match sample(rng, &logits) {
            Some(id) => id,
            None => return Err("no legal token after masking".into()),
        };
        decoder.accept_token(id).map_err(|e| e.to_string())?;
        if decoder.is_finished() {
            break;
        }
    }

    if !decoder.is_accepting() && !decoder.is_finished() {
        return Err(format!(
            "hit the {MAX_TOKENS}-token budget before reaching an accepting state"
        ));
    }
    Ok(String::from_utf8_lossy(decoder.output()).into_owned())
}

/// The control: the same fake model, sampling freely with no mask.
fn generate_unconstrained(vocab: &Vocab, rng: &mut Rng, hostile: bool) -> String {
    let mut logits = vec![0.0f32; vocab.len()];
    let mut out: Vec<u8> = Vec::new();
    for _ in 0..24 {
        fake_logits(rng, vocab, hostile, &mut logits);
        if let Some(id) = sample(rng, &logits) {
            if id == vocab.eos() {
                break;
            }
            out.extend_from_slice(vocab.bytes(id));
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

const SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "name":   { "type": "string", "maxLength": 12 },
    "age":    { "type": "integer" },
    "city":   { "enum": ["London", "Bengaluru", "Tokyo"] },
    "active": { "type": "boolean" },
    "tags":   { "type": "array", "items": { "type": "string", "maxLength": 6 }, "maxItems": 3 }
  }
}"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokenizer_path = args
        .windows(2)
        .find(|w| w[0] == "--tokenizer")
        .map(|w| w[1].clone());

    let vocab = match &tokenizer_path {
        Some(path) => match Vocab::from_tokenizer_json(path) {
            Ok(v) => {
                println!("vocabulary: {} tokens from {}", v.len(), path);
                v
            }
            Err(e) => {
                eprintln!("could not load {path}: {e}");
                std::process::exit(1);
            }
        },
        None => {
            let v = Vocab::synthetic();
            println!(
                "vocabulary: {} synthetic tokens (pass --tokenizer <path> for the real one)",
                v.len()
            );
            v
        }
    };

    println!("\nschema:\n{SCHEMA}\n");

    let t0 = Instant::now();
    let grammar = match Grammar::from_schema_str(SCHEMA) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to compile schema: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "compiled to a byte DFA: {} states, {:.1?} ({} KB transition table)",
        grammar.num_states(),
        t0.elapsed(),
        grammar.num_states() * 256 * 4 / 1024
    );

    let mut index = VocabIndex::new();

    // --- a few samples, printed --------------------------------------------
    println!("\n--- constrained generations, hostile model ---");
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    for i in 0..5 {
        match generate_constrained(&grammar, &vocab, &mut index, &mut rng, true) {
            Ok(s) => {
                let ok = serde_json::from_str::<serde_json::Value>(&s).is_ok();
                println!("  {i}: {s}   [parses: {ok}]");
            }
            Err(e) => println!("  {i}: ERROR {e}"),
        }
    }

    println!("\n--- the same model with no mask ---");
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    for i in 0..3 {
        let s = generate_unconstrained(&vocab, &mut rng, true);
        let ok = serde_json::from_str::<serde_json::Value>(&s).is_ok();
        let shown: String = s.chars().take(60).collect();
        println!("  {i}: {shown}   [parses: {ok}]");
    }

    // --- the actual measurement --------------------------------------------
    const N: usize = 500;
    println!("\n--- {N} constrained generations ---");
    let mut rng = Rng(0x00C0_FFEE_1234_5678);
    let mut valid = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let t0 = Instant::now();
    for _ in 0..N {
        match generate_constrained(&grammar, &vocab, &mut index, &mut rng, true) {
            // Validated by serde_json, an independent parser that knows nothing
            // about our automaton. Self-validation would prove nothing.
            Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(_) => valid += 1,
                Err(e) => failures.push(format!("{s}  ({e})")),
            },
            Err(e) => failures.push(e),
        }
    }
    let elapsed = t0.elapsed();

    println!("  valid: {valid}/{N}");
    for f in failures.iter().take(5) {
        println!("  FAILURE: {f}");
    }

    let s = &index.stats;
    println!("\n--- vocabulary index ---");
    println!("  DFA states total:      {}", grammar.num_states());
    println!("  states visited/cached: {}", index.cached_states());
    println!("  masks built:           {}", s.builds);
    println!("  cache hits:            {}", s.hits);
    println!(
        "  hit rate:              {:.2}%",
        100.0 * s.hits as f64 / (s.hits + s.builds).max(1) as f64
    );
    println!("  token walks:           {}", s.tokens_scanned);
    println!("  time in index builds:  {:.2?}", s.build_time);
    println!("  total generation time: {elapsed:.2?}");
    println!(
        "  mask memory:           {} KB",
        index.cached_states() * vocab.len().div_ceil(64) * 8 / 1024
    );

    if valid == N {
        println!("\n{N}/{N} parsed. Not 99.8%.");
    } else {
        println!("\nFAILED: {}/{} did not parse.", N - valid, N);
        std::process::exit(1);
    }
}
