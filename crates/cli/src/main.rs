//! `foundry` — load a checkpoint, run it, optionally constrain the output.
//!
//!     cargo run --release -p cli -- --prompt "The capital of France is" --max-tokens 20
//!     cargo run --release -p cli -- --chat --prompt "Name an Indian city." \
//!         --schema '{"type":"object","properties":{"city":{"type":"string","maxLength":16}}}'
//!
//! Hand-rolled argument parsing: the dependency budget in docs/RULES.md permits a CLI
//! crate, but a dozen flags do not need one.

use grammar::{ConstrainedDecoder, Grammar, VocabIndex};
use model::forward::Model;
use model::sampling::{Sampler, SamplingConfig};
use model::weights::Weights;
use model::{Config, KvCache, SafeTensors};
use std::io::Write;
use tokenizer::Tokenizer;

struct Args {
    model_dir: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: u64,
    chat: bool,
    schema: Option<String>,
    show_stats: bool,
    int8: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model_dir: "models/qwen2.5-0.5b".into(),
            prompt: "The capital of France is".into(),
            max_tokens: 32,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: 0xF0175D,
            chat: false,
            schema: None,
            show_stats: true,
            int8: false,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {}", argv[*i - 1]);
                std::process::exit(2);
            })
        };
        match argv[i].as_str() {
            "--model" => a.model_dir = next(&mut i),
            "--prompt" | "-p" => a.prompt = next(&mut i),
            "--max-tokens" | "-n" => a.max_tokens = next(&mut i).parse().unwrap_or(32),
            "--temperature" | "-t" => a.temperature = next(&mut i).parse().unwrap_or(0.0),
            "--top-k" => a.top_k = next(&mut i).parse().ok(),
            "--top-p" => a.top_p = next(&mut i).parse().ok(),
            "--seed" => a.seed = next(&mut i).parse().unwrap_or(0xF0175D),
            "--chat" => a.chat = true,
            "--schema" => a.schema = Some(next(&mut i)),
            "--schema-file" => {
                let path = next(&mut i);
                a.schema = Some(std::fs::read_to_string(&path).unwrap_or_else(|e| {
                    eprintln!("cannot read {path}: {e}");
                    std::process::exit(2);
                }));
            }
            "--int8" => a.int8 = true,
            "--quiet" | "-q" => a.show_stats = false,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown flag {other}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    a
}

const USAGE: &str = "\
foundry — a from-scratch LLM inference engine

  --model <dir>        checkpoint directory (default models/qwen2.5-0.5b)
  -p, --prompt <text>  the prompt
  -n, --max-tokens <n> generation budget (default 32)
  -t, --temperature <f> 0 = greedy (default)
  --top-k <n>          keep only the n most likely tokens
  --top-p <f>          nucleus sampling threshold
  --seed <n>           RNG seed, for reproducibility
  --chat               wrap the prompt in the Qwen chat template
  --schema <json>      constrain output to a JSON Schema
  --schema-file <path> same, read from a file
  --int8               quantize weights to int8 on load
  -q, --quiet          suppress timing and stats
";

/// Qwen2.5-Instruct's chat template, written out rather than interpreted from
/// `tokenizer_config.json` — that field is a Jinja template, and rendering Jinja
/// is well outside what this project should be doing.
fn apply_chat_template(prompt: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn main() {
    let args = parse_args();

    // `"nan".parse::<f32>()` succeeds and produces NaN, which `unwrap_or` never
    // catches. Left unchecked this reaches `Sampler`, where `NaN <= 0.0` is false
    // so it takes the non-greedy path, every comparison against NaN is false, and
    // sampling degrades to a confusing "no legal token to sample" — never a
    // panic, but a bad error message for what is actually invalid input. Caught
    // here, at the boundary, with a message that names the actual problem.
    if args.temperature.is_nan() {
        eprintln!("--temperature is NaN");
        std::process::exit(2);
    }

    let t_load = std::time::Instant::now();
    let cfg = match Config::from_file(format!("{}/config.json", args.model_dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let st = match SafeTensors::open(format!("{}/model.safetensors", args.model_dir)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let weights = Weights::load_with(&st, &cfg, args.int8).expect("weight layout");
    let tk =
        Tokenizer::from_file(format!("{}/tokenizer.json", args.model_dir)).expect("tokenizer.json");
    let load_time = t_load.elapsed();

    let text = if args.chat {
        apply_chat_template(&args.prompt)
    } else {
        args.prompt.clone()
    };

    // The chat template contains control tokens by design, so it goes through the
    // specials-aware encoder. A bare prompt is user content and does not.
    let prompt_ids = if args.chat {
        tk.encode_with_specials(&text)
    } else {
        tk.encode(&text)
    };

    // The forward pass asserts a non-empty sequence — reasonably, since "logits
    // for no tokens" isn't a meaningful request. An empty `--prompt` reaches this
    // with zero tokens (an empty string tokenizes to nothing), so it's caught
    // here with a clean message rather than a panic and a backtrace hint two
    // crates downstream. `--chat` can never trigger this: the template's control
    // tokens guarantee at least one id even for an empty `--prompt`.
    if prompt_ids.is_empty() {
        eprintln!("--prompt is empty (or tokenizes to nothing); nothing to run");
        std::process::exit(2);
    }

    if args.show_stats {
        eprintln!(
            "loaded {} params in {:.1?} | weights {:.0} MB {} | prompt {} tokens",
            st.total_parameters(),
            load_time,
            weights.size_bytes() as f64 / (1024.0 * 1024.0),
            if weights.is_quantized() {
                "(int8)"
            } else {
                "(f32)"
            },
            prompt_ids.len()
        );
    }

    let model = Model::new(cfg, weights);
    let cfg = &model.config;
    let mut cache = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        prompt_ids.len() + args.max_tokens + 1,
    );

    let mut sampler = Sampler::new(
        SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
        },
        args.seed,
    );

    // Constrained decoding, if a schema was given. The grammar's vocabulary is
    // built from the tokenizer we already loaded, so the two cannot disagree.
    let grammar = args.schema.as_ref().map(|s| {
        Grammar::from_schema_str(s).unwrap_or_else(|e| {
            eprintln!("bad schema: {e}");
            std::process::exit(2);
        })
    });
    let gvocab = grammar.as_ref().map(|_| {
        let n = tk.vocab_size();
        let tokens: Vec<Vec<u8>> = (0..n as u32).map(|i| tk.bytes(i).to_vec()).collect();
        let special: Vec<bool> = (0..n as u32).map(|i| tk.is_special(i)).collect();
        grammar::Vocab::from_tokens(tokens, special, tk.eos())
    });
    let mut index = VocabIndex::new();
    let mut constrained = match (&grammar, &gvocab) {
        (Some(g), Some(v)) => {
            if args.show_stats {
                eprintln!("grammar: {} DFA states", g.num_states());
            }
            Some(ConstrainedDecoder::new(&g.dfa, v, &mut index))
        }
        _ => None,
    };

    // ---- prefill ---------------------------------------------------------
    let t_prefill = std::time::Instant::now();
    let mut logits = model.forward_cached(&prompt_ids, &mut cache, None);
    let prefill_time = t_prefill.elapsed();

    if !args.chat && args.schema.is_none() {
        print!("{}", args.prompt);
        let _ = std::io::stdout().flush();
    }

    // ---- decode ----------------------------------------------------------
    let vocab_width = logits.shape()[logits.shape().len() - 1];
    let mut generated: Vec<u32> = Vec::new();
    let t_decode = std::time::Instant::now();
    let mut decode_steps = 0usize;

    for _ in 0..args.max_tokens {
        // Last row of the logits — the distribution for the *next* token.
        let rows = logits.len() / vocab_width;
        let row = &mut logits.data_mut()[(rows - 1) * vocab_width..];

        if let Some(dec) = constrained.as_mut() {
            if let Err(e) = dec.apply_mask(row) {
                eprintln!("\nconstrained decoding failed: {e}");
                std::process::exit(1);
            }
        } else {
            // No grammar active, so nothing else masks the padding tail between
            // the tokenizer's real vocabulary (151,665 ids) and the model's
            // padded output width (151,936). Without this, a sufficiently hot
            // temperature could sample a padding id and panic in `tk.decode`.
            // See `cli::mask_padding_tail`.
            cli::mask_padding_tail(row, tk.vocab_size());
        }

        let Some(next) = sampler.sample(row) else {
            eprintln!("\nno legal token to sample");
            break;
        };

        if let Some(dec) = constrained.as_mut() {
            if let Err(e) = dec.accept_token(next) {
                eprintln!("\nautomaton rejected the sampled token: {e}");
                std::process::exit(1);
            }
            if dec.is_finished() {
                break;
            }
        } else if next == tk.eos() {
            break;
        }

        generated.push(next);
        print!("{}", tk.decode(&[next]));
        let _ = std::io::stdout().flush();

        if !cache.has_room_for(1) {
            eprintln!("\n(KV cache full)");
            break;
        }
        logits = model.forward_cached(&[next], &mut cache, None);
        decode_steps += 1;
    }
    let decode_time = t_decode.elapsed();
    println!();

    if args.show_stats {
        let prefill_tps = prompt_ids.len() as f64 / prefill_time.as_secs_f64();
        eprintln!(
            "\nprefill {:.2?} ({prefill_tps:.2} tok/s) | decode {} tokens in {:.2?} ({:.2} tok/s)",
            prefill_time,
            decode_steps,
            decode_time,
            decode_steps as f64 / decode_time.as_secs_f64().max(1e-9),
        );
        eprintln!(
            "NOTE: these are single-run wall times from an unoptimised scalar build. \
             They are NOT benchmark numbers — see docs/BENCHMARKS.md and scripts/bench.sh."
        );
        if let Some(dec) = constrained.as_ref() {
            let _ = dec;
            eprintln!(
                "grammar index: {} states cached, {} builds, {} hits",
                index.cached_states(),
                index.stats.builds,
                index.stats.hits
            );
        }
    }

    // Validate constrained output with an independent parser, always.
    if args.schema.is_some() {
        let out = tk.decode(&generated);
        match serde_json::from_str::<serde_json::Value>(&out) {
            Ok(_) => eprintln!("output parses as JSON: yes"),
            Err(e) => {
                eprintln!("output parses as JSON: NO — {e}");
                std::process::exit(1);
            }
        }
    }
}
