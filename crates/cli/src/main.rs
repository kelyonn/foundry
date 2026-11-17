//! `foundry` — load a checkpoint, run it, optionally constrain the output.
//!
//!     cargo run --release -p cli -- --prompt "The capital of France is" --max-tokens 20
//!     cargo run --release -p cli -- --chat --prompt "Name an Indian city." \
//!         --schema '{"type":"object","properties":{"city":{"type":"string","maxLength":16}}}'
//!
//! Hand-rolled argument parsing: the dependency budget in docs/RULES.md permits a CLI
//! crate, but a dozen flags do not need one.

use cli::CliError;
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

/// Parse a flag's value strictly. The first version used `.parse().unwrap_or(32)`,
/// so `--max-tokens abc` silently ran with 32 and `--temperature 0,7` silently ran
/// greedy. A typo in a flag should be an error, not a different experiment.
fn parse_num<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, CliError> {
    value
        .parse()
        .map_err(|_| CliError::Usage(format!("{flag}: cannot parse {value:?}")))
}

fn parse_args() -> Result<Args, CliError> {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let mut next = || -> Result<String, CliError> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| CliError::Usage(format!("missing value for {flag}")))
        };
        match flag.as_str() {
            "--model" => a.model_dir = next()?,
            "--prompt" | "-p" => a.prompt = next()?,
            "--max-tokens" | "-n" => a.max_tokens = parse_num(&flag, &next()?)?,
            "--temperature" | "-t" => a.temperature = parse_num(&flag, &next()?)?,
            "--top-k" => a.top_k = Some(parse_num(&flag, &next()?)?),
            "--top-p" => a.top_p = Some(parse_num(&flag, &next()?)?),
            "--seed" => a.seed = parse_num(&flag, &next()?)?,
            "--chat" => a.chat = true,
            "--schema" => a.schema = Some(next()?),
            "--schema-file" => {
                let path = next()?;
                a.schema = Some(
                    std::fs::read_to_string(&path)
                        .map_err(|e| CliError::Io(format!("cannot read {path}: {e}")))?,
                );
            }
            "--int8" => a.int8 = true,
            "--quiet" | "-q" => a.show_stats = false,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                return Err(CliError::Usage(format!("unknown flag {other}\n\n{USAGE}")));
            }
        }
        i += 1;
    }
    Ok(a)
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
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(e.exit_code());
    }
}

fn run() -> Result<(), CliError> {
    let args = parse_args()?;

    // `"nan".parse::<f32>()` succeeds and produces NaN, which `unwrap_or` never
    // catches. Left unchecked this reaches `Sampler`, where `NaN <= 0.0` is false
    // so it takes the non-greedy path, every comparison against NaN is false, and
    // sampling degrades to a confusing "no legal token to sample" — never a
    // panic, but a bad error message for what is actually invalid input. Caught
    // here, at the boundary, with a message that names the actual problem.
    if args.temperature.is_nan() {
        return Err(CliError::Usage("--temperature is NaN".into()));
    }

    let t_load = std::time::Instant::now();
    let cfg = Config::from_file(format!("{}/config.json", args.model_dir))?;
    let st = SafeTensors::open(format!("{}/model.safetensors", args.model_dir))?;
    let weights = Weights::load_with(&st, &cfg, args.int8)?;
    let tk = Tokenizer::from_file(format!("{}/tokenizer.json", args.model_dir))?;
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
        return Err(CliError::Usage(
            "--prompt is empty (or tokenizes to nothing); nothing to run".into(),
        ));
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
    let grammar = args
        .schema
        .as_ref()
        .map(|s| Grammar::from_schema_str(s).map_err(|e| CliError::Schema(e.to_string())))
        .transpose()?;
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
            dec.apply_mask(row)
                .map_err(|e| CliError::Generation(e.to_string()))?;
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
            dec.accept_token(next).map_err(|e| {
                CliError::Generation(format!("automaton rejected the sampled token: {e}"))
            })?;
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
                eprintln!("output parses as JSON: NO");
                return Err(CliError::InvalidOutput(e.to_string()));
            }
        }
    }
    Ok(())
}
