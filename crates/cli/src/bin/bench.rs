//! The measurement harness. Per docs/BENCHMARKS.md and docs/RULES.md hard rule 3,
//! this is the **only** place numbers may come from.
//!
//!     cargo run --release -p cli --bin bench -- [--int8] [--runs 10]
//!
//! Rules implemented here, so they cannot be forgotten at the keyboard:
//! discard the first 3 runs, take n ≥ 10, report median and p95, and keep prefill
//! and decode separate because they are bound by different resources.

use model::forward::Model;
use model::sampling::{Sampler, SamplingConfig};
use model::weights::Weights;
use model::{Config, KvCache, SafeTensors};
use std::time::{Duration, Instant};
use tokenizer::Tokenizer;

const WARMUP: usize = 3;

struct Stats {
    median: f64,
    p95: f64,
    min: f64,
    n: usize,
}

/// Median and p95 of the *timed* runs, warmup already discarded.
fn summarise(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    // Nearest-rank p95: the smallest value at or above 95% of the data. For
    // n = 10 that is the 10th sample, i.e. the worst — which is honest for a
    // sample this small, and worth saying out loud rather than interpolating
    // to a friendlier number.
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).clamp(1, n) - 1;
    Stats {
        median: samples[n / 2],
        p95: samples[p95_idx],
        min: samples[0],
        n,
    }
}

fn peak_rss_bytes() -> u64 {
    // getrusage(RUSAGE_SELF).ru_maxrss. On macOS this is bytes; on Linux, KB.
    #[cfg(target_os = "macos")]
    {
        rusage_maxrss()
    }
    #[cfg(not(target_os = "macos"))]
    {
        rusage_maxrss() * 1024
    }
}

fn rusage_maxrss() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct Rusage {
        ru_utime: [i64; 2],
        ru_stime: [i64; 2],
        ru_maxrss: i64,
        rest: [i64; 14],
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut u = Rusage::default();
    // SAFETY: `getrusage` writes a fixed-size struct through the pointer. The
    // layout above matches the platform's `struct rusage` for the leading fields
    // we read; `rest` is padding sized to cover the remainder so the call cannot
    // write past our allocation. We only read `ru_maxrss`, which is at a stable
    // offset on both macOS and Linux.
    unsafe {
        if getrusage(0, &mut u) != 0 {
            return 0;
        }
    }
    u.ru_maxrss.max(0) as u64
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let int8 = argv.iter().any(|a| a == "--int8");
    let runs: usize = argv
        .windows(2)
        .find(|w| w[0] == "--runs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(10);
    let prompt_tokens: usize = argv
        .windows(2)
        .find(|w| w[0] == "--prompt-tokens")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(512);
    let decode_tokens: usize = argv
        .windows(2)
        .find(|w| w[0] == "--decode-tokens")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(128);
    let dir = argv
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "models/qwen2.5-0.5b".to_string());

    assert!(runs >= 10, "docs/BENCHMARKS.md requires n >= 10");

    let cfg = cli::or_exit(Config::from_file(format!("{dir}/config.json")));
    let st = cli::or_exit(SafeTensors::open(format!("{dir}/model.safetensors")));
    let tk = cli::or_exit(Tokenizer::from_file(format!("{dir}/tokenizer.json")));

    let t0 = Instant::now();
    let weights = cli::or_exit(Weights::load_with(&st, &cfg, int8));
    let load_time = t0.elapsed();
    let weight_bytes = weights.size_bytes();

    let model = Model::new(cfg, weights);
    let cfg = &model.config;

    // A deterministic prompt of the requested length. Content is irrelevant to
    // throughput; length is not.
    let base = tk.encode(
        "Foundry is an inference engine written from scratch in Rust. \
         It loads safetensors weights, tokenizes with byte-level BPE, and runs \
         a Qwen2 forward pass with grouped-query attention and a KV cache. ",
    );
    let mut prompt: Vec<u32> = Vec::with_capacity(prompt_tokens);
    while prompt.len() < prompt_tokens {
        prompt.extend_from_slice(&base);
    }
    prompt.truncate(prompt_tokens);

    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("Machine:     {}", machine());
    println!("OS:          {}", os_version());
    println!("Rust:        {} --release", rustc_version());
    println!(
        "Model:       Qwen2.5-0.5B-Instruct ({} params)",
        st.total_parameters()
    );
    println!("Precision:   {}", if int8 { "int8" } else { "f32" });
    println!("Threads:     {threads}");
    println!("Context:     {prompt_tokens} prompt -> {decode_tokens} generated");
    println!("Runs:        {runs} timed, {WARMUP} warmup discarded");
    println!("Date:        {}", today());
    println!();
    println!(
        "weights resident: {:.1} MB",
        weight_bytes as f64 / 1048576.0
    );
    println!("weight load time: {load_time:.2?}");
    println!();

    // ---- prefill ---------------------------------------------------------
    let mut prefill_samples = Vec::new();
    let mut ttft_samples = Vec::new();
    for run in 0..(runs + WARMUP) {
        let mut cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            prompt_tokens + decode_tokens + 1,
        );
        let t = Instant::now();
        let _ = model.forward_cached(&prompt, &mut cache, None);
        let dt = t.elapsed();
        if run >= WARMUP {
            prefill_samples.push(prompt_tokens as f64 / dt.as_secs_f64());
            ttft_samples.push(dt.as_secs_f64() * 1000.0);
        }
    }
    let prefill = summarise(prefill_samples);
    let ttft = summarise(ttft_samples);

    // ---- decode ----------------------------------------------------------
    let mut decode_samples = Vec::new();
    for run in 0..(runs + WARMUP) {
        let mut cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            prompt_tokens + decode_tokens + 1,
        );
        let mut logits = model.forward_cached(&prompt, &mut cache, None);
        let mut sampler = Sampler::new(SamplingConfig::greedy(), 1);
        let width = logits.shape()[logits.shape().len() - 1];

        let t = Instant::now();
        let mut produced = 0usize;
        for _ in 0..decode_tokens {
            let rows = logits.len() / width;
            let row = &logits.data()[(rows - 1) * width..];
            let Some(next) = sampler.sample(row) else {
                break;
            };
            if !cache.has_room_for(1) {
                break;
            }
            logits = model.forward_cached(&[next], &mut cache, None);
            produced += 1;
        }
        let dt = t.elapsed();
        if run >= WARMUP {
            decode_samples.push(produced as f64 / dt.as_secs_f64());
        }
    }
    let decode = summarise(decode_samples);

    // ---- KV cache footprint ---------------------------------------------
    let kv_4096 = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        4096,
    );

    println!(
        "{:<28} {:>10} {:>10} {:>10}",
        "metric", "median", "p95", "best"
    );
    println!("{}", "-".repeat(62));
    println!(
        "{:<28} {:>10.2} {:>10.2} {:>10.2}",
        "prefill tok/s", prefill.median, prefill.p95, prefill.min
    );
    println!(
        "{:<28} {:>10.2} {:>10.2} {:>10.2}",
        "decode tok/s", decode.median, decode.p95, decode.min
    );
    println!(
        "{:<28} {:>10.1} {:>10.1} {:>10.1}",
        "time to first token (ms)", ttft.median, ttft.p95, ttft.min
    );
    println!();
    println!(
        "weights resident       {:>10.1} MB",
        weight_bytes as f64 / 1048576.0
    );
    println!(
        "KV cache @ 4096 ctx    {:>10.1} MB",
        kv_4096.size_bytes() as f64 / 1048576.0
    );
    println!(
        "peak RSS               {:>10.1} MB",
        peak_rss_bytes() as f64 / 1048576.0
    );
    println!();
    println!(
        "note: p95 over n={} is the nearest-rank value, i.e. effectively the worst run.",
        prefill.n
    );
    let _ = Duration::from_secs(0);
}

fn machine() -> String {
    let brand = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(0);
    let mem = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| format!("{} GB", b / 1024 / 1024 / 1024))
        .unwrap_or_else(|| "unknown RAM".into());
    format!("{brand}, {cores} cores, {mem}")
}

fn os_version() -> String {
    std::process::Command::new("uname")
        .arg("-sr")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn today() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}
