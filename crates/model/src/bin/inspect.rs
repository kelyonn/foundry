//! Print what is actually in a checkpoint.
//!
//! Hard rule 4 says the file on disk wins over the docs. This is the tool that
//! settles the argument.
//!
//!     cargo run -p model --bin inspect -- models/qwen2.5-0.5b

use model::{Config, SafeTensors};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen2.5-0.5b".to_string());

    let cfg = match Config::from_file(format!("{dir}/config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    println!("--- config.json ---");
    println!("  hidden_size            {}", cfg.hidden_size);
    println!("  intermediate_size      {}", cfg.intermediate_size);
    println!("  num_hidden_layers      {}", cfg.num_hidden_layers);
    println!("  num_attention_heads    {}", cfg.num_attention_heads);
    println!("  num_key_value_heads    {}", cfg.num_key_value_heads);
    println!("  vocab_size             {}", cfg.vocab_size);
    println!("  rope_theta             {}", cfg.rope_theta);
    println!("  rms_norm_eps           {}", cfg.rms_norm_eps);
    println!("  tie_word_embeddings    {}", cfg.tie_word_embeddings);
    println!("  derived head_dim       {}", cfg.head_dim());
    println!("  derived GQA group size {}", cfg.gqa_group_size());

    let st = match SafeTensors::open(format!("{dir}/model.safetensors")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    println!("\n--- model.safetensors ---");
    println!("  {} tensors", st.len());
    println!("  {} parameters", st.total_parameters());

    // Layer 0 in full, then just the count for the rest — 24 identical layers
    // printed in full is noise.
    println!("\n  layer 0 and the non-layer tensors:");
    for name in st.names() {
        let is_layer = name.starts_with("model.layers.");
        let is_layer_zero = name.starts_with("model.layers.0.");
        if is_layer && !is_layer_zero {
            continue;
        }
        let info = st.info(name).unwrap();
        println!("    {:<52} {:?} {:?}", name, info.shape, info.dtype);
    }

    // Does q/k/v really have a bias and o_proj really not? docs/MODEL.md claims so.
    println!("\n  bias check (docs/MODEL.md claims q/k/v yes, o_proj and MLP no):");
    for suffix in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ] {
        let has = st.info(&format!("model.layers.0.{suffix}.bias")).is_some();
        println!("    {suffix:<20} bias: {has}");
    }

    let tied = st.info("lm_head.weight").is_none();
    println!("\n  lm_head.weight absent (i.e. embeddings tied): {tied}");
    if tied != cfg.tie_word_embeddings {
        println!(
            "    MISMATCH: config says tie_word_embeddings = {}",
            cfg.tie_word_embeddings
        );
    }
}
