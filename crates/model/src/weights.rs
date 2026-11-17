//! Weights, resolved from the safetensors file into the shapes the forward pass
//! wants.
//!
//! Loading is eager and up front: every tensor becomes `f32` at construction. That
//! turns ~376 MB of `bf16` on disk into ~2 GB resident, which is the price of
//! docs/ARCHITECTURE.md's "everything is f32 in v1" rule.

use crate::config::Config;
use crate::safetensors::{LoadError, SafeTensors};
use tensor::quant::{linear_q, QTensor};
use tensor::Tensor;

/// A projection matrix, at whatever precision it was loaded in.
///
/// An enum rather than a trait object: there are exactly two cases, the dispatch
/// is once per matmul (not per element), and the shapes are needed at load time.
/// Making the forward pass call `.matmul()` means precision is a *loading*
/// decision that the model code never sees — which is what keeps the int8 path
/// from becoming a parallel implementation that can drift.
pub enum LinearWeight {
    F32(Tensor),
    Q8(QTensor),
}

impl LinearWeight {
    pub fn matmul(&self, x: &Tensor, bias: Option<&Tensor>) -> Tensor {
        match self {
            LinearWeight::F32(w) => tensor::ops::linear(x, w, bias),
            LinearWeight::Q8(w) => linear_q(x, w, bias),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            LinearWeight::F32(w) => w.shape(),
            LinearWeight::Q8(w) => w.shape(),
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            LinearWeight::F32(w) => w.len() * 4,
            LinearWeight::Q8(w) => w.size_bytes(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, LinearWeight::Q8(_))
    }

    /// One row, as `f32`. Used for the embedding lookup, which is a gather rather
    /// than a matmul — dequantizing a single row per token is negligible next to
    /// keeping a second `f32` copy of a 136M-parameter table.
    pub fn row_f32(&self, i: usize) -> Vec<f32> {
        match self {
            LinearWeight::F32(w) => w.row(i).to_vec(),
            LinearWeight::Q8(w) => w.row_f32(i),
        }
    }

    fn load(st: &SafeTensors, name: &str, quantize: bool) -> Result<LinearWeight, LoadError> {
        let t = st.tensor(name)?;
        Ok(if quantize {
            LinearWeight::Q8(QTensor::quantize(&t))
        } else {
            LinearWeight::F32(t)
        })
    }
}

pub struct LayerWeights {
    /// Norm weights stay `f32` always. They are `[896]` — 3.5 KB against a
    /// 4.3 MB projection — so quantizing them would save nothing measurable while
    /// injecting error directly into the residual stream's scale.
    pub input_layernorm: Tensor,
    pub q_proj: LinearWeight,
    pub q_bias: Tensor,
    pub k_proj: LinearWeight,
    pub k_bias: Tensor,
    pub v_proj: LinearWeight,
    pub v_bias: Tensor,
    pub o_proj: LinearWeight,
    pub post_attention_layernorm: Tensor,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

pub struct Weights {
    pub embed_tokens: LinearWeight,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,
    /// `None` when `tie_word_embeddings` is true, which it is for this model —
    /// the output projection reuses `embed_tokens`.
    pub lm_head: Option<LinearWeight>,
}

impl Weights {
    /// Load at full precision.
    pub fn load(st: &SafeTensors, cfg: &Config) -> Result<Weights, LoadError> {
        Weights::load_with(st, cfg, false)
    }

    /// Load, optionally quantizing every projection to int8.
    ///
    /// Biases and norm weights are never quantized — see `LayerWeights`.
    pub fn load_quantized(st: &SafeTensors, cfg: &Config) -> Result<Weights, LoadError> {
        Weights::load_with(st, cfg, true)
    }

    pub fn load_with(st: &SafeTensors, cfg: &Config, quantize: bool) -> Result<Weights, LoadError> {
        let embed_tokens = LinearWeight::load(st, "model.embed_tokens.weight", quantize)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
            layers.push(LayerWeights {
                input_layernorm: st.tensor(&p("input_layernorm.weight"))?,
                // Qwen2 has bias on q/k/v but not on o_proj or the MLP. This is a
                // real architectural quirk, not an oversight — asking for a bias
                // that does not exist fails loudly here rather than producing a
                // silently wrong projection.
                q_proj: LinearWeight::load(st, &p("self_attn.q_proj.weight"), quantize)?,
                q_bias: st.tensor(&p("self_attn.q_proj.bias"))?,
                k_proj: LinearWeight::load(st, &p("self_attn.k_proj.weight"), quantize)?,
                k_bias: st.tensor(&p("self_attn.k_proj.bias"))?,
                v_proj: LinearWeight::load(st, &p("self_attn.v_proj.weight"), quantize)?,
                v_bias: st.tensor(&p("self_attn.v_proj.bias"))?,
                o_proj: LinearWeight::load(st, &p("self_attn.o_proj.weight"), quantize)?,
                post_attention_layernorm: st.tensor(&p("post_attention_layernorm.weight"))?,
                gate_proj: LinearWeight::load(st, &p("mlp.gate_proj.weight"), quantize)?,
                up_proj: LinearWeight::load(st, &p("mlp.up_proj.weight"), quantize)?,
                down_proj: LinearWeight::load(st, &p("mlp.down_proj.weight"), quantize)?,
            });
        }

        let final_norm = st.tensor("model.norm.weight")?;

        // With tied embeddings the checkpoint simply has no `lm_head.weight`; the
        // output projection is the embedding matrix transposed. For a 0.5B model
        // that matters — the embedding is 151936 × 896 ≈ 136M parameters, over a
        // quarter of the total, and tying halves it.
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(LinearWeight::load(st, "lm_head.weight", quantize)?)
        };

        Ok(Weights {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Total weight bytes resident, for the memory comparison.
    pub fn size_bytes(&self) -> usize {
        let mut n = self.embed_tokens.size_bytes() + self.final_norm.len() * 4;
        for l in &self.layers {
            n += l.q_proj.size_bytes()
                + l.k_proj.size_bytes()
                + l.v_proj.size_bytes()
                + l.o_proj.size_bytes()
                + l.gate_proj.size_bytes()
                + l.up_proj.size_bytes()
                + l.down_proj.size_bytes();
            n += (l.input_layernorm.len()
                + l.post_attention_layernorm.len()
                + l.q_bias.len()
                + l.k_bias.len()
                + l.v_bias.len())
                * 4;
        }
        n += self.lm_head.as_ref().map(|w| w.size_bytes()).unwrap_or(0);
        n
    }

    pub fn is_quantized(&self) -> bool {
        self.embed_tokens.is_quantized()
    }

    /// The matrix used for the output projection, honouring weight tying.
    pub fn output_projection(&self) -> &LinearWeight {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }
}
