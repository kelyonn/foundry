//! Weight loading, the Qwen2 forward pass, KV cache and sampling.
//!
//! Oracle: per-layer activation parity against HuggingFace, within `1e-4` max
//! absolute error. See docs/TESTING.md.

pub mod config;
pub mod forward;
pub mod kv_cache;
pub mod npy;
pub mod safetensors;
pub mod sampling;
pub mod weights;

pub use config::Config;
pub use forward::{Model, Trace};
pub use kv_cache::KvCache;
pub use safetensors::{Dtype, LoadError, SafeTensors, TensorInfo};
pub use sampling::{Sampler, SamplingConfig};
pub use weights::Weights;
