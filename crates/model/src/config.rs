//! Model configuration, read from `config.json`.
//!
//! Per docs/RULES.md hard rule 4: **the file on disk wins.** docs/MODEL.md records what
//! we expect, but nothing here defaults to those expectations — every field is read
//! from the file, and a missing field is an error rather than a guess. A silent
//! default is how you spend a day debugging a forward pass that was configured
//! wrong rather than implemented wrong.

use serde_json::Value;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
}

impl Config {
    /// `hidden_size / num_attention_heads`. 896 / 14 = 64 for this model.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// How many query heads share each KV head. 14 / 2 = 7 here.
    ///
    /// This is the grouped-query attention ratio. Query head `i` reads KV head
    /// `i / group_size` — docs/ARCHITECTURE.md warns that getting this mapping
    /// backwards produces output that is *almost* coherent, which is the worst
    /// failure mode available because it does not look like a bug.
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Config, ConfigError> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError(format!("cannot read config.json: {e}")))?;
        let v: Value =
            serde_json::from_str(&raw).map_err(|e| ConfigError(format!("invalid JSON: {e}")))?;
        Config::from_value(&v)
    }

    pub fn from_value(v: &Value) -> Result<Config, ConfigError> {
        fn usize_field(v: &Value, key: &str) -> Result<usize, ConfigError> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| ConfigError(format!("missing or non-integer field `{key}`")))
        }
        fn f32_field(v: &Value, key: &str) -> Result<f32, ConfigError> {
            v.get(key)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| ConfigError(format!("missing or non-numeric field `{key}`")))
        }

        let cfg = Config {
            hidden_size: usize_field(v, "hidden_size")?,
            intermediate_size: usize_field(v, "intermediate_size")?,
            num_hidden_layers: usize_field(v, "num_hidden_layers")?,
            num_attention_heads: usize_field(v, "num_attention_heads")?,
            num_key_value_heads: usize_field(v, "num_key_value_heads")?,
            vocab_size: usize_field(v, "vocab_size")?,
            max_position_embeddings: usize_field(v, "max_position_embeddings")?,
            rope_theta: f32_field(v, "rope_theta")?,
            rms_norm_eps: f32_field(v, "rms_norm_eps")?,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .ok_or_else(|| ConfigError("missing field `tie_word_embeddings`".into()))?,
        };

        // Structural invariants the rest of the code assumes. Checked once here so
        // a violation is a clear message rather than a shape panic 200 lines later.
        //
        // Zero counts get an explicit check before the divisibility checks below.
        // `is_multiple_of` does not panic on a zero divisor (`n.is_multiple_of(0)`
        // is `false` unless `n` is also `0`), so a lone zero is already caught —
        // but `hidden_size == 0 && num_attention_heads == 0` slips through both
        // checks, because `0.is_multiple_of(0)` is defined as `true`. Left
        // unguarded, `head_dim()` would later divide `0 / 0` and panic with a bare
        // "attempt to divide by zero" instead of this message.
        if cfg.num_attention_heads == 0 {
            return Err(ConfigError("num_attention_heads must be nonzero".into()));
        }
        if cfg.num_key_value_heads == 0 {
            return Err(ConfigError("num_key_value_heads must be nonzero".into()));
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            return Err(ConfigError(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                cfg.hidden_size, cfg.num_attention_heads
            )));
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(ConfigError(format!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_like() -> Value {
        serde_json::json!({
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true
        })
    }

    #[test]
    fn derives_head_dim_and_gqa_ratio() {
        let c = Config::from_value(&qwen_like()).unwrap();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.gqa_group_size(), 7);
    }

    #[test]
    fn a_missing_field_is_an_error_not_a_default() {
        let mut v = qwen_like();
        v.as_object_mut().unwrap().remove("rope_theta");
        assert!(Config::from_value(&v).is_err());
    }

    #[test]
    fn rejects_inconsistent_head_counts() {
        let mut v = qwen_like();
        v["num_attention_heads"] = serde_json::json!(15);
        assert!(Config::from_value(&v).is_err());
    }

    /// A lone zero head count is already caught by the divisibility checks
    /// (`is_multiple_of` doesn't panic on zero), but both being zero at once
    /// slips past them — `0.is_multiple_of(0)` is `true`. This pins the explicit
    /// guard rather than relying on that coincidence.
    #[test]
    fn rejects_all_zero_head_counts_without_panicking() {
        let mut v = qwen_like();
        v["hidden_size"] = serde_json::json!(0);
        v["num_attention_heads"] = serde_json::json!(0);
        v["num_key_value_heads"] = serde_json::json!(0);
        assert!(Config::from_value(&v).is_err());
    }

    #[test]
    fn rejects_zero_kv_heads() {
        let mut v = qwen_like();
        v["num_key_value_heads"] = serde_json::json!(0);
        assert!(Config::from_value(&v).is_err());
    }
}
