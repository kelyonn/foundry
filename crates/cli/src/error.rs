//! The CLI's error type.
//!
//! Library crates already return their own typed errors (`ConfigError`, `LoadError`,
//! `TokenizerError`, `SchemaError`, `DecodeError`). This enum is the one place they
//! are gathered, so `main` can be `fn run() -> Result<(), CliError>` and every
//! failure path — a missing file, a malformed schema, a bad flag — leaves through
//! the same door with a message that names the problem and an exit code that
//! distinguishes "you called it wrong" (2) from "it failed" (1).
//!
//! What stays a panic, deliberately: shape and length *invariants* inside the
//! tensor and model crates (`assert!` on `[m, k] @ [n, k]`). Those are bugs in
//! this code, not bad input, and the CLI checks the user-reachable preconditions
//! (empty prompt, KV-cache capacity) before it gets there. See D24.

use std::fmt;

#[derive(Debug)]
pub enum CliError {
    /// A bad flag, a bad flag value, or a bad combination. Exit code 2.
    Usage(String),
    Config(model::config::ConfigError),
    Load(model::LoadError),
    Tokenizer(tokenizer::TokenizerError),
    /// The user's JSON Schema did not compile.
    Schema(String),
    /// The constrained decoder refused a step it should not have.
    Generation(String),
    /// Constrained output failed independent validation. This is the failure the
    /// whole project exists to rule out, so it gets its own variant.
    InvalidOutput(String),
    Io(String),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Usage(m) => write!(f, "{m}"),
            CliError::Config(e) => write!(f, "{e}"),
            CliError::Load(e) => write!(f, "{e}"),
            CliError::Tokenizer(e) => write!(f, "{e}"),
            CliError::Schema(m) => write!(f, "bad schema: {m}"),
            CliError::Generation(m) => write!(f, "constrained decoding failed: {m}"),
            CliError::InvalidOutput(m) => write!(f, "output does not parse as JSON: {m}"),
            CliError::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<model::config::ConfigError> for CliError {
    fn from(e: model::config::ConfigError) -> Self {
        CliError::Config(e)
    }
}
impl From<model::LoadError> for CliError {
    fn from(e: model::LoadError) -> Self {
        CliError::Load(e)
    }
}
impl From<tokenizer::TokenizerError> for CliError {
    fn from(e: tokenizer::TokenizerError) -> Self {
        CliError::Tokenizer(e)
    }
}

/// For the small measurement binaries (`bench`, `perplexity`, `json_compare`,
/// `parity`): print `error: …` and exit instead of unwinding with a backtrace hint.
pub fn or_exit<T, E: Into<CliError>>(r: Result<T, E>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            let e: CliError = e.into();
            eprintln!("error: {e}");
            std::process::exit(e.exit_code());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_errors_exit_2_and_everything_else_exits_1() {
        assert_eq!(CliError::Usage("x".into()).exit_code(), 2);
        assert_eq!(CliError::Schema("x".into()).exit_code(), 1);
        assert_eq!(CliError::Io("x".into()).exit_code(), 1);
    }

    #[test]
    fn display_names_the_problem() {
        let e = CliError::Schema("minItems > maxItems".into());
        assert!(e.to_string().contains("minItems > maxItems"));
    }
}
