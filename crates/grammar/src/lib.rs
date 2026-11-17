//! Grammar-constrained decoding.
//!
//! ```text
//! JSON Schema  ──schema.rs──▶  Pattern  ──dfa.rs──▶  DFA over bytes
//!                                                        │
//!                                    index.rs ◀──────────┘
//!                                       │  state → legal token bitmask (lazy)
//!                                       ▼
//!                              logits[illegal] = -inf  →  sample  →  advance
//! ```
//!
//! The guarantee: at every step, a token is legal only if the bytes it spells
//! keep the document on a path to a complete, valid parse. Since the sampler can
//! only choose legal tokens, and generation can only stop in an accepting state,
//! invalid output is not unlikely — it is unreachable.
//!
//! Nothing here knows what a model is. The interface to the rest of Foundry is a
//! mutable slice of logits.

pub mod dfa;
pub mod index;
pub mod pattern;
pub mod schema;
pub mod vocab;

pub use dfa::{Dfa, StateId};
pub use index::{ConstrainedDecoder, DecodeError, IndexStats, TokenMask, VocabIndex};
pub use pattern::{ByteSet, Pattern};
pub use schema::{compile_schema, SchemaError};
pub use vocab::Vocab;

/// A compiled grammar: the automaton plus the pattern it came from.
pub struct Grammar {
    pub dfa: Dfa,
}

impl Grammar {
    /// Compile a JSON Schema document.
    pub fn from_schema(schema: &serde_json::Value) -> Result<Grammar, SchemaError> {
        let pattern = compile_schema(schema)?;
        Ok(Grammar {
            dfa: Dfa::compile(&pattern),
        })
    }

    pub fn from_schema_str(schema: &str) -> Result<Grammar, Box<dyn std::error::Error>> {
        let value: serde_json::Value = serde_json::from_str(schema)?;
        Ok(Grammar::from_schema(&value)?)
    }

    /// Compile a raw pattern, bypassing the JSON frontend. Used by tests.
    pub fn from_pattern(pattern: &Pattern) -> Grammar {
        Grammar {
            dfa: Dfa::compile(pattern),
        }
    }

    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }
}
