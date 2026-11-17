//! The intermediate grammar: a small regular-expression IR over **bytes**.
//!
//! This sits between the JSON Schema frontend (`schema.rs`) and the automaton
//! backend (`dfa.rs`). Keeping it separate means the schema frontend never has to
//! think about automata, and the automaton never has to think about JSON.
//!
//! The alphabet is `u8`, all 256 of them — never `char`. See docs/GRAMMAR.md
//! ("Why bytes, not characters"). Every construct here is closed over regular
//! languages, which is what makes the DFA conversion in `dfa.rs` possible.

/// A set of bytes, as a 256-bit bitmap (4 × u64 = 32 bytes, `Copy`).
///
/// This is the label on an automaton edge. A dense bitmap rather than a range
/// list because the hot operation is `contains(byte)` during construction and we
/// want it branch-free.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ByteSet {
    bits: [u64; 4],
}

impl ByteSet {
    pub const fn empty() -> Self {
        ByteSet { bits: [0; 4] }
    }

    /// All 256 byte values. Used for the `.` in a string body.
    pub const fn full() -> Self {
        ByteSet {
            bits: [u64::MAX; 4],
        }
    }

    pub fn single(b: u8) -> Self {
        let mut s = Self::empty();
        s.insert(b);
        s
    }

    /// Inclusive range, e.g. `range(b'0', b'9')`.
    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }

    pub fn insert(&mut self, b: u8) {
        self.bits[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    pub fn contains(&self, b: u8) -> bool {
        self.bits[(b >> 6) as usize] & (1u64 << (b & 63)) != 0
    }

    pub fn union(mut self, other: &ByteSet) -> Self {
        for i in 0..4 {
            self.bits[i] |= other.bits[i];
        }
        self
    }

    /// Set difference: everything in `self` that is not in `other`.
    pub fn subtract(mut self, other: &ByteSet) -> Self {
        for i in 0..4 {
            self.bits[i] &= !other.bits[i];
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    pub fn len(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0u16..=255).filter_map(move |b| {
            let b = b as u8;
            if self.contains(b) {
                Some(b)
            } else {
                None
            }
        })
    }
}

/// A regular expression over bytes.
///
/// Deliberately minimal: this is everything needed to express the JSON subset in
/// `schema.rs`, and nothing more. Anything not expressible here is outside the
/// regular languages and is a documented limitation, not a bug.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// Matches the empty string. Identity for `Concat`.
    Empty,
    /// Matches exactly one byte drawn from the set.
    Class(ByteSet),
    /// Matches a fixed byte string. `Literal(b"null")` is 4 edges in sequence.
    Literal(Vec<u8>),
    Concat(Vec<Pattern>),
    Alt(Vec<Pattern>),
    /// `inner` repeated between `min` and `max` times (`None` = unbounded).
    ///
    /// Bounded repetition is expanded by *copying* the inner pattern during NFA
    /// construction — this is why `maxLength: 10000` would be expensive, and why
    /// we bound nesting depth rather than allowing arbitrary recursion.
    Repeat {
        inner: Box<Pattern>,
        min: u32,
        max: Option<u32>,
    },
}

impl Pattern {
    pub fn lit(s: &str) -> Pattern {
        Pattern::Literal(s.as_bytes().to_vec())
    }

    pub fn class(set: ByteSet) -> Pattern {
        Pattern::Class(set)
    }

    pub fn concat(parts: Vec<Pattern>) -> Pattern {
        Pattern::Concat(parts)
    }

    pub fn alt(parts: Vec<Pattern>) -> Pattern {
        Pattern::Alt(parts)
    }

    pub fn star(self) -> Pattern {
        Pattern::Repeat {
            inner: Box::new(self),
            min: 0,
            max: None,
        }
    }

    pub fn plus(self) -> Pattern {
        Pattern::Repeat {
            inner: Box::new(self),
            min: 1,
            max: None,
        }
    }

    pub fn opt(self) -> Pattern {
        Pattern::Repeat {
            inner: Box::new(self),
            min: 0,
            max: Some(1),
        }
    }

    pub fn repeat(self, min: u32, max: Option<u32>) -> Pattern {
        Pattern::Repeat {
            inner: Box::new(self),
            min,
            max,
        }
    }
}
