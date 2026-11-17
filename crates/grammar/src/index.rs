//! The vocabulary index and the masking loop — where the automaton meets the
//! sampler.
//!
//! The question asked once per decoded token:
//!
//! > Given DFA state `s`, which of the vocabulary's tokens can I emit without
//! > leaving the language?
//!
//! Answering it from scratch costs one `step_token` per token in the vocabulary.
//! Answering it from a cache costs one hash lookup. The whole design of this
//! module is about making sure we mostly do the second thing.

use crate::dfa::{Dfa, StateId};
use crate::vocab::Vocab;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// A bitset over token ids: bit `i` set means token `i` is legal.
///
/// One bit per token — 151,936 tokens is 2,374 `u64`s, about 19 KB per cached
/// state. Cheap enough that caching every state we actually visit is a
/// non-decision; the reason we do not precompute *all* states is time, not space.
#[derive(Clone)]
pub struct TokenMask {
    bits: Vec<u64>,
    allowed: u32,
}

impl TokenMask {
    fn new(vocab_size: usize) -> Self {
        TokenMask {
            bits: vec![0; vocab_size.div_ceil(64)],
            allowed: 0,
        }
    }

    #[inline]
    fn set(&mut self, id: u32) {
        let w = (id >> 6) as usize;
        let bit = 1u64 << (id & 63);
        if self.bits[w] & bit == 0 {
            self.bits[w] |= bit;
            self.allowed += 1;
        }
    }

    #[inline]
    pub fn allows(&self, id: u32) -> bool {
        self.bits[(id >> 6) as usize] & (1u64 << (id & 63)) != 0
    }

    /// How many tokens are legal here. This number is the story: at a state in
    /// the middle of a fixed key like `"name"`, it is often 1.
    pub fn allowed_count(&self) -> u32 {
        self.allowed
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, &word)| {
            (0..64u32).filter_map(move |b| {
                if word & (1u64 << b) != 0 {
                    Some(w as u32 * 64 + b)
                } else {
                    None
                }
            })
        })
    }
}

#[derive(Default, Clone, Debug)]
pub struct IndexStats {
    /// States whose mask we computed from scratch.
    pub builds: u32,
    /// Lookups served from the cache.
    pub hits: u32,
    /// Total `step_token` calls across all builds — the real work done.
    pub tokens_scanned: u64,
    pub build_time: Duration,
}

/// Lazily-built, cached map from DFA state to legal token set.
pub struct VocabIndex {
    cache: HashMap<StateId, Rc<TokenMask>>,
    pub stats: IndexStats,
}

impl VocabIndex {
    pub fn new() -> Self {
        VocabIndex {
            cache: HashMap::new(),
            stats: IndexStats::default(),
        }
    }

    /// The legal token set at `state`, computed on first visit and cached after.
    ///
    /// Lazy because the full table is `num_states × vocab_size` membership tests
    /// and a typical generation visits a small fraction of the states. Paying for
    /// the states you actually reach turns a multi-second precompute into a few
    /// milliseconds amortised across a generation — and the second generation
    /// against the same schema pays nothing at all.
    ///
    /// The EOS token is deliberately *not* included here, in any state. Its
    /// legality depends on whether the current state is accepting, which is a
    /// property of the state, not of the vocabulary — so it is applied at mask
    /// time in `ConstrainedDecoder::apply_mask`. Baking it in here would mean
    /// caching two variants of every mask for no reason.
    pub fn mask_for(&mut self, dfa: &Dfa, vocab: &Vocab, state: StateId) -> Rc<TokenMask> {
        if let Some(mask) = self.cache.get(&state) {
            self.stats.hits += 1;
            return Rc::clone(mask);
        }

        let started = Instant::now();
        let mut mask = TokenMask::new(vocab.len());
        for id in 0..vocab.len() as u32 {
            if vocab.is_special(id) {
                continue;
            }
            let bytes = vocab.bytes(id);
            if bytes.is_empty() {
                continue;
            }
            // Legal iff every byte of the token stays out of the dead state.
            // Note the token may pass through several states on the way; only
            // the landing state matters to the caller.
            if dfa.step_token(state, bytes).is_some() {
                mask.set(id);
            }
        }

        self.stats.builds += 1;
        self.stats.tokens_scanned += vocab.len() as u64;
        self.stats.build_time += started.elapsed();

        let mask = Rc::new(mask);
        self.cache.insert(state, Rc::clone(&mask));
        mask
    }

    pub fn cached_states(&self) -> usize {
        self.cache.len()
    }
}

impl Default for VocabIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum DecodeError {
    /// No token in the vocabulary can continue the document from here, and here
    /// is not a valid stopping point. The grammar admits a continuation but the
    /// tokenizer has no way to spell it.
    ///
    /// Detected explicitly and raised loudly rather than silently emitting
    /// something invalid or looping forever. A crash in a demo is recoverable;
    /// a hang is not, and invalid output would defeat the entire premise.
    DeadEnd { state: StateId, emitted: String },
    /// The sampler picked a token the mask forbade — i.e. someone sampled from
    /// unmasked logits. Checked rather than assumed, because this is the failure
    /// mode that would quietly break the guarantee.
    IllegalToken { id: u32, state: StateId },
    /// Logit slice length does not match the vocabulary.
    LogitLenMismatch { got: usize, want: usize },
    /// A token was offered after the document was already complete.
    AlreadyFinished,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::DeadEnd { state, emitted } => write!(
                f,
                "dead end at DFA state {state}: no legal token, and the document is incomplete. \
                 emitted so far: {emitted:?}"
            ),
            DecodeError::IllegalToken { id, state } => {
                write!(
                    f,
                    "token {id} is illegal in DFA state {state} — sampled from unmasked logits?"
                )
            }
            DecodeError::LogitLenMismatch { got, want } => {
                write!(f, "logits have length {got}, vocabulary has {want} tokens")
            }
            DecodeError::AlreadyFinished => write!(f, "generation already finished"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Drives one constrained generation: holds the automaton, the current state,
/// and the bytes emitted so far.
pub struct ConstrainedDecoder<'a> {
    dfa: &'a Dfa,
    vocab: &'a Vocab,
    index: &'a mut VocabIndex,
    state: StateId,
    output: Vec<u8>,
    finished: bool,
}

impl<'a> ConstrainedDecoder<'a> {
    pub fn new(dfa: &'a Dfa, vocab: &'a Vocab, index: &'a mut VocabIndex) -> Self {
        let state = dfa.start();
        ConstrainedDecoder {
            dfa,
            vocab,
            index,
            state,
            output: Vec::new(),
            finished: false,
        }
    }

    pub fn state(&self) -> StateId {
        self.state
    }

    /// True when the bytes emitted so far are a complete, valid document.
    /// Only here is EOS permitted.
    pub fn is_accepting(&self) -> bool {
        self.dfa.is_accepting(self.state)
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn output(&self) -> &[u8] {
        &self.output
    }

    /// Set every illegal token's logit to `-inf`, in place.
    ///
    /// **Why `-inf` before softmax rather than zeroing probabilities after.**
    /// `exp(-inf) == 0` exactly, so an illegal token's probability is zero and
    /// the remaining mass renormalises for free inside the softmax we were going
    /// to run anyway. Zeroing after softmax leaves the distribution unnormalised
    /// and forces a second pass; worse, it interacts badly with top-k and top-p,
    /// which would have already selected their candidate set from a distribution
    /// including tokens that are about to be deleted. Masking first means every
    /// downstream sampling knob operates on the legal distribution.
    ///
    /// # The logit vector may be wider than the vocabulary
    ///
    /// For Qwen2.5, `config.json` says `vocab_size: 151936` — but `tokenizer.json`
    /// defines only 151,665 tokens (151,643 base plus 22 added). The embedding
    /// matrix is padded up to 128 × 1187 so the output projection is a
    /// nicely-aligned matmul, and the extra 271 rows correspond to no token at all.
    ///
    /// The forward pass therefore produces 151,936 logits, of which the last 271
    /// are meaningless. They are not tokens; nothing can be decoded from them. So
    /// this accepts any slice at least as wide as the vocabulary and masks the
    /// tail unconditionally. Rejecting the wider slice would be wrong, and
    /// silently ignoring the tail would leave 271 samplable ids that decode to
    /// nothing.
    ///
    /// Returns the number of tokens left legal.
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<u32, DecodeError> {
        if self.finished {
            return Err(DecodeError::AlreadyFinished);
        }
        if logits.len() < self.vocab.len() {
            return Err(DecodeError::LogitLenMismatch {
                got: logits.len(),
                want: self.vocab.len(),
            });
        }

        let mask = self.index.mask_for(self.dfa, self.vocab, self.state);
        let accepting = self.is_accepting();
        let eos = self.vocab.eos();

        if mask.allowed_count() == 0 && !accepting {
            return Err(DecodeError::DeadEnd {
                state: self.state,
                emitted: String::from_utf8_lossy(&self.output).into_owned(),
            });
        }

        let vocab_len = self.vocab.len();
        for (id, logit) in logits.iter_mut().enumerate() {
            let legal = if id >= vocab_len {
                false // padding row, not a token
            } else if id as u32 == eos {
                accepting
            } else {
                mask.allows(id as u32)
            };
            if !legal {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(mask.allowed_count() + u32::from(accepting))
    }

    /// Advance the automaton by a chosen token.
    ///
    /// Rechecks legality rather than trusting the caller — the guarantee is only
    /// worth something if it cannot be bypassed by a sampler that forgot to mask.
    pub fn accept_token(&mut self, id: u32) -> Result<(), DecodeError> {
        if self.finished {
            return Err(DecodeError::AlreadyFinished);
        }
        if id == self.vocab.eos() {
            if !self.is_accepting() {
                return Err(DecodeError::IllegalToken {
                    id,
                    state: self.state,
                });
            }
            self.finished = true;
            return Ok(());
        }
        if self.vocab.is_special(id) {
            return Err(DecodeError::IllegalToken {
                id,
                state: self.state,
            });
        }

        let bytes = self.vocab.bytes(id);
        match self.dfa.step_token(self.state, bytes) {
            Some(next) => {
                self.output.extend_from_slice(bytes);
                self.state = next;
                Ok(())
            }
            None => Err(DecodeError::IllegalToken {
                id,
                state: self.state,
            }),
        }
    }
}
