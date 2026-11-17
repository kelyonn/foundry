//! The KV cache.
//!
//! # Why this exists
//!
//! Attention at position `p` needs the keys and values of every position `≤ p`.
//! Those are a pure function of the tokens already fixed, so recomputing them each
//! step turns generation into O(n²) *projections* on top of the O(n²) attention
//! that is unavoidable. Caching them makes decoding one token cost one row of
//! projections instead of `n`.
//!
//! # Why it is a *KV* cache and not a QKV cache
//!
//! Queries are not cached, and cannot be: the query for position `p` is used once,
//! at step `p`, and never referred to again. Keys and values are read by every
//! later position. That asymmetry is the whole reason the cache holds exactly two
//! of the three projections.
//!
//! # Layout
//!
//! `[layer][kv_head][position][head_dim]`, preallocated at `max_seq`, per
//! docs/ARCHITECTURE.md. Position is the second-innermost axis so that one head's
//! history is contiguous — that is the order attention scans it in.
//!
//! Preallocated rather than grown, because a `Vec` doubling mid-generation means a
//! realloc and a copy at exactly the moment throughput is being measured.
//!
//! # Size
//!
//! `layers × 2 × kv_heads × max_seq × head_dim × 4` bytes. For this model at
//! `max_seq = 2048`: `24 × 2 × 2 × 2048 × 64 × 4` ≈ **50 MB**.
//!
//! Note the `kv_heads = 2`, not `num_attention_heads = 14`. That is what grouped-
//! query attention actually buys: the cache is 7× smaller than it would be with
//! multi-head attention. It does **not** reduce attention FLOPs — every one of the
//! 14 query heads still does its own full dot product against the shared keys. GQA
//! is a memory-bandwidth optimization wearing an attention-shape costume.

#[derive(Debug)]
pub struct CacheFull {
    pub max_seq: usize,
    pub wanted: usize,
}

impl std::fmt::Display for CacheFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KV cache holds {} positions, needed {}",
            self.max_seq, self.wanted
        )
    }
}

impl std::error::Error for CacheFull {}

pub struct KvCache {
    /// `k[layer]` is `[kv_head][max_seq][head_dim]`, flattened.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
    max_seq: usize,
    n_kv: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(layers: usize, n_kv: usize, head_dim: usize, max_seq: usize) -> KvCache {
        let per_layer = n_kv * max_seq * head_dim;
        KvCache {
            k: (0..layers).map(|_| vec![0.0; per_layer]).collect(),
            v: (0..layers).map(|_| vec![0.0; per_layer]).collect(),
            len: 0,
            max_seq,
            n_kv,
            head_dim,
        }
    }

    /// Positions currently held — i.e. the absolute position the next token takes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn size_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * self.n_kv * self.max_seq * self.head_dim * 4
    }

    /// Forget everything. Does not free — the point of preallocating is reuse.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append `seq` positions of keys and values for one layer.
    ///
    /// `k_hsd` and `v_hsd` are `[kv_head][seq * head_dim]`, i.e. exactly what
    /// `split_heads` produces. Writing happens at `self.len`, so the caller must
    /// advance the length once per *token step*, not once per layer — see
    /// `advance`.
    pub fn append_layer(&mut self, layer: usize, k_hsd: &[Vec<f32>], v_hsd: &[Vec<f32>]) {
        let seq = k_hsd[0].len() / self.head_dim;
        let base = self.len;
        for h in 0..self.n_kv {
            for s in 0..seq {
                let dst = (h * self.max_seq + base + s) * self.head_dim;
                let src = s * self.head_dim;
                self.k[layer][dst..dst + self.head_dim]
                    .copy_from_slice(&k_hsd[h][src..src + self.head_dim]);
                self.v[layer][dst..dst + self.head_dim]
                    .copy_from_slice(&v_hsd[h][src..src + self.head_dim]);
            }
        }
    }

    /// All cached keys for one layer and head, as `[total_positions][head_dim]`.
    pub fn keys(&self, layer: usize, head: usize, total: usize) -> &[f32] {
        let start = head * self.max_seq * self.head_dim;
        &self.k[layer][start..start + total * self.head_dim]
    }

    pub fn values(&self, layer: usize, head: usize, total: usize) -> &[f32] {
        let start = head * self.max_seq * self.head_dim;
        &self.v[layer][start..start + total * self.head_dim]
    }

    /// Commit `seq` new positions. Called once after all layers have appended.
    pub fn advance(&mut self, seq: usize) -> Result<(), CacheFull> {
        if self.len + seq > self.max_seq {
            return Err(CacheFull {
                max_seq: self.max_seq,
                wanted: self.len + seq,
            });
        }
        self.len += seq;
        Ok(())
    }

    /// Would `seq` more positions fit?
    pub fn has_room_for(&self, seq: usize) -> bool {
        self.len + seq <= self.max_seq
    }
}
