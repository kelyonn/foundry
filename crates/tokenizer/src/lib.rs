//! Byte-level BPE: encode, decode, round-trip.
//!
//! Three stages, and they are separable — which matters, because when IDs disagree
//! with HuggingFace the first question is always *which stage*:
//!
//! 1. **Pre-tokenization.** Split the text into chunks that BPE is never allowed to
//!    merge across. This is a regex in the reference implementation; here it is a
//!    hand-written scanner, because the pattern uses features (lookahead, Unicode
//!    property classes) that would otherwise mean pulling in a regex engine.
//! 2. **Byte-level mapping.** Each chunk's UTF-8 bytes are mapped to printable
//!    characters, so the merge table can be plain text. See `byte_encoder`.
//! 3. **The BPE merge loop.** Repeatedly apply the highest-priority adjacent merge
//!    until none applies.
//!
//! Decoding is the inverse and is far simpler — which is why `crates/grammar` only
//! ever needed id→bytes and could be built with no encoder at all.

pub mod pretokenize;

use std::collections::HashMap;

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tokenizer error: {}", self.0)
    }
}

impl std::error::Error for TokenizerError {}

pub struct Tokenizer {
    /// id → the bytes that token spells.
    tokens: Vec<Vec<u8>>,
    /// The byte-level *string* form of each token → id, for the merge loop.
    token_to_id: HashMap<String, u32>,
    /// (left, right) → rank. Lower rank wins; rank is the line number in the
    /// merges list, i.e. the order the merges were learned during training.
    merges: HashMap<(String, String), u32>,
    special: Vec<bool>,
    /// Special token text → id, so control tokens can be emitted deliberately.
    specials_by_text: HashMap<String, u32>,
    eos: u32,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn bytes(&self, id: u32) -> &[u8] {
        &self.tokens[id as usize]
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special[id as usize]
    }

    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.specials_by_text.get(text).copied()
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Tokenizer, TokenizerError> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| TokenizerError(format!("cannot read tokenizer.json: {e}")))?;
        let json: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| TokenizerError(format!("invalid JSON: {e}")))?;

        let model = json
            .get("model")
            .ok_or_else(|| TokenizerError("missing `model`".into()))?;
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TokenizerError("missing `model.vocab`".into()))?;

        let (byte_encoder, byte_decoder) = byte_alphabet();

        let mut max_id = 0u32;
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::with_capacity(vocab.len());

        for (text, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| TokenizerError(format!("vocab id for {text:?} is not an integer")))?
                as u32;
            max_id = max_id.max(id);
            token_to_id.insert(text.clone(), id);

            let mut bytes = Vec::with_capacity(text.len());
            for ch in text.chars() {
                match byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            entries.push((id, bytes));
        }

        // Merges are ordered: index 0 was learned first and binds most tightly.
        let mut merges = HashMap::new();
        if let Some(list) = model.get("merges").and_then(|m| m.as_array()) {
            for (rank, entry) in list.iter().enumerate() {
                // Two encodings exist in the wild: "a b" and ["a", "b"].
                let pair = match entry {
                    serde_json::Value::String(s) => {
                        let mut it = s.splitn(2, ' ');
                        match (it.next(), it.next()) {
                            (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                            _ => continue,
                        }
                    }
                    serde_json::Value::Array(a) if a.len() == 2 => {
                        match (a[0].as_str(), a[1].as_str()) {
                            (Some(x), Some(y)) => (x.to_string(), y.to_string()),
                            _ => continue,
                        }
                    }
                    _ => continue,
                };
                merges.insert(pair, rank as u32);
            }
        }
        if merges.is_empty() {
            return Err(TokenizerError("no merges found in tokenizer.json".into()));
        }

        let mut specials: Vec<(u32, String)> = Vec::new();
        if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                let (Some(id), Some(content)) = (
                    tok.get("id").and_then(|v| v.as_u64()),
                    tok.get("content").and_then(|v| v.as_str()),
                ) else {
                    continue;
                };
                max_id = max_id.max(id as u32);
                specials.push((id as u32, content.to_string()));
            }
        }

        let n = max_id as usize + 1;
        let mut tokens = vec![Vec::new(); n];
        let mut special = vec![false; n];
        for (id, bytes) in entries {
            tokens[id as usize] = bytes;
        }

        let mut eos = None;
        let mut specials_by_text = HashMap::new();
        for (id, content) in &specials {
            tokens[*id as usize] = content.as_bytes().to_vec();
            special[*id as usize] = true;
            specials_by_text.insert(content.clone(), *id);
            if content == "<|im_end|>" || (content == "<|endoftext|>" && eos.is_none()) {
                eos = Some(*id);
            }
        }
        for (id, slot) in special.iter_mut().enumerate() {
            if tokens[id].is_empty() {
                *slot = true;
            }
        }

        let eos = eos.ok_or_else(|| TokenizerError("could not identify an EOS token".into()))?;

        Ok(Tokenizer {
            tokens,
            token_to_id,
            merges,
            special,
            specials_by_text,
            eos,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Encode text to token ids, treating control-token text as **literal text**.
    ///
    /// `encode("<|im_end|>")` yields the bytes of that string, not the control id.
    ///
    /// # Why this is the default, and where it diverges from HuggingFace
    ///
    /// HuggingFace's `encode` scans for added-token strings *before* BPE, so
    /// `<|im_end|>` appearing anywhere in the input — including in text a user
    /// typed — becomes the real end-of-turn token. That is a live prompt-injection
    /// vector: paste the right string into a chat box and you end the assistant's
    /// turn. The reference itself offers `split_special_tokens=True` to disable it,
    /// which is exactly this function.
    ///
    /// So the two behaviours both exist here, and the choice is which one is easy
    /// to reach by accident. Untrusted content is the common case, so it gets the
    /// short name; `encode_with_specials` is for trusted prompt templates and is
    /// what the parity tests exercise, since it is what HuggingFace does.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for chunk in pretokenize::split(text) {
            self.encode_chunk(chunk, &mut out);
        }
        out
    }

    /// Encode text, recognising control-token strings as control tokens.
    ///
    /// This matches HuggingFace's default `encode`. Use it for prompt templates you
    /// control; never for content that came from a user. See `encode`.
    ///
    /// Matching is longest-first at each position, so a token whose text is a
    /// prefix of another cannot shadow it.
    pub fn encode_with_specials(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            // Find the earliest control token; ties broken by length, longest wins.
            //
            // Zero-length needles are skipped outright. `str::find("")` always
            // succeeds at position 0, so an added token with empty `content` (a
            // malformed or adversarial tokenizer.json, not this checkpoint — see
            // `refuses_to_hang_on_an_empty_added_token`) would match at `at = 0`
            // with `len = 0` on every iteration, `rest` would never shrink, and
            // the loop would spin forever. A hang is a worse failure than an
            // empty needle simply never being reachable as a real control token.
            let mut hit: Option<(usize, usize, u32)> = None; // (start, len, id)
            for (needle, &id) in &self.specials_by_text {
                if needle.is_empty() {
                    continue;
                }
                if let Some(at) = rest.find(needle.as_str()) {
                    let better = match hit {
                        None => true,
                        Some((best_at, best_len, _)) => {
                            at < best_at || (at == best_at && needle.len() > best_len)
                        }
                    };
                    if better {
                        hit = Some((at, needle.len(), id));
                    }
                }
            }

            match hit {
                Some((at, len, id)) => {
                    if at > 0 {
                        for chunk in pretokenize::split(&rest[..at]) {
                            self.encode_chunk(chunk, &mut out);
                        }
                    }
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    for chunk in pretokenize::split(rest) {
                        self.encode_chunk(chunk, &mut out);
                    }
                    break;
                }
            }
        }

        out
    }

    /// BPE within a single pre-tokenized chunk.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        // Map the chunk's raw bytes into the printable alphabet, one symbol per byte.
        let mut symbols: Vec<String> = chunk
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[b as usize].to_string())
            .collect();

        if symbols.is_empty() {
            return;
        }

        // Repeatedly merge the adjacent pair with the lowest rank. O(n²) in the
        // chunk length, but pre-tokenization keeps chunks to a handful of bytes,
        // so this never becomes the bottleneck.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
            if symbols.len() == 1 {
                break;
            }
        }

        for sym in symbols {
            match self.token_to_id.get(&sym) {
                Some(&id) => out.push(id),
                None => {
                    // Should be unreachable: every single byte is in the vocabulary,
                    // and merges only ever produce vocabulary entries. If it happens,
                    // fall back to per-byte rather than dropping text on the floor.
                    for ch in sym.chars() {
                        let b = self.byte_decoder[&ch];
                        let single = self.byte_encoder[b as usize].to_string();
                        if let Some(&id) = self.token_to_id.get(&single) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Decode ids back to bytes. Concatenation — that is the whole algorithm.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(&self.tokens[id as usize]);
        }
        out
    }

    /// Decode to a string. Lossy, because a *prefix* of a generation can end
    /// mid-character — that is normal during streaming, not an error.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// GPT-2's byte-level alphabet, in both directions.
///
/// All 256 byte values are mapped onto printable Unicode characters so the vocab
/// and merge tables can live in a JSON file as text. Bytes that are already
/// printable map to themselves; the other 68 are shifted to U+0100 and upward.
/// This is why a space appears as `Ġ` (0x20 → U+0120) in the vocabulary.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut direct = [false; 256];
    for b in b'!'..=b'~' {
        direct[b as usize] = true;
    }
    for b in 0xA1u8..=0xAC {
        direct[b as usize] = true;
    }
    for b in 0xAEu8..=0xFF {
        direct[b as usize] = true;
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    let mut shifted = 0u32;
    for b in 0..=255u8 {
        let ch = if direct[b as usize] {
            b as char
        } else {
            let c = char::from_u32(256 + shifted).expect("in range");
            shifted += 1;
            c
        };
        encoder[b as usize] = ch;
        decoder.insert(ch, b);
    }
    (encoder, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the empty-needle hang in `encode_with_specials`.
    ///
    /// Not reachable with the real Qwen `tokenizer.json` (checked: zero
    /// empty-content `added_tokens`), but nothing stops a malformed or
    /// adversarial one from declaring `{"content": "", "id": N}`. Before the
    /// fix, `str::find("")` matches at position 0 with length 0 on every
    /// iteration, `rest` never shrinks, and the loop spins forever — a hang, not
    /// a crash, and per docs/RULES.md's stated position on the grammar crate's
    /// dead-end handling, a hang is the worse failure of the two.
    ///
    /// Run on a background thread with a bounded join so that if this bug is
    /// ever reintroduced, the test *fails on a timeout* rather than hanging the
    /// whole suite — a assert inside an infinite loop never gets the chance to
    /// run.
    #[test]
    fn refuses_to_hang_on_an_empty_added_token() {
        let (byte_encoder, byte_decoder) = byte_alphabet();
        let mut specials_by_text = HashMap::new();
        specials_by_text.insert(String::new(), 0u32); // the malformed entry
        specials_by_text.insert("<|im_end|>".to_string(), 1u32);

        let tk = Tokenizer {
            tokens: vec![b"<|im_end|>".to_vec(), b"<|im_end|>".to_vec()],
            token_to_id: HashMap::new(),
            merges: HashMap::new(),
            special: vec![true, true],
            specials_by_text,
            eos: 1,
            byte_encoder,
            byte_decoder,
        };

        let handle = std::thread::spawn(move || {
            tk.encode_with_specials("hello <|im_end|> world");
        });
        // Generous relative to how fast this should actually run (microseconds);
        // tight enough that a real hang fails the test instead of the harness.
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(
            handle.is_finished(),
            "encode_with_specials did not terminate — the empty-needle hang is back"
        );
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (enc, dec) = byte_alphabet();
        assert_eq!(dec.len(), 256, "every byte needs a distinct character");
        for b in 0..=255u8 {
            assert_eq!(dec[&enc[b as usize]], b);
        }
        // The documented landmark: space becomes Ġ.
        assert_eq!(enc[0x20], 'Ġ');
    }
}
