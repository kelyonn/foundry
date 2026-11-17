//! The tokenizer vocabulary, as far as the grammar layer needs to care about it.
//!
//! The grammar layer needs exactly one thing from a tokenizer: **token id → the
//! bytes that token spells**. Not the merges, not the pre-tokenizer regex, not
//! the encoder. Keeping the dependency this thin is why constrained decoding can
//! be developed and tested with no model and no encoder in the loop.

use std::collections::HashMap;

/// Token id → its byte spelling.
pub struct Vocab {
    tokens: Vec<Vec<u8>>,
    /// Control tokens (`<|im_end|>`, `<|endoftext|>`, …). These are never legal
    /// as grammar content — their *surface text* is not what they mean. Emitting
    /// the literal bytes `<|im_end|>` into a JSON string would be a bug, not a
    /// generation.
    special: Vec<bool>,
    eos: u32,
}

impl Vocab {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn bytes(&self, id: u32) -> &[u8] {
        &self.tokens[id as usize]
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special[id as usize]
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Best-effort rendering for logs and the demo. Tokens are byte strings and
    /// need not be valid UTF-8 on their own, which is the entire point — so this
    /// is lossy on purpose and must never be used for anything but display.
    pub fn display(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.tokens[id as usize]).into_owned()
    }

    /// Build directly from token spellings.
    ///
    /// Lets a caller that already has a tokenizer hand its table over, rather than
    /// parsing `tokenizer.json` a second time. More importantly it removes any
    /// chance of the two disagreeing about an id — a masker and a decoder that
    /// number tokens differently would produce output that is valid according to
    /// the automaton and garbage on screen.
    pub fn from_tokens(tokens: Vec<Vec<u8>>, special: Vec<bool>, eos: u32) -> Vocab {
        assert_eq!(
            tokens.len(),
            special.len(),
            "token and special-flag tables must be the same length"
        );
        assert!((eos as usize) < tokens.len(), "eos id is out of range");
        Vocab {
            tokens,
            special,
            eos,
        }
    }

    /// Load from a HuggingFace `tokenizer.json`.
    ///
    /// Reads `model.vocab` (token text → id) and `added_tokens` (the control
    /// tokens). Token text is in GPT-2's byte-level alphabet, so it goes through
    /// `byte_decoder` to recover the real bytes — see below.
    pub fn from_tokenizer_json(path: &str) -> Result<Vocab, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&raw)?;

        let vocab_map = json
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or("tokenizer.json: missing model.vocab")?;

        let decoder = byte_decoder();

        let mut max_id = 0u32;
        let mut pairs: Vec<(u32, Vec<u8>)> = Vec::with_capacity(vocab_map.len());
        for (text, id) in vocab_map {
            let id = id.as_u64().ok_or("vocab id is not an integer")? as u32;
            max_id = max_id.max(id);
            let mut bytes = Vec::with_capacity(text.len());
            for ch in text.chars() {
                match decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // A character outside the byte-level alphabet means this
                    // entry is not a byte-level token (some vocabs mix them in).
                    // Fall back to its UTF-8 encoding.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            pairs.push((id, bytes));
        }

        let mut specials: Vec<(u32, String)> = Vec::new();
        if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                let id = match tok.get("id").and_then(|v| v.as_u64()) {
                    Some(i) => i as u32,
                    None => continue,
                };
                let content = tok
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                max_id = max_id.max(id);
                specials.push((id, content));
            }
        }

        let n = max_id as usize + 1;
        let mut tokens = vec![Vec::new(); n];
        let mut special = vec![false; n];
        for (id, bytes) in pairs {
            tokens[id as usize] = bytes;
        }
        let mut eos = None;
        for (id, content) in &specials {
            tokens[*id as usize] = content.as_bytes().to_vec();
            special[*id as usize] = true;
            // Qwen2.5-Instruct ends a turn with <|im_end|>, so it wins outright.
            // <|endoftext|> is only a fallback, for a base model that has no
            // chat template.
            if content == "<|im_end|>" || (content == "<|endoftext|>" && eos.is_none()) {
                eos = Some(*id);
            }
        }

        // Ids present in neither map are holes in the vocabulary. They exist in
        // real tokenizers (reserved slots). Treat them as special so they are
        // never emitted.
        for id in 0..n {
            if tokens[id].is_empty() && !special[id] {
                special[id] = true;
            }
        }

        let eos = eos.ok_or("tokenizer.json: could not identify an EOS token")?;
        Ok(Vocab {
            tokens,
            special,
            eos,
        })
    }

    /// A small hand-built vocabulary, for tests and for running the demo without
    /// a downloaded tokenizer.
    ///
    /// Deliberately nasty: it contains multi-byte tokens that straddle JSON
    /// structure (`{"`, `":`, `",`), and tokens that are UTF-8 *fragments* —
    /// which is exactly the case a character-level DFA cannot handle.
    pub fn synthetic() -> Vocab {
        let mut tokens: Vec<Vec<u8>> = Vec::new();

        // Every single byte, so any document is at least spellable.
        for b in 0..=255u8 {
            tokens.push(vec![b]);
        }

        // Multi-byte merges of the kind BPE actually learns on JSON.
        for s in [
            "{\"",
            "\":",
            "\",",
            "\":\"",
            "\"}",
            "\"]",
            "true",
            "false",
            "null",
            "name",
            "age",
            "city",
            "email",
            "id",
            "score",
            "tags",
            "active",
            "London",
            "Kalyan",
            "Bengaluru",
            "\":[",
            "],\"",
            "},{",
            "0",
            "1",
            "2",
            "25",
            "42",
            "100",
            "3.14",
            "-",
        ] {
            tokens.push(s.as_bytes().to_vec());
        }

        // A three-byte character split across two tokens: ₹ is E2 82 B9.
        tokens.push(vec![0xE2, 0x82]); // first two bytes only — not valid UTF-8
        tokens.push(vec![0xB9]); // the continuation byte that finishes it
        tokens.push("₹".as_bytes().to_vec()); // and the whole thing as one token

        let eos = tokens.len() as u32;
        tokens.push(b"<|im_end|>".to_vec());

        let mut special = vec![false; tokens.len()];
        special[eos as usize] = true;

        Vocab {
            tokens,
            special,
            eos,
        }
    }
}

/// GPT-2's byte-level alphabet, reversed: printable character → the byte it stands for.
///
/// Byte-level BPE cannot put raw control bytes in a JSON vocabulary file, so it
/// maps all 256 byte values onto printable Unicode characters. Bytes that are
/// already printable map to themselves; the other 68 are shifted into U+0100 and
/// upward. This is why a space shows up as `Ġ` (0x20 → U+0120) in the vocab file.
fn byte_decoder() -> HashMap<char, u8> {
    let mut direct: Vec<u8> = Vec::new();
    direct.extend(b'!'..=b'~');
    direct.extend(0xA1u8..=0xAC);
    direct.extend(0xAEu8..=0xFF);

    let mut map = HashMap::with_capacity(256);
    let mut shifted = 0u32;
    for b in 0..=255u8 {
        if direct.contains(&b) {
            map.insert(b as char, b);
        } else {
            let ch = char::from_u32(256 + shifted).expect("in range");
            map.insert(ch, b);
            shifted += 1;
        }
    }
    map
}
