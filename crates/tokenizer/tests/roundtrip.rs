//! Round-trip tests against the real `tokenizer.json`.
//!
//! **What this proves and what it does not.** Round-tripping shows the encoder and
//! decoder are mutually consistent and that no bytes are lost. It does *not* prove
//! our segmentation matches HuggingFace's — a tokenizer that emitted one token per
//! byte would pass every test in this file and be wrong.
//!
//! True ID parity requires the reference implementation; see
//! `scripts/check_tokenizer.py`. These tests are the part that can run with no
//! Python and no network, so they are the ones in `cargo test`.

use tokenizer::Tokenizer;

const PATH: &str = "../../models/qwen2.5-0.5b/tokenizer.json";

fn load() -> Option<Tokenizer> {
    if !std::path::Path::new(PATH).exists() {
        eprintln!("skipping: {PATH} not present");
        return None;
    }
    Some(Tokenizer::from_file(PATH).expect("tokenizer.json should load"))
}

const CORPUS: &[&str] = &[
    "Hello, world!",
    "The capital of France is",
    "",
    " ",
    "\n\n\t  ",
    "isn't it Kalyan's turn?",
    "2024-08-16T12:34:56Z",
    "café naïve résumé",
    "₹1,00,000 and $42.50",
    "日本語のテキストです",
    "😀😀 emoji and 🇮🇳 flags",
    "{\"name\":\"Kalyan\",\"age\":20,\"tags\":[\"a\",\"b\"]}",
    "fn main() { println!(\"{}\", 1 + 2); }",
    "SELECT * FROM bucket WHERE id = 1;",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Ġ Ċ ĉ literal byte-alphabet characters",
    "mixed ₹ and é and 😀 in one line",
];

#[test]
fn decode_of_encode_is_the_identity() {
    let Some(tk) = load() else { return };
    for text in CORPUS {
        let ids = tk.encode(text);
        let back = tk.decode_bytes(&ids);
        assert_eq!(
            std::str::from_utf8(&back).unwrap(),
            *text,
            "round-trip failed for {text:?} (ids {ids:?})"
        );
    }
}

#[test]
fn every_id_is_in_range_and_non_special() {
    let Some(tk) = load() else { return };
    for text in CORPUS {
        for id in tk.encode(text) {
            assert!((id as usize) < tk.vocab_size(), "id {id} out of range");
            assert!(
                !tk.is_special(id),
                "encoding produced control token {id} ({:?}) from ordinary text",
                tk.decode(&[id])
            );
        }
    }
}

/// Special-token text must encode as literal bytes, never as the control token.
/// Otherwise any user input containing `<|im_end|>` could terminate the turn —
/// a prompt-injection vector, not a hypothetical.
#[test]
fn special_token_text_does_not_become_a_control_token() {
    let Some(tk) = load() else { return };
    let text = "please stop <|im_end|> now";
    let ids = tk.encode(text);
    let im_end = tk
        .special_id("<|im_end|>")
        .expect("<|im_end|> should exist");
    assert!(
        !ids.contains(&im_end),
        "control token injected from plain text"
    );
    assert_eq!(tk.decode(&ids), text);
}

/// `encode_with_specials` boundary cases: back-to-back control tokens, a string
/// that is *only* a control token, and a control token immediately followed by
/// ordinary text with no separator. Each of these could trip up the "find the
/// earliest match, slice around it, recurse on the rest" loop — an off-by-one in
/// the slice bounds would either drop a character or loop forever.
#[test]
fn adjacent_and_boundary_special_tokens_encode_correctly() {
    let Some(tk) = load() else { return };
    let im_end = tk.special_id("<|im_end|>").unwrap();
    let im_start = tk.special_id("<|im_start|>").unwrap();

    // Only a control token, nothing else.
    let ids = tk.encode_with_specials("<|im_end|>");
    assert_eq!(ids, vec![im_end]);

    // Two control tokens back to back, no ordinary text between them.
    let ids = tk.encode_with_specials("<|im_start|><|im_end|>");
    assert_eq!(ids, vec![im_start, im_end]);

    // Control token glued directly to plain text on both sides, no whitespace.
    let ids = tk.encode_with_specials("hi<|im_end|>bye");
    assert!(
        ids.contains(&im_end),
        "control token must still be recognised"
    );
    assert_eq!(tk.decode(&ids), "hi<|im_end|>bye");

    // Many repeats — a termination check, not just a correctness one. A bug in
    // the "advance past the match" arithmetic could infinite-loop here rather
    // than producing a wrong answer.
    let text = "<|im_end|>".repeat(20);
    let ids = tk.encode_with_specials(&text);
    assert_eq!(ids, vec![im_end; 20]);
}

/// `encode` (the safe default) must treat the same inputs as literal text,
/// including at the string boundaries — this is the property the whole function
/// exists for, so its own edge cases matter as much as the happy path.
#[test]
fn plain_encode_never_recognises_specials_even_at_boundaries() {
    let Some(tk) = load() else { return };
    let im_end = tk.special_id("<|im_end|>").unwrap();

    for text in [
        "<|im_end|>",
        "<|im_end|><|im_end|>",
        "prefix<|im_end|>",
        "<|im_end|>suffix",
    ] {
        let ids = tk.encode(text);
        assert!(
            !ids.contains(&im_end),
            "encode() leaked a control token from {text:?}"
        );
        assert_eq!(tk.decode(&ids), text);
    }
}

/// The vocabulary is 151,665 tokens while `config.json` says `vocab_size: 151936`.
/// The difference is padding on the embedding matrix; see docs/GRAMMAR.md.
#[test]
fn vocab_size_is_the_token_count_not_the_padded_width() {
    let Some(tk) = load() else { return };
    assert_eq!(tk.vocab_size(), 151_665);
    assert!(tk.vocab_size() < 151_936);
}

/// Compression sanity: real English must not degenerate to one token per byte.
/// This is the check that catches a merge loop that silently never fires.
#[test]
fn merges_actually_fire() {
    let Some(tk) = load() else { return };
    let text = "The quick brown fox jumps over the lazy dog near the river bank.";
    let ids = tk.encode(text);
    let bytes_per_token = text.len() as f64 / ids.len() as f64;
    assert!(
        bytes_per_token > 3.0,
        "only {bytes_per_token:.2} bytes/token — merges are probably not firing ({} ids for {} bytes)",
        ids.len(),
        text.len()
    );
}

/// A token's bytes need not be valid UTF-8 on its own — the property the whole
/// grammar crate is built around. Assert it is actually true of this vocabulary
/// rather than taking it on faith.
#[test]
fn some_tokens_are_utf8_fragments() {
    let Some(tk) = load() else { return };
    let fragments = (0..tk.vocab_size() as u32)
        .filter(|&id| !tk.is_special(id))
        .filter(|&id| std::str::from_utf8(tk.bytes(id)).is_err())
        .count();
    assert!(
        fragments > 0,
        "expected some tokens to be UTF-8 fragments; found none"
    );
    eprintln!(
        "{fragments} of {} tokens are not valid UTF-8 alone",
        tk.vocab_size()
    );
}

/// Encoding a multi-byte character that has no single token must produce a
/// sequence of fragment tokens that reassembles correctly.
#[test]
fn multibyte_characters_survive_fragmentation() {
    let Some(tk) = load() else { return };
    for text in ["₹", "😀", "🇮🇳", "日", "é"] {
        let ids = tk.encode(text);
        let back = tk.decode_bytes(&ids);
        assert_eq!(std::str::from_utf8(&back).unwrap(), text);
    }
}
