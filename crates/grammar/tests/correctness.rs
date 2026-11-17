//! Correctness oracles for constrained decoding.
//!
//! Per docs/TESTING.md, every stage needs something objective that says you're
//! wrong. Here that is `serde_json` — an independent parser that knows nothing
//! about our automaton. Validating our output with our own DFA would be circular
//! and would prove nothing.
//!
//! Per docs/RULES.md hard rule 5, constrained decoding correctness is binary. These
//! tests assert exact counts, never rates.

use grammar::pattern::{ByteSet, Pattern};
use grammar::{ConstrainedDecoder, Dfa, Grammar, Vocab, VocabIndex};

// ---------------------------------------------------------------------------
// Automaton construction
// ---------------------------------------------------------------------------

#[test]
fn literal_matches_exactly_itself() {
    let dfa = Dfa::compile(&Pattern::lit("null"));
    assert!(dfa.matches(b"null"));
    assert!(!dfa.matches(b"nul"));
    assert!(!dfa.matches(b"nulll"));
    assert!(!dfa.matches(b"NULL"));
    assert!(!dfa.matches(b""));
}

#[test]
fn alternation_and_repetition() {
    let dfa = Dfa::compile(&Pattern::Alt(vec![
        Pattern::lit("true"),
        Pattern::lit("false"),
    ]));
    assert!(dfa.matches(b"true"));
    assert!(dfa.matches(b"false"));
    assert!(!dfa.matches(b"truefalse"));

    // (ab){2,3}
    let dfa = Dfa::compile(&Pattern::lit("ab").repeat(2, Some(3)));
    assert!(!dfa.matches(b"ab"));
    assert!(dfa.matches(b"abab"));
    assert!(dfa.matches(b"ababab"));
    assert!(!dfa.matches(b"abababab"));

    let dfa = Dfa::compile(&Pattern::class(ByteSet::range(b'0', b'9')).plus());
    assert!(dfa.matches(b"7"));
    assert!(dfa.matches(b"1234567890"));
    assert!(!dfa.matches(b""));
    assert!(!dfa.matches(b"12a"));
}

/// Alive-but-doomed states must not survive compilation.
///
/// `ab` where the `b` branch is unreachable to any accept: after consuming `a`
/// there is exactly one continuation. If pruning were skipped, a state whose
/// every path dies would still look "alive" to the masker, which would allow a
/// token leading into it and then get stuck. This is the guarantee's foundation:
/// legality means *an accepting state is still reachable*, not merely *this one
/// byte was accepted*.
#[test]
fn doomed_states_are_pruned() {
    // (abc | abd) but with the `d` branch made unsatisfiable by an empty class.
    let unsatisfiable = Pattern::Concat(vec![
        Pattern::lit("ab"),
        Pattern::class(ByteSet::empty()),
        Pattern::lit("zzz"),
    ]);
    let live = Pattern::lit("abc");
    let dfa = Dfa::compile(&Pattern::Alt(vec![live, unsatisfiable]));

    assert!(dfa.matches(b"abc"));

    // Walking "ab" must leave us somewhere that only accepts 'c'. If the doomed
    // branch survived, some other byte would also be accepted here.
    let after_ab = dfa.step_token(dfa.start(), b"ab").expect("ab is a prefix");
    let legal: Vec<u8> = (0..=255u8)
        .filter(|&b| dfa.transition(after_ab, b).is_some())
        .collect();
    assert_eq!(legal, vec![b'c']);
}

#[test]
fn empty_language_compiles_to_a_dead_automaton() {
    let dfa = Dfa::compile(&Pattern::Alt(vec![]));
    assert!(!dfa.matches(b""));
    assert!(!dfa.matches(b"anything"));
    assert!(!dfa.is_accepting(dfa.start()));
}

// ---------------------------------------------------------------------------
// UTF-8: the reason the alphabet is u8
// ---------------------------------------------------------------------------

fn string_dfa() -> Dfa {
    let schema: serde_json::Value =
        serde_json::from_str(r#"{"type":"string","maxLength":8}"#).unwrap();
    Grammar::from_schema(&schema).unwrap().dfa
}

#[test]
fn accepts_well_formed_utf8_of_every_length() {
    let dfa = string_dfa();
    for s in ["\"a\"", "\"é\"", "\"₹\"", "\"😀\"", "\"aé₹😀\"", "\"\""] {
        assert!(dfa.matches(s.as_bytes()), "should accept {s:?}");
    }
}

#[test]
fn rejects_ill_formed_utf8() {
    let dfa = string_dfa();

    // A lone continuation byte.
    assert!(!dfa.matches(&[b'"', 0x80, b'"']));
    // A 2-byte lead with no continuation.
    assert!(!dfa.matches(&[b'"', 0xC3, b'"']));
    // Overlong encoding of '/' (0xC0 0xAF) — the classic security bug.
    assert!(!dfa.matches(&[b'"', 0xC0, 0xAF, b'"']));
    // A UTF-16 surrogate half (U+D800 as ED A0 80), illegal in UTF-8.
    assert!(!dfa.matches(&[b'"', 0xED, 0xA0, 0x80, b'"']));
    // Beyond U+10FFFF.
    assert!(!dfa.matches(&[b'"', 0xF5, 0x80, 0x80, 0x80, b'"']));
    // Raw control character — JSON requires these escaped.
    assert!(!dfa.matches(&[b'"', 0x0A, b'"']));
    // Unescaped quote and backslash.
    assert!(!dfa.matches(br#""a"b""#));
    assert!(!dfa.matches(b"\"\\\""));
}

/// `\uXXXX` escapes must respect UTF-16 surrogate pairing.
///
/// Regression test for a bug that only the *real* 151,936-token vocabulary could
/// reach: the synthetic vocabulary has no token that spells a lone surrogate, so
/// 500/500 passed while the grammar was still wrong. With the Qwen vocabulary the
/// same schema failed 9 times in 500 on strings like `"\uDafe"`.
#[test]
fn lone_surrogate_escapes_are_rejected() {
    let dfa = string_dfa();

    // Ordinary escapes and non-surrogate code units.
    assert!(dfa.matches(br#""\n""#));
    assert!(dfa.matches(br#""A""#));
    assert!(dfa.matches(br#""\uD7FF""#)); // just below the surrogate block
    assert!(dfa.matches(br#""\uE000""#)); // just above it
    assert!(dfa.matches(br#""\uFFFF""#));

    // A high surrogate is legal only when its low partner follows immediately.
    assert!(dfa.matches(br#""\uD83D\uDE00""#)); // an emoji, as a surrogate pair
    assert!(!dfa.matches(br#""\uD83D""#), "lone high surrogate");
    assert!(!dfa.matches(br#""\uDE00""#), "lone low surrogate");
    assert!(
        !dfa.matches(br#""\uD83Dx""#),
        "high surrogate not followed by an escape"
    );
    assert!(
        !dfa.matches(br#""\uD83D\u0041""#),
        "partner is not a low surrogate"
    );
    assert!(
        !dfa.matches(br#""\uDafe""#),
        "the literal failure from the demo"
    );

    // Mid-pair is not a valid stopping point: after a high surrogate the only
    // legal continuation is the backslash beginning its partner.
    let s = dfa
        .step_token(dfa.start(), br#""\uD83D"#)
        .expect("high surrogate is a legal prefix");
    assert!(!dfa.is_accepting(s));
    let legal: Vec<u8> = (0..=255u8)
        .filter(|&b| dfa.transition(s, b).is_some())
        .collect();
    assert_eq!(legal, vec![b'\\']);
}

/// The situation a `char`-level DFA cannot represent.
///
/// `₹` is three bytes (E2 82 B9). Feed the first two, then the third, as if they
/// had arrived as two separate tokens. The intermediate position is a perfectly
/// ordinary DFA state — but it is not a character boundary, so there is no `char`
/// to transition on. This test is the concrete form of the argument in
/// docs/GRAMMAR.md.
#[test]
fn a_character_may_be_split_across_tokens() {
    let dfa = string_dfa();

    let s0 = dfa.step_token(dfa.start(), b"\"").unwrap();
    let s1 = dfa
        .step_token(s0, &[0xE2, 0x82])
        .expect("partial char is a legal prefix");
    assert!(
        !dfa.is_accepting(s1),
        "mid-character is never a valid stopping point"
    );

    // From mid-character, exactly the continuation bytes 0x80..=0xBF may follow.
    let legal: Vec<u8> = (0..=255u8)
        .filter(|&b| dfa.transition(s1, b).is_some())
        .collect();
    assert_eq!(legal, (0x80..=0xBF).collect::<Vec<u8>>());

    let s2 = dfa.step_token(s1, &[0xB9]).unwrap();
    let s3 = dfa.step_token(s2, b"\"").unwrap();
    assert!(dfa.is_accepting(s3));
}

// ---------------------------------------------------------------------------
// Masking and the decode loop
// ---------------------------------------------------------------------------

/// The id of the token spelling exactly one byte `b`. Looked up rather than
/// assumed, so the tests do not silently depend on the synthetic vocabulary's
/// layout.
fn single_byte_token(vocab: &Vocab, b: u8) -> u32 {
    (0..vocab.len() as u32)
        .find(|&id| !vocab.is_special(id) && vocab.bytes(id) == [b])
        .unwrap_or_else(|| panic!("no single-byte token for {b:#04x}"))
}

#[test]
fn eos_is_legal_only_in_accepting_states() {
    let vocab = Vocab::synthetic();
    let grammar = Grammar::from_schema_str(r#"{"type":"boolean"}"#).unwrap();
    let mut index = VocabIndex::new();
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, &vocab, &mut index);
    let mut logits = vec![0.0f32; vocab.len()];

    // At the start, nothing has been emitted — stopping now would produce "".
    decoder.apply_mask(&mut logits).unwrap();
    assert!(
        logits[vocab.eos() as usize].is_infinite(),
        "EOS must be masked before the document is complete"
    );
    assert!(decoder.accept_token(vocab.eos()).is_err());

    // Emit "true" and the document is complete.
    for b in b"true" {
        let id = single_byte_token(&vocab, *b);
        decoder.accept_token(id).unwrap();
    }
    assert!(decoder.is_accepting());

    let mut logits = vec![0.0f32; vocab.len()];
    decoder.apply_mask(&mut logits).unwrap();
    assert!(
        logits[vocab.eos() as usize].is_finite(),
        "EOS must now be legal"
    );
    decoder.accept_token(vocab.eos()).unwrap();
    assert!(decoder.is_finished());
}

/// The mask must be the *only* thing standing between the sampler and invalid
/// output — so the decoder rechecks, and refuses a token the mask forbade.
#[test]
fn decoder_rejects_a_token_the_mask_forbade() {
    let vocab = Vocab::synthetic();
    let grammar = Grammar::from_schema_str(r#"{"type":"boolean"}"#).unwrap();
    let mut index = VocabIndex::new();
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, &vocab, &mut index);

    let illegal = single_byte_token(&vocab, b'x');
    assert!(decoder.accept_token(illegal).is_err());
    assert!(decoder.output().is_empty());
}

#[test]
fn mask_narrows_to_a_single_token_inside_a_fixed_key() {
    let vocab = Vocab::synthetic();
    let grammar =
        Grammar::from_schema_str(r#"{"type":"object","properties":{"zz":{"type":"boolean"}}}"#)
            .unwrap();
    let mut index = VocabIndex::new();
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, &vocab, &mut index);

    // After `{"z`, the only continuation the grammar admits is `z`.
    for b in b"{\"z" {
        decoder.accept_token(single_byte_token(&vocab, *b)).unwrap();
    }
    let mut logits = vec![0.0f32; vocab.len()];
    let allowed = decoder.apply_mask(&mut logits).unwrap();
    let legal: Vec<String> = (0..vocab.len() as u32)
        .filter(|&i| logits[i as usize].is_finite())
        .map(|i| vocab.display(i))
        .collect();
    assert_eq!(allowed, legal.len() as u32);
    assert_eq!(legal, vec!["z".to_string()]);
}

/// A padded logit vector must have its tail masked, not be rejected.
///
/// Qwen2.5's `config.json` declares `vocab_size: 151936` while `tokenizer.json`
/// defines 151,665 tokens — the embedding matrix is padded to a multiple of 128
/// for alignment. The forward pass will hand over the wider vector, and the extra
/// rows decode to nothing, so they must never be samplable.
#[test]
fn padding_rows_beyond_the_vocabulary_are_masked() {
    let vocab = Vocab::synthetic();
    let grammar = Grammar::from_schema_str(r#"{"type":"boolean"}"#).unwrap();
    let mut index = VocabIndex::new();
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, &vocab, &mut index);

    let padded = vocab.len() + 271;
    let mut logits = vec![0.0f32; padded];
    decoder.apply_mask(&mut logits).expect("wider is fine");

    for (id, l) in logits.iter().enumerate().skip(vocab.len()) {
        assert!(l.is_infinite(), "padding row {id} must be masked");
    }

    // Narrower than the vocabulary is still an error — that one is a real bug.
    let mut short = vec![0.0f32; vocab.len() - 1];
    assert!(decoder.apply_mask(&mut short).is_err());
}

/// The cache must be a pure memo: the answer for a state cannot depend on when
/// it was asked.
#[test]
fn index_cache_returns_identical_masks() {
    let vocab = Vocab::synthetic();
    let grammar = Grammar::from_schema_str(r#"{"type":"integer"}"#).unwrap();
    let mut index = VocabIndex::new();

    let start = grammar.dfa.start();
    let first = index.mask_for(&grammar.dfa, &vocab, start);
    let builds_after_first = index.stats.builds;
    let second = index.mask_for(&grammar.dfa, &vocab, start);

    assert_eq!(
        index.stats.builds, builds_after_first,
        "second call must hit the cache"
    );
    assert_eq!(index.stats.hits, 1);
    let a: Vec<u32> = first.iter().collect();
    let b: Vec<u32> = second.iter().collect();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// End to end: the binary guarantee
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() >> 33) as u32 % n.max(1)
    }
}

/// Generate by picking uniformly at random from whatever the mask permits.
///
/// This is the strongest form of the claim: the "model" has no preference at all,
/// so nothing about the output's validity can be attributed to it.
fn generate(grammar: &Grammar, vocab: &Vocab, index: &mut VocabIndex, rng: &mut Rng) -> String {
    let mut decoder = ConstrainedDecoder::new(&grammar.dfa, vocab, index);
    let mut logits = vec![0.0f32; vocab.len()];

    for _ in 0..512 {
        logits.iter_mut().for_each(|l| *l = 0.0);
        decoder.apply_mask(&mut logits).expect("no dead ends");
        let legal: Vec<u32> = (0..vocab.len() as u32)
            .filter(|&i| logits[i as usize].is_finite())
            .collect();
        assert!(!legal.is_empty());
        let pick = legal[rng.below(legal.len() as u32) as usize];
        decoder.accept_token(pick).unwrap();
        if decoder.is_finished() {
            break;
        }
    }
    assert!(
        decoder.is_finished() || decoder.is_accepting(),
        "ran out of budget in a non-accepting state"
    );
    String::from_utf8_lossy(decoder.output()).into_owned()
}

fn schemas() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "flat object",
            r#"{"type":"object","properties":{"name":{"type":"string","maxLength":8},"age":{"type":"integer"}}}"#,
        ),
        (
            "enum",
            r#"{"type":"object","properties":{"city":{"enum":["London","Bengaluru","Tokyo"]}}}"#,
        ),
        (
            "array of strings",
            r#"{"type":"array","items":{"type":"string","maxLength":4},"maxItems":3}"#,
        ),
        (
            "array of integers",
            r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":4}"#,
        ),
        (
            "nested object",
            r#"{"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"},"tags":{"type":"array","items":{"type":"string","maxLength":4},"maxItems":2}}},"ok":{"type":"boolean"}}}"#,
        ),
        (
            "number",
            r#"{"type":"object","properties":{"score":{"type":"number"}}}"#,
        ),
        (
            "null and const",
            r#"{"type":"object","properties":{"nothing":{"type":"null"},"kind":{"const":"user"}}}"#,
        ),
    ]
}

#[test]
fn every_constrained_generation_parses() {
    let vocab = Vocab::synthetic();
    for (label, schema) in schemas() {
        let grammar = Grammar::from_schema_str(schema).unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut index = VocabIndex::new();
        let mut rng = Rng(0xABCD_0000_1111_2222);

        for i in 0..300 {
            let out = generate(&grammar, &vocab, &mut index, &mut rng);
            let parsed = serde_json::from_str::<serde_json::Value>(&out);
            assert!(
                parsed.is_ok(),
                "{label} #{i}: {out:?} did not parse: {:?}",
                parsed.err()
            );
        }
    }
}

/// Structure, not just parseability: the document must have the shape the schema
/// asked for. A generator that emitted `0` for every schema would pass the parse
/// test and fail this one.
#[test]
fn generated_documents_match_the_schema_shape() {
    let vocab = Vocab::synthetic();
    let schema_str = r#"{"type":"object","properties":{"active":{"type":"boolean"},"age":{"type":"integer"},"city":{"enum":["London","Tokyo"]}}}"#;
    let grammar = Grammar::from_schema_str(schema_str).unwrap();
    let mut index = VocabIndex::new();
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    for _ in 0..200 {
        let out = generate(&grammar, &vocab, &mut index, &mut rng);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let obj = v.as_object().expect("must be an object");
        assert_eq!(obj.len(), 3, "exactly the declared keys: {out}");
        assert!(obj["active"].is_boolean(), "{out}");
        assert!(obj["age"].is_i64(), "{out}");
        let city = obj["city"].as_str().unwrap();
        assert!(city == "London" || city == "Tokyo", "{out}");
    }
}

/// Non-ASCII content must survive the whole pipeline, including when a character
/// is assembled from several tokens.
#[test]
fn non_ascii_output_is_valid_utf8_and_parses() {
    let vocab = Vocab::synthetic();
    let grammar = Grammar::from_schema_str(r#"{"type":"string","maxLength":6}"#).unwrap();
    let mut index = VocabIndex::new();
    let mut rng = Rng(0x0FED_CBA9_8765_4321);

    let mut saw_multibyte = false;
    for _ in 0..400 {
        let mut decoder = ConstrainedDecoder::new(&grammar.dfa, &vocab, &mut index);
        let mut logits = vec![0.0f32; vocab.len()];
        for _ in 0..64 {
            logits.iter_mut().for_each(|l| *l = 0.0);
            decoder.apply_mask(&mut logits).unwrap();
            let legal: Vec<u32> = (0..vocab.len() as u32)
                .filter(|&i| logits[i as usize].is_finite())
                .collect();
            let pick = legal[rng.below(legal.len() as u32) as usize];
            decoder.accept_token(pick).unwrap();
            if decoder.is_finished() {
                break;
            }
        }
        if !decoder.is_finished() && !decoder.is_accepting() {
            continue;
        }
        let bytes = decoder.output();
        // The strong assertion: raw bytes are valid UTF-8. Nothing in the loop
        // ever checked for character boundaries — the automaton did it.
        let text = std::str::from_utf8(bytes).expect("output must be valid UTF-8");
        if text.chars().any(|c| c.len_utf8() > 1) {
            saw_multibyte = true;
        }
        serde_json::from_str::<serde_json::Value>(text).expect("must parse");
    }
    assert!(
        saw_multibyte,
        "test is vacuous if no multi-byte character was produced"
    );
}

// ---------------------------------------------------------------------------
// Schema edge cases
// ---------------------------------------------------------------------------

/// `enum` mixing types (string, number, bool, null in one list) must compile:
/// each alternative is just its own canonical literal, and the JSON Schema spec
/// permits a heterogeneous enum even though it is unusual in practice.
#[test]
fn enum_with_mixed_value_types_compiles_and_matches_each_literal() {
    let schema = r#"{"enum": ["London", 42, true, null]}"#;
    let grammar = Grammar::from_schema_str(schema).unwrap();
    assert!(grammar.dfa.matches(b"\"London\""));
    assert!(grammar.dfa.matches(b"42"));
    assert!(grammar.dfa.matches(b"true"));
    assert!(grammar.dfa.matches(b"null"));
    // Nothing else is legal — not even a value present in none of the four.
    assert!(!grammar.dfa.matches(b"false"));
    assert!(!grammar.dfa.matches(b"43"));
    assert!(!grammar.dfa.matches(b"\"Tokyo\""));
}

/// An object with an explicit empty `properties: {}` is `{}` and only `{}` — not
/// an object accepting arbitrary content. Free-form objects are out of scope
/// (schema.rs says so), so this is the one way to spell "an object with nothing
/// in it," and it should not be confused with "any object."
#[test]
fn object_with_no_properties_matches_only_empty_braces() {
    let schema = r#"{"type":"object","properties":{}}"#;
    let grammar = Grammar::from_schema_str(schema).unwrap();
    assert!(grammar.dfa.matches(b"{}"));
    assert!(!grammar.dfa.matches(b"{\"x\":1}"));
    assert!(!grammar.dfa.matches(b""));
}

/// `array` with `maxItems: 0` is `[]` and only `[]`.
#[test]
fn array_with_max_items_zero_matches_only_empty_brackets() {
    let schema = r#"{"type":"array","items":{"type":"integer"},"maxItems":0}"#;
    let grammar = Grammar::from_schema_str(schema).unwrap();
    assert!(grammar.dfa.matches(b"[]"));
    assert!(!grammar.dfa.matches(b"[1]"));
}

/// `minItems > maxItems` is a contradiction the schema compiler should reject at
/// compile time, not one that should silently produce a DFA accepting nothing
/// (which would look like "compiled fine" right up until every generation hits a
/// DeadEnd at runtime).
#[test]
fn contradictory_item_bounds_are_rejected_at_compile_time() {
    let schema = r#"{"type":"array","items":{"type":"boolean"},"minItems":5,"maxItems":2}"#;
    assert!(Grammar::from_schema_str(schema).is_err());
}

/// `minLength > maxLength` is the identical contradiction on `string` that
/// `contradictory_item_bounds_are_rejected_at_compile_time` checks for `array`.
/// Found during a bug-hunt audit: the two code paths used to disagree on this —
/// `array` rejected it, `string` silently clamped `min` down to `max` and
/// compiled a schema that quietly accepted less than half of what was asked.
#[test]
fn contradictory_string_length_bounds_are_rejected_at_compile_time() {
    let schema = r#"{"type":"string","minLength":10,"maxLength":3}"#;
    assert!(Grammar::from_schema_str(schema).is_err());

    // minLength == maxLength is not a contradiction and must still compile.
    let schema = r#"{"type":"string","minLength":3,"maxLength":3}"#;
    let grammar = Grammar::from_schema_str(schema).unwrap();
    assert!(grammar.dfa.matches(b"\"abc\""));
    assert!(!grammar.dfa.matches(b"\"ab\""));
    assert!(!grammar.dfa.matches(b"\"abcd\""));
}

/// A large `maxLength`/`maxItems` is a compile-time denial-of-service, not just
/// an oddly big schema. Measured before this ceiling existed:
/// `{"type":"string","maxLength":4096}` compiled to 192,515 states in 34.5s and
/// a ~197 MB transition table — from one line of attacker-supplied JSON. Both
/// fields must be rejected above the ceiling, and a schema at or under it must
/// still compile (this is a resource limit, not a ban on legitimately large
/// strings/arrays).
#[test]
fn oversized_length_bounds_are_rejected_before_compilation() {
    assert!(Grammar::from_schema_str(r#"{"type":"string","maxLength":100000}"#).is_err());
    assert!(Grammar::from_schema_str(
        r#"{"type":"array","items":{"type":"boolean"},"maxItems":100000}"#
    )
    .is_err());

    // At the ceiling: still compiles, and without the multi-minute blowup an
    // unbounded schema would cause. The measured uncontended cost at this size is
    // ~2-4s (per the docs/schema.rs table: 1,024 was 8.4s, 4,096 was 34.5s, and
    // this crate's growth is superlinear, so 512 sits well below the 1,024
    // figure). 60s leaves generous room for a loaded CI machine while still
    // catching the failure mode this test exists for — the ceiling silently
    // regressing back toward "no ceiling" and reintroducing the DoS.
    let t0 = std::time::Instant::now();
    let grammar = Grammar::from_schema_str(r#"{"type":"string","maxLength":512}"#).unwrap();
    assert!(grammar
        .dfa
        .matches(&[b"\"", &[b'a'; 512][..], b"\""].concat()));
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(60),
        "compiling at the ceiling took {:?} — investigate before assuming CI load",
        t0.elapsed()
    );
}

/// `const` on its own (no `type`) must work — `const` short-circuits the type
/// dispatch entirely, so this checks that path doesn't accidentally require
/// `type` to also be present.
#[test]
fn const_alone_with_no_type_field_compiles() {
    let schema = r#"{"const": "fixed"}"#;
    let grammar = Grammar::from_schema_str(schema).unwrap();
    assert!(grammar.dfa.matches(b"\"fixed\""));
    assert!(!grammar.dfa.matches(b"\"other\""));
}
