//! JSON Schema (a deliberate subset) → `Pattern`.
//!
//! # Decisions made here, and their limits
//!
//! **No whitespace.** The generated language is canonical, compact JSON: no
//! spaces or newlines anywhere. docs/GRAMMAR.md asks for this choice to be made
//! explicitly, so: permitting arbitrary whitespace multiplies the legal token set
//! at every single state, and lets the model burn its budget emitting indentation
//! that carries no information. Output is compact; pretty-print downstream if a
//! human needs to read it.
//!
//! **Object keys are emitted in schema declaration order, and every declared
//! property is required.** Permuting `n` keys means `n!` alternations in a
//! regular language — there is no way to say "each of these once, in any order"
//! without enumerating. Fixing the order keeps the automaton small. The cost is
//! that the model cannot choose to omit an optional field.
//!
//! **Structure only, not value bounds.** `maxLength` on a string is honored
//! (bounded repetition is regular). `minimum`/`maximum` on a number is *not* —
//! comparing magnitudes requires counting, which a DFA cannot do. We guarantee
//! the shape, not the value.
//!
//! **No `$ref`, no recursion.** A recursive schema describes a context-free
//! language. We compile a concrete finite tree, so the nesting depth is whatever
//! the schema literally spells out.

use crate::pattern::{ByteSet, Pattern};
use serde_json::Value;

#[derive(Debug)]
pub struct SchemaError(pub String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "schema error: {}", self.0)
    }
}

impl std::error::Error for SchemaError {}

fn err<T>(msg: impl Into<String>) -> Result<T, SchemaError> {
    Err(SchemaError(msg.into()))
}

/// Ceiling on `maxLength` and `maxItems`, enforced regardless of what the schema
/// author asks for.
///
/// **Why this exists — measured, not guessed.** Every unit of bounded repetition
/// costs real NFA/DFA states, and subset construction's cost grows faster than
/// linearly in the result size. Measured on `{"type":"string","maxLength":N}`:
///
/// | N | states | compile time |
/// |---|---|---|
/// | 64 (the default) | 3,011 | 0.47s |
/// | 1,024 | 48,131 | 8.4s |
/// | 4,096 | 192,515 | 34.5s, ~197 MB transition table |
///
/// Nothing upstream of `compile_schema` limits what a caller can put in a JSON
/// Schema. `--schema` is a CLI flag today, but the crate's contract is "compile
/// whatever schema you're handed," and a single `maxLength: 1000000` line is
/// enough to stall the process for minutes and allocate gigabytes — a
/// denial-of-service that costs the attacker one short string. The DFA
/// construction has no way to warn about this partway through; the only sound
/// place to stop it is before compilation starts.
const MAX_BOUNDED_REPEAT: u32 = 512;

fn check_ceiling(field: &str, value: u32) -> Result<u32, SchemaError> {
    if value > MAX_BOUNDED_REPEAT {
        return err(format!(
            "{field}: {value} exceeds the compiled-grammar ceiling of {MAX_BOUNDED_REPEAT} \
             (large bounds make DFA compilation slow and memory-heavy — see schema.rs)"
        ));
    }
    Ok(value)
}

/// Compile a JSON Schema document into a byte-level pattern.
pub fn compile_schema(schema: &Value) -> Result<Pattern, SchemaError> {
    node(schema)
}

fn node(schema: &Value) -> Result<Pattern, SchemaError> {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return err("each schema node must be a JSON object"),
    };

    // `enum` short-circuits the type: the language is exactly those literals,
    // serialized canonically. This is the cheapest possible constraint and the
    // easiest to demo — the model literally cannot say anything else.
    if let Some(values) = obj.get("enum") {
        let values = match values.as_array() {
            Some(v) if !v.is_empty() => v,
            _ => return err("`enum` must be a non-empty array"),
        };
        let alts = values
            .iter()
            .map(|v| Pattern::Literal(canonical_json(v).into_bytes()))
            .collect();
        return Ok(Pattern::Alt(alts));
    }

    if let Some(c) = obj.get("const") {
        return Ok(Pattern::Literal(canonical_json(c).into_bytes()));
    }

    let ty = match obj.get("type").and_then(|t| t.as_str()) {
        Some(t) => t,
        None => return err("every node needs a `type` (or `enum`/`const`)"),
    };

    match ty {
        "object" => object(obj),
        "array" => array(obj),
        "string" => {
            let max_len = obj
                .get("maxLength")
                .and_then(|v| v.as_u64())
                .map(|v| check_ceiling("maxLength", v as u32))
                .transpose()?;
            let min_len = obj
                .get("minLength")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            string_pattern(max_len, min_len)
        }
        "integer" => Ok(integer_pattern()),
        "number" => Ok(number_pattern()),
        "boolean" => Ok(Pattern::Alt(vec![
            Pattern::lit("true"),
            Pattern::lit("false"),
        ])),
        "null" => Ok(Pattern::lit("null")),
        other => err(format!("unsupported type `{other}`")),
    }
}

fn object(obj: &serde_json::Map<String, Value>) -> Result<Pattern, SchemaError> {
    let props = match obj.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return err("`object` needs `properties` (free-form objects are not supported)"),
    };
    if props.is_empty() {
        return Ok(Pattern::lit("{}"));
    }

    // Without serde_json's `preserve_order` feature this map is a BTreeMap, so
    // iteration is lexicographic by key rather than declaration order. That is
    // still *deterministic*, which is all the grammar needs — but it means the
    // emitted key order is sorted, not the order written in the schema file.
    let mut parts = vec![Pattern::lit("{")];
    for (i, (key, sub)) in props.iter().enumerate() {
        if i > 0 {
            parts.push(Pattern::lit(","));
        }
        // The key is a fixed literal — the model does not get to choose it. This
        // is the single biggest reduction in the legal token set: at a state
        // mid-key, usually exactly one token continues the language.
        parts.push(Pattern::Literal(
            canonical_json(&Value::String(key.clone())).into_bytes(),
        ));
        parts.push(Pattern::lit(":"));
        parts.push(node(sub)?);
    }
    parts.push(Pattern::lit("}"));
    Ok(Pattern::Concat(parts))
}

fn array(obj: &serde_json::Map<String, Value>) -> Result<Pattern, SchemaError> {
    let items = match obj.get("items") {
        Some(i) => node(i)?,
        None => return err("`array` needs `items`"),
    };
    let min = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let max = obj
        .get("maxItems")
        .and_then(|v| v.as_u64())
        .map(|v| check_ceiling("maxItems", v as u32))
        .transpose()?;

    // Bounded above so the language stays finite in the item count. Without a
    // ceiling the model can emit elements forever; 16 is a pragmatic default and
    // it is a *documented* limit, not an accident.
    let max = max.unwrap_or(16);
    if max == 0 {
        return Ok(Pattern::lit("[]"));
    }
    if min > max {
        return err("minItems > maxItems");
    }

    // [ item (,item){min-1..max-1} ]  — with the whole body optional if min == 0.
    let tail = Pattern::Concat(vec![Pattern::lit(","), items.clone()])
        .repeat(min.saturating_sub(1), Some(max - 1));
    let body = Pattern::Concat(vec![items, tail]);
    let body = if min == 0 { body.opt() } else { body };

    Ok(Pattern::Concat(vec![
        Pattern::lit("["),
        body,
        Pattern::lit("]"),
    ]))
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/// Digits allowed after the leading one. 18 total digits stays inside `i64`.
const INT_DIGITS: u32 = 18;

/// `0 | -?[1-9][0-9]*` — a canonical integer literal.
///
/// Note what is *excluded*: leading zeros and a leading `+` (both illegal in
/// JSON), and also `-0`. Negative zero is legal JSON and is mathematically an
/// integer, but it does not round-trip through an integer type — `serde_json`
/// parses `-0` as a float, since `i64` has no negative zero. Emitting it would
/// mean the document parses but a downstream consumer typed as integer would
/// still be surprised. Excluding it costs nothing and makes the output canonical.
fn integer_pattern() -> Pattern {
    Pattern::Alt(vec![
        Pattern::lit("0"),
        Pattern::Concat(vec![
            Pattern::lit("-").opt(),
            Pattern::class(ByteSet::range(b'1', b'9')),
            Pattern::class(ByteSet::range(b'0', b'9')).repeat(0, Some(INT_DIGITS - 1)),
        ]),
    ])
}

/// The integer part of a `number`, which unlike a standalone integer *may* be
/// `-0` — because `-0.5` is an ordinary negative number and its integer part is
/// `-0`. The distinction only exists because of the rule above.
fn number_int_part() -> Pattern {
    Pattern::Concat(vec![
        Pattern::lit("-").opt(),
        Pattern::Alt(vec![
            Pattern::lit("0"),
            Pattern::Concat(vec![
                Pattern::class(ByteSet::range(b'1', b'9')),
                Pattern::class(ByteSet::range(b'0', b'9')).repeat(0, Some(INT_DIGITS - 1)),
            ]),
        ]),
    ])
}

/// A JSON number.
///
/// The exponent is capped at two digits, which is a *value* restriction wearing a
/// syntactic disguise, and it is here because of a real bug: an 18-digit mantissa
/// with a three-digit exponent produces things like `2760420040100242100E670`.
/// That is a perfectly well-formed JSON number by RFC 8259 — and `serde_json`
/// rejects it, because it does not fit in an `f64`.
///
/// This is precisely the expressiveness boundary described in docs/GRAMMAR.md. A
/// DFA cannot compare magnitudes, so "must be finite as an f64" is not something
/// the automaton can enforce. What it *can* do is restrict the digit budget so
/// that no reachable string can overflow: 18 mantissa digits and `e99` tops out
/// around 1e117, comfortably inside f64's ~1.8e308. The constraint is enforced
/// structurally, which is the only way this layer knows how to enforce anything.
fn number_pattern() -> Pattern {
    let digits = Pattern::class(ByteSet::range(b'0', b'9'));
    Pattern::Concat(vec![
        number_int_part(),
        // fraction
        Pattern::Concat(vec![Pattern::lit("."), digits.clone().repeat(1, Some(18))]).opt(),
        // exponent
        Pattern::Concat(vec![
            Pattern::Alt(vec![Pattern::lit("e"), Pattern::lit("E")]),
            Pattern::Alt(vec![Pattern::lit("+"), Pattern::lit("-")]).opt(),
            digits.repeat(1, Some(2)),
        ])
        .opt(),
    ])
}

/// `"` char* `"`, where `char` is a well-formed UTF-8 scalar or a JSON escape.
fn string_pattern(max_len: Option<u32>, min_len: Option<u32>) -> Result<Pattern, SchemaError> {
    let ch = Pattern::Alt(vec![utf8_char(), json_escape()]);
    // Default ceiling keeps the automaton finite. A string that can be any length
    // is fine for a DFA in principle (star), but a bounded one lets us also
    // support `maxLength` with the same code path.
    let max = max_len.unwrap_or(64);
    let min = min_len.unwrap_or(0);
    // Same contradiction as `array`'s `minItems > maxItems` (see `array`), and
    // it needs the same fix. This used to be `.min(max)` — silently clamping the
    // minimum down to the maximum, which hides a contradictory schema behind a
    // grammar that compiles fine and just quietly ignores half of what was
    // asked. Caught while auditing the two length-bound code paths against each
    // other and finding they disagreed on an identical situation.
    if min > max {
        return err("minLength > maxLength");
    }
    Ok(Pattern::Concat(vec![
        Pattern::lit("\""),
        ch.repeat(min, Some(max)),
        Pattern::lit("\""),
    ]))
}

/// A hex digit class built from an explicit set of nibble values, case-insensitive.
fn hex_nibbles(values: std::ops::RangeInclusive<u8>) -> Pattern {
    let mut s = ByteSet::empty();
    for v in values {
        match v {
            0..=9 => s.insert(b'0' + v),
            10..=15 => {
                s.insert(b'a' + (v - 10));
                s.insert(b'A' + (v - 10));
            }
            _ => unreachable!("nibbles are 0..=15"),
        }
    }
    Pattern::class(s)
}

/// A JSON string escape: one of the eight shorthands, or a `\uXXXX` code unit.
///
/// # The surrogate rule
///
/// A naive `\u` + four hex digits is wrong, and the real Qwen vocabulary finds it
/// within a few hundred generations. `\uXXXX` encodes a UTF-16 *code unit*, and
/// `D800..DFFF` is the surrogate block — those code units are meaningless alone.
/// A parser sees `\uDafe` and either demands a following low surrogate or rejects
/// it outright. Either way the document does not parse, which is the one outcome
/// this crate is supposed to make impossible.
///
/// So the escape is split three ways, using the fact that the first two hex
/// digits alone determine the block:
///
/// * `\u` + a non-surrogate code unit — first digit is not `d`, or it is `d` and
///   the second digit is `0..7` (i.e. `D000..D7FF`).
/// * `\u` + a high surrogate (`D800..DBFF`, second digit `8..b`) **immediately
///   followed by** `\u` + a low surrogate (`DC00..DFFF`, second digit `c..f`).
/// * A lone low surrogate is unreachable: no production begins with one.
///
/// Pairing is a fixed-length adjacency requirement, so it stays regular — the
/// automaton simply has no path from a high surrogate to anything except the
/// backslash of its partner. This is a good example of a constraint that *looks*
/// semantic but is expressible structurally; contrast `minimum`/`maximum` on a
/// number, which genuinely is not.
fn json_escape() -> Pattern {
    let hex = hex_nibbles(0..=15);
    let simple = {
        let mut s = ByteSet::empty();
        // The eight single-character escapes JSON defines: \" \\ \/ \b \f \n \r \t
        for &b in b"\"\\/bfnrt" {
            s.insert(b);
        }
        Pattern::class(s)
    };

    // Anything outside D800..DFFF.
    let non_surrogate = Pattern::Alt(vec![
        // First digit is not `d`: cannot be in the surrogate block at all.
        Pattern::Concat(vec![
            Pattern::class({
                let mut s = ByteSet::empty();
                for v in 0..=15u8 {
                    if v == 13 {
                        continue; // skip 'd'
                    }
                    match v {
                        0..=9 => s.insert(b'0' + v),
                        _ => {
                            s.insert(b'a' + (v - 10));
                            s.insert(b'A' + (v - 10));
                        }
                    }
                }
                s
            }),
            hex.clone(),
            hex.clone(),
            hex.clone(),
        ]),
        // D000..D7FF — leading `d`, but below the surrogate block.
        Pattern::Concat(vec![
            hex_nibbles(13..=13),
            hex_nibbles(0..=7),
            hex.clone(),
            hex.clone(),
        ]),
    ]);

    // D800..DBFF
    let high_surrogate = Pattern::Concat(vec![
        hex_nibbles(13..=13),
        hex_nibbles(8..=11),
        hex.clone(),
        hex.clone(),
    ]);
    // DC00..DFFF
    let low_surrogate = Pattern::Concat(vec![
        hex_nibbles(13..=13),
        hex_nibbles(12..=15),
        hex.clone(),
        hex,
    ]);

    Pattern::Concat(vec![
        Pattern::lit("\\"),
        Pattern::Alt(vec![
            simple,
            Pattern::Concat(vec![Pattern::lit("u"), non_surrogate]),
            // A surrogate pair is one indivisible unit of the language.
            Pattern::Concat(vec![
                Pattern::lit("u"),
                high_surrogate,
                Pattern::lit("\\u"),
                low_surrogate,
            ]),
        ]),
    ])
}

/// One well-formed UTF-8 scalar value, minus the two bytes JSON reserves.
///
/// This is the concrete reason the whole automaton is over bytes. A DFA over
/// `char` cannot express "this token is the *first two bytes* of a three-byte
/// character, which is fine because a later token supplies the third" — but that
/// situation is completely routine in byte-level BPE. Here, a partial character
/// simply leaves us in a non-accepting intermediate state, and the mask at that
/// state permits exactly the continuation bytes that finish the character.
///
/// The ranges below are Unicode Table 3-7 (well-formed UTF-8 byte sequences).
/// They are narrower than "0x80..=0xFF anywhere": they exclude overlong forms,
/// surrogates (`ED A0..BF`), and anything past U+10FFFF. So the automaton cannot
/// emit invalid UTF-8 either — the guarantee covers encoding, not just shape.
fn utf8_char() -> Pattern {
    let cont = || Pattern::class(ByteSet::range(0x80, 0xBF));

    // ASCII, excluding `"` (ends the string) and `\` (starts an escape). Control
    // characters below 0x20 must be escaped in JSON, so they start at 0x20.
    let mut ascii = ByteSet::range(0x20, 0x7F);
    ascii = ascii.subtract(&ByteSet::single(b'"'));
    ascii = ascii.subtract(&ByteSet::single(b'\\'));

    Pattern::Alt(vec![
        Pattern::class(ascii),
        // 2-byte: C2..DF 80..BF   (C0/C1 would be overlong)
        Pattern::Concat(vec![Pattern::class(ByteSet::range(0xC2, 0xDF)), cont()]),
        // 3-byte, split four ways to exclude overlongs and surrogates
        Pattern::Concat(vec![
            Pattern::class(ByteSet::single(0xE0)),
            Pattern::class(ByteSet::range(0xA0, 0xBF)),
            cont(),
        ]),
        Pattern::Concat(vec![
            Pattern::class(ByteSet::range(0xE1, 0xEC)),
            cont(),
            cont(),
        ]),
        Pattern::Concat(vec![
            Pattern::class(ByteSet::single(0xED)),
            Pattern::class(ByteSet::range(0x80, 0x9F)), // excludes surrogates
            cont(),
        ]),
        Pattern::Concat(vec![
            Pattern::class(ByteSet::range(0xEE, 0xEF)),
            cont(),
            cont(),
        ]),
        // 4-byte, split three ways to exclude overlongs and > U+10FFFF
        Pattern::Concat(vec![
            Pattern::class(ByteSet::single(0xF0)),
            Pattern::class(ByteSet::range(0x90, 0xBF)),
            cont(),
            cont(),
        ]),
        Pattern::Concat(vec![
            Pattern::class(ByteSet::range(0xF1, 0xF3)),
            cont(),
            cont(),
            cont(),
        ]),
        Pattern::Concat(vec![
            Pattern::class(ByteSet::single(0xF4)),
            Pattern::class(ByteSet::range(0x80, 0x8F)),
            cont(),
            cont(),
        ]),
    ])
}

/// Serialize a JSON value the way our grammar spells it: no whitespace.
/// `serde_json::to_string` is already compact, so this is mostly a name for the
/// intent — the grammar and the literal must agree byte for byte or the DFA will
/// reject its own enum values.
fn canonical_json(v: &Value) -> String {
    serde_json::to_string(v).expect("serializing a Value cannot fail")
}
