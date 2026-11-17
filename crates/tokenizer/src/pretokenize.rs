//! Pre-tokenization: the split that BPE is never allowed to merge across.
//!
//! Qwen2.5's `tokenizer.json` specifies this pattern:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)      1. contraction suffixes
//! |[^\r\n\p{L}\p{N}]?\p{L}+         2. optional lead char, then letters
//! |\p{N}                            3. ONE digit — note: not \p{N}+
//! | ?[^\s\p{L}\p{N}]+[\r\n]*        4. optional space, then symbols
//! |\s*[\r\n]+                       5. whitespace ending in newlines
//! |\s+(?!\S)                        6. trailing whitespace run
//! |\s+                              7. any whitespace
//! ```
//!
//! It is implemented as a hand-written scanner rather than with a regex engine,
//! because the pattern needs a negative lookahead (`(?!\S)`) and Unicode property
//! classes — pulling in `fancy-regex` for one pattern is more dependency than this
//! deserves, and writing it out makes the semantics inspectable.
//!
//! **Two details that would silently break parity if missed:**
//!
//! *Alternation is leftmost-**first**, not leftmost-longest.* Alternatives are tried
//! in written order and the first that matches wins, even if a later one would match
//! more. So `'s` is a contraction before it is a symbol-then-letters.
//!
//! *`\p{N}` has no quantifier.* Every digit becomes its own chunk, so `2024` is four
//! chunks and can never merge into one token. GPT-2 used `\p{N}+` and cl100k uses
//! `\p{N}{1,3}`; copying either from memory would produce a tokenizer that looks
//! right and disagrees with the reference on every number.
//!
//! ## Known approximation
//!
//! `\p{L}` is the Unicode general category *Letter*. Rust's std exposes
//! `char::is_alphabetic()`, which is the *Alphabetic* derived property — that is
//! `L + Nl + Other_Alphabetic`, so it is slightly wider. The two agree on every
//! script in common use; they can disagree on characters like combining marks that
//! carry Other_Alphabetic. `\p{N}` and `\s` map exactly onto `is_numeric()` and
//! `is_whitespace()`, so those are not approximations. If ID parity against
//! HuggingFace ever fails on exotic input, this is the first place to look.

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// Split `text` into pre-token chunks. Concatenating the result reproduces the
/// input exactly — the split is total and lossless.
pub fn split(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < text.len() {
        let rest = &text[pos..];
        let len = match_one(rest);
        // `match_one` always returns at least one character's worth, so this
        // terminates. The fallback exists so a gap in the alternatives can never
        // produce an infinite loop.
        let len = if len == 0 {
            rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1)
        } else {
            len
        };
        out.push(&rest[..len]);
        pos += len;
    }

    out
}

/// Byte length of the match at the start of `s`, trying alternatives in order.
fn match_one(s: &str) -> usize {
    if let Some(n) = match_contraction(s) {
        return n;
    }
    if let Some(n) = match_letters(s) {
        return n;
    }
    if let Some(n) = match_single_digit(s) {
        return n;
    }
    if let Some(n) = match_symbols(s) {
        return n;
    }
    if let Some(n) = match_newline_run(s) {
        return n;
    }
    if let Some(n) = match_trailing_space(s) {
        return n;
    }
    match_any_space(s).unwrap_or(0)
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
fn match_contraction(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'\'') {
        return None;
    }
    // Longest first among same-prefix options so `'ll` is not cut to `'l`.
    for suffix in ["re", "ve", "ll", "s", "t", "m", "d"] {
        let n = suffix.len();
        if b.len() > n && s[1..1 + n].eq_ignore_ascii_case(suffix) {
            return Some(1 + n);
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+`
///
/// The optional leading character is anything that is not a newline, letter or
/// digit — a space qualifies, which is why ` hello` is one chunk and why almost
/// every word token in the vocabulary begins with `Ġ`.
fn match_letters(s: &str) -> Option<usize> {
    let mut it = s.char_indices();
    let (_, first) = it.next()?;

    let mut consumed = 0usize;
    let mut rest_start = 0usize;

    if !is_letter(first) {
        // Try it as the optional lead character.
        if first == '\r' || first == '\n' || is_number(first) {
            return None;
        }
        consumed = first.len_utf8();
        rest_start = consumed;
    }

    // One or more letters must follow.
    let mut letters = 0usize;
    for (i, c) in s[rest_start..].char_indices() {
        if is_letter(c) {
            letters += c.len_utf8();
        } else {
            let _ = i;
            break;
        }
    }
    if letters == 0 {
        return None;
    }
    Some(consumed + letters)
}

/// `\p{N}` — exactly one digit, never a run.
fn match_single_digit(s: &str) -> Option<usize> {
    let c = s.chars().next()?;
    if is_number(c) {
        Some(c.len_utf8())
    } else {
        None
    }
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
fn match_symbols(s: &str) -> Option<usize> {
    let mut consumed = 0usize;
    let bytes = s.as_bytes();

    // Optional single leading space (the literal ` ?`, not `\s?`).
    if bytes.first() == Some(&b' ') {
        consumed = 1;
    }

    let mut symbols = 0usize;
    for c in s[consumed..].chars() {
        if !is_space(c) && !is_letter(c) && !is_number(c) {
            symbols += c.len_utf8();
        } else {
            break;
        }
    }
    if symbols == 0 {
        return None;
    }
    consumed += symbols;

    // Trailing `[\r\n]*`.
    for c in s[consumed..].chars() {
        if c == '\r' || c == '\n' {
            consumed += c.len_utf8();
        } else {
            break;
        }
    }
    Some(consumed)
}

/// `\s*[\r\n]+`
///
/// Greedy `\s*` with backtracking: the match must *end* on a newline run, so this
/// takes all whitespace up to and including the final contiguous newlines.
fn match_newline_run(s: &str) -> Option<usize> {
    // Furthest point reachable by `\s*`.
    let mut ws_end = 0usize;
    for c in s.chars() {
        if is_space(c) {
            ws_end += c.len_utf8();
        } else {
            break;
        }
    }
    if ws_end == 0 {
        return None;
    }
    // Backtrack to the last position where the remainder of the whitespace run is
    // all newlines and non-empty.
    let ws = &s[..ws_end];
    let mut last_non_newline_end = 0usize;
    for (i, c) in ws.char_indices() {
        if c != '\r' && c != '\n' {
            last_non_newline_end = i + c.len_utf8();
        }
    }
    if last_non_newline_end == ws_end {
        return None; // no trailing newlines
    }
    Some(ws_end)
}

/// `\s+(?!\S)`
///
/// Backtracking semantics matter here. `\s+` grabs the whole whitespace run; if a
/// non-whitespace character follows, the lookahead fails and the engine gives back
/// one character at a time until the *next* character is whitespace. Net effect: a
/// run of length `L` followed by text matches `L - 1` characters, leaving the final
/// space to be picked up by rule 2 or 4 as the lead character of the next word.
/// That is why ` hello` tokenizes as one chunk rather than two.
fn match_trailing_space(s: &str) -> Option<usize> {
    let mut chars: Vec<(usize, char)> = Vec::new();
    let mut ws_end = 0usize;
    for (i, c) in s.char_indices() {
        if is_space(c) {
            chars.push((i, c));
            ws_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if chars.is_empty() {
        return None;
    }
    if ws_end == s.len() {
        return Some(ws_end); // run reaches the end; lookahead trivially succeeds
    }
    // Give back the final whitespace character.
    let (last_start, _) = *chars.last().unwrap();
    if last_start == 0 {
        None // would leave an empty match
    } else {
        Some(last_start)
    }
}

/// `\s+`
fn match_any_space(s: &str) -> Option<usize> {
    let mut n = 0usize;
    for c in s.chars() {
        if is_space(c) {
            n += c.len_utf8();
        } else {
            break;
        }
    }
    if n == 0 {
        None
    } else {
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitting_is_lossless() {
        for text in [
            "Hello, world!",
            "  leading and trailing  ",
            "line one\nline two\r\n\r\nthree",
            "café ₹100 — 😀 mixed",
            "isn't it Kalyan's?",
            "",
            "12345",
            "\t\ttabs",
        ] {
            let parts = split(text);
            assert_eq!(parts.concat(), text, "lossless split failed for {text:?}");
        }
    }

    #[test]
    fn a_space_attaches_to_the_following_word() {
        assert_eq!(split("hello world"), vec!["hello", " world"]);
        assert_eq!(split("a b c"), vec!["a", " b", " c"]);
    }

    /// The unquantified `\p{N}`: every digit is its own chunk.
    #[test]
    fn digits_split_individually() {
        assert_eq!(split("2024"), vec!["2", "0", "2", "4"]);
        assert_eq!(split("v2"), vec!["v", "2"]);
        assert_eq!(split(" 42"), vec![" ", "4", "2"]);
    }

    #[test]
    fn contractions_are_their_own_chunks() {
        assert_eq!(split("isn't"), vec!["isn", "'t"]);
        assert_eq!(split("Kalyan's"), vec!["Kalyan", "'s"]);
        assert_eq!(split("we'll"), vec!["we", "'ll"]);
        // Case-insensitive per (?i:).
        assert_eq!(split("WE'LL"), vec!["WE", "'LL"]);
    }

    #[test]
    fn a_run_of_spaces_keeps_all_but_the_last() {
        // "   a" → "  " then " a": the trailing-space rule gives back one space,
        // which the letters rule then picks up as its lead character.
        assert_eq!(split("   a"), vec!["  ", " a"]);
        // At end of input there is nothing to give back.
        assert_eq!(split("a   "), vec!["a", "   "]);
    }

    #[test]
    fn newlines_group_with_preceding_whitespace() {
        assert_eq!(split("a  \n\nb"), vec!["a", "  \n\n", "b"]);
        assert_eq!(split("x\r\ny"), vec!["x", "\r\n", "y"]);
    }

    #[test]
    fn symbols_take_an_optional_leading_space() {
        assert_eq!(split("a !!"), vec!["a", " !!"]);
        assert_eq!(split("{\"k\":1}"), vec!["{\"", "k", "\":", "1", "}"]);
    }

    #[test]
    fn non_ascii_letters_are_letters() {
        assert_eq!(split("café"), vec!["café"]);
        assert_eq!(split("日本語"), vec!["日本語"]);
        assert_eq!(split(" ₹"), vec![" ₹"]);
    }
}
