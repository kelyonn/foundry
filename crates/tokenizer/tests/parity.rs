//! ID parity against HuggingFace.
//!
//! This is the oracle docs/ARCHITECTURE.md names for the tokenizer crate. Fixtures
//! are produced by `scripts/dump_tokenizer_ids.py`; regenerate them if the corpus
//! or the model changes.
//!
//! Round-trip (`tests/roundtrip.rs`) proves self-consistency. This proves we
//! segment the same way the reference does, which is a strictly stronger claim and
//! the one that matters — a tokenizer that disagrees with the model's training
//! segmentation produces valid-looking IDs that the model has never seen in that
//! arrangement.

use tokenizer::Tokenizer;

const TOKENIZER: &str = "../../models/qwen2.5-0.5b/tokenizer.json";
const FIXTURE: &str = "../../tests/fixtures/tokenizer_ids.json";

struct Case {
    text: String,
    ids: Vec<u32>,
}

fn load() -> Option<(Tokenizer, Vec<Case>)> {
    if !std::path::Path::new(TOKENIZER).exists() || !std::path::Path::new(FIXTURE).exists() {
        eprintln!("skipping: run scripts/dump_tokenizer_ids.py first");
        return None;
    }
    let tk = Tokenizer::from_file(TOKENIZER).expect("tokenizer.json should load");
    let raw = std::fs::read_to_string(FIXTURE).unwrap();
    let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let cases = json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            text: c["text"].as_str().unwrap().to_string(),
            ids: c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
        })
        .collect();
    Some((tk, cases))
}

#[test]
fn encoder_matches_huggingface_ids_exactly() {
    let Some((tk, cases)) = load() else { return };

    let mut failures = Vec::new();
    for case in &cases {
        // `encode_with_specials`, not `encode`: the fixtures come from
        // HuggingFace's default, which extracts control tokens from raw text.
        // Our safe default deliberately does not — see `Tokenizer::encode`.
        let got = tk.encode_with_specials(&case.text);
        if got != case.ids {
            failures.push(format!(
                "text: {:?}\n     expected: {:?}\n     got:      {:?}\n     expected decoded: {:?}\n     got decoded:      {:?}",
                case.text,
                case.ids,
                got,
                case.ids.iter().map(|&i| tk.decode(&[i])).collect::<Vec<_>>(),
                got.iter().map(|&i| tk.decode(&[i])).collect::<Vec<_>>(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} cases disagree with HuggingFace:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
}

/// The prompt used for the forward-pass fixtures must tokenize identically, or
/// every downstream parity number is measuring the wrong input.
#[test]
fn the_parity_prompt_tokenizes_correctly() {
    let Some((tk, _)) = load() else { return };
    let meta_path = "../../tests/fixtures/qwen05b/meta.json";
    if !std::path::Path::new(meta_path).exists() {
        eprintln!("skipping: no forward-pass fixtures");
        return;
    }
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let expected: Vec<u32> = meta["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let text = meta["templated_text"].as_str().unwrap();

    assert_eq!(
        tk.encode(text),
        expected,
        "the forward-pass prompt does not tokenize to the reference ids"
    );
}
