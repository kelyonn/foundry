//! The one piece of `foundry`'s logic that needs to be independently testable
//! rather than living inline in `main`: masking the padding tail of a logit row.

pub mod error;
pub use error::{or_exit, CliError};

/// Set every logit at or beyond `vocab_size` to `-inf`, in place.
///
/// # The bug this fixes
///
/// `config.json` declares `vocab_size: 151936` — the padded width of the
/// embedding matrix — but the real tokenizer defines only 151,665 tokens (see
/// docs/GRAMMAR.md). The forward pass therefore emits 271 logits with no
/// corresponding token. `crates/grammar`'s `apply_mask` already masks this tail
/// unconditionally for constrained decoding. The CLI's **unconstrained** path did
/// not: it sampled from the full row, so a high-temperature or top-p run that
/// landed in the padding tail would call `tokenizer::decode` with an out-of-range
/// id and panic. Measured on the actual checkpoint the padding logits sit well
/// below every real token's (see `padding_tail_is_uniformly_dominated_in_practice`
/// in `tests/`), which is why greedy decoding never surfaced it — but "the model
/// happens not to prefer it today" is not the same guarantee as "it cannot be
/// sampled," and sampling code should not depend on the former.
///
/// This is called on every decode step regardless of whether a grammar is active,
/// so the unconstrained path gets the same guarantee the constrained path already
/// had.
pub fn mask_padding_tail(row: &mut [f32], vocab_size: usize) {
    if vocab_size < row.len() {
        for logit in &mut row[vocab_size..] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_only_the_tail() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mask_padding_tail(&mut row, 3);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 2.0);
        assert_eq!(row[2], 3.0);
        assert!(row[3].is_infinite() && row[3] < 0.0);
        assert!(row[4].is_infinite() && row[4] < 0.0);
    }

    #[test]
    fn a_row_no_wider_than_the_vocabulary_is_untouched() {
        let mut row = vec![1.0, 2.0, 3.0];
        mask_padding_tail(&mut row, 3);
        assert_eq!(row, vec![1.0, 2.0, 3.0]);

        let mut row2 = vec![1.0, 2.0];
        mask_padding_tail(&mut row2, 5); // vocab_size > row.len(): nothing to mask
        assert_eq!(row2, vec![1.0, 2.0]);
    }

    /// The property that actually matters: after masking, greedy argmax and any
    /// temperature/top-k/top-p sample can never land in the padding region,
    /// regardless of what values were there.
    #[test]
    fn masked_region_is_never_selected_by_the_sampler() {
        use model::sampling::{argmax, Sampler, SamplingConfig};

        // Deliberately make the padding tail the largest values in the row, so a
        // sampler that ignored the mask would certainly pick one of them.
        let mut row = vec![0.1, 0.2, 0.3, 999.0, 999.0, 999.0];
        mask_padding_tail(&mut row, 3);

        assert!(argmax(&row).unwrap() < 3);

        let mut sampler = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: None,
            },
            42,
        );
        for _ in 0..500 {
            assert!(sampler.sample(&row).unwrap() < 3);
        }
    }
}
