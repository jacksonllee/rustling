//! Random word segmenter.

use crate::persistence::ModelError;
use rand::RngExt;

// ---------------------------------------------------------------------------
// BaseRandomSegmenter
// ---------------------------------------------------------------------------

/// Core random segmentation behavior with default implementations.
///
/// Implementors provide required methods that grant access to internal state.
/// All segmentation logic is provided as defaults.
pub trait BaseRandomSegmenter: Sized + Clone {
    fn prob(&self) -> f64;
    fn from_prob(prob: f64) -> Self;

    /// Segment the given unsegmented sentences.
    fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        sent_strs
            .into_iter()
            .map(|sent_str| self.predict_sent(&sent_str))
            .collect()
    }

    /// Segment a single unsegmented sentence randomly.
    fn predict_sent(&self, sent_str: &str) -> Vec<String> {
        let mut rng = rand::rng();
        self.predict_sent_seeded(sent_str, &mut rng)
    }

    /// Segment a single unsegmented sentence with a seeded random generator.
    fn predict_sent_seeded<R: RngExt>(&self, sent_str: &str, rng: &mut R) -> Vec<String> {
        let chars: Vec<char> = sent_str.chars().collect();
        if chars.is_empty() {
            return vec![];
        }

        let segment_or_not: Vec<bool> = (0..chars.len().saturating_sub(1))
            .map(|_| self.prob() > rng.random::<f64>())
            .collect();

        let boundaries: Vec<usize> = segment_or_not
            .iter()
            .enumerate()
            .filter_map(|(i, &seg)| if seg { Some(i + 1) } else { None })
            .collect();

        let mut sent = Vec::new();
        let mut starts = vec![0];
        starts.extend(&boundaries);
        let mut ends = boundaries.clone();
        ends.push(chars.len());

        for (start, end) in starts.iter().zip(ends.iter()) {
            let word: String = chars[*start..*end].iter().collect();
            sent.push(word);
        }

        sent
    }

    /// Segment unsegmented sentences and return words with character offsets.
    fn predict_with_offsets(&self, sent_strs: Vec<String>) -> Vec<Vec<(String, (usize, usize))>> {
        let words = self.predict(sent_strs);
        crate::wordseg::attach_offsets(words)
    }
}

// ---------------------------------------------------------------------------
// Pure Rust struct
// ---------------------------------------------------------------------------

/// A random segmenter.
///
/// Segmentation is predicted at random at each potential word
/// boundary independently for a given probability. No training is required.
///
/// For Python, use [`PyRandomSegmenter`].
#[derive(Clone, Debug)]
pub struct RandomSegmenter {
    prob: f64,
}

impl BaseRandomSegmenter for RandomSegmenter {
    fn prob(&self) -> f64 {
        self.prob
    }
    fn from_prob(prob: f64) -> Self {
        Self { prob }
    }
}

impl RandomSegmenter {
    /// Create a new random segmenter.
    ///
    /// # Arguments
    ///
    /// * `prob` - The probability from [0, 1) that segmentation occurs between
    ///   two symbols.
    pub fn new(prob: f64) -> Result<Self, ModelError> {
        if !(0.0..1.0).contains(&prob) {
            return Err(ModelError::ValidationError(format!(
                "prob must be from [0, 1): {}",
                prob
            )));
        }
        Ok(Self { prob })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_new_valid() {
        let segmenter = RandomSegmenter::new(0.5);
        assert!(segmenter.is_ok());
        let segmenter = segmenter.unwrap();
        assert!((segmenter.prob - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_valid_zero() {
        let segmenter = RandomSegmenter::new(0.0);
        assert!(segmenter.is_ok());
    }

    #[test]
    fn test_new_invalid_prob_negative() {
        let result = RandomSegmenter::new(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_prob_one() {
        let result = RandomSegmenter::new(1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_prob_greater_than_one() {
        let result = RandomSegmenter::new(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_prob_zero_no_segmentation() {
        let segmenter = RandomSegmenter::new(0.0).unwrap();
        let result = segmenter.predict(vec!["hello".to_string()]);
        assert_eq!(result, vec![vec!["hello"]]);
    }

    #[test]
    fn test_predict_empty_input() {
        let segmenter = RandomSegmenter::new(0.5).unwrap();
        let result = segmenter.predict(vec!["".to_string()]);
        assert_eq!(result, vec![Vec::<String>::new()]);
    }

    #[test]
    fn test_predict_single_char() {
        let segmenter = RandomSegmenter::new(0.5).unwrap();
        let result = segmenter.predict(vec!["a".to_string()]);
        assert_eq!(result, vec![vec!["a"]]);
    }

    #[test]
    fn test_predict_seeded_deterministic() {
        let segmenter = RandomSegmenter::new(0.5).unwrap();

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let result1 = segmenter.predict_sent_seeded("hello", &mut rng1);
        let result2 = segmenter.predict_sent_seeded("hello", &mut rng2);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_predict_unicode() {
        let segmenter = RandomSegmenter::new(0.0).unwrap();
        let result = segmenter.predict(vec!["你好".to_string()]);
        assert_eq!(result, vec![vec!["你好"]]);
    }

    #[test]
    fn test_predict_multiple_sentences() {
        let segmenter = RandomSegmenter::new(0.0).unwrap();
        let result = segmenter.predict(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(result, vec![vec!["hello"], vec!["world"]]);
    }
}
