//! HMM-based word segmenter using BMES tagging (supervised and unsupervised).

use super::bmes::{bmes_to_words, words_to_bmes};
use crate::hmm::{BaseHiddenMarkovModel, HiddenMarkovModel};
use crate::persistence::ModelError;
use crate::seq_feature::{SeqFeatureTemplate, default_segmenter_hmm_features};
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of BMES states.
const N_STATES: usize = 4;

/// Map a state index back to its BMES label string.
///
/// BMES labels sorted alphabetically: B=0, E=1, M=2, S=3.
fn index_to_bmes_label(index: usize) -> &'static str {
    match index {
        0 => "B",
        1 => "E",
        2 => "M",
        3 => "S",
        _ => panic!("Unknown state index: {}", index),
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Core HMM segmenter behavior with default implementations.
///
/// The segmenter delegates vocabulary building, parameter estimation, Viterbi
/// decoding, and save/load to [`BaseHiddenMarkovModel`]. It provides the
/// word↔char↔BMES conversion layer on top.
pub trait BaseHiddenMarkovModelSegmenter: Sized + Clone {
    // --- Required: storage contract ---
    fn hmm(&self) -> &HiddenMarkovModel;
    fn hmm_mut(&mut self) -> &mut HiddenMarkovModel;
    fn from_hmm(hmm: HiddenMarkovModel) -> Self;

    // --- Default: business logic ---

    /// Train the segmenter from supervised segmented sentences.
    fn fit_segmented(&mut self, sents: Vec<Vec<String>>) {
        let labeled_sents: Vec<Vec<(char, &str)>> =
            sents.iter().map(|sent| words_to_bmes(sent)).collect();

        let filtered: Vec<&Vec<(char, &str)>> =
            labeled_sents.iter().filter(|s| !s.is_empty()).collect();

        if filtered.is_empty() {
            self.hmm_mut().set_fitted(true);
            return;
        }

        // Build sequences (chars as strings) and labels (BMES tags).
        let mut sequences: Vec<Vec<String>> = filtered
            .iter()
            .map(|s| s.iter().map(|(c, _)| c.to_string()).collect())
            .collect();
        let mut labels: Vec<Vec<String>> = filtered
            .iter()
            .map(|s| s.iter().map(|(_, l)| l.to_string()).collect())
            .collect();

        // Ensure all 4 BMES labels are present so that fit_labeled always
        // discovers exactly N_STATES states with the correct alphabetical
        // ordering: B=0, E=1, M=2, S=3.
        sequences.push(vec!["_".into(), "_".into(), "_".into(), "_".into()]);
        labels.push(vec!["B".into(), "M".into(), "E".into(), "S".into()]);

        // fit_labeled handles vocab building, counting, smoothing, normalization.
        self.hmm_mut().fit_labeled(sequences, labels).unwrap();
    }

    /// Train the segmenter from unsegmented sentences using Baum-Welch EM.
    ///
    /// Each string in `sent_strs` is an unsegmented sentence. Characters are
    /// extracted and passed to the underlying HMM's Baum-Welch algorithm.
    /// If the model was previously fitted (e.g., via `fit_segmented`),
    /// the existing parameters serve as EM initialization (warm start).
    fn fit_unsegmented(&mut self, sent_strs: Vec<String>) {
        let sequences: Vec<Vec<String>> = sent_strs
            .iter()
            .map(|s| s.chars().map(|c| c.to_string()).collect())
            .filter(|s: &Vec<String>| !s.is_empty())
            .collect();
        if sequences.is_empty() {
            return;
        }
        self.hmm_mut().fit_unlabeled(sequences);
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    fn save_to_path(&self, path: &str) -> Result<(), ModelError> {
        self.hmm().save_to_path(path)
    }

    fn load_from_path(&mut self, path: &str) -> Result<(), ModelError> {
        self.hmm_mut().load_from_path(path)
    }

    /// Save the model to a FlatBuffers binary stream.
    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<(), ModelError> {
        self.hmm().save_to_writer(writer)
    }

    /// Load the model from a FlatBuffers byte slice.
    fn load_from_reader<R: Read>(&mut self, reader: R) -> Result<(), ModelError> {
        self.hmm_mut().load_from_reader(reader)
    }

    /// Compute log-likelihood of segmented sentences under the model.
    ///
    /// Each sentence is a list of words (same format as `fit_segmented` input).
    /// Returns one log-likelihood per sentence using the Forward algorithm.
    fn score(&self, sents: Vec<Vec<String>>) -> Result<Vec<f64>, ModelError> {
        if !self.hmm().fitted() {
            return Err(ModelError::ValidationError(
                "Model has not been fitted yet.".to_string(),
            ));
        }
        let sequences: Vec<Vec<String>> = sents
            .iter()
            .map(|sent| {
                sent.iter()
                    .flat_map(|w| w.chars())
                    .map(|c| c.to_string())
                    .collect()
            })
            .collect();
        self.hmm().score(sequences)
    }

    /// Segment unsegmented sentence strings.
    fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        if sent_strs.is_empty() {
            return Vec::new();
        }

        // Gather chars per sentence.
        let chars_per_sent: Vec<Vec<char>> =
            sent_strs.iter().map(|s| s.chars().collect()).collect();

        // Not fitted → fall back to single-character words.
        if !self.hmm().fitted() {
            return chars_per_sent
                .iter()
                .map(|chars| chars.iter().map(|c| c.to_string()).collect())
                .collect();
        }

        // Build observation sequences (chars as single-char strings).
        let sequences: Vec<Vec<String>> = chars_per_sent
            .iter()
            .map(|chars| chars.iter().map(|c| c.to_string()).collect())
            .collect();

        // Batch Viterbi decode via the HMM.
        let state_paths = self.hmm().predict(sequences).unwrap();

        // Convert state indices → BMES labels → words.
        chars_per_sent
            .iter()
            .zip(state_paths.iter())
            .map(|(chars, path)| {
                if chars.is_empty() {
                    return Vec::new();
                }
                let tags: Vec<&str> = path.iter().map(|&idx| index_to_bmes_label(idx)).collect();
                bmes_to_words(chars, &tags)
            })
            .collect()
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

/// An HMM-based word segmenter using supervised BMES tagging.
///
/// For Python, use [`PyHiddenMarkovModelSegmenter`].
#[derive(Clone, Debug)]
pub struct HiddenMarkovModelSegmenter {
    hmm: HiddenMarkovModel,
}

impl BaseHiddenMarkovModelSegmenter for HiddenMarkovModelSegmenter {
    fn hmm(&self) -> &HiddenMarkovModel {
        &self.hmm
    }
    fn hmm_mut(&mut self) -> &mut HiddenMarkovModel {
        &mut self.hmm
    }
    fn from_hmm(hmm: HiddenMarkovModel) -> Self {
        Self { hmm }
    }
}

impl Default for HiddenMarkovModelSegmenter {
    fn default() -> Self {
        Self::new(None, None, None, None, None)
    }
}

impl HiddenMarkovModelSegmenter {
    /// Create a new HMM-based word segmenter.
    ///
    /// # Parameters
    ///
    /// * `n_iter` – Maximum EM iterations for unsupervised fitting (default 1).
    /// * `tolerance` – Convergence threshold for EM (default 0.0).
    /// * `gamma` – Lidstone smoothing parameter (default 1.0 = Laplace). Must be > 0.
    /// * `random_seed` – Optional seed for reproducible random initialization.
    /// * `features` – Custom feature templates (default: segmenter features).
    pub fn new(
        n_iter: Option<usize>,
        tolerance: Option<f64>,
        gamma: Option<f64>,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Self {
        let templates = features.unwrap_or_else(default_segmenter_hmm_features);
        Self {
            hmm: HiddenMarkovModel::new(
                N_STATES,
                n_iter.unwrap_or(1),
                tolerance.unwrap_or(0.0),
                gamma.unwrap_or(1.0),
                random_seed,
                Some(templates),
            )
            .unwrap(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn training_data() -> Vec<Vec<String>> {
        vec![
            vec!["你好".into(), "世界".into()],
            vec!["我".into(), "喜歡".into(), "你".into()],
            vec![
                "他".into(),
                "是".into(),
                "一".into(),
                "個".into(),
                "好".into(),
                "人".into(),
            ],
        ]
    }

    #[test]
    fn test_new() {
        let segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        assert!(!segmenter.hmm().fitted());
    }

    #[test]
    fn test_predict_before_fit_falls_back_to_chars() {
        let segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        let result = segmenter.predict(vec!["你好".into()]);
        assert_eq!(result, vec![vec!["你".to_string(), "好".to_string()]]);
    }

    #[test]
    fn test_fit_segmented_sets_fitted() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        assert!(segmenter.hmm().fitted());
    }

    #[test]
    fn test_fit_segmented_and_predict_preserves_characters() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let result = segmenter.predict(vec!["你好世界".into()]);
        let reconstructed: String = result[0].iter().cloned().collect();
        assert_eq!(reconstructed, "你好世界");
    }

    #[test]
    fn test_predict_empty_input() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let result = segmenter.predict(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_predict_empty_string() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let result = segmenter.predict(vec!["".into()]);
        assert_eq!(result, vec![Vec::<String>::new()]);
    }

    #[test]
    fn test_fit_segmented_empty_data() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(vec![]);
        assert!(segmenter.hmm().fitted());
    }

    #[test]
    fn test_predict_multiple_sentences() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let result = segmenter.predict(vec!["你好".into(), "世界".into()]);
        assert_eq!(result.len(), 2);
        let r0: String = result[0].iter().cloned().collect();
        let r1: String = result[1].iter().cloned().collect();
        assert_eq!(r0, "你好");
        assert_eq!(r1, "世界");
    }

    #[test]
    fn test_deterministic() {
        let mut seg1 = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        seg1.fit_segmented(training_data());
        let result1 = seg1.predict(vec!["你好世界".into()]);

        let mut seg2 = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        seg2.fit_segmented(training_data());
        let result2 = seg2.predict(vec!["你好世界".into()]);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_save_and_load() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        let path_str = path.to_str().unwrap();

        segmenter.save_to_path(path_str).unwrap();

        let mut loaded = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        loaded.load_from_path(path_str).unwrap();

        let input = vec!["你好世界".into(), "我喜歡你".into()];
        assert_eq!(segmenter.predict(input.clone()), loaded.predict(input));
    }

    #[test]
    fn test_fit_unsegmented_after_fit_segmented() {
        let mut segmenter =
            HiddenMarkovModelSegmenter::new(Some(5), Some(1e-4), None, Some(42), None);
        segmenter.fit_segmented(training_data());
        segmenter.fit_unsegmented(vec!["你好世界我喜歡你".into()]);
        let result = segmenter.predict(vec!["你好世界".into()]);
        let reconstructed: String = result[0].iter().cloned().collect();
        assert_eq!(reconstructed, "你好世界");
    }

    #[test]
    fn test_fit_unsegmented_deterministic_with_seed() {
        let data = training_data();
        let unseg = vec!["你好世界我喜歡你".into(), "他是一個好人".into()];

        let mut s1 = HiddenMarkovModelSegmenter::new(Some(3), None, None, Some(42), None);
        s1.fit_segmented(data.clone());
        s1.fit_unsegmented(unseg.clone());

        let mut s2 = HiddenMarkovModelSegmenter::new(Some(3), None, None, Some(42), None);
        s2.fit_segmented(data);
        s2.fit_unsegmented(unseg);

        assert_eq!(
            s1.predict(vec!["你好世界".into()]),
            s2.predict(vec!["你好世界".into()])
        );
    }

    #[test]
    fn test_fit_unsegmented_empty_input() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(Some(5), None, None, Some(42), None);
        segmenter.fit_segmented(training_data());
        let before = segmenter.predict(vec!["你好世界".into()]);
        segmenter.fit_unsegmented(vec![]);
        let after = segmenter.predict(vec!["你好世界".into()]);
        assert_eq!(before, after);
    }

    #[test]
    fn test_score_not_fitted() {
        let segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        let result = segmenter.score(vec![vec!["你好".into(), "世界".into()]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_empty_input() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let result = segmenter.score(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_score_returns_finite_values() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        let scores = segmenter
            .score(vec![
                vec!["你好".into(), "世界".into()],
                vec!["我".into(), "喜歡".into(), "你".into()],
            ])
            .unwrap();
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(s.is_finite(), "score should be finite, got {}", s);
        }
    }

    #[test]
    fn test_score_training_vs_arbitrary() {
        let mut segmenter = HiddenMarkovModelSegmenter::new(None, None, None, None, None);
        segmenter.fit_segmented(training_data());
        // Training-data segmentation.
        let good = segmenter
            .score(vec![vec!["你好".into(), "世界".into()]])
            .unwrap()[0];
        // Arbitrary segmentation of the same characters.
        let bad = segmenter
            .score(vec![vec!["你".into(), "好世".into(), "界".into()]])
            .unwrap()[0];
        // Both should be finite (Forward algorithm sums over all state paths,
        // so the score depends on the character sequence, not the segmentation).
        assert!(good.is_finite());
        assert!(bad.is_finite());
        // The character sequence is the same, so scores should be equal.
        assert!(
            (good - bad).abs() < 1e-10,
            "same char sequence should give same score: {} vs {}",
            good,
            bad
        );
    }

    #[test]
    fn test_index_to_bmes_label_roundtrip() {
        for (expected_idx, label) in [(0, "B"), (1, "E"), (2, "M"), (3, "S")] {
            assert_eq!(index_to_bmes_label(expected_idx), label);
        }
    }
}
