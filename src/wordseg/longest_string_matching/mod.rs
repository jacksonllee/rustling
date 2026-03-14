//! Longest string matching word segmenter.

#[cfg(feature = "pyo3")]
mod py;
#[cfg(feature = "pyo3")]
pub use py::PyLongestStringMatching;

use crate::trie::Trie;
use flatbuffers;
use std::io::Write;

// FlatBuffers generated code (produced by build.rs from src/wordseg/longest_string_matching/model.fbs).
#[allow(dead_code, unused_imports, clippy::all)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/lsm/model_generated.rs"));
}

use crate::persistence::ModelError;

// ---------------------------------------------------------------------------
// BaseLongestStringMatching
// ---------------------------------------------------------------------------

/// Core longest-string-matching behavior with default implementations.
///
/// Implementors provide required methods that grant access to internal state.
/// All segmentation logic is provided as defaults.
pub trait BaseLongestStringMatching: Sized + Clone + Sync {
    fn max_word_length(&self) -> usize;
    fn trie(&self) -> &Trie<char, ()>;
    fn trie_mut(&mut self) -> &mut Trie<char, ()>;
    fn from_parts(max_word_length: usize, trie: Trie<char, ()>) -> Self;

    /// Train the model with the input segmented sentences.
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        *self.trie_mut() = Trie::new();
        for sent in sents {
            for word in sent {
                if word.chars().count() > 1 {
                    self.trie_mut().insert_seq(word.chars());
                }
            }
        }
    }

    /// Segment the given unsegmented sentences.
    fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            sent_strs
                .into_par_iter()
                .with_min_len(16)
                .map(|sent_str| self.predict_sent(&sent_str))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            sent_strs
                .into_iter()
                .map(|sent_str| self.predict_sent(&sent_str))
                .collect()
        }
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    fn save_to_path(&self, path: &str) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        save_lsm_flatbuffers(self, &mut buf)?;
        crate::persistence::save_zstd(path, &buf)
    }

    fn load_from_path(&mut self, path: &str) -> Result<(), ModelError> {
        let bytes = crate::persistence::load_zstd(path, "longest string matching model")?;
        load_lsm_flatbuffers(self, &bytes)
    }

    /// Segment a single unsegmented sentence using the trie.
    fn predict_sent(&self, sent_str: &str) -> Vec<String> {
        let chars: Vec<char> = sent_str.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }

        let estimated_words = (chars.len() / 3).max(1);
        let mut sent_predicted = Vec::with_capacity(estimated_words);

        let mut i = 0;

        while i < chars.len() {
            let remaining = &chars[i..];
            let max_len = std::cmp::min(remaining.len(), self.max_word_length());
            let match_len = self.trie().longest_match(remaining, max_len);

            if match_len > 0 {
                let word: String = chars[i..i + match_len].iter().collect();
                sent_predicted.push(word);
                i += match_len;
            } else {
                sent_predicted.push(chars[i].to_string());
                i += 1;
            }
        }

        sent_predicted
    }

    /// Segment unsegmented sentences and return words with character offsets.
    fn predict_with_offsets(&self, sent_strs: Vec<String>) -> Vec<Vec<(String, (usize, usize))>> {
        let words = self.predict(sent_strs);
        crate::wordseg::attach_offsets(words)
    }
}

// ---------------------------------------------------------------------------
// FlatBuffers save / load
// ---------------------------------------------------------------------------

fn save_lsm_flatbuffers<T: BaseLongestStringMatching, W: Write>(
    model: &T,
    writer: &mut W,
) -> Result<(), ModelError> {
    use generated::rustling::lsm_fbs as fbs;

    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(512 * 1024);

    let mut words: Vec<String> = model
        .trie()
        .all_sequences()
        .into_iter()
        .map(|chars| chars.into_iter().collect())
        .collect();
    words.sort();

    let word_strs: Vec<_> = words.iter().map(|w| builder.create_string(w)).collect();
    let words_fb = builder.create_vector(&word_strs);

    let lsm = fbs::LsmModel::create(
        &mut builder,
        &fbs::LsmModelArgs {
            max_word_length: model.max_word_length() as u32,
            words: Some(words_fb),
        },
    );
    builder.finish(lsm, None);

    writer
        .write_all(builder.finished_data())
        .map_err(|e| ModelError::Io(format!("Failed to write FlatBuffers data: {e}")))
}

fn load_lsm_flatbuffers<T: BaseLongestStringMatching>(
    model: &mut T,
    bytes: &[u8],
) -> Result<(), ModelError> {
    use generated::rustling::lsm_fbs as fbs;

    let opts = crate::persistence::flatbuffers_verifier_opts();
    let lsm = flatbuffers::root_with_opts::<fbs::LsmModel>(&opts, bytes)
        .map_err(|e| ModelError::ParseError(format!("Invalid FlatBuffers LSM data: {e}")))?;

    let mut trie = Trie::new();
    for word in lsm.words().iter() {
        trie.insert_seq(word.chars());
    }
    *model.trie_mut() = trie;

    Ok(())
}

// ---------------------------------------------------------------------------
// Pure Rust struct
// ---------------------------------------------------------------------------

/// Longest string matching segmenter.
///
/// This model constructs predicted words by moving from left to right
/// along an unsegmented sentence and finding the longest matching words,
/// constrained by a maximum word length parameter.
///
/// For Python, use [`PyLongestStringMatching`].
#[derive(Clone, Debug)]
pub struct LongestStringMatching {
    max_word_length: usize,
    trie: Trie<char, ()>,
}

impl BaseLongestStringMatching for LongestStringMatching {
    fn max_word_length(&self) -> usize {
        self.max_word_length
    }
    fn trie(&self) -> &Trie<char, ()> {
        &self.trie
    }
    fn trie_mut(&mut self) -> &mut Trie<char, ()> {
        &mut self.trie
    }
    fn from_parts(max_word_length: usize, trie: Trie<char, ()>) -> Self {
        Self {
            max_word_length,
            trie,
        }
    }
}

impl LongestStringMatching {
    /// Create a new longest string matching segmenter.
    ///
    /// # Arguments
    ///
    /// * `max_word_length` - Maximum word length in the segmented sentences during prediction.
    ///   Must be >= 2 to be meaningful.
    pub fn new(max_word_length: usize) -> Result<Self, ModelError> {
        if max_word_length < 2 {
            return Err(ModelError::ValidationError(format!(
                "max_word_length must be >= 2 to be meaningful: {}",
                max_word_length
            )));
        }
        Ok(Self {
            max_word_length,
            trie: Trie::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let model = LongestStringMatching::new(4);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.max_word_length, 4);
    }

    #[test]
    fn test_new_invalid_max_word_length() {
        let result = LongestStringMatching::new(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_basic() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![
            vec![
                "this".to_string(),
                "is".to_string(),
                "a".to_string(),
                "sentence".to_string(),
            ],
            vec![
                "that".to_string(),
                "is".to_string(),
                "not".to_string(),
                "a".to_string(),
                "sentence".to_string(),
            ],
        ]);

        let result = model.predict(vec!["thatisadog".to_string(), "thisisnotacat".to_string()]);

        assert_eq!(
            result,
            vec![
                vec!["that", "is", "a", "d", "o", "g"],
                vec!["this", "is", "not", "a", "c", "a", "t"],
            ]
        );
    }

    #[test]
    fn test_empty_input() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![vec!["hello".to_string(), "world".to_string()]]);

        let result = model.predict(vec!["".to_string()]);
        assert_eq!(result, vec![Vec::<String>::new()]);
    }

    #[test]
    fn test_no_training_data() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![]);

        let result = model.predict(vec!["hello".to_string()]);
        assert_eq!(result, vec![vec!["h", "e", "l", "l", "o"]]);
    }

    #[test]
    fn test_single_char_words_ignored_in_training() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![vec![
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ]]);

        let result = model.predict(vec!["abab".to_string()]);
        assert_eq!(result, vec![vec!["ab", "ab"]]);
    }

    #[test]
    fn test_unicode_chars() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![vec!["你好".to_string(), "世界".to_string()]]);

        let result = model.predict(vec!["你好世界".to_string()]);
        assert_eq!(result, vec![vec!["你好", "世界"]]);
    }

    #[test]
    fn test_max_word_length_constraint() {
        let mut model = LongestStringMatching::new(3).unwrap();
        model.fit(vec![vec!["hello".to_string()]]);

        let result = model.predict(vec!["hello".to_string()]);
        assert_eq!(result, vec![vec!["h", "e", "l", "l", "o"]]);
    }

    #[test]
    fn test_save_and_load() {
        let mut model = LongestStringMatching::new(4).unwrap();
        model.fit(vec![
            vec![
                "this".to_string(),
                "is".to_string(),
                "a".to_string(),
                "sentence".to_string(),
            ],
            vec![
                "that".to_string(),
                "is".to_string(),
                "not".to_string(),
                "a".to_string(),
                "sentence".to_string(),
            ],
        ]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let path_str = path.to_str().unwrap();

        model.save_to_path(path_str).unwrap();

        let mut loaded = LongestStringMatching::new(4).unwrap();
        loaded.load_from_path(path_str).unwrap();

        let input = vec!["thatisadog".to_string(), "thisisnotacat".to_string()];
        assert_eq!(model.predict(input.clone()), loaded.predict(input));
    }
}
