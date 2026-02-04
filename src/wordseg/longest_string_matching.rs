//! Longest string matching word segmenter.

use crate::trie::Trie;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Longest string matching segmenter.
///
/// This model constructs predicted words by moving from left to right
/// along an unsegmented sentence and finding the longest matching words,
/// constrained by a maximum word length parameter.
#[pyclass]
#[derive(Clone)]
pub struct LongestStringMatching {
    max_word_length: usize,
    trie: Trie<char>,
}

#[pymethods]
impl LongestStringMatching {
    /// Initialize a longest string matching segmenter.
    ///
    /// # Arguments
    ///
    /// * `max_word_length` - Maximum word length in the segmented sentences during prediction.
    ///                       Must be >= 2 to be meaningful.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If max_word_length is < 2.
    #[new]
    #[pyo3(signature = (*, max_word_length))]
    pub fn new(max_word_length: usize) -> PyResult<Self> {
        if max_word_length < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "max_word_length must be >= 2 to be meaningful: {}",
                max_word_length
            )));
        }
        Ok(Self {
            max_word_length,
            trie: Trie::new(),
        })
    }

    /// Train the model with the input segmented sentences.
    ///
    /// No cleaning or preprocessing (e.g., normalizing upper/lowercase,
    /// tokenization) is performed on the training data.
    ///
    /// # Arguments
    ///
    /// * `sents` - An iterable of segmented sentences (each sentence is a list of words).
    pub fn fit(&mut self, sents: Vec<Vec<String>>) {
        self.trie = Trie::new();

        for sent in sents {
            for word in sent {
                // Don't waste memory for words of length 1,
                // which are practically useless for this algorithm.
                if word.chars().count() > 1 {
                    self.trie.insert(word.chars());
                }
            }
        }
    }

    /// Segment the given unsegmented sentences.
    ///
    /// # Arguments
    ///
    /// * `sent_strs` - An iterable of unsegmented sentences.
    ///
    /// # Returns
    ///
    /// A list of segmented sentences.
    pub fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        // Use parallel iteration for better performance on multiple sentences
        sent_strs
            .into_par_iter()
            .map(|sent_str| self.predict_sent(&sent_str))
            .collect()
    }
}

impl LongestStringMatching {
    /// Segment a single unsegmented sentence using the trie.
    fn predict_sent(&self, sent_str: &str) -> Vec<String> {
        let chars: Vec<char> = sent_str.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }

        // Pre-allocate with estimated capacity (assume avg word length of 3-4 chars)
        let estimated_words = (chars.len() / 3).max(1);
        let mut sent_predicted = Vec::with_capacity(estimated_words);

        let mut i = 0;

        while i < chars.len() {
            // Use trie to find longest match starting at position i
            let remaining = &chars[i..];
            let max_len = std::cmp::min(remaining.len(), self.max_word_length);
            let match_len = self.trie.longest_match(remaining, max_len);

            if match_len > 0 {
                // Found a word in the trie
                let word: String = chars[i..i + match_len].iter().collect();
                sent_predicted.push(word);
                i += match_len;
            } else {
                // No match found, emit single character
                sent_predicted.push(chars[i].to_string());
                i += 1;
            }
        }

        sent_predicted
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
        // Single-character words should be ignored in training
        model.fit(vec![vec![
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ]]);

        // "ab" should be recognized, but not single chars
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
        // Train with a word longer than max_word_length
        model.fit(vec![vec!["hello".to_string()]]);

        // Even though "hello" is in training, we can only match up to 3 chars
        // "hel" is not in training, so we fall back character by character
        let result = model.predict(vec!["hello".to_string()]);
        assert_eq!(result, vec![vec!["h", "e", "l", "l", "o"]]);
    }
}
