//! Word segmentation models.
//!
//! This module provides word segmentation models that can be trained on
//! segmented sentences and used to predict segmentation of unsegmented text.
//!
//! ## Example
//!
//! ```rust
//! use rustling::wordseg::{LongestStringMatching, RandomSegmenter};
//! use rustling::wordseg::{BaseLongestStringMatching, BaseRandomSegmenter};
//!
//! // Longest String Matching
//! let mut model = LongestStringMatching::new(4).unwrap();
//! model.fit(vec![
//!     vec!["this".into(), "is".into(), "a".into(), "sentence".into()],
//!     vec!["that".into(), "is".into(), "not".into(), "a".into(), "sentence".into()],
//! ]);
//! let result = model.predict(vec!["thatisadog".into(), "thisisnotacat".into()]);
//! println!("{:?}", result);
//! // [["that", "is", "a", "d", "o", "g"], ["this", "is", "not", "a", "c", "a", "t"]]
//!
//! // Random Segmenter (no training needed)
//! let segmenter = RandomSegmenter::new(0.3).unwrap();
//! let result = segmenter.predict(vec!["helloworld".into()]);
//! println!("{:?}", result);
//! // e.g., [["hel", "lo", "wor", "ld"]] (varies due to randomness)
//! ```

pub(crate) mod bmes;
mod dag_hmm;
#[cfg(feature = "pyo3")]
mod dag_hmm_py;
mod hmm;
#[cfg(feature = "pyo3")]
mod hmm_py;
mod longest_string_matching;
mod random_segmenter;
#[cfg(feature = "pyo3")]
mod random_segmenter_py;

pub use dag_hmm::DagHmmSegmenter;
#[cfg(feature = "pyo3")]
pub use dag_hmm_py::PyDagHmmSegmenter;
pub use hmm::{BaseHiddenMarkovModelSegmenter, HiddenMarkovModelSegmenter};
#[cfg(feature = "pyo3")]
pub use hmm_py::PyHiddenMarkovModelSegmenter;
#[cfg(feature = "pyo3")]
pub use longest_string_matching::PyLongestStringMatching;
pub use longest_string_matching::{BaseLongestStringMatching, LongestStringMatching};
pub use random_segmenter::{BaseRandomSegmenter, RandomSegmenter};
#[cfg(feature = "pyo3")]
pub use random_segmenter_py::PyRandomSegmenter;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Attach (start, end) character offsets to segmented words.
///
/// Words are assumed contiguous and covering the full input.
/// Offsets use exclusive end (Python slice convention).
pub fn attach_offsets(sents: Vec<Vec<String>>) -> Vec<Vec<(String, (usize, usize))>> {
    sents
        .into_iter()
        .map(|words| {
            let mut offset = 0usize;
            words
                .into_iter()
                .map(|word| {
                    let start = offset;
                    offset += word.chars().count();
                    (word, (start, offset))
                })
                .collect()
        })
        .collect()
}

/// Register the wordseg submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let wordseg_module = PyModule::new(parent_module.py(), "wordseg")?;
    wordseg_module.add_class::<PyDagHmmSegmenter>()?;
    wordseg_module.add_class::<PyHiddenMarkovModelSegmenter>()?;
    wordseg_module.add_class::<PyLongestStringMatching>()?;
    wordseg_module.add_class::<PyRandomSegmenter>()?;
    parent_module.add_submodule(&wordseg_module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attach_offsets_basic() {
        let words = vec![vec!["你好".into(), "世界".into()]];
        let result = attach_offsets(words);
        assert_eq!(
            result,
            vec![vec![("你好".into(), (0, 2)), ("世界".into(), (2, 4)),]]
        );
    }

    #[test]
    fn test_attach_offsets_ascii() {
        let words = vec![vec!["hello".into(), "world".into()]];
        let result = attach_offsets(words);
        assert_eq!(
            result,
            vec![vec![("hello".into(), (0, 5)), ("world".into(), (5, 10)),]]
        );
    }

    #[test]
    fn test_attach_offsets_multiple_sents() {
        let words = vec![
            vec!["ab".into(), "cd".into()],
            vec!["x".into(), "yz".into()],
        ];
        let result = attach_offsets(words);
        assert_eq!(
            result,
            vec![
                vec![("ab".into(), (0, 2)), ("cd".into(), (2, 4))],
                vec![("x".into(), (0, 1)), ("yz".into(), (1, 3))],
            ]
        );
    }

    #[test]
    fn test_attach_offsets_empty() {
        let result = attach_offsets(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_attach_offsets_empty_sent() {
        let result = attach_offsets(vec![vec![]]);
        assert_eq!(result, vec![Vec::<(String, (usize, usize))>::new()]);
    }
}
