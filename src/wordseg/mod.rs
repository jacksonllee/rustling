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
mod hmm;
mod longest_string_matching;
mod random_segmenter;

pub use dag_hmm::{DagHmmSegmenter, PyDagHmmSegmenter};
pub use hmm::{
    BaseHiddenMarkovModelSegmenter, HiddenMarkovModelSegmenter, PyHiddenMarkovModelSegmenter,
};
pub use longest_string_matching::{
    BaseLongestStringMatching, LongestStringMatching, PyLongestStringMatching,
};
pub use random_segmenter::{BaseRandomSegmenter, PyRandomSegmenter, RandomSegmenter};

use pyo3::prelude::*;

/// Register the wordseg submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let wordseg_module = PyModule::new(parent_module.py(), "wordseg")?;
    wordseg_module.add_class::<PyDagHmmSegmenter>()?;
    wordseg_module.add_class::<PyHiddenMarkovModelSegmenter>()?;
    wordseg_module.add_class::<PyLongestStringMatching>()?;
    wordseg_module.add_class::<PyRandomSegmenter>()?;
    parent_module.add_submodule(&wordseg_module)?;
    Ok(())
}
