//! Word segmentation models.
//!
//! This module provides word segmentation models that can be trained on
//! segmented sentences and used to predict segmentation of unsegmented text.

mod longest_string_matching;
mod random_segmenter;

pub use longest_string_matching::LongestStringMatching;
pub use random_segmenter::RandomSegmenter;

use pyo3::prelude::*;

/// Register the wordseg submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let wordseg_module = PyModule::new(parent_module.py(), "wordseg")?;
    wordseg_module.add_class::<LongestStringMatching>()?;
    wordseg_module.add_class::<RandomSegmenter>()?;
    parent_module.add_submodule(&wordseg_module)?;
    Ok(())
}
