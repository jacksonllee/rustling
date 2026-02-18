//! N-gram counting.
//!
//! This module provides an n-gram counter for counting n-gram
//! frequencies from sequential data.

mod counter;

pub use counter::Ngrams;

use pyo3::prelude::*;

/// Register the ngram submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let ngram_module = PyModule::new(parent_module.py(), "ngram")?;
    ngram_module.add_class::<Ngrams>()?;
    parent_module.add_submodule(&ngram_module)?;
    Ok(())
}
