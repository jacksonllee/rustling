//! # Rustling
//!
//! Rustling is a blazingly fast library for computational linguistics.

use pyo3::prelude::*;

pub mod chat;
pub mod lm;
pub mod ngram;
pub mod tagging;
pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    chat::register_module(m)?;
    lm::register_module(m)?;
    ngram::register_module(m)?;
    tagging::register_module(m)?;
    wordseg::register_module(m)?;
    Ok(())
}
