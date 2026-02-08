//! # Rustling
//!
//! Rustling is a blazingly fast library for computational linguistics.

use pyo3::prelude::*;

pub mod lm;
pub mod tagging;
pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lm::register_module(m)?;
    tagging::register_module(m)?;
    wordseg::register_module(m)?;
    Ok(())
}
