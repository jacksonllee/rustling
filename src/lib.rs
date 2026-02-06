//! # Rustling
//!
//! Rustling is a blazingly fast library of tools for computational linguistics.

use pyo3::prelude::*;

pub mod taggers;
pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    taggers::register_module(m)?;
    wordseg::register_module(m)?;
    Ok(())
}
