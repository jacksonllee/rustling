#![doc = include_str!("../README.md")]

use pyo3::prelude::*;

pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    wordseg::register_module(m)?;
    Ok(())
}
