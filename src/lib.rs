//! # Rustling
//!
//! Rustling is a blazingly fast library for computational linguistics.
//! It is written in Rust, with Python bindings.
//!
//! Rustling is fully functional for both Rust-only and Python-only users.
//! The objects defined and exposed in Rust correspond
//! to the same ones in Python under the comparable namespace.
//! For documentation, especially details about linguistics and modeling,
//! please see the Python docs: <https://rustling.readthedocs.io>

use pyo3::prelude::*;

pub mod chat;
pub mod hmm;
pub mod lm;
pub mod ngram;
pub mod perceptron_pos_tagger;
pub mod persistence;
pub mod seq_feature;
pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    chat::register_module(m)?;
    hmm::register_module(m)?;
    seq_feature::register_module(m)?;
    lm::register_module(m)?;
    ngram::register_module(m)?;
    perceptron_pos_tagger::register_module(m)?;
    wordseg::register_module(m)?;
    Ok(())
}
