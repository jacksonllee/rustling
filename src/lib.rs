//! # Rustling
//!
//! Rustling is a blazingly fast library for computational linguistics.
//! It aims to provide flexible and efficient tools to facilitate further research.
//! It is written in Rust, with Python bindings.
//!
//! Rustling is fully functional for both Rust-only and Python-only users.
//! The objects defined and exposed in Rust correspond
//! to the same ones in Python under the comparable namespace.
//! For documentation, especially details about linguistics and modeling,
//! please see the Python docs: <https://docs.rustling.io>

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub mod chat;
pub mod conllu;
pub mod elan;
pub mod hmm;
pub mod lm;
pub mod ngram;
pub mod perceptron_pos_tagger;
pub mod persistence;
pub mod prelude;
pub mod seq_feature;
pub mod sources;
pub mod srt;
pub mod textgrid;
pub mod trie;
pub mod wordseg;

/// A Python module implemented in Rust.
#[cfg(feature = "pyo3")]
#[pymodule]
#[pyo3(name = "_lib_name")]
fn rustling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    chat::register_module(m)?;
    conllu::register_module(m)?;
    elan::register_module(m)?;
    hmm::register_module(m)?;
    srt::register_module(m)?;
    textgrid::register_module(m)?;
    seq_feature::register_module(m)?;
    lm::register_module(m)?;
    ngram::register_module(m)?;
    perceptron_pos_tagger::register_module(m)?;
    wordseg::register_module(m)?;
    Ok(())
}
