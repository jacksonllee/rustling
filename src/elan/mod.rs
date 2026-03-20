//! ELAN (.eaf) file parsing.
//!
//! This module provides a parser for ELAN annotation files
//! and data structures for accessing tiers and annotations.

mod chat_writer;
mod reader;
#[cfg(feature = "pyo3")]
mod reader_py;
mod srt_writer;
mod textgrid_writer;

pub use reader::{
    Annotation, BaseElan, Elan, ElanError, ElanFile, Tier, WriteError, parse_eaf_str,
    serialize_eaf_file,
};
#[cfg(feature = "pyo3")]
pub use reader_py::{PyAnnotation, PyElan, PyTier};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Register the elan submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let elan_module = PyModule::new(parent_module.py(), "elan")?;
    elan_module.add_class::<PyElan>()?;
    elan_module.add_class::<PyTier>()?;
    elan_module.add_class::<PyAnnotation>()?;
    parent_module.add_submodule(&elan_module)?;
    Ok(())
}
