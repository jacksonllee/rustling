//! TextGrid (Praat) file parsing.
//!
//! This module provides a parser for Praat TextGrid annotation files
//! and data structures for accessing tiers, intervals, and points.

mod chat_writer;
mod elan_writer;
mod reader;
#[cfg(feature = "pyo3")]
mod reader_py;
mod srt_writer;

pub use reader::{
    BaseTextGrid, Interval, Point, TextGrid, TextGridError, TextGridFile, TextGridTier, WriteError,
    parse_textgrid_str, serialize_textgrid_file,
};
#[cfg(feature = "pyo3")]
pub use reader_py::{PyInterval, PyIntervalTier, PyPoint, PyTextGrid, PyTextTier};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Register the textgrid submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let textgrid_module = PyModule::new(parent_module.py(), "textgrid")?;
    textgrid_module.add_class::<PyTextGrid>()?;
    textgrid_module.add_class::<PyIntervalTier>()?;
    textgrid_module.add_class::<PyTextTier>()?;
    textgrid_module.add_class::<PyInterval>()?;
    textgrid_module.add_class::<PyPoint>()?;
    parent_module.add_submodule(&textgrid_module)?;
    Ok(())
}
