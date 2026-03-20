//! CHAT parsing.
//!
//! This module provides a parser for CHAT transcription files
//! (CHILDES/TalkBank format) and data structures for accessing
//! utterances, tokens, and annotations.

mod clean_utterance;
mod conllu_writer;
mod elan_writer;
pub(crate) mod header;
#[cfg(feature = "pyo3")]
mod header_py;
mod ipsyn;
mod reader;
#[cfg(feature = "pyo3")]
mod reader_py;
mod srt_writer;
mod textgrid_writer;
mod utterance;
#[cfg(feature = "pyo3")]
mod utterance_py;
pub(crate) mod validation;

pub use header::{Age, ChangeableHeader, Headers, Participant};
pub use reader::{
    BaseChat, Chat, ChatError, ChatFile, MisalignmentInfo, WriteError, filter_file_paths,
    serialize_chat_file,
};
#[cfg(feature = "pyo3")]
pub use reader_py::{BasePyChat, PyChat};
pub use utterance::{BaseToken, BaseUtterance, Gra, Token, Utterance, Utterances};
#[cfg(feature = "pyo3")]
pub use utterance_py::{PyToken, PyUtterance, PyUtterances};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Register the chat submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let chat_module = PyModule::new(parent_module.py(), "chat")?;
    chat_module.add_class::<PyChat>()?;
    chat_module.add_class::<PyToken>()?;
    chat_module.add_class::<Gra>()?;
    chat_module.add_class::<PyUtterance>()?;
    chat_module.add_class::<PyUtterances>()?;
    chat_module.add_class::<Headers>()?;
    chat_module.add_class::<Participant>()?;
    chat_module.add_class::<Age>()?;
    chat_module.add_class::<ChangeableHeader>()?;
    parent_module.add_submodule(&chat_module)?;
    Ok(())
}
