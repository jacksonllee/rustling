//! CHAT parsing.
//!
//! This module provides a parser for CHAT transcription files
//! (CHILDES/TalkBank format) and data structures for accessing
//! utterances, tokens, and annotations.

mod clean_utterance;
pub(crate) mod header;
mod ipsyn;
mod reader;
mod utterance;

pub use header::{Age, ChangeableHeader, Headers, Participant};
pub use reader::Chat;
pub use utterance::{Gra, Token, Utterance, Utterances};

use pyo3::prelude::*;

/// Register the chat submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let chat_module = PyModule::new(parent_module.py(), "chat")?;
    chat_module.add_class::<Chat>()?;
    chat_module.add_class::<Token>()?;
    chat_module.add_class::<Gra>()?;
    chat_module.add_class::<Utterance>()?;
    chat_module.add_class::<Utterances>()?;
    chat_module.add_class::<Headers>()?;
    chat_module.add_class::<Participant>()?;
    chat_module.add_class::<Age>()?;
    chat_module.add_class::<ChangeableHeader>()?;
    parent_module.add_submodule(&chat_module)?;
    Ok(())
}
