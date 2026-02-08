//! Language models.
//!
//! This module provides n-gram language models that can be trained on
//! tokenized text and used to score and generate word sequences.
//!
//! ## Example
//!
//! ```rust
//! use rustling::lm::LanguageModel;
//!
//! // Create a bigram MLE language model
//! let mut model = LanguageModel::new_mle(2).unwrap();
//! model.fit(vec![
//!     vec!["the".into(), "cat".into(), "sat".into()],
//!     vec!["the".into(), "dog".into(), "ran".into()],
//! ]);
//! let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
//! assert!((score - 0.5).abs() < 1e-9);
//! ```

mod model;

pub use model::LanguageModel;

use pyo3::prelude::*;

/// Register the lm submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let lm_module = PyModule::new(parent_module.py(), "lm")?;
    lm_module.add_class::<LanguageModel>()?;
    parent_module.add_submodule(&lm_module)?;
    Ok(())
}
