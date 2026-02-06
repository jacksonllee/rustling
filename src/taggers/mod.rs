//! POS taggers.
//!
//! This module provides part-of-speech taggers that can be trained on
//! tagged sentences and used to predict POS tags for new text.
//!
//! ## Example
//!
//! ```rust
//! use rustling::taggers::AveragedPerceptronTagger;
//!
//! // Create a tagger with default parameters
//! // (frequency_threshold=20, ambiguity_threshold=0.97, n_iter=5)
//! let mut tagger = AveragedPerceptronTagger::new(20, 0.97, 5);
//!
//! // Training data: Vec of sentences, each sentence is Vec of (word, tag) tuples
//! let training_data = vec![
//!     vec![("I".to_string(), "PRP".to_string()),
//!          ("love".to_string(), "VBP".to_string()),
//!          ("Rust".to_string(), "NNP".to_string())],
//!     vec![("Rust".to_string(), "NNP".to_string()),
//!          ("is".to_string(), "VBZ".to_string()),
//!          ("fast".to_string(), "JJ".to_string())],
//! ];
//!
//! // Train the tagger
//! tagger.train(training_data);
//!
//! // Tag a sentence
//! let sentence = vec!["I".to_string(), "love".to_string(), "Rust".to_string()];
//! let tags = tagger.tag(sentence);
//! println!("{:?}", tags);
//! // ["PRP", "VBP", "NNP"]
//! ```

mod averaged_perceptron_tagger;

pub use averaged_perceptron_tagger::AveragedPerceptronTagger;

use pyo3::prelude::*;

/// Register the taggers submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let taggers_module = PyModule::new(parent_module.py(), "taggers")?;
    taggers_module.add_class::<AveragedPerceptronTagger>()?;
    parent_module.add_submodule(&taggers_module)?;
    Ok(())
}
