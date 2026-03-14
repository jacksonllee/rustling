use pyo3::prelude::*;
use std::path::PathBuf;

use super::{BaseLongestStringMatching, LongestStringMatching};
use crate::persistence::pathbuf_to_string;
use crate::trie::Trie;

/// Python-exposed wrapper. Python users see this as `LongestStringMatching`.
#[pyclass(name = "LongestStringMatching", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyLongestStringMatching {
    pub inner: LongestStringMatching,
}

impl BaseLongestStringMatching for PyLongestStringMatching {
    fn max_word_length(&self) -> usize {
        self.inner.max_word_length()
    }
    fn trie(&self) -> &Trie<char, ()> {
        self.inner.trie()
    }
    fn trie_mut(&mut self) -> &mut Trie<char, ()> {
        self.inner.trie_mut()
    }
    fn from_parts(max_word_length: usize, trie: Trie<char, ()>) -> Self {
        Self {
            inner: LongestStringMatching::from_parts(max_word_length, trie),
        }
    }
}

#[pymethods]
impl PyLongestStringMatching {
    /// Initialize a longest string matching segmenter.
    ///
    /// # Arguments
    ///
    /// * `max_word_length` - Maximum word length in the segmented sentences during prediction.
    ///                       Must be >= 2 to be meaningful.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If max_word_length is < 2.
    #[new]
    #[pyo3(signature = (*, max_word_length))]
    fn new(max_word_length: usize) -> PyResult<Self> {
        LongestStringMatching::new(max_word_length)
            .map(|inner| Self { inner })
            .map_err(PyErr::from)
    }

    /// Train the model with the input segmented sentences.
    ///
    /// No cleaning or preprocessing (e.g., normalizing upper/lowercase,
    /// tokenization) is performed on the training data.
    ///
    /// # Arguments
    ///
    /// * `sents` - An iterable of segmented sentences (each sentence is a list of words).
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        BaseLongestStringMatching::fit(self, sents);
    }

    /// Segment the given unsegmented sentences.
    ///
    /// # Arguments
    ///
    /// * `sent_strs` - An iterable of unsegmented sentences.
    /// * `offsets` - If true, return each word as (word, (start, end)).
    ///
    /// # Returns
    ///
    /// A list of segmented sentences.
    #[pyo3(signature = (sent_strs, *, offsets=false))]
    fn predict(
        &self,
        py: Python<'_>,
        sent_strs: Vec<String>,
        offsets: bool,
    ) -> PyResult<Py<PyAny>> {
        let words = BaseLongestStringMatching::predict(self, sent_strs);
        if offsets {
            let with_offsets = super::super::attach_offsets(words);
            Ok(with_offsets.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(words.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Save the model to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the model will be saved as a JSON file.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.save_to_path(&path).map_err(PyErr::from)
    }

    /// Load a model from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the model, stored as a JSON file, is located.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.load_from_path(&path).map_err(PyErr::from)
    }
}
