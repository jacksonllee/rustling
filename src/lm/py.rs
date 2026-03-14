use std::path::PathBuf;

use pyo3::prelude::*;

use super::{BaseLanguageModel, Laplace, Lidstone, MLE, Smoothing, Vocabulary};
use crate::persistence::pathbuf_to_string;
use crate::trie::CountTrie;

// ---------------------------------------------------------------------------
// PyMLE
// ---------------------------------------------------------------------------

/// Python-exposed MLE language model.
#[pyclass(name = "MLE", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyMLE {
    pub inner: MLE,
}

impl BaseLanguageModel for PyMLE {
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn smoothing(&self) -> &Smoothing {
        self.inner.smoothing()
    }
    fn smoothing_name(&self) -> &str {
        self.inner.smoothing_name()
    }
    fn vocabulary(&self) -> &Vocabulary {
        self.inner.vocabulary()
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        self.inner.vocabulary_mut()
    }
    fn counts(&self) -> &CountTrie<String> {
        self.inner.counts()
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        self.inner.counts_mut()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.inner.set_fitted(fitted);
    }
}

#[pymethods]
impl PyMLE {
    /// Initialize an MLE language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    #[new]
    #[pyo3(signature = (*, order))]
    fn new(order: usize) -> PyResult<Self> {
        let inner = MLE::new(order).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Train the language model on tokenized sentences.
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        BaseLanguageModel::fit(self, sents);
    }

    /// Return the probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::score(self, word, context).map_err(PyErr::from)
    }

    /// Return the probability of a word given a context, without OOV mapping.
    #[pyo3(signature = (word, context=None))]
    fn unmasked_score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::unmasked_score(self, word, context).map_err(PyErr::from)
    }

    /// Return the log (base 2) probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::logscore(self, word, context).map_err(PyErr::from)
    }

    /// Generate words from the language model.
    #[pyo3(signature = (*, num_words = 1, text_seed = None, random_seed = None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        BaseLanguageModel::generate(self, num_words, text_seed, random_seed).map_err(PyErr::from)
    }

    /// The order of the n-gram model.
    #[getter]
    fn order(&self) -> usize {
        BaseLanguageModel::order(self)
    }

    /// The vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        BaseLanguageModel::vocab_size(self)
    }

    /// Save the model to a zstd-compressed FlatBuffers binary.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.save_to_path(&path).map_err(PyErr::from)
    }

    /// Load a model from a zstd-compressed FlatBuffers binary.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.load_from_path(&path).map_err(PyErr::from)
    }
}

// ---------------------------------------------------------------------------
// PyLidstone
// ---------------------------------------------------------------------------

/// Python-exposed Lidstone language model.
#[pyclass(name = "Lidstone", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyLidstone {
    pub inner: Lidstone,
}

impl BaseLanguageModel for PyLidstone {
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn smoothing(&self) -> &Smoothing {
        self.inner.smoothing()
    }
    fn smoothing_name(&self) -> &str {
        self.inner.smoothing_name()
    }
    fn vocabulary(&self) -> &Vocabulary {
        self.inner.vocabulary()
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        self.inner.vocabulary_mut()
    }
    fn counts(&self) -> &CountTrie<String> {
        self.inner.counts()
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        self.inner.counts_mut()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.inner.set_fitted(fitted);
    }
}

#[pymethods]
impl PyLidstone {
    /// Initialize a Lidstone language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    /// * `gamma` - The smoothing parameter. Must be > 0.
    #[new]
    #[pyo3(signature = (*, order, gamma))]
    fn new(order: usize, gamma: f64) -> PyResult<Self> {
        let inner = Lidstone::new(order, gamma).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Train the language model on tokenized sentences.
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        BaseLanguageModel::fit(self, sents);
    }

    /// Return the probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::score(self, word, context).map_err(PyErr::from)
    }

    /// Return the probability of a word given a context, without OOV mapping.
    #[pyo3(signature = (word, context=None))]
    fn unmasked_score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::unmasked_score(self, word, context).map_err(PyErr::from)
    }

    /// Return the log (base 2) probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::logscore(self, word, context).map_err(PyErr::from)
    }

    /// Generate words from the language model.
    #[pyo3(signature = (*, num_words = 1, text_seed = None, random_seed = None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        BaseLanguageModel::generate(self, num_words, text_seed, random_seed).map_err(PyErr::from)
    }

    /// The order of the n-gram model.
    #[getter]
    fn order(&self) -> usize {
        BaseLanguageModel::order(self)
    }

    /// The vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        BaseLanguageModel::vocab_size(self)
    }

    /// The gamma (smoothing) parameter.
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Save the model to a zstd-compressed FlatBuffers binary.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.save_to_path(&path).map_err(PyErr::from)
    }

    /// Load a model from a zstd-compressed FlatBuffers binary.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.load_from_path(&path).map_err(PyErr::from)
    }
}

// ---------------------------------------------------------------------------
// PyLaplace
// ---------------------------------------------------------------------------

/// Python-exposed Laplace language model.
#[pyclass(name = "Laplace", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyLaplace {
    pub inner: Laplace,
}

impl BaseLanguageModel for PyLaplace {
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn smoothing(&self) -> &Smoothing {
        self.inner.smoothing()
    }
    fn smoothing_name(&self) -> &str {
        self.inner.smoothing_name()
    }
    fn vocabulary(&self) -> &Vocabulary {
        self.inner.vocabulary()
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        self.inner.vocabulary_mut()
    }
    fn counts(&self) -> &CountTrie<String> {
        self.inner.counts()
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        self.inner.counts_mut()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.inner.set_fitted(fitted);
    }
}

#[pymethods]
impl PyLaplace {
    /// Initialize a Laplace language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    #[new]
    #[pyo3(signature = (*, order))]
    fn new(order: usize) -> PyResult<Self> {
        let inner = Laplace::new(order).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Train the language model on tokenized sentences.
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        BaseLanguageModel::fit(self, sents);
    }

    /// Return the probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::score(self, word, context).map_err(PyErr::from)
    }

    /// Return the probability of a word given a context, without OOV mapping.
    #[pyo3(signature = (word, context=None))]
    fn unmasked_score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::unmasked_score(self, word, context).map_err(PyErr::from)
    }

    /// Return the log (base 2) probability of a word given a context.
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        BaseLanguageModel::logscore(self, word, context).map_err(PyErr::from)
    }

    /// Generate words from the language model.
    #[pyo3(signature = (*, num_words = 1, text_seed = None, random_seed = None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        BaseLanguageModel::generate(self, num_words, text_seed, random_seed).map_err(PyErr::from)
    }

    /// The order of the n-gram model.
    #[getter]
    fn order(&self) -> usize {
        BaseLanguageModel::order(self)
    }

    /// The vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        BaseLanguageModel::vocab_size(self)
    }

    /// Save the model to a zstd-compressed FlatBuffers binary.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.save_to_path(&path).map_err(PyErr::from)
    }

    /// Load a model from a zstd-compressed FlatBuffers binary.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.load_from_path(&path).map_err(PyErr::from)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the lm submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let lm_module = PyModule::new(parent_module.py(), "lm")?;
    lm_module.add_class::<PyMLE>()?;
    lm_module.add_class::<PyLidstone>()?;
    lm_module.add_class::<PyLaplace>()?;
    parent_module.add_submodule(&lm_module)?;
    Ok(())
}
