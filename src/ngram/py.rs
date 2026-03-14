use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::{BaseNgrams, Ngrams};
use crate::trie::CountTrie;

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-exposed wrapper. Python users see this as `Ngrams`.
#[pyclass(name = "Ngrams", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyNgrams {
    pub inner: Ngrams,
}

impl BaseNgrams for PyNgrams {
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn min_order(&self) -> usize {
        self.inner.min_order()
    }
    fn counts(&self) -> &CountTrie<String> {
        self.inner.counts()
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        self.inner.counts_mut()
    }
    fn totals(&self) -> &Vec<u64> {
        self.inner.totals()
    }
    fn totals_mut(&mut self) -> &mut Vec<u64> {
        self.inner.totals_mut()
    }
    fn from_parts(
        order: usize,
        min_order: usize,
        counts: CountTrie<String>,
        totals: Vec<u64>,
    ) -> Self {
        Self {
            inner: Ngrams::from_parts(order, min_order, counts, totals),
        }
    }
}

/// Convert n-gram pairs to a Python list of (tuple, count).
fn pairs_to_pylist(py: Python<'_>, pairs: Vec<(Vec<String>, u64)>) -> PyResult<Py<PyAny>> {
    let result = PyList::empty(py);
    for (ngram, count) in pairs {
        let tuple = PyTuple::new(py, &ngram)?;
        result.append((tuple, count))?;
    }
    Ok(result.into_any().unbind())
}

#[pymethods]
impl PyNgrams {
    /// Create a new empty Ngrams.
    ///
    /// # Arguments
    ///
    /// * `n` - The n-gram order (1 for unigrams, 2 for bigrams, etc.). Must be >= 1.
    #[new]
    #[pyo3(signature = (n, *, min_n=None))]
    fn new(n: usize, min_n: Option<usize>) -> PyResult<Self> {
        Ngrams::new(n, min_n)
            .map(|inner| Self { inner })
            .map_err(PyErr::from)
    }

    /// Count n-grams from a single sequence.
    ///
    /// Extracts all n-grams of the configured order from the sequence
    /// and increments their counts. N-grams do not cross sequence boundaries.
    fn count(&mut self, seq: Vec<String>) {
        BaseNgrams::count(self, seq);
    }

    /// Count n-grams from multiple sequences.
    ///
    /// Each sequence is treated independently (n-grams do not cross boundaries).
    fn count_seqs(&mut self, seqs: Vec<Vec<String>>) {
        BaseNgrams::count_seqs(self, seqs);
    }

    /// Return the count for a specific n-gram.
    ///
    /// Returns 0 if the n-gram has not been observed.
    fn get(&self, ngram: Vec<String>) -> u64 {
        BaseNgrams::get(self, ngram)
    }

    /// Return the n most common n-grams with their counts.
    ///
    /// If n is None, returns all n-grams sorted by count (descending).
    #[pyo3(signature = (n=None, *, order=None))]
    fn most_common(
        &self,
        py: Python<'_>,
        n: Option<usize>,
        order: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let pairs = self.most_common_items(n, order).map_err(PyErr::from)?;
        pairs_to_pylist(py, pairs)
    }

    /// Return all (n-gram, count) pairs.
    #[pyo3(signature = (*, order=None))]
    fn items(&self, py: Python<'_>, order: Option<usize>) -> PyResult<Py<PyAny>> {
        let pairs = self.items_list(order).map_err(PyErr::from)?;
        pairs_to_pylist(py, pairs)
    }

    /// Return the total number of n-gram tokens counted.
    #[pyo3(signature = (*, order=None))]
    fn total(&self, order: Option<usize>) -> PyResult<u64> {
        BaseNgrams::total(self, order).map_err(PyErr::from)
    }

    /// The n-gram order.
    #[getter]
    fn n(&self) -> usize {
        self.order()
    }

    /// The minimum n-gram order.
    #[getter]
    fn min_n(&self) -> usize {
        self.min_order()
    }

    fn __getitem__(&self, ngram: Vec<String>) -> u64 {
        BaseNgrams::get(self, ngram)
    }

    fn __len__(&self) -> usize {
        BaseNgrams::len(self)
    }

    fn __contains__(&self, ngram: Vec<String>) -> bool {
        BaseNgrams::contains(self, ngram)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ngrams = self.all_ngrams();
        let result = PyList::empty(py);
        for ngram in ngrams {
            let tuple = PyTuple::new(py, &ngram)?;
            result.append(tuple)?;
        }
        Ok(result.call_method0("__iter__")?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }

    fn __add__(&self, other: &PyNgrams) -> PyResult<PyNgrams> {
        BaseNgrams::add(self, other).map_err(PyErr::from)
    }

    fn __iadd__(&mut self, other: &PyNgrams) -> PyResult<()> {
        BaseNgrams::iadd(self, other).map_err(PyErr::from)
    }

    /// Convert to a Python ``collections.Counter``.
    ///
    /// Returns a ``Counter`` mapping n-gram tuples to their counts.
    #[pyo3(signature = (*, order=None))]
    fn to_counter(&self, py: Python<'_>, order: Option<usize>) -> PyResult<Py<PyAny>> {
        let effective_order = order.unwrap_or(self.order());
        self.validate_order(Some(effective_order))
            .map_err(PyErr::from)?;
        let counter_type = py.import("collections")?.getattr("Counter")?;
        let dict = PyDict::new(py);
        for (ngram, count) in self.counts().all_counts() {
            if ngram.len() == effective_order {
                let tuple = PyTuple::new(py, &ngram)?;
                dict.set_item(tuple, count)?;
            }
        }
        Ok(counter_type.call1((dict,))?.unbind())
    }

    /// Clear all counts.
    fn clear(&mut self) {
        BaseNgrams::clear(self);
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the ngram submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let ngram_module = PyModule::new(parent_module.py(), "ngram")?;
    ngram_module.add_class::<PyNgrams>()?;
    parent_module.add_submodule(&ngram_module)?;
    Ok(())
}
