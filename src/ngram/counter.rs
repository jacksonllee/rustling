//! N-gram counter implementation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::trie::CountTrie;

/// An n-gram counter for counting n-gram frequencies.
///
/// Accumulates n-gram counts from sequences of elements. N-grams
/// do not cross sequence boundaries.
#[pyclass]
#[derive(Clone)]
pub struct Ngrams {
    order: usize,
    min_order: usize,
    counts: CountTrie<String>,
    totals: Vec<u64>,
}

#[pymethods]
impl Ngrams {
    /// Create a new empty Ngrams.
    ///
    /// # Arguments
    ///
    /// * `n` - The n-gram order (1 for unigrams, 2 for bigrams, etc.). Must be >= 1.
    #[new]
    #[pyo3(signature = (n, *, min_n=None))]
    pub fn new(n: usize, min_n: Option<usize>) -> PyResult<Self> {
        if n < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("n must be >= 1"));
        }
        let min_order = min_n.unwrap_or(n);
        if min_order < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_n must be >= 1",
            ));
        }
        if min_order > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_n must be <= n",
            ));
        }
        let num_orders = n - min_order + 1;
        Ok(Self {
            order: n,
            min_order,
            counts: CountTrie::new(),
            totals: vec![0u64; num_orders],
        })
    }

    /// Count n-grams from a single sequence.
    ///
    /// Extracts all n-grams of the configured order from the sequence
    /// and increments their counts. N-grams do not cross sequence boundaries.
    pub fn count(&mut self, seq: Vec<String>) {
        for k in self.min_order..=self.order {
            if seq.len() < k {
                continue;
            }
            let idx = k - self.min_order;
            for window in seq.windows(k) {
                self.counts.increment(window.iter().cloned());
                self.totals[idx] += 1;
            }
        }
    }

    /// Count n-grams from multiple sequences.
    ///
    /// Each sequence is treated independently (n-grams do not cross boundaries).
    pub fn count_seqs(&mut self, seqs: Vec<Vec<String>>) {
        for seq in seqs {
            self.count(seq);
        }
    }

    /// Return the count for a specific n-gram.
    ///
    /// Returns 0 if the n-gram has not been observed.
    pub fn get(&self, ngram: Vec<String>) -> u64 {
        self.counts.get_count(ngram)
    }

    /// Return the n most common n-grams with their counts.
    ///
    /// If n is None, returns all n-grams sorted by count (descending).
    #[pyo3(signature = (n=None, *, order=None))]
    pub fn most_common(
        &self,
        py: Python<'_>,
        n: Option<usize>,
        order: Option<usize>,
    ) -> PyResult<PyObject> {
        self.validate_order(order)?;
        let mut pairs = self.counts.all_counts();
        if let Some(k) = order {
            pairs.retain(|(ngram, _)| ngram.len() == k);
        }
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(limit) = n {
            pairs.truncate(limit);
        }
        let result = PyList::empty(py);
        for (ngram, count) in pairs {
            let tuple = PyTuple::new(py, &ngram)?;
            result.append((tuple, count))?;
        }
        Ok(result.into_any().unbind())
    }

    /// Return all (n-gram, count) pairs.
    #[pyo3(signature = (*, order=None))]
    pub fn items(&self, py: Python<'_>, order: Option<usize>) -> PyResult<PyObject> {
        self.validate_order(order)?;
        let pairs = self.counts.all_counts();
        let result = PyList::empty(py);
        for (ngram, count) in pairs {
            if let Some(k) = order
                && ngram.len() != k
            {
                continue;
            }
            let tuple = PyTuple::new(py, &ngram)?;
            result.append((tuple, count))?;
        }
        Ok(result.into_any().unbind())
    }

    /// Return the total number of n-gram tokens counted.
    #[pyo3(signature = (*, order=None))]
    pub fn total(&self, order: Option<usize>) -> PyResult<u64> {
        match order {
            None => Ok(self.totals.iter().sum()),
            Some(k) => {
                self.validate_order(Some(k))?;
                Ok(self.totals[k - self.min_order])
            }
        }
    }

    /// The n-gram order.
    #[getter]
    pub fn n(&self) -> usize {
        self.order
    }

    /// The minimum n-gram order.
    #[getter]
    pub fn min_n(&self) -> usize {
        self.min_order
    }

    fn __getitem__(&self, ngram: Vec<String>) -> u64 {
        self.counts.get_count(ngram)
    }

    fn __len__(&self) -> usize {
        self.counts.len()
    }

    fn __contains__(&self, ngram: Vec<String>) -> bool {
        self.counts.get_count(ngram) > 0
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pairs = self.counts.all_counts();
        let result = PyList::empty(py);
        for (ngram, _) in pairs {
            let tuple = PyTuple::new(py, &ngram)?;
            result.append(tuple)?;
        }
        Ok(result.call_method0("__iter__")?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let total: u64 = self.totals.iter().sum();
        if self.min_order == self.order {
            format!(
                "Ngrams(n={}, unique={}, total={})",
                self.order,
                self.counts.len(),
                total
            )
        } else {
            format!(
                "Ngrams(n={}, min_n={}, unique={}, total={})",
                self.order,
                self.min_order,
                self.counts.len(),
                total
            )
        }
    }

    fn __add__(&self, other: &Ngrams) -> PyResult<Ngrams> {
        if self.order != other.order || self.min_order != other.min_order {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot add Ngrams with different orders \
                 (n={}, min_n={}) vs (n={}, min_n={})",
                self.order, self.min_order, other.order, other.min_order
            )));
        }
        let mut result = self.clone();
        for (ngram, count) in other.counts.all_counts() {
            let idx = ngram.len() - self.min_order;
            for _ in 0..count {
                result.counts.increment(ngram.iter().cloned());
            }
            result.totals[idx] += count;
        }
        Ok(result)
    }

    fn __iadd__(&mut self, other: &Ngrams) -> PyResult<()> {
        if self.order != other.order || self.min_order != other.min_order {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot add Ngrams with different orders \
                 (n={}, min_n={}) vs (n={}, min_n={})",
                self.order, self.min_order, other.order, other.min_order
            )));
        }
        for (ngram, count) in other.counts.all_counts() {
            let idx = ngram.len() - self.min_order;
            for _ in 0..count {
                self.counts.increment(ngram.iter().cloned());
            }
            self.totals[idx] += count;
        }
        Ok(())
    }

    /// Convert to a Python ``collections.Counter``.
    ///
    /// Returns a ``Counter`` mapping n-gram tuples to their counts.
    #[pyo3(signature = (*, order=None))]
    pub fn to_counter(&self, py: Python<'_>, order: Option<usize>) -> PyResult<PyObject> {
        let effective_order = order.unwrap_or(self.order);
        self.validate_order(Some(effective_order))?;
        let counter_type = py.import("collections")?.getattr("Counter")?;
        let dict = PyDict::new(py);
        for (ngram, count) in self.counts.all_counts() {
            if ngram.len() == effective_order {
                let tuple = PyTuple::new(py, &ngram)?;
                dict.set_item(tuple, count)?;
            }
        }
        Ok(counter_type.call1((dict,))?.unbind())
    }

    /// Clear all counts.
    pub fn clear(&mut self) {
        self.counts.clear();
        for t in &mut self.totals {
            *t = 0;
        }
    }
}

impl Ngrams {
    fn validate_order(&self, order: Option<usize>) -> PyResult<()> {
        if let Some(k) = order
            && (k < self.min_order || k > self.order)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "order must be between {} and {}",
                self.min_order, self.order
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(counter: &Ngrams) -> u64 {
        counter.totals.iter().sum()
    }

    #[test]
    fn test_new_valid() {
        let counter = Ngrams::new(1, None).unwrap();
        assert_eq!(counter.order, 1);
        assert_eq!(counter.min_order, 1);
        assert_eq!(total(&counter), 0);

        let counter = Ngrams::new(3, None).unwrap();
        assert_eq!(counter.order, 3);
        assert_eq!(counter.min_order, 3);
    }

    #[test]
    fn test_new_invalid() {
        let result = Ngrams::new(0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_min_n() {
        let counter = Ngrams::new(3, Some(1)).unwrap();
        assert_eq!(counter.order, 3);
        assert_eq!(counter.min_order, 1);
        assert_eq!(counter.totals.len(), 3);
    }

    #[test]
    fn test_new_min_n_defaults_to_n() {
        let counter = Ngrams::new(3, None).unwrap();
        assert_eq!(counter.min_order, 3);
        assert_eq!(counter.totals.len(), 1);
    }

    #[test]
    fn test_new_min_n_invalid() {
        assert!(Ngrams::new(3, Some(0)).is_err());
        assert!(Ngrams::new(3, Some(4)).is_err());
    }

    #[test]
    fn test_count_unigrams() {
        let mut counter = Ngrams::new(1, None).unwrap();
        counter.count(vec!["the".into(), "cat".into(), "sat".into(), "the".into()]);

        assert_eq!(counter.get(vec!["the".into()]), 2);
        assert_eq!(counter.get(vec!["cat".into()]), 1);
        assert_eq!(counter.get(vec!["sat".into()]), 1);
        assert_eq!(total(&counter), 4);
        assert_eq!(counter.counts.len(), 3);
    }

    #[test]
    fn test_count_bigrams() {
        let mut counter = Ngrams::new(2, None).unwrap();
        counter.count(vec![
            "the".into(),
            "cat".into(),
            "sat".into(),
            "the".into(),
            "cat".into(),
        ]);

        assert_eq!(counter.get(vec!["the".into(), "cat".into()]), 2);
        assert_eq!(counter.get(vec!["cat".into(), "sat".into()]), 1);
        assert_eq!(counter.get(vec!["sat".into(), "the".into()]), 1);
        assert_eq!(total(&counter), 4);
    }

    #[test]
    fn test_count_sentence_too_short() {
        let mut counter = Ngrams::new(3, None).unwrap();
        counter.count(vec!["the".into(), "cat".into()]);

        assert_eq!(total(&counter), 0);
        assert_eq!(counter.counts.len(), 0);
    }

    #[test]
    fn test_count_seqs() {
        let mut counter = Ngrams::new(1, None).unwrap();
        counter.count_seqs(vec![
            vec!["the".into(), "cat".into()],
            vec!["the".into(), "dog".into()],
        ]);

        assert_eq!(counter.get(vec!["the".into()]), 2);
        assert_eq!(counter.get(vec!["cat".into()]), 1);
        assert_eq!(counter.get(vec!["dog".into()]), 1);
        assert_eq!(total(&counter), 4);
    }

    #[test]
    fn test_count_no_cross_boundary() {
        let mut counter = Ngrams::new(2, None).unwrap();
        counter.count(vec!["a".into(), "b".into()]);
        counter.count(vec!["c".into(), "d".into()]);

        // "b c" should NOT exist since they come from separate count() calls
        assert_eq!(counter.get(vec!["b".into(), "c".into()]), 0);
        assert_eq!(counter.get(vec!["a".into(), "b".into()]), 1);
        assert_eq!(counter.get(vec!["c".into(), "d".into()]), 1);
    }

    #[test]
    fn test_get_missing() {
        let counter = Ngrams::new(1, None).unwrap();
        assert_eq!(counter.get(vec!["nonexistent".into()]), 0);
    }

    #[test]
    fn test_len() {
        let mut counter = Ngrams::new(1, None).unwrap();
        assert_eq!(counter.counts.len(), 0);

        counter.count(vec!["a".into(), "b".into(), "a".into()]);
        assert_eq!(counter.counts.len(), 2); // "a" and "b"
    }

    #[test]
    fn test_clear() {
        let mut counter = Ngrams::new(1, None).unwrap();
        counter.count(vec!["a".into(), "b".into()]);
        assert_eq!(total(&counter), 2);

        counter.clear();
        assert_eq!(total(&counter), 0);
        assert_eq!(counter.counts.len(), 0);
        assert_eq!(counter.get(vec!["a".into()]), 0);
    }

    #[test]
    fn test_merge_same_order() {
        let mut c1 = Ngrams::new(1, None).unwrap();
        c1.count(vec!["a".into(), "b".into()]);

        let mut c2 = Ngrams::new(1, None).unwrap();
        c2.count(vec!["b".into(), "c".into()]);

        let merged = c1.__add__(&c2).unwrap();
        assert_eq!(merged.get(vec!["a".into()]), 1);
        assert_eq!(merged.get(vec!["b".into()]), 2);
        assert_eq!(merged.get(vec!["c".into()]), 1);
        assert_eq!(total(&merged), 4);
    }

    #[test]
    fn test_merge_different_order_fails() {
        let c1 = Ngrams::new(1, None).unwrap();
        let c2 = Ngrams::new(2, None).unwrap();
        assert!(c1.__add__(&c2).is_err());
    }

    #[test]
    fn test_iadd() {
        let mut c1 = Ngrams::new(1, None).unwrap();
        c1.count(vec!["a".into()]);

        let mut c2 = Ngrams::new(1, None).unwrap();
        c2.count(vec!["a".into(), "b".into()]);

        c1.__iadd__(&c2).unwrap();
        assert_eq!(c1.get(vec!["a".into()]), 2);
        assert_eq!(c1.get(vec!["b".into()]), 1);
        assert_eq!(total(&c1), 3);
    }

    // All ngram tests

    #[test]
    fn test_count_all_ngrams() {
        let mut counter = Ngrams::new(3, Some(1)).unwrap();
        counter.count(vec!["a".into(), "b".into(), "c".into()]);

        // Unigrams
        assert_eq!(counter.get(vec!["a".into()]), 1);
        assert_eq!(counter.get(vec!["b".into()]), 1);
        assert_eq!(counter.get(vec!["c".into()]), 1);
        // Bigrams
        assert_eq!(counter.get(vec!["a".into(), "b".into()]), 1);
        assert_eq!(counter.get(vec!["b".into(), "c".into()]), 1);
        // Trigrams
        assert_eq!(counter.get(vec!["a".into(), "b".into(), "c".into()]), 1);

        // Per-order totals: 3 unigrams + 2 bigrams + 1 trigram
        assert_eq!(counter.totals[0], 3);
        assert_eq!(counter.totals[1], 2);
        assert_eq!(counter.totals[2], 1);
        assert_eq!(total(&counter), 6);
        assert_eq!(counter.counts.len(), 6);
    }

    #[test]
    fn test_count_all_ngrams_short_sequence() {
        let mut counter = Ngrams::new(3, Some(1)).unwrap();
        counter.count(vec!["a".into()]);

        assert_eq!(counter.get(vec!["a".into()]), 1);
        assert_eq!(counter.totals[0], 1); // unigrams
        assert_eq!(counter.totals[1], 0); // bigrams
        assert_eq!(counter.totals[2], 0); // trigrams
    }

    #[test]
    fn test_count_all_ngrams_min_n_equals_n() {
        // Should behave identically to single-order
        let mut counter = Ngrams::new(2, Some(2)).unwrap();
        counter.count(vec!["a".into(), "b".into(), "c".into()]);

        assert_eq!(counter.get(vec!["a".into()]), 0); // no unigrams
        assert_eq!(counter.get(vec!["a".into(), "b".into()]), 1);
        assert_eq!(counter.get(vec!["b".into(), "c".into()]), 1);
        assert_eq!(total(&counter), 2);
    }

    #[test]
    fn test_merge_all_ngrams() {
        let mut c1 = Ngrams::new(2, Some(1)).unwrap();
        c1.count(vec!["a".into(), "b".into()]);

        let mut c2 = Ngrams::new(2, Some(1)).unwrap();
        c2.count(vec!["b".into(), "c".into()]);

        let merged = c1.__add__(&c2).unwrap();
        assert_eq!(merged.get(vec!["b".into()]), 2);
        assert_eq!(merged.get(vec!["a".into(), "b".into()]), 1);
        assert_eq!(merged.get(vec!["b".into(), "c".into()]), 1);
        assert_eq!(merged.totals[0], 4); // unigram totals
        assert_eq!(merged.totals[1], 2); // bigram totals
    }

    #[test]
    fn test_merge_different_min_order_fails() {
        let c1 = Ngrams::new(3, Some(1)).unwrap();
        let c2 = Ngrams::new(3, Some(2)).unwrap();
        assert!(c1.__add__(&c2).is_err());
    }

    #[test]
    fn test_clear_all_ngrams() {
        let mut counter = Ngrams::new(3, Some(1)).unwrap();
        counter.count(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(total(&counter), 6);

        counter.clear();
        assert_eq!(total(&counter), 0);
        assert_eq!(counter.totals, vec![0, 0, 0]);
        assert_eq!(counter.counts.len(), 0);
    }
}
