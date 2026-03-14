//! N-gram counting.
//!
//! This module provides an n-gram counter for counting n-gram
//! frequencies from sequential data.

#[cfg(feature = "pyo3")]
mod py;

use crate::persistence::ModelError;
use crate::trie::CountTrie;

#[cfg(feature = "pyo3")]
pub use py::PyNgrams;
#[cfg(feature = "pyo3")]
pub(crate) use py::register_module;

// ---------------------------------------------------------------------------
// BaseNgrams
// ---------------------------------------------------------------------------

/// Core n-gram counting behavior with default implementations.
///
/// Implementors provide required methods that grant access to the
/// underlying storage. All counting and query logic is provided as defaults.
pub trait BaseNgrams: Sized + Clone {
    fn order(&self) -> usize;
    fn min_order(&self) -> usize;
    fn counts(&self) -> &CountTrie<String>;
    fn counts_mut(&mut self) -> &mut CountTrie<String>;
    fn totals(&self) -> &Vec<u64>;
    fn totals_mut(&mut self) -> &mut Vec<u64>;
    fn from_parts(
        order: usize,
        min_order: usize,
        counts: CountTrie<String>,
        totals: Vec<u64>,
    ) -> Self;

    // -----------------------------------------------------------------------
    // Counting
    // -----------------------------------------------------------------------

    /// Count n-grams from a single sequence.
    fn count(&mut self, seq: Vec<String>) {
        for k in self.min_order()..=self.order() {
            if seq.len() < k {
                continue;
            }
            let idx = k - self.min_order();
            for window in seq.windows(k) {
                self.counts_mut().increment(window.iter().cloned());
                self.totals_mut()[idx] += 1;
            }
        }
    }

    /// Count n-grams from multiple sequences.
    fn count_seqs(&mut self, seqs: Vec<Vec<String>>) {
        for seq in seqs {
            self.count(seq);
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the count for a specific n-gram.
    fn get(&self, ngram: Vec<String>) -> u64 {
        self.counts().get_count(ngram)
    }

    /// Validate that an order is within the valid range.
    fn validate_order(&self, order: Option<usize>) -> Result<(), ModelError> {
        if let Some(k) = order
            && (k < self.min_order() || k > self.order())
        {
            return Err(ModelError::ValidationError(format!(
                "order must be between {} and {}",
                self.min_order(),
                self.order()
            )));
        }
        Ok(())
    }

    /// Return the n most common n-grams with their counts.
    fn most_common_items(
        &self,
        n: Option<usize>,
        order: Option<usize>,
    ) -> Result<Vec<(Vec<String>, u64)>, ModelError> {
        self.validate_order(order)?;
        let mut pairs = self.counts().all_counts();
        if let Some(k) = order {
            pairs.retain(|(ngram, _)| ngram.len() == k);
        }
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some(limit) = n {
            pairs.truncate(limit);
        }
        Ok(pairs)
    }

    /// Return all (n-gram, count) pairs.
    fn items_list(&self, order: Option<usize>) -> Result<Vec<(Vec<String>, u64)>, ModelError> {
        self.validate_order(order)?;
        let pairs = self.counts().all_counts();
        match order {
            Some(k) => Ok(pairs
                .into_iter()
                .filter(|(ngram, _)| ngram.len() == k)
                .collect()),
            None => Ok(pairs),
        }
    }

    /// Return the total number of n-gram tokens counted.
    fn total(&self, order: Option<usize>) -> Result<u64, ModelError> {
        match order {
            None => Ok(self.totals().iter().sum()),
            Some(k) => {
                self.validate_order(Some(k))?;
                Ok(self.totals()[k - self.min_order()])
            }
        }
    }

    /// Number of unique n-grams.
    fn len(&self) -> usize {
        self.counts().len()
    }

    /// Whether no n-grams have been counted.
    fn is_empty(&self) -> bool {
        self.counts().len() == 0
    }

    /// Whether a specific n-gram has been observed.
    fn contains(&self, ngram: Vec<String>) -> bool {
        self.counts().get_count(ngram) > 0
    }

    /// Return all n-grams (without counts).
    fn all_ngrams(&self) -> Vec<Vec<String>> {
        self.counts()
            .all_counts()
            .into_iter()
            .map(|(ngram, _)| ngram)
            .collect()
    }

    /// Return a string representation.
    fn repr_string(&self) -> String {
        let total: u64 = self.totals().iter().sum();
        if self.min_order() == self.order() {
            format!(
                "Ngrams(n={}, unique={}, total={})",
                self.order(),
                self.counts().len(),
                total
            )
        } else {
            format!(
                "Ngrams(n={}, min_n={}, unique={}, total={})",
                self.order(),
                self.min_order(),
                self.counts().len(),
                total
            )
        }
    }

    /// Add two n-gram counters together, returning a new counter.
    fn add(&self, other: &Self) -> Result<Self, ModelError> {
        if self.order() != other.order() || self.min_order() != other.min_order() {
            return Err(ModelError::ValidationError(format!(
                "Cannot add Ngrams with different orders \
                 (n={}, min_n={}) vs (n={}, min_n={})",
                self.order(),
                self.min_order(),
                other.order(),
                other.min_order()
            )));
        }
        let mut result = self.clone();
        for (ngram, count) in other.counts().all_counts() {
            let idx = ngram.len() - self.min_order();
            for _ in 0..count {
                result.counts_mut().increment(ngram.iter().cloned());
            }
            result.totals_mut()[idx] += count;
        }
        Ok(result)
    }

    /// Add another n-gram counter into this one in-place.
    fn iadd(&mut self, other: &Self) -> Result<(), ModelError> {
        if self.order() != other.order() || self.min_order() != other.min_order() {
            return Err(ModelError::ValidationError(format!(
                "Cannot add Ngrams with different orders \
                 (n={}, min_n={}) vs (n={}, min_n={})",
                self.order(),
                self.min_order(),
                other.order(),
                other.min_order()
            )));
        }
        for (ngram, count) in other.counts().all_counts() {
            let idx = ngram.len() - self.min_order();
            for _ in 0..count {
                self.counts_mut().increment(ngram.iter().cloned());
            }
            self.totals_mut()[idx] += count;
        }
        Ok(())
    }

    /// Clear all counts.
    fn clear(&mut self) {
        self.counts_mut().clear();
        for t in self.totals_mut() {
            *t = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Ngrams (pure Rust)
// ---------------------------------------------------------------------------

/// An n-gram counter for counting n-gram frequencies.
///
/// Accumulates n-gram counts from sequences of elements. N-grams
/// do not cross sequence boundaries.
///
/// For Python, use [`PyNgrams`].
#[derive(Clone, Debug)]
pub struct Ngrams {
    order: usize,
    min_order: usize,
    pub(crate) counts: CountTrie<String>,
    pub(crate) totals: Vec<u64>,
}

impl BaseNgrams for Ngrams {
    fn order(&self) -> usize {
        self.order
    }
    fn min_order(&self) -> usize {
        self.min_order
    }
    fn counts(&self) -> &CountTrie<String> {
        &self.counts
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        &mut self.counts
    }
    fn totals(&self) -> &Vec<u64> {
        &self.totals
    }
    fn totals_mut(&mut self) -> &mut Vec<u64> {
        &mut self.totals
    }
    fn from_parts(
        order: usize,
        min_order: usize,
        counts: CountTrie<String>,
        totals: Vec<u64>,
    ) -> Self {
        Self {
            order,
            min_order,
            counts,
            totals,
        }
    }
}

impl Ngrams {
    /// Create a new empty Ngrams.
    ///
    /// # Arguments
    ///
    /// * `n` - The n-gram order (1 for unigrams, 2 for bigrams, etc.). Must be >= 1.
    /// * `min_n` - Minimum n-gram order. Must be >= 1 and <= n. Defaults to n.
    pub fn new(n: usize, min_n: Option<usize>) -> Result<Self, ModelError> {
        if n < 1 {
            return Err(ModelError::ValidationError("n must be >= 1".to_string()));
        }
        let min_order = min_n.unwrap_or(n);
        if min_order < 1 {
            return Err(ModelError::ValidationError(
                "min_n must be >= 1".to_string(),
            ));
        }
        if min_order > n {
            return Err(ModelError::ValidationError(
                "min_n must be <= n".to_string(),
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

        let merged = c1.add(&c2).unwrap();
        assert_eq!(merged.get(vec!["a".into()]), 1);
        assert_eq!(merged.get(vec!["b".into()]), 2);
        assert_eq!(merged.get(vec!["c".into()]), 1);
        assert_eq!(total(&merged), 4);
    }

    #[test]
    fn test_merge_different_order_fails() {
        let c1 = Ngrams::new(1, None).unwrap();
        let c2 = Ngrams::new(2, None).unwrap();
        assert!(c1.add(&c2).is_err());
    }

    #[test]
    fn test_iadd() {
        let mut c1 = Ngrams::new(1, None).unwrap();
        c1.count(vec!["a".into()]);

        let mut c2 = Ngrams::new(1, None).unwrap();
        c2.count(vec!["a".into(), "b".into()]);

        c1.iadd(&c2).unwrap();
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

        let merged = c1.add(&c2).unwrap();
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
        assert!(c1.add(&c2).is_err());
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
