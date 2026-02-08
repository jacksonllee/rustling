//! Language model implementation.

use std::collections::HashSet;

use pyo3::prelude::*;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;

use crate::trie::CountTrie;

/// Smoothing method for language model probability estimation.
#[derive(Clone, Debug)]
enum Smoothing {
    /// Maximum Likelihood Estimation (no smoothing).
    Mle,
    /// Lidstone (additive) smoothing with parameter gamma.
    Lidstone { gamma: f64 },
}

/// A vocabulary of known words, with OOV mapping to `<UNK>`.
#[derive(Clone, Debug)]
struct Vocabulary {
    words: HashSet<String>,
}

const UNK_LABEL: &str = "<UNK>";
const BOS_LABEL: &str = "<s>";
const EOS_LABEL: &str = "</s>";

impl Vocabulary {
    fn new() -> Self {
        Self {
            words: HashSet::new(),
        }
    }

    /// Build vocabulary from training data.
    fn build(sents: &[Vec<String>]) -> Self {
        let mut words = HashSet::new();
        for sent in sents {
            for word in sent {
                words.insert(word.clone());
            }
        }
        // Always include special tokens
        words.insert(UNK_LABEL.to_string());
        words.insert(BOS_LABEL.to_string());
        words.insert(EOS_LABEL.to_string());
        Self { words }
    }

    /// Look up a word: return it if known, otherwise return `<UNK>`.
    fn lookup(&self, word: &str) -> String {
        if self.words.contains(word) {
            word.to_string()
        } else {
            UNK_LABEL.to_string()
        }
    }

    /// Number of unique words in the vocabulary (including special tokens).
    fn len(&self) -> usize {
        self.words.len()
    }
}

/// An n-gram language model.
///
/// Supports MLE, Lidstone, and Laplace smoothing methods.
/// This translates NLTK's `LanguageModel` into Rust, using a counting trie
/// instead of NLTK's `NgramCounter` for n-gram storage.
#[pyclass(subclass)]
#[derive(Clone)]
pub struct LanguageModel {
    order: usize,
    smoothing: Smoothing,
    vocabulary: Vocabulary,
    counts: CountTrie<String>,
    fitted: bool,
}

#[pymethods]
impl LanguageModel {
    /// Initialize a language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    /// * `smoothing` - The smoothing method: "mle", "lidstone", or "laplace".
    /// * `gamma` - The smoothing parameter for Lidstone. Must be > 0.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If order < 1, smoothing is unknown, or gamma <= 0.
    #[new]
    #[pyo3(signature = (*, order, smoothing = "mle", gamma = 1.0))]
    pub fn new(order: usize, smoothing: &str, gamma: f64) -> PyResult<Self> {
        if order < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "order must be >= 1",
            ));
        }
        let smoothing = match smoothing {
            "mle" => Smoothing::Mle,
            "lidstone" => {
                if gamma <= 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("gamma must be > 0"));
                }
                Smoothing::Lidstone { gamma }
            }
            "laplace" => Smoothing::Lidstone { gamma: 1.0 },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown smoothing method: '{}'. Use 'mle', 'lidstone', or 'laplace'.",
                    smoothing
                )));
            }
        };
        Ok(Self {
            order,
            smoothing,
            vocabulary: Vocabulary::new(),
            counts: CountTrie::new(),
            fitted: false,
        })
    }

    /// Train the language model on tokenized sentences.
    ///
    /// Each sentence is a list of tokens. The model extracts n-grams of all orders
    /// from 1 to `self.order` and counts their occurrences. Sentences are
    /// automatically padded with `<s>` and `</s>` tokens.
    ///
    /// # Arguments
    ///
    /// * `sents` - An iterable of tokenized sentences.
    pub fn fit(&mut self, sents: Vec<Vec<String>>) {
        self.vocabulary = Vocabulary::build(&sents);
        self.counts = CountTrie::new();

        for sent in &sents {
            // Pad: (order-1) <s> tokens at start, one </s> at end
            let mut padded: Vec<String> = Vec::with_capacity(self.order - 1 + sent.len() + 1);
            for _ in 0..self.order.saturating_sub(1) {
                padded.push(BOS_LABEL.to_string());
            }
            for word in sent {
                padded.push(word.clone());
            }
            padded.push(EOS_LABEL.to_string());

            // Extract and count n-grams of all orders from 1 to self.order
            for n in 1..=self.order {
                for window in padded.windows(n) {
                    self.counts.increment(window.iter().cloned());
                }
            }
        }

        self.fitted = true;
    }

    /// Return the probability of a word given a context.
    ///
    /// Maps out-of-vocabulary words to `<UNK>` via the vocabulary, then computes
    /// the model-specific probability.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to score.
    /// * `context` - The preceding context words. If `None`, computes unigram probability.
    ///
    /// # Returns
    ///
    /// The probability P(word | context).
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the model has not been fitted yet.
    #[pyo3(signature = (word, context=None))]
    pub fn score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        if !self.fitted {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Model has not been fitted yet.",
            ));
        }
        let word = self.vocabulary.lookup(&word);
        let context: Vec<String> = context
            .unwrap_or_default()
            .iter()
            .map(|w| self.vocabulary.lookup(w))
            .collect();
        Ok(self.compute_score(&word, &context))
    }

    /// Return the probability of a word given a context, without OOV mapping.
    ///
    /// Unlike `score`, this method does not map out-of-vocabulary words to `<UNK>`.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to score.
    /// * `context` - The preceding context words. If `None`, computes unigram probability.
    ///
    /// # Returns
    ///
    /// The probability P(word | context).
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the model has not been fitted yet.
    #[pyo3(signature = (word, context=None))]
    pub fn unmasked_score(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        if !self.fitted {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Model has not been fitted yet.",
            ));
        }
        let context = context.unwrap_or_default();
        Ok(self.compute_score(&word, &context))
    }

    /// Return the log (base 2) probability of a word given a context.
    ///
    /// Maps out-of-vocabulary words to `<UNK>` via the vocabulary.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to score.
    /// * `context` - The preceding context words. If `None`, computes unigram log-probability.
    ///
    /// # Returns
    ///
    /// log2(P(word | context)). Returns negative infinity if probability is 0.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the model has not been fitted yet.
    #[pyo3(signature = (word, context=None))]
    pub fn logscore(&self, word: String, context: Option<Vec<String>>) -> PyResult<f64> {
        let s = self.score(word, context)?;
        if s == 0.0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(s.log2())
        }
    }

    /// Generate words from the language model.
    ///
    /// Uses weighted random sampling from the conditional distribution.
    /// Generation stops early if `</s>` (end-of-sentence) is sampled or
    /// if no continuations are available for the current context.
    ///
    /// # Arguments
    ///
    /// * `num_words` - Number of words to generate.
    /// * `text_seed` - Optional seed text (context to start from).
    ///   Defaults to beginning-of-sentence context.
    /// * `random_seed` - Optional random seed for reproducibility.
    ///
    /// # Returns
    ///
    /// A list of generated words.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the model has not been fitted yet.
    #[pyo3(signature = (*, num_words = 1, text_seed = None, random_seed = None))]
    pub fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        if !self.fitted {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Model has not been fitted yet.",
            ));
        }

        let mut rng: Box<dyn rand::RngCore> = match random_seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::rng()),
        };

        // Default seed is (order-1) <s> tokens
        let mut context: Vec<String> = text_seed.unwrap_or_else(|| {
            (0..self.order.saturating_sub(1))
                .map(|_| BOS_LABEL.to_string())
                .collect()
        });

        let mut generated = Vec::with_capacity(num_words);

        for _ in 0..num_words {
            // Get the context window (last order-1 words)
            let ctx_start = if context.len() >= self.order.saturating_sub(1) {
                context.len() - self.order.saturating_sub(1)
            } else {
                0
            };
            let ctx = &context[ctx_start..];

            // Get all continuations from the trie
            let children = self.counts.children_with_counts(ctx.iter().cloned());
            if children.is_empty() {
                break;
            }

            let words: Vec<String> = children.iter().map(|(w, _)| w.clone()).collect();
            let weights: Vec<f64> = children.iter().map(|(_, c)| *c as f64).collect();

            let dist = WeightedIndex::new(&weights).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Sampling error: {}", e))
            })?;

            let idx = dist.sample(&mut *rng);
            let word = words[idx].clone();

            if word == EOS_LABEL {
                break;
            }

            context.push(word.clone());
            generated.push(word);
        }

        Ok(generated)
    }

    /// The order of the n-gram model.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// The vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl LanguageModel {
    /// Compute score without OOV mapping (shared logic for score and unmasked_score).
    fn compute_score(&self, word: &str, context: &[String]) -> f64 {
        // Trim context to at most (order - 1)
        let ctx = if context.len() >= self.order {
            &context[context.len() - (self.order - 1)..]
        } else {
            context
        };

        let mut ngram: Vec<String> = ctx.to_vec();
        ngram.push(word.to_string());

        let word_count = self.counts.get_count(ngram.iter().cloned()) as f64;
        let context_count = self.counts.children_count_sum(ctx.iter().cloned()) as f64;

        match &self.smoothing {
            Smoothing::Mle => {
                if context_count == 0.0 {
                    0.0
                } else {
                    word_count / context_count
                }
            }
            Smoothing::Lidstone { gamma } => {
                let vocab_size = self.vocabulary.len() as f64;
                let numerator = word_count + gamma;
                let denominator = context_count + vocab_size * gamma;
                if denominator == 0.0 {
                    0.0
                } else {
                    numerator / denominator
                }
            }
        }
    }

    /// Create an MLE language model (Rust API convenience constructor).
    pub fn new_mle(order: usize) -> PyResult<Self> {
        Self::new(order, "mle", 1.0)
    }

    /// Create a Lidstone language model (Rust API convenience constructor).
    pub fn new_lidstone(order: usize, gamma: f64) -> PyResult<Self> {
        Self::new(order, "lidstone", gamma)
    }

    /// Create a Laplace language model (Rust API convenience constructor).
    pub fn new_laplace(order: usize) -> PyResult<Self> {
        Self::new(order, "laplace", 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_data() -> Vec<Vec<String>> {
        vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
            vec!["the".into(), "cat".into(), "ran".into()],
        ]
    }

    #[test]
    fn test_new_mle() {
        let model = LanguageModel::new_mle(2).unwrap();
        assert_eq!(model.order, 2);
        assert!(!model.fitted);
    }

    #[test]
    fn test_new_invalid_order() {
        let result = LanguageModel::new(0, "mle", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_smoothing() {
        let result = LanguageModel::new(2, "unknown", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_lidstone_invalid_gamma() {
        let result = LanguageModel::new(2, "lidstone", 0.0);
        assert!(result.is_err());
        let result = LanguageModel::new(2, "lidstone", -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_builds_vocabulary() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());
        assert!(model.fitted);
        // Training words + <UNK>, <s>, </s>
        // Words: the, cat, sat, dog, ran = 5 + 3 special = 8
        assert_eq!(model.vocabulary.len(), 8);
    }

    #[test]
    fn test_score_before_fit() {
        let model = LanguageModel::new_mle(2).unwrap();
        let result = model.score("cat".into(), Some(vec!["the".into()]));
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_before_fit() {
        let model = LanguageModel::new_mle(2).unwrap();
        let result = model.generate(5, None, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_mle_bigram_score() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        // In training: "the cat" x2, "the dog" x1
        // Also "the" is padded as "<s> the" x3
        // P(cat | the) = count(the, cat) / count(the, *) where * includes cat, dog, cat
        // count(the, cat) = 2, count(the, dog) = 1, count(the, cat) + count(the, dog) = 3
        // (the context "the" is followed by: cat(2), dog(1))
        // BUT we also have "the" followed by </s>? No, not for bigrams.
        // Padded sentences: ["<s>", "the", "cat", "sat", "</s>"]
        //                   ["<s>", "the", "dog", "ran", "</s>"]
        //                   ["<s>", "the", "cat", "ran", "</s>"]
        // Bigrams with context "the": (the, cat) x2, (the, dog) x1
        // So P(cat | the) = 2/3
        let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        assert!((score - 2.0 / 3.0).abs() < 1e-9);

        // P(dog | the) = 1/3
        let score = model.score("dog".into(), Some(vec!["the".into()])).unwrap();
        assert!((score - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_mle_unseen_is_zero() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        // "fish" is OOV, mapped to <UNK>. P(<UNK> | the) = 0
        let score = model
            .score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mle_unigram() {
        let mut model = LanguageModel::new_mle(1).unwrap();
        model.fit(training_data());

        // Unigram model, no padding for order=1 (order-1 = 0 <s> tokens)
        // But still has </s> at end.
        // Padded: ["the", "cat", "sat", "</s>"], ["the", "dog", "ran", "</s>"],
        //         ["the", "cat", "ran", "</s>"]
        // Unigram counts: the=3, cat=2, sat=1, dog=1, ran=2, </s>=3
        // Total = 12
        // P(the) = 3/12 = 0.25
        let score = model.score("the".into(), None).unwrap();
        assert!((score - 3.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_vs_unmasked_score() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        // For in-vocabulary words, score and unmasked_score should be the same
        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = model
            .unmasked_score("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);

        // For OOV words, score maps to <UNK> but unmasked_score doesn't
        let s1 = model
            .score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        let s2 = model
            .unmasked_score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        // Both are 0 in MLE (neither <UNK> nor "fish" follows "the")
        assert_eq!(s1, 0.0);
        assert_eq!(s2, 0.0);
    }

    #[test]
    fn test_lidstone_unseen_nonzero() {
        let mut model = LanguageModel::new_lidstone(2, 0.5).unwrap();
        model.fit(training_data());

        // With Lidstone smoothing, unseen n-grams get nonzero probability
        let score = model
            .score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_lidstone_score_formula() {
        let mut model = LanguageModel::new_lidstone(2, 0.5).unwrap();
        model.fit(training_data());

        // P(cat | the) = (count(the, cat) + gamma) / (count(the, *) + |V| * gamma)
        // count(the, cat) = 2, count(the, *) = 3, |V| = 8, gamma = 0.5
        // P = (2 + 0.5) / (3 + 8 * 0.5) = 2.5 / 7.0
        let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        assert!((score - 2.5 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_laplace_is_lidstone_gamma_one() {
        let mut laplace = LanguageModel::new_laplace(2).unwrap();
        let mut lidstone = LanguageModel::new_lidstone(2, 1.0).unwrap();
        let data = training_data();
        laplace.fit(data.clone());
        lidstone.fit(data);

        for word in &["cat", "dog", "sat", "ran", "fish"] {
            for ctx in &[vec!["the".into()], vec!["cat".into()]] {
                let s1 = laplace.score(word.to_string(), Some(ctx.clone())).unwrap();
                let s2 = lidstone.score(word.to_string(), Some(ctx.clone())).unwrap();
                assert!(
                    (s1 - s2).abs() < 1e-9,
                    "Mismatch for word={} ctx={:?}: {} vs {}",
                    word,
                    ctx,
                    s1,
                    s2
                );
            }
        }
    }

    #[test]
    fn test_logscore() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let logscore = model
            .logscore("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((logscore - score.log2()).abs() < 1e-9);
    }

    #[test]
    fn test_logscore_zero_is_neg_inf() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        let logscore = model
            .logscore("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!(logscore.is_infinite() && logscore.is_sign_negative());
    }

    #[test]
    fn test_generate_deterministic_with_seed() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        let result1 = model.generate(5, None, Some(42)).unwrap();
        let result2 = model.generate(5, None, Some(42)).unwrap();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_generate_returns_words() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        let result = model.generate(3, None, Some(42)).unwrap();
        assert!(!result.is_empty());
        assert!(result.len() <= 3);
        // All generated words should be real words (not <s> or </s>)
        for word in &result {
            assert_ne!(word, BOS_LABEL);
            assert_ne!(word, EOS_LABEL);
        }
    }

    #[test]
    fn test_generate_with_text_seed() {
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        let result = model
            .generate(2, Some(vec!["the".into()]), Some(42))
            .unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vocabulary_lookup() {
        let vocab = Vocabulary::build(&[vec!["hello".into(), "world".into()]]);
        assert_eq!(vocab.lookup("hello"), "hello");
        assert_eq!(vocab.lookup("unknown"), UNK_LABEL);
    }

    #[test]
    fn test_context_trimming() {
        // For a bigram model, context longer than 1 should be trimmed
        let mut model = LanguageModel::new_mle(2).unwrap();
        model.fit(training_data());

        // These should give the same result since bigram only uses last 1 context word
        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = model
            .score("cat".into(), Some(vec!["blah".into(), "the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);
    }
}
