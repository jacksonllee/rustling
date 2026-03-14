//! Language models.
//!
//! This module provides n-gram language models that can be trained on
//! tokenized text and used to score and generate word sequences.
//!
//! ## Example
//!
//! ```rust
//! use rustling::lm::MLE;
//! use rustling::lm::BaseLanguageModel;
//!
//! // Create a bigram MLE language model
//! let mut model = MLE::new(2).unwrap();
//! model.fit(vec![
//!     vec!["the".into(), "cat".into(), "sat".into()],
//!     vec!["the".into(), "dog".into(), "ran".into()],
//! ]);
//! let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
//! assert!((score - 0.5).abs() < 1e-9);
//! ```

use std::collections::HashSet;
use std::io::Write;

use flatbuffers;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;

use crate::trie::CountTrie;

#[cfg(feature = "pyo3")]
mod py;
#[cfg(feature = "pyo3")]
pub(crate) use py::register_module;
#[cfg(feature = "pyo3")]
pub use py::{PyLaplace, PyLidstone, PyMLE};

// FlatBuffers generated code (produced by build.rs from src/lm/model.fbs).
#[allow(dead_code, unused_imports, clippy::all)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/lm/model_generated.rs"));
}

/// Smoothing method for language model probability estimation.
#[derive(Clone, Debug)]
pub enum Smoothing {
    /// Maximum Likelihood Estimation (no smoothing).
    Mle,
    /// Lidstone (additive) smoothing with parameter gamma.
    Lidstone { gamma: f64 },
}

impl Smoothing {
    /// Return the gamma parameter, if applicable.
    pub fn gamma(&self) -> Option<f64> {
        match self {
            Smoothing::Mle => None,
            Smoothing::Lidstone { gamma } => Some(*gamma),
        }
    }
}

/// A vocabulary of known words, with OOV mapping to `<UNK>`.
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    words: HashSet<String>,
}

pub const UNK_LABEL: &str = "<UNK>";
pub const BOS_LABEL: &str = "<s>";
pub const EOS_LABEL: &str = "</s>";

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build vocabulary from training data.
    pub fn build(sents: &[Vec<String>]) -> Self {
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
    pub fn lookup(&self, word: &str) -> String {
        if self.words.contains(word) {
            word.to_string()
        } else {
            UNK_LABEL.to_string()
        }
    }

    /// Number of unique words in the vocabulary (including special tokens).
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Get a reference to the internal word set (for serialization).
    pub fn words(&self) -> &HashSet<String> {
        &self.words
    }

    /// Build vocabulary from a set of words (for deserialization).
    pub fn from_words(words: HashSet<String>) -> Self {
        Self { words }
    }
}

// ---------------------------------------------------------------------------
// BaseLanguageModel
// ---------------------------------------------------------------------------

/// Core language model behavior with default implementations.
///
/// Implementors provide required methods that grant access to the
/// underlying model state. All scoring, training, and generation logic
/// is provided as defaults.
pub trait BaseLanguageModel: Sized + Clone {
    fn order(&self) -> usize;
    fn smoothing(&self) -> &Smoothing;
    fn vocabulary(&self) -> &Vocabulary;
    fn vocabulary_mut(&mut self) -> &mut Vocabulary;
    fn counts(&self) -> &CountTrie<String>;
    fn counts_mut(&mut self) -> &mut CountTrie<String>;
    fn fitted(&self) -> bool;
    fn set_fitted(&mut self, fitted: bool);

    /// Return a string identifying the smoothing type for this model
    /// (used in save/load validation). One of: "mle", "lidstone", "laplace".
    fn smoothing_name(&self) -> &str;

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train the language model on tokenized sentences.
    fn fit(&mut self, sents: Vec<Vec<String>>) {
        *self.vocabulary_mut() = Vocabulary::build(&sents);
        *self.counts_mut() = CountTrie::new();

        for sent in &sents {
            let mut padded: Vec<String> = Vec::with_capacity(self.order() - 1 + sent.len() + 1);
            for _ in 0..self.order().saturating_sub(1) {
                padded.push(BOS_LABEL.to_string());
            }
            for word in sent {
                padded.push(word.clone());
            }
            padded.push(EOS_LABEL.to_string());

            for n in 1..=self.order() {
                for window in padded.windows(n) {
                    self.counts_mut().increment(window.iter().cloned());
                }
            }
        }

        self.set_fitted(true);
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Compute score without OOV mapping.
    fn compute_score(&self, word: &str, context: &[String]) -> f64 {
        let ctx = if context.len() >= self.order() {
            &context[context.len() - (self.order() - 1)..]
        } else {
            context
        };

        let mut ngram: Vec<String> = ctx.to_vec();
        ngram.push(word.to_string());

        let word_count = self.counts().get_count(ngram.iter().cloned()) as f64;
        let context_count = self.counts().children_count_sum(ctx.iter().cloned()) as f64;

        match self.smoothing() {
            Smoothing::Mle => {
                if context_count == 0.0 {
                    0.0
                } else {
                    word_count / context_count
                }
            }
            Smoothing::Lidstone { gamma } => {
                let vocab_size = self.vocabulary().len() as f64;
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

    /// Return the probability of a word given a context.
    fn score(&self, word: String, context: Option<Vec<String>>) -> Result<f64, ModelError> {
        if !self.fitted() {
            return Err(ModelError::ValidationError(
                "Model has not been fitted yet.".to_string(),
            ));
        }
        let word = self.vocabulary().lookup(&word);
        let context: Vec<String> = context
            .unwrap_or_default()
            .iter()
            .map(|w| self.vocabulary().lookup(w))
            .collect();
        Ok(self.compute_score(&word, &context))
    }

    /// Return the probability of a word given a context, without OOV mapping.
    fn unmasked_score(
        &self,
        word: String,
        context: Option<Vec<String>>,
    ) -> Result<f64, ModelError> {
        if !self.fitted() {
            return Err(ModelError::ValidationError(
                "Model has not been fitted yet.".to_string(),
            ));
        }
        let context = context.unwrap_or_default();
        Ok(self.compute_score(&word, &context))
    }

    /// Return the log (base 2) probability of a word given a context.
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> Result<f64, ModelError> {
        let s = self.score(word, context)?;
        if s == 0.0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(s.log2())
        }
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /// Generate words from the language model.
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Result<Vec<String>, ModelError> {
        if !self.fitted() {
            return Err(ModelError::ValidationError(
                "Model has not been fitted yet.".to_string(),
            ));
        }

        let mut rng: Box<dyn rand::Rng> = match random_seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::rng()),
        };

        let mut context: Vec<String> = text_seed.unwrap_or_else(|| {
            (0..self.order().saturating_sub(1))
                .map(|_| BOS_LABEL.to_string())
                .collect()
        });

        let mut generated = Vec::with_capacity(num_words);

        for _ in 0..num_words {
            let ctx_start = if context.len() >= self.order().saturating_sub(1) {
                context.len() - self.order().saturating_sub(1)
            } else {
                0
            };
            let ctx = &context[ctx_start..];

            let children = self.counts().children_with_counts(ctx.iter().cloned());
            if children.is_empty() {
                break;
            }

            let words: Vec<String> = children.iter().map(|(w, _)| w.clone()).collect();
            let weights: Vec<f64> = children.iter().map(|(_, c)| *c as f64).collect();

            let dist = WeightedIndex::new(&weights)
                .map_err(|e| ModelError::ValidationError(format!("Sampling error: {}", e)))?;

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

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// The vocabulary size (including special tokens).
    fn vocab_size(&self) -> usize {
        self.vocabulary().len()
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    fn save_to_path(&self, path: &str) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        save_lm_flatbuffers(self, &mut buf)?;
        crate::persistence::save_zstd(path, &buf)
    }

    fn load_from_path(&mut self, path: &str) -> Result<(), ModelError> {
        let bytes = crate::persistence::load_zstd(path, "language model")?;
        load_lm_flatbuffers(self, &bytes)
    }
}

// ---------------------------------------------------------------------------
// FlatBuffers save / load
// ---------------------------------------------------------------------------

/// Save an LM to a FlatBuffers binary stream.
fn save_lm_flatbuffers<T: BaseLanguageModel, W: Write>(
    model: &T,
    writer: &mut W,
) -> Result<(), ModelError> {
    use generated::rustling::lm_fbs as fbs;

    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(1024 * 1024);

    // vocabulary (sorted).
    let mut vocab_words: Vec<&String> = model.vocabulary().words().iter().collect();
    vocab_words.sort();
    let vocab_strs: Vec<_> = vocab_words
        .iter()
        .map(|w| builder.create_string(w))
        .collect();
    let vocab_fb = builder.create_vector(&vocab_strs);

    // ngrams from CountTrie.
    let mut all_counts = model.counts().all_counts();
    all_counts.sort_by(|a, b| a.0.cmp(&b.0));
    let fb_ngrams: Vec<_> = all_counts
        .iter()
        .map(|(ngram, count)| {
            let ngram_strs: Vec<_> = ngram.iter().map(|w| builder.create_string(w)).collect();
            let ngram_fb = builder.create_vector(&ngram_strs);
            fbs::NgramEntry::create(
                &mut builder,
                &fbs::NgramEntryArgs {
                    ngram: Some(ngram_fb),
                    count: *count,
                },
            )
        })
        .collect();
    let ngrams_fb = builder.create_vector(&fb_ngrams);

    let smoothing_name = builder.create_string(model.smoothing_name());
    let gamma = match model.smoothing() {
        Smoothing::Mle => 0.0,
        Smoothing::Lidstone { gamma } => *gamma,
    };

    let lm = fbs::LmModel::create(
        &mut builder,
        &fbs::LmModelArgs {
            order: model.order() as u32,
            smoothing: Some(smoothing_name),
            gamma,
            vocabulary: Some(vocab_fb),
            ngrams: Some(ngrams_fb),
        },
    );
    builder.finish(lm, None);

    writer
        .write_all(builder.finished_data())
        .map_err(|e| ModelError::Io(format!("Failed to write FlatBuffers data: {e}")))
}

/// Load an LM from a FlatBuffers byte slice.
fn load_lm_flatbuffers<T: BaseLanguageModel>(lm: &mut T, bytes: &[u8]) -> Result<(), ModelError> {
    use generated::rustling::lm_fbs as fbs;

    let opts = crate::persistence::flatbuffers_verifier_opts();
    let model = flatbuffers::root_with_opts::<fbs::LmModel>(&opts, bytes)
        .map_err(|e| ModelError::ParseError(format!("Invalid FlatBuffers LM data: {e}")))?;

    // Validate smoothing
    let file_smoothing = model.smoothing();
    if file_smoothing != lm.smoothing_name() {
        return Err(ModelError::ParseError(format!(
            "Smoothing type mismatch: file has '{file_smoothing}' but this model is '{}'",
            lm.smoothing_name()
        )));
    }

    // Validate order
    let file_order = model.order() as usize;
    if file_order != lm.order() {
        return Err(ModelError::ParseError(format!(
            "Order mismatch: file has {file_order} but this model has {}",
            lm.order()
        )));
    }

    // Validate gamma for Lidstone
    if let Smoothing::Lidstone { gamma } = lm.smoothing() {
        let file_gamma = model.gamma();
        if (file_gamma - gamma).abs() > 1e-15 {
            return Err(ModelError::ParseError(format!(
                "Gamma mismatch: file has {file_gamma} but this model has {gamma}"
            )));
        }
    }

    // Load vocabulary
    let vocab_words: HashSet<String> = model.vocabulary().iter().map(|s| s.to_owned()).collect();
    *lm.vocabulary_mut() = Vocabulary::from_words(vocab_words);

    // Load counts
    *lm.counts_mut() = CountTrie::new();
    for entry in model.ngrams().iter() {
        let ngram: Vec<String> = entry.ngram().iter().map(|s| s.to_owned()).collect();
        lm.counts_mut()
            .insert_count(ngram.into_iter(), entry.count());
    }

    lm.set_fitted(true);
    Ok(())
}

use crate::persistence::ModelError;

// ---------------------------------------------------------------------------
// Pure Rust structs
// ---------------------------------------------------------------------------

/// Maximum Likelihood Estimation language model (no smoothing).
///
/// For Python, use [`PyMLE`].
#[derive(Clone)]
pub struct MLE {
    order: usize,
    smoothing: Smoothing,
    vocabulary: Vocabulary,
    counts: CountTrie<String>,
    fitted: bool,
}

impl BaseLanguageModel for MLE {
    fn order(&self) -> usize {
        self.order
    }
    fn smoothing(&self) -> &Smoothing {
        &self.smoothing
    }
    fn smoothing_name(&self) -> &str {
        "mle"
    }
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
    }
    fn counts(&self) -> &CountTrie<String> {
        &self.counts
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        &mut self.counts
    }
    fn fitted(&self) -> bool {
        self.fitted
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.fitted = fitted;
    }
}

impl MLE {
    /// Create a new MLE language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    pub fn new(order: usize) -> Result<Self, ModelError> {
        if order < 1 {
            return Err(ModelError::ValidationError(
                "order must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            order,
            smoothing: Smoothing::Mle,
            vocabulary: Vocabulary::new(),
            counts: CountTrie::new(),
            fitted: false,
        })
    }
}

/// Lidstone (additive) smoothing language model.
///
/// For Python, use [`PyLidstone`].
#[derive(Clone)]
pub struct Lidstone {
    order: usize,
    gamma: f64,
    smoothing: Smoothing,
    vocabulary: Vocabulary,
    counts: CountTrie<String>,
    fitted: bool,
}

impl BaseLanguageModel for Lidstone {
    fn order(&self) -> usize {
        self.order
    }
    fn smoothing(&self) -> &Smoothing {
        &self.smoothing
    }
    fn smoothing_name(&self) -> &str {
        "lidstone"
    }
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
    }
    fn counts(&self) -> &CountTrie<String> {
        &self.counts
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        &mut self.counts
    }
    fn fitted(&self) -> bool {
        self.fitted
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.fitted = fitted;
    }
}

impl Lidstone {
    /// Create a new Lidstone language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    /// * `gamma` - The smoothing parameter. Must be > 0.
    pub fn new(order: usize, gamma: f64) -> Result<Self, ModelError> {
        if order < 1 {
            return Err(ModelError::ValidationError(
                "order must be >= 1".to_string(),
            ));
        }
        if gamma <= 0.0 {
            return Err(ModelError::ValidationError("gamma must be > 0".to_string()));
        }
        Ok(Self {
            order,
            gamma,
            smoothing: Smoothing::Lidstone { gamma },
            vocabulary: Vocabulary::new(),
            counts: CountTrie::new(),
            fitted: false,
        })
    }

    /// The gamma (smoothing) parameter.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// Laplace (add-one) smoothing language model (Lidstone with gamma=1).
///
/// For Python, use [`PyLaplace`].
#[derive(Clone)]
pub struct Laplace {
    order: usize,
    smoothing: Smoothing,
    vocabulary: Vocabulary,
    counts: CountTrie<String>,
    fitted: bool,
}

impl BaseLanguageModel for Laplace {
    fn order(&self) -> usize {
        self.order
    }
    fn smoothing(&self) -> &Smoothing {
        &self.smoothing
    }
    fn smoothing_name(&self) -> &str {
        "laplace"
    }
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
    }
    fn counts(&self) -> &CountTrie<String> {
        &self.counts
    }
    fn counts_mut(&mut self) -> &mut CountTrie<String> {
        &mut self.counts
    }
    fn fitted(&self) -> bool {
        self.fitted
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.fitted = fitted;
    }
}

impl Laplace {
    /// Create a new Laplace language model.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the n-gram model (e.g., 2 for bigram). Must be >= 1.
    pub fn new(order: usize) -> Result<Self, ModelError> {
        if order < 1 {
            return Err(ModelError::ValidationError(
                "order must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            order,
            smoothing: Smoothing::Lidstone { gamma: 1.0 },
            vocabulary: Vocabulary::new(),
            counts: CountTrie::new(),
            fitted: false,
        })
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
        let model = MLE::new(2).unwrap();
        assert_eq!(model.order, 2);
        assert!(!model.fitted);
    }

    #[test]
    fn test_new_invalid_order() {
        let result = MLE::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_lidstone_invalid_gamma() {
        let result = Lidstone::new(2, 0.0);
        assert!(result.is_err());
        let result = Lidstone::new(2, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_builds_vocabulary() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());
        assert!(model.fitted);
        // Training words + <UNK>, <s>, </s>
        // Words: the, cat, sat, dog, ran = 5 + 3 special = 8
        assert_eq!(model.vocabulary.len(), 8);
    }

    #[test]
    fn test_score_before_fit() {
        let model = MLE::new(2).unwrap();
        let result = model.score("cat".into(), Some(vec!["the".into()]));
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_before_fit() {
        let model = MLE::new(2).unwrap();
        let result = model.generate(5, None, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_mle_bigram_score() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

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
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        // "fish" is OOV, mapped to <UNK>. P(<UNK> | the) = 0
        let score = model
            .score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mle_unigram() {
        let mut model = MLE::new(1).unwrap();
        model.fit(training_data());

        // Unigram counts: the=3, cat=2, sat=1, dog=1, ran=2, </s>=3
        // Total = 12
        // P(the) = 3/12 = 0.25
        let score = model.score("the".into(), None).unwrap();
        assert!((score - 3.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_vs_unmasked_score() {
        let mut model = MLE::new(2).unwrap();
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
        let mut model = Lidstone::new(2, 0.5).unwrap();
        model.fit(training_data());

        // With Lidstone smoothing, unseen n-grams get nonzero probability
        let score = model
            .score("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_lidstone_score_formula() {
        let mut model = Lidstone::new(2, 0.5).unwrap();
        model.fit(training_data());

        // P(cat | the) = (count(the, cat) + gamma) / (count(the, *) + |V| * gamma)
        // count(the, cat) = 2, count(the, *) = 3, |V| = 8, gamma = 0.5
        // P = (2 + 0.5) / (3 + 8 * 0.5) = 2.5 / 7.0
        let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        assert!((score - 2.5 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_laplace_is_lidstone_gamma_one() {
        let mut laplace = Laplace::new(2).unwrap();
        let mut lidstone = Lidstone::new(2, 1.0).unwrap();
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
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        let score = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let logscore = model
            .logscore("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((logscore - score.log2()).abs() < 1e-9);
    }

    #[test]
    fn test_logscore_zero_is_neg_inf() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        let logscore = model
            .logscore("fish".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!(logscore.is_infinite() && logscore.is_sign_negative());
    }

    #[test]
    fn test_generate_deterministic_with_seed() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        let result1 = model.generate(5, None, Some(42)).unwrap();
        let result2 = model.generate(5, None, Some(42)).unwrap();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_generate_returns_words() {
        let mut model = MLE::new(2).unwrap();
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
        let mut model = MLE::new(2).unwrap();
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
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        // These should give the same result since bigram only uses last 1 context word
        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = model
            .score("cat".into(), Some(vec!["blah".into(), "the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);
    }

    #[test]
    fn test_save_and_load_mle() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mle_model.bin");
        let path_str = path.to_str().unwrap();

        model.save_to_path(path_str).unwrap();

        let mut loaded = MLE::new(2).unwrap();
        loaded.load_from_path(path_str).unwrap();

        assert!(loaded.fitted());
        assert_eq!(loaded.vocabulary().len(), model.vocabulary().len());

        // Verify scores match
        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = loaded
            .score("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);
    }

    #[test]
    fn test_save_and_load_lidstone() {
        let mut model = Lidstone::new(2, 0.5).unwrap();
        model.fit(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lidstone_model.bin");
        let path_str = path.to_str().unwrap();

        model.save_to_path(path_str).unwrap();

        let mut loaded = Lidstone::new(2, 0.5).unwrap();
        loaded.load_from_path(path_str).unwrap();

        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = loaded
            .score("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);
    }

    #[test]
    fn test_save_and_load_laplace() {
        let mut model = Laplace::new(2).unwrap();
        model.fit(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace_model.bin");
        let path_str = path.to_str().unwrap();

        model.save_to_path(path_str).unwrap();

        let mut loaded = Laplace::new(2).unwrap();
        loaded.load_from_path(path_str).unwrap();

        let s1 = model.score("cat".into(), Some(vec!["the".into()])).unwrap();
        let s2 = loaded
            .score("cat".into(), Some(vec!["the".into()]))
            .unwrap();
        assert!((s1 - s2).abs() < 1e-9);
    }

    #[test]
    fn test_load_smoothing_mismatch() {
        let mut model = MLE::new(2).unwrap();
        model.fit(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let path_str = path.to_str().unwrap();

        model.save_to_path(path_str).unwrap();

        // Try loading into a Lidstone model -- should fail
        let mut wrong = Lidstone::new(2, 0.5).unwrap();
        let result = wrong.load_from_path(path_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let mut model = MLE::new(2).unwrap();
        let result = model.load_from_path("/nonexistent/path/model.bin");
        assert!(result.is_err());
    }
}
