//! POS tagging.
//!
//! This module provides a part-of-speech tagger that can be trained on
//! tagged sentences and used to predict POS tags for new text.
//!
//! ## Example
//!
//! ```rust
//! use rustling::perceptron_pos_tagger::AveragedPerceptron;
//! use rustling::perceptron_pos_tagger::BaseTagger;
//!
//! // Create a tagger with default parameters
//! // (frequency_threshold=20, ambiguity_threshold=0.97, n_iter=5)
//! let mut tagger = AveragedPerceptron::new(20, 0.97, 5, None, None);
//!
//! // Training data: sequences and their corresponding tags
//! let sequences = vec![
//!     vec!["I".to_string(), "love".to_string(), "Rust".to_string()],
//!     vec!["Rust".to_string(), "is".to_string(), "fast".to_string()],
//! ];
//! let tags = vec![
//!     vec!["PRP".to_string(), "VBP".to_string(), "NNP".to_string()],
//!     vec!["NNP".to_string(), "VBZ".to_string(), "JJ".to_string()],
//! ];
//!
//! // Fit the tagger
//! tagger.fit(sequences, tags);
//!
//! // Predict tags for sentences
//! let sentences = vec![vec!["I".to_string(), "love".to_string(), "Rust".to_string()]];
//! let predicted = tagger.predict(sentences);
//! println!("{:?}", predicted);
//! // [["PRP", "VBP", "NNP"]]
//! ```

//! Averaged perceptron tagger.
//!
//! This is a modified version based on the textblob-aptagger codebase
//! (MIT license), with original implementation by Matthew Honnibal:
//! <https://github.com/sloria/textblob-aptagger>

#[cfg(feature = "pyo3")]
mod py;
#[cfg(feature = "pyo3")]
pub use py::PyAveragedPerceptron;
#[cfg(feature = "pyo3")]
pub(crate) use py::register_module;

use crate::seq_feature::{
    FeatureBuffer, SeqFeatureConfig, SeqFeatureTemplate, default_tagger_ap_features,
    extract_features,
};
use flatbuffers;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::Write;

// FlatBuffers generated code (produced by build.rs from src/perceptron_pos_tagger/model.fbs).
#[allow(dead_code, unused_imports, clippy::all)]
mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/perceptron_pos_tagger/model_generated.rs"
    ));
}

/// An averaged perceptron.
///
/// This is the internal model used by the tagger. It maintains feature weights
/// and supports training with weight averaging for better generalization.
#[derive(Clone, Default)]
pub struct PerceptronModel {
    pub(crate) weights: FxHashMap<String, FxHashMap<String, f64>>,
    pub(crate) classes: FxHashSet<String>,
    pub(crate) classes_sorted: Vec<String>,
    totals: FxHashMap<(String, String), f64>,
    tstamps: FxHashMap<(String, String), u64>,
    pub(crate) i: u64,
}

impl PerceptronModel {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn finalize_classes(&mut self) {
        self.classes_sorted = self.classes.iter().cloned().collect();
        self.classes_sorted.sort();
    }

    pub(crate) fn predict(&self, features: &[&str]) -> &str {
        let mut scores: FxHashMap<&str, f64> = FxHashMap::default();

        for feat in features {
            if let Some(feat_weights) = self.weights.get(*feat) {
                for (label, weight) in feat_weights {
                    *scores.entry(label.as_str()).or_insert(0.0) += weight;
                }
            }
        }

        let mut best_class: &str = "";
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes_sorted {
            let score = scores.get(class.as_str()).copied().unwrap_or(0.0);
            if score > best_score || (score == best_score && class.as_str() > best_class) {
                best_score = score;
                best_class = class.as_str();
            }
        }

        best_class
    }

    pub(crate) fn update(&mut self, truth: &str, guess: &str, features: &[String]) {
        self.i += 1;
        if truth == guess {
            return;
        }

        for f in features {
            let truth_weight = self
                .weights
                .get(f)
                .and_then(|w| w.get(truth))
                .copied()
                .unwrap_or(0.0);
            let guess_weight = self
                .weights
                .get(f)
                .and_then(|w| w.get(guess))
                .copied()
                .unwrap_or(0.0);

            self.upd_feat(truth, f, truth_weight, 1.0);
            self.upd_feat(guess, f, guess_weight, -1.0);
        }
    }

    fn upd_feat(&mut self, c: &str, f: &str, w: f64, v: f64) {
        let param = (f.to_string(), c.to_string());
        let tstamp = self.tstamps.get(&param).copied().unwrap_or(0);
        let total = self.totals.entry(param.clone()).or_insert(0.0);
        *total += (self.i - tstamp) as f64 * w;
        self.tstamps.insert(param.clone(), self.i);
        self.weights
            .entry(f.to_string())
            .or_default()
            .insert(c.to_string(), w + v);
    }

    pub(crate) fn average_weights(&mut self) {
        for (feat, weights) in &mut self.weights {
            let mut new_feat_weights: FxHashMap<String, f64> = FxHashMap::default();
            for (clas, weight) in weights.iter() {
                let param = (feat.clone(), clas.clone());
                let mut total = self.totals.get(&param).copied().unwrap_or(0.0);
                let tstamp = self.tstamps.get(&param).copied().unwrap_or(0);
                total += (self.i - tstamp) as f64 * weight;
                let averaged = (total / self.i as f64 * 1000.0).round() / 1000.0;
                if averaged != 0.0 {
                    new_feat_weights.insert(clas.clone(), averaged);
                }
            }
            *weights = new_feat_weights;
        }
        self.totals.clear();
        self.totals.shrink_to_fit();
        self.tstamps.clear();
        self.tstamps.shrink_to_fit();
    }
}

// ---------------------------------------------------------------------------
// BaseTagger
// ---------------------------------------------------------------------------

use crate::persistence::ModelError;

/// Core tagger behavior with default implementations.
///
/// Implementors provide required methods that grant access to the
/// underlying model state. All prediction, training, and I/O logic
/// is provided as defaults.
pub trait BaseTagger: Sized + Clone + Sync {
    fn frequency_threshold(&self) -> u32;
    fn ambiguity_threshold(&self) -> f64;
    fn n_iter(&self) -> u32;
    fn random_seed(&self) -> Option<u64>;
    fn model(&self) -> &PerceptronModel;
    fn model_mut(&mut self) -> &mut PerceptronModel;
    fn tagdict_ref(&self) -> &FxHashMap<String, String>;
    fn tagdict_mut(&mut self) -> &mut FxHashMap<String, String>;
    fn classes_ref(&self) -> &FxHashSet<String>;
    fn classes_mut(&mut self) -> &mut FxHashSet<String>;
    fn features(&self) -> &SeqFeatureConfig;

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Predict tags for the sequences.
    fn predict(&self, sequences: Vec<Vec<String>>) -> Vec<Vec<String>> {
        let predict_one = |words: &Vec<String>| -> Vec<String> {
            let n = words.len();
            if n == 0 {
                return Vec::new();
            }

            let word_refs: Vec<&str> = words.iter().map(|w| w.as_str()).collect();
            let mut label_strs: Vec<String> = Vec::with_capacity(n);
            let mut tags = Vec::with_capacity(n);
            let mut feature_buf = FeatureBuffer::new();

            for (i, word) in words.iter().enumerate() {
                let tag = if let Some(t) = self.tagdict_ref().get(word) {
                    t.clone()
                } else {
                    let labels: Vec<&str> = label_strs.iter().map(|s| s.as_str()).collect();
                    feature_buf.clear();
                    extract_features(&mut feature_buf, self.features(), &word_refs, i, &labels);
                    self.model().predict(&feature_buf.keys()).to_string()
                };

                label_strs.push(tag.clone());
                tags.push(tag);
            }

            tags
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            sequences
                .par_iter()
                .with_min_len(16)
                .map(predict_one)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            sequences.iter().map(predict_one).collect()
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Fit a model.
    fn fit(&mut self, sequences: Vec<Vec<String>>, tags: Vec<Vec<String>>) {
        self.make_tagdict(&sequences, &tags);
        self.model_mut().classes = self.classes_ref().clone();
        self.model_mut().finalize_classes();

        let mut indices: Vec<usize> = (0..sequences.len()).collect();
        let mut rng: Box<dyn rand::Rng> = match self.random_seed() {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::rng()),
        };

        let mut feature_buf = FeatureBuffer::new();

        for _iter in 0..self.n_iter() {
            for &idx in &indices {
                let words = &sequences[idx];
                let sent_tags = &tags[idx];
                let n = words.len();
                let word_refs: Vec<&str> = words.iter().map(|w| w.as_str()).collect();
                let mut label_strs: Vec<String> = Vec::with_capacity(n);

                for (i, (word, tag)) in words.iter().zip(sent_tags.iter()).enumerate() {
                    let guess = if let Some(t) = self.tagdict_ref().get(word) {
                        t.clone()
                    } else {
                        let labels: Vec<&str> = label_strs.iter().map(|s| s.as_str()).collect();
                        feature_buf.clear();
                        extract_features(&mut feature_buf, self.features(), &word_refs, i, &labels);
                        let guess = self.model().predict(&feature_buf.keys()).to_string();
                        self.model_mut().update(tag, &guess, feature_buf.features());
                        guess
                    };

                    label_strs.push(guess);
                }
            }

            indices.shuffle(&mut rng);
        }

        self.model_mut().average_weights();
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    fn save_to_path(&self, path: &str) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        save_perceptron_flatbuffers(self, &mut buf)?;
        crate::persistence::save_zstd(path, &buf)
    }

    fn load_from_path(&mut self, path: &str) -> Result<(), ModelError> {
        let bytes = crate::persistence::load_zstd(path, "tagger model")?;
        load_perceptron_flatbuffers(self, &bytes)
    }

    // -----------------------------------------------------------------------
    // Tag dictionary
    // -----------------------------------------------------------------------

    /// Make a tag dictionary for single-tag words.
    fn make_tagdict(&mut self, sequences: &[Vec<String>], tags: &[Vec<String>]) {
        let mut counts: FxHashMap<String, FxHashMap<String, u32>> = FxHashMap::default();

        for (words, sent_tags) in sequences.iter().zip(tags.iter()) {
            for (word, tag) in words.iter().zip(sent_tags.iter()) {
                *counts
                    .entry(word.clone())
                    .or_default()
                    .entry(tag.clone())
                    .or_insert(0) += 1;
                self.classes_mut().insert(tag.clone());
            }
        }

        for (word, tag_freqs) in &counts {
            let (best_tag, mode) = tag_freqs
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(tag, count)| (tag.clone(), *count))
                .unwrap_or_default();

            let n: u32 = tag_freqs.values().sum();
            let above_freq_threshold = n >= self.frequency_threshold();
            let unambiguous = (mode as f64 / n as f64) >= self.ambiguity_threshold();

            if above_freq_threshold && unambiguous {
                self.tagdict_mut().insert(word.clone(), best_tag);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FlatBuffers save / load
// ---------------------------------------------------------------------------

/// Save a perceptron tagger model to a FlatBuffers binary stream.
pub(crate) fn save_perceptron_flatbuffers<T: BaseTagger, W: Write>(
    tagger: &T,
    writer: &mut W,
) -> Result<(), ModelError> {
    use generated::rustling::perceptron_fbs as fbs;

    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(4 * 1024 * 1024);

    // Build feature_entries from weights.
    let mut feature_keys: Vec<&String> = tagger.model().weights.keys().collect();
    feature_keys.sort();

    let fb_entries: Vec<_> = feature_keys
        .iter()
        .map(|feat_key| {
            let class_weights = &tagger.model().weights[*feat_key];
            let mut class_keys: Vec<&String> = class_weights.keys().collect();
            class_keys.sort();
            let fb_class_weights: Vec<_> = class_keys
                .iter()
                .map(|class_name| {
                    let weight = class_weights[*class_name] as f32;
                    let class_name_fb = builder.create_string(class_name);
                    fbs::ClassWeight::create(
                        &mut builder,
                        &fbs::ClassWeightArgs {
                            class_name: Some(class_name_fb),
                            weight,
                        },
                    )
                })
                .collect();
            let class_weights_fb = builder.create_vector(&fb_class_weights);
            let feat_key_fb = builder.create_string(feat_key);
            fbs::FeatureEntry::create(
                &mut builder,
                &fbs::FeatureEntryArgs {
                    feature_key: Some(feat_key_fb),
                    class_weights: Some(class_weights_fb),
                },
            )
        })
        .collect();
    let feature_entries_fb = builder.create_vector(&fb_entries);

    // Build classes vector (sorted).
    let mut classes: Vec<&String> = tagger.classes_ref().iter().collect();
    classes.sort();
    let classes_str: Vec<_> = classes.iter().map(|s| builder.create_string(s)).collect();
    let classes_fb = builder.create_vector(&classes_str);

    // Build tagdict as parallel sorted key/value arrays.
    let mut tagdict_pairs: Vec<(&String, &String)> = tagger.tagdict_ref().iter().collect();
    tagdict_pairs.sort_by_key(|(k, _)| *k);
    let tagdict_keys_str: Vec<_> = tagdict_pairs
        .iter()
        .map(|(k, _)| builder.create_string(k))
        .collect();
    let tagdict_vals_str: Vec<_> = tagdict_pairs
        .iter()
        .map(|(_, v)| builder.create_string(v))
        .collect();
    let tagdict_keys_fb = builder.create_vector(&tagdict_keys_str);
    let tagdict_vals_fb = builder.create_vector(&tagdict_vals_str);

    // Serialize feature templates as JSON string.
    let features_json = serde_json::to_string(&tagger.features().templates)
        .map_err(|e| ModelError::Io(format!("Failed to serialize features: {e}")))?;
    let features_json_fb = builder.create_string(&features_json);

    let model = fbs::PerceptronModel::create(
        &mut builder,
        &fbs::PerceptronModelArgs {
            feature_entries: Some(feature_entries_fb),
            classes: Some(classes_fb),
            tagdict_keys: Some(tagdict_keys_fb),
            tagdict_values: Some(tagdict_vals_fb),
            features_json: Some(features_json_fb),
        },
    );
    builder.finish(model, None);

    writer
        .write_all(builder.finished_data())
        .map_err(|e| ModelError::Io(format!("Failed to write FlatBuffers data: {e}")))
}

/// Load a perceptron tagger model from a FlatBuffers byte slice.
pub(crate) fn load_perceptron_flatbuffers<T: BaseTagger>(
    tagger: &mut T,
    bytes: &[u8],
) -> Result<(), ModelError> {
    use generated::rustling::perceptron_fbs as fbs;

    let opts = crate::persistence::flatbuffers_verifier_opts();
    let model = flatbuffers::root_with_opts::<fbs::PerceptronModel>(&opts, bytes)
        .map_err(|e| ModelError::ParseError(format!("Invalid FlatBuffers perceptron data: {e}")))?;

    // Load weights: feature_key → { class_name → weight (f64) }.
    let mut weights: FxHashMap<String, FxHashMap<String, f64>> = FxHashMap::default();
    for entry in model.feature_entries().iter() {
        let feat_key = entry.feature_key().to_owned();
        let mut class_map: FxHashMap<String, f64> = FxHashMap::default();
        for cw in entry.class_weights().iter() {
            class_map.insert(cw.class_name().to_owned(), cw.weight() as f64);
        }
        weights.insert(feat_key, class_map);
    }
    tagger.model_mut().weights = weights;

    // Load classes.
    let classes: FxHashSet<String> = model.classes().iter().map(|s| s.to_owned()).collect();
    *tagger.classes_mut() = classes;
    tagger.model_mut().classes = tagger.classes_ref().clone();
    tagger.model_mut().finalize_classes();

    // Load tagdict.
    let keys = model.tagdict_keys();
    let vals = model.tagdict_values();
    let tagdict: FxHashMap<String, String> = keys
        .iter()
        .zip(vals.iter())
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    *tagger.tagdict_mut() = tagdict;

    Ok(())
}

// ---------------------------------------------------------------------------
// Pure Rust struct
// ---------------------------------------------------------------------------

/// A part-of-speech tagger using an averaged perceptron model.
///
/// For Python, use [`PyAveragedPerceptron`].
#[derive(Clone)]
pub struct AveragedPerceptron {
    frequency_threshold: u32,
    ambiguity_threshold: f64,
    n_iter: u32,
    random_seed: Option<u64>,
    features: SeqFeatureConfig,
    model: PerceptronModel,
    tagdict: FxHashMap<String, String>,
    classes: FxHashSet<String>,
}

impl BaseTagger for AveragedPerceptron {
    fn frequency_threshold(&self) -> u32 {
        self.frequency_threshold
    }
    fn ambiguity_threshold(&self) -> f64 {
        self.ambiguity_threshold
    }
    fn n_iter(&self) -> u32 {
        self.n_iter
    }
    fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }
    fn model(&self) -> &PerceptronModel {
        &self.model
    }
    fn model_mut(&mut self) -> &mut PerceptronModel {
        &mut self.model
    }
    fn tagdict_ref(&self) -> &FxHashMap<String, String> {
        &self.tagdict
    }
    fn tagdict_mut(&mut self) -> &mut FxHashMap<String, String> {
        &mut self.tagdict
    }
    fn classes_ref(&self) -> &FxHashSet<String> {
        &self.classes
    }
    fn classes_mut(&mut self) -> &mut FxHashSet<String> {
        &mut self.classes
    }
    fn features(&self) -> &SeqFeatureConfig {
        &self.features
    }
}

impl AveragedPerceptron {
    /// Create a new tagger.
    pub fn new(
        frequency_threshold: u32,
        ambiguity_threshold: f64,
        n_iter: u32,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Self {
        let templates = features.unwrap_or_else(default_tagger_ap_features);
        Self {
            frequency_threshold,
            ambiguity_threshold,
            n_iter,
            random_seed,
            features: SeqFeatureConfig::new(templates),
            model: PerceptronModel::new(),
            tagdict: FxHashMap::default(),
            classes: FxHashSet::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq_feature;

    #[test]
    fn test_new() {
        let tagger = AveragedPerceptron::new(10, 0.95, 5, None, None);
        assert_eq!(tagger.frequency_threshold, 10);
        assert!(tagger.tagdict.is_empty());
        assert!(tagger.classes.is_empty());
    }

    #[test]
    fn test_predict_empty() {
        let tagger = AveragedPerceptron::new(10, 0.95, 5, None, None);
        let result = tagger.predict(vec![]);
        assert!(result.is_empty());

        let result = tagger.predict(vec![vec![]]);
        assert_eq!(result, vec![Vec::<String>::new()]);
    }

    #[test]
    fn test_fit_and_predict() {
        let mut tagger = AveragedPerceptron::new(1, 0.9, 2, None, None);

        let sequences = vec![
            vec!["I".to_string(), "love".to_string(), "cats".to_string()],
            vec!["You".to_string(), "love".to_string(), "dogs".to_string()],
        ];
        let tags = vec![
            vec!["PRON".to_string(), "VERB".to_string(), "NOUN".to_string()],
            vec!["PRON".to_string(), "VERB".to_string(), "NOUN".to_string()],
        ];

        tagger.fit(sequences, tags);

        assert!(!tagger.classes.is_empty());

        let words = vec![vec![
            "I".to_string(),
            "love".to_string(),
            "cats".to_string(),
        ]];
        let result = tagger.predict(words);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_first_char() {
        assert_eq!(seq_feature::first_char("hello"), "h");
        assert_eq!(seq_feature::first_char("世界"), "世");
        assert_eq!(seq_feature::first_char(""), "");
    }

    #[test]
    fn test_final_char() {
        assert_eq!(seq_feature::final_char("hello"), "o");
        assert_eq!(seq_feature::final_char("世界"), "界");
        assert_eq!(seq_feature::final_char(""), "");
    }
}
