//! Averaged perceptron tagger.
//!
//! This is a modified version based on the textblob-aptagger codebase
//! (MIT license), with original implementation by Matthew Honnibal:
//! <https://github.com/sloria/textblob-aptagger>

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rng;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

/// An averaged perceptron.
///
/// This is the internal model used by the POSTagger. It maintains feature weights
/// and supports training with weight averaging for better generalization.
///
/// Optimized version using FxHashMap for faster hashing.
#[derive(Clone, Default)]
struct AveragedPerceptron {
    /// Each feature (key) gets its own weight vector (value).
    /// Uses FxHashMap for faster hashing of string keys.
    weights: FxHashMap<String, FxHashMap<String, f64>>,
    /// The set of all class labels.
    classes: FxHashSet<String>,
    /// Sorted classes for stable prediction ordering.
    classes_sorted: Vec<String>,
    /// The accumulated values, for the averaging.
    /// Keyed by (feature, class) tuples.
    totals: FxHashMap<(String, String), f64>,
    /// The last time the feature was changed, for the averaging.
    /// Keyed by (feature, class) tuples.
    tstamps: FxHashMap<(String, String), u64>,
    /// Number of instances seen.
    i: u64,
}

impl AveragedPerceptron {
    fn new() -> Self {
        Self::default()
    }

    /// Finalize classes after training setup - sorts them for stable iteration.
    fn finalize_classes(&mut self) {
        self.classes_sorted = self.classes.iter().cloned().collect();
        self.classes_sorted.sort();
    }

    /// Return the best label for the given features.
    ///
    /// It's computed based on the dot-product between the features and current weights.
    /// Optimized: avoids allocating a scores HashMap when possible.
    fn predict(&self, features: &[&str]) -> &str {
        let mut scores: FxHashMap<&str, f64> = FxHashMap::default();

        for feat in features {
            if let Some(feat_weights) = self.weights.get(*feat) {
                for (label, weight) in feat_weights {
                    *scores.entry(label.as_str()).or_insert(0.0) += weight;
                }
            }
        }

        // Return the class with the highest score.
        // Use pre-sorted classes for stable ordering.
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

    /// Update the feature weights.
    fn update(&mut self, truth: &str, guess: &str, features: &[String]) {
        self.i += 1;
        if truth == guess {
            return;
        }

        for f in features {
            // Get current weights for truth and guess before making any changes
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

            // Update for truth class (positive)
            self.upd_feat(truth, f, truth_weight, 1.0);

            // Update for guess class (negative)
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

    /// Average weights from all iterations.
    fn average_weights(&mut self) {
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
        // Clear temporaries after averaging to free memory
        self.totals.clear();
        self.totals.shrink_to_fit();
        self.tstamps.clear();
        self.tstamps.shrink_to_fit();
    }
}

/// A part-of-speech tagger using an averaged perceptron model.
///
/// This is a modified version based on the textblob-aptagger codebase
/// (MIT license), with original implementation by Matthew Honnibal:
/// <https://github.com/sloria/textblob-aptagger>
#[pyclass]
#[derive(Clone)]
pub struct AveragedPerceptronTagger {
    /// Frequency threshold for tag dictionary.
    frequency_threshold: u32,
    /// Ambiguity threshold for tag dictionary.
    ambiguity_threshold: f64,
    /// Number of training iterations.
    n_iter: u32,
    /// The averaged perceptron model.
    model: AveragedPerceptron,
    /// Dictionary mapping words to their most likely tags.
    tagdict: FxHashMap<String, String>,
    /// Set of all POS tag classes.
    classes: FxHashSet<String>,
}

const START: [&str; 2] = ["-START-", "-START2-"];
const END: [&str; 2] = ["-END-", "-END2-"];

#[pymethods]
impl AveragedPerceptronTagger {
    /// Initialize a part-of-speech tagger.
    ///
    /// # Arguments
    ///
    /// * `frequency_threshold` - A good number of words are almost unambiguously associated with
    ///   a given tag. If these words have a frequency of occurrence above this threshold in the
    ///   training data, they are directly associated with their tag in the model.
    /// * `ambiguity_threshold` - A good number of words are almost unambiguously associated with
    ///   a given tag. If the ratio of (# of occurrences of this word with this tag) /
    ///   (# of occurrences of this word) in the training data is equal to or greater than this
    ///   threshold, then this word is directly associated with the tag in the model.
    /// * `n_iter` - Number of times the training phase iterates through the data.
    ///   At each new iteration, the data is randomly shuffled.
    #[new]
    #[pyo3(signature = (*, frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5))]
    pub fn new(frequency_threshold: u32, ambiguity_threshold: f64, n_iter: u32) -> Self {
        Self {
            frequency_threshold,
            ambiguity_threshold,
            n_iter,
            model: AveragedPerceptron::new(),
            tagdict: FxHashMap::default(),
            classes: FxHashSet::default(),
        }
    }

    /// Tag the words.
    ///
    /// # Arguments
    ///
    /// * `words` - A segmented sentence or phrase, where each word is a string.
    ///
    /// # Returns
    ///
    /// The list of predicted tags.
    pub fn tag(&self, words: Vec<String>) -> Vec<String> {
        let n = words.len();
        let mut tags = Vec::with_capacity(n);

        if n == 0 {
            return tags;
        }

        let mut prev = START[0].to_string();
        let mut prev2 = START[1].to_string();

        // Reusable buffer for features to avoid repeated allocations
        let mut feature_buf = FeatureBuffer::new();

        for i in 0..n {
            let word = &words[i];

            let tag = if let Some(t) = self.tagdict.get(word) {
                t.clone()
            } else {
                // Get context words directly without building a context vector
                let i_m2 = if i >= 2 {
                    words[i - 2].as_str()
                } else if i == 1 {
                    START[0]
                } else {
                    START[1]
                };
                let i_m1 = if i >= 1 {
                    words[i - 1].as_str()
                } else {
                    START[0]
                };
                let i_p1 = words.get(i + 1).map(|s| s.as_str()).unwrap_or(END[0]);
                let i_p2 = words.get(i + 2).map(|s| s.as_str()).unwrap_or(END[1]);

                feature_buf.clear();
                self.get_features_into(
                    &mut feature_buf,
                    word,
                    i_m2,
                    i_m1,
                    i_p1,
                    i_p2,
                    &prev,
                    &prev2,
                );
                self.model.predict(&feature_buf.keys()).to_string()
            };

            prev2 = prev;
            prev = tag.clone();
            tags.push(tag);
        }

        tags
    }

    /// Train a model.
    ///
    /// # Arguments
    ///
    /// * `tagged_sents` - A list of segmented and tagged sentences for training.
    ///   Each sentence is a list of (word, tag) tuples.
    pub fn train(&mut self, tagged_sents: Vec<Vec<(String, String)>>) {
        self.make_tagdict(&tagged_sents);
        self.model.classes = self.classes.clone();
        self.model.finalize_classes();

        let mut tagged_sents = tagged_sents;
        let mut rng = rng();

        // Reusable buffer for features
        let mut feature_buf = FeatureBuffer::new();

        for _iter in 0..self.n_iter {
            for tagged_sent in &tagged_sents {
                let n = tagged_sent.len();
                let mut prev = START[0].to_string();
                let mut prev2 = START[1].to_string();

                for i in 0..n {
                    let (word, tag) = &tagged_sent[i];

                    let guess = if let Some(t) = self.tagdict.get(word) {
                        t.clone()
                    } else {
                        // Get context words directly
                        let i_m2 = if i >= 2 {
                            tagged_sent[i - 2].0.as_str()
                        } else if i == 1 {
                            START[0]
                        } else {
                            START[1]
                        };
                        let i_m1 = if i >= 1 {
                            tagged_sent[i - 1].0.as_str()
                        } else {
                            START[0]
                        };
                        let i_p1 = tagged_sent
                            .get(i + 1)
                            .map(|(w, _)| w.as_str())
                            .unwrap_or(END[0]);
                        let i_p2 = tagged_sent
                            .get(i + 2)
                            .map(|(w, _)| w.as_str())
                            .unwrap_or(END[1]);

                        feature_buf.clear();
                        self.get_features_into(
                            &mut feature_buf,
                            word,
                            i_m2,
                            i_m1,
                            i_p1,
                            i_p2,
                            &prev,
                            &prev2,
                        );
                        let guess = self.model.predict(&feature_buf.keys()).to_string();
                        self.model.update(tag, &guess, feature_buf.features());
                        guess
                    };

                    prev2 = prev;
                    prev = guess;
                }
            }

            tagged_sents.shuffle(&mut rng);
        }

        self.model.average_weights();
    }

    /// Save the model to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the model will be saved as a JSON file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        // Convert FxHashMap to HashMap for JSON serialization
        let weights: HashMap<String, HashMap<String, f64>> = self
            .model
            .weights
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.iter().map(|(k2, v2)| (k2.clone(), *v2)).collect(),
                )
            })
            .collect();
        let tagdict: HashMap<String, String> = self
            .tagdict
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let model_data = serde_json::json!({
            "weights": weights,
            "tagdict": tagdict,
            "classes": self.classes.iter().collect::<Vec<_>>(),
        });

        let file = std::fs::File::create(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create file: {}", e))
        })?;
        serde_json::to_writer_pretty(file, &model_data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write JSON: {}", e))
        })?;

        Ok(())
    }

    /// Load a model from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the model, stored as a JSON file, is located.
    pub fn load(&mut self, path: &str) -> PyResult<()> {
        let file = std::fs::File::open(path).map_err(|_| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Can't locate tagger model {}",
                path
            ))
        })?;

        let data: serde_json::Value = serde_json::from_reader(file).map_err(|e| {
            pyo3::exceptions::PyEnvironmentError::new_err(format!(
                "A file is detected at {}, but it cannot be read as a tagger model. \
                 The tagger model JSON file may be corrupted for some reason. Error: {}",
                path, e
            ))
        })?;

        // Load weights (deserialize to std HashMap, then convert)
        if let Some(weights) = data.get("weights") {
            let std_weights: HashMap<String, HashMap<String, f64>> =
                serde_json::from_value(weights.clone()).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to parse weights: {}",
                        e
                    ))
                })?;
            self.model.weights = std_weights
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect();
        }

        // Load tagdict
        if let Some(tagdict) = data.get("tagdict") {
            let std_tagdict: HashMap<String, String> = serde_json::from_value(tagdict.clone())
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to parse tagdict: {}",
                        e
                    ))
                })?;
            self.tagdict = std_tagdict.into_iter().collect();
        }

        // Load classes
        if let Some(classes) = data.get("classes") {
            let classes_vec: Vec<String> =
                serde_json::from_value(classes.clone()).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to parse classes: {}",
                        e
                    ))
                })?;
            self.classes = classes_vec.into_iter().collect();
        }

        self.model.classes = self.classes.clone();
        self.model.finalize_classes();

        Ok(())
    }

    /// Get the model's weights dictionary.
    ///
    /// # Returns
    ///
    /// A dictionary mapping features to their weight vectors.
    #[getter]
    fn weights(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (feat, weights) in &self.model.weights {
            let inner_dict = PyDict::new(py);
            for (class, weight) in weights {
                inner_dict.set_item(class, weight)?;
            }
            dict.set_item(feat, inner_dict)?;
        }
        Ok(dict.into())
    }

    /// Get the tag dictionary.
    ///
    /// # Returns
    ///
    /// A dictionary mapping words to their most likely tags.
    #[getter]
    fn tagdict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (word, tag) in &self.tagdict {
            dict.set_item(word, tag)?;
        }
        Ok(dict.into())
    }

    /// Get the set of POS tag classes.
    ///
    /// # Returns
    ///
    /// A set of all tag classes in the model.
    #[getter]
    fn classes(&self) -> HashSet<String> {
        self.classes.iter().cloned().collect()
    }
}

/// A reusable buffer for building feature strings.
/// Avoids repeated allocations during tagging/training.
struct FeatureBuffer {
    features: Vec<String>,
}

impl FeatureBuffer {
    fn new() -> Self {
        Self {
            features: Vec::with_capacity(16),
        }
    }

    fn clear(&mut self) {
        self.features.clear();
    }

    fn push(&mut self, feature: String) {
        self.features.push(feature);
    }

    fn keys(&self) -> Vec<&str> {
        self.features.iter().map(|s| s.as_str()).collect()
    }

    fn features(&self) -> &[String] {
        &self.features
    }
}

impl AveragedPerceptronTagger {
    /// Map tokens into a feature representation.
    /// Optimized: writes features into a reusable buffer.
    #[allow(clippy::too_many_arguments)]
    fn get_features_into(
        &self,
        buf: &mut FeatureBuffer,
        word: &str,
        i_m2: &str,
        i_m1: &str,
        i_p1: &str,
        i_p2: &str,
        prev: &str,
        prev2: &str,
    ) {
        // It's useful to have a constant feature, which acts sort of like a prior.
        buf.push("bias".to_string());

        // Current word features
        buf.push(format!("i word's first char {}", first_char(word)));
        buf.push(format!("i word's final char {}", final_char(word)));

        // Previous word features (i-1)
        buf.push(format!("i-1 word's first char {}", first_char(i_m1)));
        buf.push(format!("i-1 word's final char {}", final_char(i_m1)));
        buf.push(format!("i-1 tag {}", prev));

        // Second previous word features (i-2)
        buf.push(format!("i-2 word's first char {}", first_char(i_m2)));
        buf.push(format!("i-2 word's final char {}", final_char(i_m2)));
        buf.push(format!("i-2 tag {}", prev2));

        // Next word features (i+1)
        buf.push(format!("i+1 word's first char {}", first_char(i_p1)));
        buf.push(format!("i+1 word's final char {}", final_char(i_p1)));

        // Second next word features (i+2)
        buf.push(format!("i+2 word's first char {}", first_char(i_p2)));
        buf.push(format!("i+2 word's final char {}", final_char(i_p2)));
    }

    /// Make a tag dictionary for single-tag words.
    fn make_tagdict(&mut self, tagged_sents: &[Vec<(String, String)>]) {
        let mut counts: FxHashMap<String, FxHashMap<String, u32>> = FxHashMap::default();

        for tagged_sent in tagged_sents {
            for (word, tag) in tagged_sent {
                *counts
                    .entry(word.clone())
                    .or_default()
                    .entry(tag.clone())
                    .or_insert(0) += 1;
                self.classes.insert(tag.clone());
            }
        }

        for (word, tag_freqs) in &counts {
            // Find the most frequent tag and its count
            let (best_tag, mode) = tag_freqs
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(tag, count)| (tag.clone(), *count))
                .unwrap_or_default();

            let n: u32 = tag_freqs.values().sum();
            let above_freq_threshold = n >= self.frequency_threshold;
            let unambiguous = (mode as f64 / n as f64) >= self.ambiguity_threshold;

            if above_freq_threshold && unambiguous {
                self.tagdict.insert(word.clone(), best_tag);
            }
        }
    }
}

/// Get the first character of a string, or an empty string if the string is empty.
#[inline]
fn first_char(s: &str) -> &str {
    s.chars().next().map(|c| &s[..c.len_utf8()]).unwrap_or("")
}

/// Get the final character of a string, or an empty string if the string is empty.
#[inline]
fn final_char(s: &str) -> &str {
    s.chars()
        .next_back()
        .map(|c| &s[s.len() - c.len_utf8()..])
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tagger = AveragedPerceptronTagger::new(10, 0.95, 5);
        assert_eq!(tagger.frequency_threshold, 10);
        assert!(tagger.tagdict.is_empty());
        assert!(tagger.classes.is_empty());
    }

    #[test]
    fn test_tag_empty() {
        let tagger = AveragedPerceptronTagger::new(10, 0.95, 5);
        let tags = tagger.tag(vec![]);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_first_char() {
        assert_eq!(first_char("hello"), "h");
        assert_eq!(first_char("世界"), "世");
        assert_eq!(first_char(""), "");
    }

    #[test]
    fn test_final_char() {
        assert_eq!(final_char("hello"), "o");
        assert_eq!(final_char("世界"), "界");
        assert_eq!(final_char(""), "");
    }

    #[test]
    fn test_train_and_tag() {
        let mut tagger = AveragedPerceptronTagger::new(1, 0.9, 2);

        let training_data = vec![
            vec![
                ("I".to_string(), "PRON".to_string()),
                ("love".to_string(), "VERB".to_string()),
                ("cats".to_string(), "NOUN".to_string()),
            ],
            vec![
                ("You".to_string(), "PRON".to_string()),
                ("love".to_string(), "VERB".to_string()),
                ("dogs".to_string(), "NOUN".to_string()),
            ],
        ];

        tagger.train(training_data);

        // The tagger should have learned something
        assert!(!tagger.classes.is_empty());

        // Test tagging
        let words = vec!["I".to_string(), "love".to_string(), "cats".to_string()];
        let tags = tagger.tag(words);
        assert_eq!(tags.len(), 3);
    }
}
