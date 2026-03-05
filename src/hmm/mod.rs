//! Hidden Markov Model.
//!
//! A Hidden Markov Model with both unsupervised (Baum-Welch EM) and supervised
//! (labeled counting with configurable Lidstone smoothing) training, Viterbi
//! decoding, and Forward-algorithm scoring.
//! All probabilities are stored in log-space for numerical stability.
//!
//! Also includes shared I/O utilities for serializing and deserializing HMM
//! model parameters in FlatBuffers binary format.

use crate::seq_feature::{
    SeqFeatureConfig, SeqFeatureTemplate, SeqTransform, default_tagger_hmm_features,
    extract_observation, extract_observation_cow, validate_templates,
};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::{StdRng, SysRng};
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::PathBuf;

// FlatBuffers generated code (produced by build.rs from src/hmm/model.fbs).
#[allow(dead_code, unused_imports, clippy::all)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/hmm/model_generated.rs"));
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Numerically stable log-sum-exp.
fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

use crate::persistence::{ModelError, pathbuf_to_string};

// ---------------------------------------------------------------------------
// Model data returned by load
// ---------------------------------------------------------------------------

/// Parsed HMM model data returned by [`load_hmm_flatbuffers`].
pub(crate) struct HmmModelData {
    pub log_initial: Vec<f64>,
    pub log_transition: Vec<Vec<f64>>,
    pub feature_vocabs: Vec<FxHashMap<String, usize>>,
    pub feature_log_emissions: Vec<Vec<Vec<f64>>>,
    pub state_labels: Option<Vec<String>>,
}

/// Data bundle passed to [`save_hmm_flatbuffers`].
pub(crate) struct HmmSaveData<'a> {
    pub log_initial: &'a [f64],
    pub log_transition: &'a [Vec<f64>],
    pub feature_vocabs: &'a [FxHashMap<String, usize>],
    pub feature_log_emissions: &'a [Vec<Vec<f64>>],
    pub templates: &'a [SeqFeatureTemplate],
    pub n_states: Option<usize>,
    pub state_labels: Option<&'a [String]>,
}

// ---------------------------------------------------------------------------
// FlatBuffers save / load
// ---------------------------------------------------------------------------

/// Save HMM model data to a FlatBuffers binary stream.
///
/// The output is a raw FlatBuffers buffer (no gzip, no length prefix).
/// Use [`load_hmm_flatbuffers`] to read it back.
pub(crate) fn save_hmm_flatbuffers<W: Write>(
    writer: &mut W,
    data: &HmmSaveData,
) -> Result<(), ModelError> {
    use generated::rustling::hmm_fbs as fbs;

    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(64 * 1024 * 1024);

    // Build feature_vocabs — must be built before builder is borrowed for scalars.
    let fb_feature_vocabs: Vec<_> = data
        .feature_vocabs
        .iter()
        .map(|vocab| {
            let entries: Vec<_> = vocab
                .iter()
                .map(|(key, &emission_idx)| {
                    let key_fb = builder.create_string(key);
                    fbs::VocabEntry::create(
                        &mut builder,
                        &fbs::VocabEntryArgs {
                            key: Some(key_fb),
                            emission_idx: emission_idx as u32,
                        },
                    )
                })
                .collect();
            let entries_fb = builder.create_vector(&entries);
            fbs::FeatureVocab::create(
                &mut builder,
                &fbs::FeatureVocabArgs {
                    entries: Some(entries_fb),
                },
            )
        })
        .collect();

    // Build feature_emissions.
    let fb_emissions: Vec<_> = data
        .feature_log_emissions
        .iter()
        .map(|feature_em| {
            let n_states = feature_em.len() as u32;
            let vocab_size = feature_em.first().map(|s| s.len()).unwrap_or(0) as u32;
            let flat: Vec<f32> = feature_em
                .iter()
                .flat_map(|row| row.iter().map(|&x| x as f32))
                .collect();
            let values_fb = builder.create_vector(&flat);
            fbs::EmissionMatrix::create(
                &mut builder,
                &fbs::EmissionMatrixArgs {
                    n_states,
                    vocab_size,
                    values: Some(values_fb),
                },
            )
        })
        .collect();

    // Simple vectors.
    let log_initial_f32: Vec<f32> = data.log_initial.iter().map(|&x| x as f32).collect();
    let log_initial_fb = builder.create_vector(&log_initial_f32);
    let flat_transition: Vec<f32> = data
        .log_transition
        .iter()
        .flat_map(|row| row.iter().map(|&x| x as f32))
        .collect();
    let log_transition_fb = builder.create_vector(&flat_transition);
    let state_labels_fb = data.state_labels.map(|labels| {
        let strs: Vec<_> = labels.iter().map(|s| builder.create_string(s)).collect();
        builder.create_vector(&strs)
    });
    let features_json = serde_json::to_string(data.templates)
        .map_err(|e| ModelError::Io(format!("Failed to serialize features: {e}")))?;
    let features_json_fb = builder.create_string(&features_json);
    let feature_vocabs_fb = builder.create_vector(&fb_feature_vocabs);
    let feature_emissions_fb = builder.create_vector(&fb_emissions);

    let model = fbs::HmmModel::create(
        &mut builder,
        &fbs::HmmModelArgs {
            n_states: data.n_states.unwrap_or(0) as u32,
            state_labels: state_labels_fb,
            log_initial: Some(log_initial_fb),
            log_transition: Some(log_transition_fb),
            feature_vocabs: Some(feature_vocabs_fb),
            feature_emissions: Some(feature_emissions_fb),
            features_json: Some(features_json_fb),
        },
    );
    builder.finish(model, None);

    writer
        .write_all(builder.finished_data())
        .map_err(|e| ModelError::Io(format!("Failed to write FlatBuffers data: {e}")))
}

/// Load HMM model data from a FlatBuffers byte slice.
///
/// `n_features` must match the number of feature templates the HMM was
/// initialized with; used to validate the loaded data.
pub(crate) fn load_hmm_flatbuffers(
    bytes: &[u8],
    n_features: usize,
    expected_n_states: Option<usize>,
) -> Result<HmmModelData, ModelError> {
    use generated::rustling::hmm_fbs as fbs;

    let opts = crate::persistence::flatbuffers_verifier_opts();
    let model = flatbuffers::root_with_opts::<fbs::HmmModel>(&opts, bytes)
        .map_err(|e| ModelError::ParseError(format!("Invalid FlatBuffers HMM data: {e}")))?;

    let n_states = model.n_states() as usize;
    if let Some(expected) = expected_n_states
        && n_states != expected
    {
        return Err(ModelError::ParseError(format!(
            "n_states mismatch: expected {expected}, got {n_states}"
        )));
    }

    // (required) fields return the type directly, not Option.
    let log_initial: Vec<f64> = model.log_initial().iter().map(|x| x as f64).collect();

    let log_transition_flat: Vec<f64> = model.log_transition().iter().map(|x| x as f64).collect();
    let log_transition: Vec<Vec<f64>> = log_transition_flat
        .chunks(n_states)
        .map(|c: &[f64]| c.to_vec())
        .collect();

    let state_labels: Option<Vec<String>> = model
        .state_labels()
        .map(|sl| sl.iter().map(|s| s.to_owned()).collect());

    let fb_vocabs = model.feature_vocabs();
    if fb_vocabs.len() != n_features {
        return Err(ModelError::ParseError(format!(
            "Expected {n_features} feature vocabs, got {}",
            fb_vocabs.len()
        )));
    }
    let feature_vocabs: Vec<FxHashMap<String, usize>> = fb_vocabs
        .iter()
        .map(|fv: fbs::FeatureVocab<'_>| {
            fv.entries()
                .iter()
                .map(|e: fbs::VocabEntry<'_>| (e.key().to_owned(), e.emission_idx() as usize))
                .collect()
        })
        .collect();

    let fb_emissions = model.feature_emissions();
    if fb_emissions.len() != n_features {
        return Err(ModelError::ParseError(format!(
            "Expected {n_features} emission matrices, got {}",
            fb_emissions.len()
        )));
    }
    let feature_log_emissions: Vec<Vec<Vec<f64>>> = fb_emissions
        .iter()
        .map(|em: fbs::EmissionMatrix<'_>| {
            let ns = em.n_states() as usize;
            let vs = em.vocab_size() as usize;
            let flat: Vec<f64> = em.values().iter().map(|x| x as f64).collect();
            if vs == 0 {
                vec![Vec::new(); ns]
            } else {
                flat.chunks(vs)
                    .take(ns)
                    .map(|r: &[f64]| r.to_vec())
                    .collect()
            }
        })
        .collect();

    Ok(HmmModelData {
        log_initial,
        log_transition,
        feature_vocabs,
        feature_log_emissions,
        state_labels,
    })
}

// ---------------------------------------------------------------------------
// Standalone Viterbi decode
// ---------------------------------------------------------------------------

/// Flatten a `Vec<Vec<f64>>` transition matrix into a contiguous `Vec<f64>`
/// with row-major layout `[i * n_states + j]`.
fn flatten_transition(log_transition: &[Vec<f64>], n_states: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(n_states * n_states);
    for row in log_transition.iter().take(n_states) {
        flat.extend_from_slice(row);
    }
    flat
}

/// Reusable working-memory buffers for [`viterbi_decode`].
pub(crate) struct ViterbiBuffers {
    viterbi: Vec<f64>,
    backptr: Vec<usize>,
    frame_emit: Vec<f64>,
    path: Vec<usize>,
}

impl ViterbiBuffers {
    fn new() -> Self {
        Self {
            viterbi: Vec::new(),
            backptr: Vec::new(),
            frame_emit: Vec::new(),
            path: Vec::new(),
        }
    }
}

/// Viterbi decoding: find the most probable state sequence.
///
/// All probabilities are in log-space. Pre-computes frame emission log-probs
/// into a contiguous buffer for cache-friendly access (like hmmlearn's
/// `framelogprob`), then runs the standard Viterbi DP.
///
/// `flat_log_transition` has row-major layout `[i * n_states + j]`.
/// `encoded` has feature-major layout `[f * t_len + t]`.
///
/// Uses reusable buffers from `bufs` to avoid per-call allocation.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn viterbi_decode(
    n_states: usize,
    log_initial: &[f64],
    flat_log_transition: &[f64],
    encoded: &[Option<usize>],
    t_len: usize,
    n_features: usize,
    feature_log_emissions: &[Vec<Vec<f64>>],
    unknown_log_probs: &[f64],
    bufs: &mut ViterbiBuffers,
) -> Vec<usize> {
    let total = t_len * n_states;

    // Pre-compute frame emission log-probs: frame_emit[t * n_states + state].
    bufs.frame_emit.clear();
    bufs.frame_emit.resize(total, 0.0);
    for t in 0..t_len {
        let base = t * n_states;
        for state in 0..n_states {
            let mut log_prob = 0.0;
            for f in 0..n_features {
                match encoded[f * t_len + t] {
                    Some(k) => log_prob += feature_log_emissions[f][state][k],
                    None => log_prob += unknown_log_probs[f],
                }
            }
            bufs.frame_emit[base + state] = log_prob;
        }
    }

    // Prepare flat viterbi and backptr buffers.
    bufs.viterbi.clear();
    bufs.viterbi.resize(total, f64::NEG_INFINITY);
    bufs.backptr.clear();
    bufs.backptr.resize(total, 0usize);

    // Initialization: t=0.
    for i in 0..n_states {
        bufs.viterbi[i] = log_initial[i] + bufs.frame_emit[i];
    }

    // Forward pass: t=1..T-1.
    for t in 1..t_len {
        let prev_base = (t - 1) * n_states;
        let curr_base = t * n_states;
        for j in 0..n_states {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_state = 0usize;
            for i in 0..n_states {
                let score = bufs.viterbi[prev_base + i] + flat_log_transition[i * n_states + j];
                if score > best_score {
                    best_score = score;
                    best_state = i;
                }
            }
            bufs.viterbi[curr_base + j] = best_score + bufs.frame_emit[curr_base + j];
            bufs.backptr[curr_base + j] = best_state;
        }
    }

    // Backtrace.
    bufs.path.clear();
    bufs.path.resize(t_len, 0usize);
    let last_base = (t_len - 1) * n_states;
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..n_states {
        if bufs.viterbi[last_base + i] > best_val {
            best_val = bufs.viterbi[last_base + i];
            best_idx = i;
        }
    }
    bufs.path[t_len - 1] = best_idx;
    for t in (0..t_len - 1).rev() {
        bufs.path[t] = bufs.backptr[(t + 1) * n_states + bufs.path[t + 1]];
    }

    std::mem::take(&mut bufs.path)
}

// ---------------------------------------------------------------------------
// BaseHiddenMarkovModel trait
// ---------------------------------------------------------------------------

/// Core HMM behavior with default implementations.
///
/// Accumulator for the E-step of the Baum-Welch (EM) algorithm.
struct EmAccumulator {
    log_initial_acc: Vec<f64>,
    log_trans_acc: Vec<Vec<f64>>,
    log_emit_acc: Vec<Vec<Vec<f64>>>,
    total_log_likelihood: f64,
}

/// Uses per-feature emission tracks: each feature template has its own
/// vocabulary and emission matrix. Combined emission is the sum of
/// per-feature log-probabilities.
#[allow(clippy::needless_range_loop)]
pub trait BaseHiddenMarkovModel: Sized + Clone + Sync {
    fn n_states(&self) -> usize;
    fn set_n_states(&mut self, n: usize);
    fn n_iter(&self) -> usize;
    fn tolerance(&self) -> f64;
    fn gamma(&self) -> f64;
    fn random_seed(&self) -> Option<u64>;
    fn fitted(&self) -> bool;
    fn set_fitted(&mut self, fitted: bool);
    fn log_initial(&self) -> &Vec<f64>;
    fn log_initial_mut(&mut self) -> &mut Vec<f64>;
    fn log_transition(&self) -> &Vec<Vec<f64>>;
    fn log_transition_mut(&mut self) -> &mut Vec<Vec<f64>>;
    fn features(&self) -> &SeqFeatureConfig;
    fn feature_vocabs(&self) -> &Vec<FxHashMap<String, usize>>;
    fn feature_vocabs_mut(&mut self) -> &mut Vec<FxHashMap<String, usize>>;
    fn feature_log_emissions(&self) -> &Vec<Vec<Vec<f64>>>;
    fn feature_log_emissions_mut(&mut self) -> &mut Vec<Vec<Vec<f64>>>;
    fn state_labels(&self) -> &Option<Vec<String>>;
    fn state_labels_mut(&mut self) -> &mut Option<Vec<String>>;

    /// Build per-feature vocabulary mappings from observation sequences.
    fn build_vocab(&mut self, sequences: &[Vec<String>]) {
        let n_features = self.features().templates.len();
        let mut vocabs: Vec<FxHashMap<String, usize>> =
            (0..n_features).map(|_| FxHashMap::default()).collect();
        let mut next_indices: Vec<usize> = vec![0; n_features];

        for seq in sequences {
            let obs: Vec<&str> = seq.iter().map(|s| s.as_str()).collect();
            for t in 0..obs.len() {
                for (f, template) in self.features().templates.iter().enumerate() {
                    let feat_val = extract_observation(template, &obs, t);
                    if !vocabs[f].contains_key(&feat_val) {
                        vocabs[f].insert(feat_val, next_indices[f]);
                        next_indices[f] += 1;
                    }
                }
            }
        }
        *self.feature_vocabs_mut() = vocabs;
    }

    /// Extend per-feature vocabulary mappings with new observations.
    ///
    /// Existing vocabulary entries and their indices are preserved. Only new
    /// observations not already in the vocabulary are added with fresh indices.
    /// Returns the number of new vocab entries added per feature.
    fn extend_vocab(&mut self, sequences: &[Vec<String>]) -> Vec<usize> {
        let n_features = self.features().templates.len();
        let mut vocabs: Vec<FxHashMap<String, usize>> = self.feature_vocabs().to_vec();
        let mut new_counts = vec![0usize; n_features];

        for seq in sequences {
            let obs: Vec<&str> = seq.iter().map(|s| s.as_str()).collect();
            for t in 0..obs.len() {
                for (f, template) in self.features().templates.iter().enumerate() {
                    let feat_val = extract_observation(template, &obs, t);
                    if !vocabs[f].contains_key(&feat_val) {
                        let next_idx = vocabs[f].len();
                        vocabs[f].insert(feat_val, next_idx);
                        new_counts[f] += 1;
                    }
                }
            }
        }
        *self.feature_vocabs_mut() = vocabs;
        new_counts
    }

    /// Extend emission matrices to accommodate new vocabulary entries.
    ///
    /// For each feature whose vocabulary grew, appends columns to each state's
    /// emission row. New entries receive the minimum existing log-emission for
    /// that state/feature row (a conservative small probability).
    fn extend_emissions_for_new_vocab(&mut self, new_counts: &[usize]) {
        let n = self.n_states();
        for (f, &count) in new_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            for i in 0..n {
                let min_val = self.feature_log_emissions()[f][i]
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                let init_val = if min_val.is_finite() {
                    min_val
                } else {
                    -(self.feature_vocabs()[f].len() as f64).ln()
                };
                self.feature_log_emissions_mut()[f][i].extend(std::iter::repeat_n(init_val, count));
            }
        }
    }

    /// Randomly initialize π, A, and per-feature B.
    fn initialize_parameters(&mut self) {
        let n = self.n_states();
        let n_features = self.features().templates.len();

        let mut rng: StdRng = match self.random_seed() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::try_from_rng(&mut SysRng).unwrap(),
        };

        let uniform = Uniform::new(0.0_f64, 1.0).unwrap();

        // Initialize π
        let raw: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
        let total: f64 = raw.iter().sum();
        *self.log_initial_mut() = raw.iter().map(|r| (r / total).ln()).collect();

        // Initialize A
        let mut log_trans = vec![vec![0.0; n]; n];
        for row in &mut log_trans {
            let raw_row: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
            let row_total: f64 = raw_row.iter().sum();
            for (j, r) in raw_row.iter().enumerate() {
                row[j] = (r / row_total).ln();
            }
        }
        *self.log_transition_mut() = log_trans;

        // Initialize per-feature B
        let mut all_log_emit: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let m = self.feature_vocabs()[f].len();
            let mut log_emit = vec![vec![0.0; m]; n];
            for row in &mut log_emit {
                let raw_row: Vec<f64> = (0..m).map(|_| uniform.sample(&mut rng)).collect();
                let row_total: f64 = raw_row.iter().sum();
                for (k, r) in raw_row.iter().enumerate() {
                    row[k] = (r / row_total).ln();
                }
            }
            all_log_emit.push(log_emit);
        }
        *self.feature_log_emissions_mut() = all_log_emit;
    }

    /// Compute combined emission log-probability for a state at a time step.
    fn combined_emission(
        &self,
        state: usize,
        encoded: &[Vec<Option<usize>>],
        t: usize,
        unknown_log_probs: &[f64],
    ) -> f64 {
        let mut log_prob = 0.0;
        for (f, obs_idx) in encoded.iter().enumerate() {
            match obs_idx[t] {
                Some(k) => log_prob += self.feature_log_emissions()[f][state][k],
                None => log_prob += unknown_log_probs[f],
            }
        }
        log_prob
    }

    /// Encode a sequence into per-feature index vectors.
    fn encode_sequence(&self, observations: &[String]) -> Vec<Vec<Option<usize>>> {
        let obs: Vec<&str> = observations.iter().map(|s| s.as_str()).collect();
        self.features()
            .templates
            .iter()
            .enumerate()
            .map(|(f, template)| {
                (0..obs.len())
                    .map(|t| {
                        let feat_val = extract_observation_cow(template, &obs, t);
                        self.feature_vocabs()[f].get(feat_val.as_ref()).copied()
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute unknown log-probs per feature.
    fn unknown_log_probs(&self) -> Vec<f64> {
        self.feature_vocabs()
            .iter()
            .map(|v| {
                if v.is_empty() {
                    0.0
                } else {
                    -(v.len() as f64).ln()
                }
            })
            .collect()
    }

    /// Train the model. If labels are provided, uses supervised counting.
    /// If labels are None, uses Baum-Welch EM.
    fn fit(
        &mut self,
        sequences: Vec<Vec<String>>,
        labels: Option<Vec<Vec<String>>>,
    ) -> Result<(), String> {
        match labels {
            None => {
                self.fit_unlabeled(sequences);
                Ok(())
            }
            Some(labels) => self.fit_labeled(sequences, labels),
        }
    }

    /// Train the model using the Baum-Welch (EM) algorithm (unsupervised).
    ///
    /// If the model is already fitted (e.g., from a prior supervised `fit` call),
    /// the existing parameters are used as the EM initialization (warm start).
    /// Otherwise, parameters are randomly initialized (cold start).
    fn fit_unlabeled(&mut self, sequences: Vec<Vec<String>>) {
        let sequences: Vec<Vec<String>> = sequences.into_iter().filter(|s| !s.is_empty()).collect();
        if sequences.is_empty() {
            self.set_fitted(true);
            return;
        }

        let already_fitted = self.fitted();

        if already_fitted {
            // Warm start: extend vocab and emissions, keep existing parameters.
            let new_counts = self.extend_vocab(&sequences);
            self.extend_emissions_for_new_vocab(&new_counts);
        } else {
            // Cold start: build vocab from scratch and randomly initialize.
            self.build_vocab(&sequences);
            self.initialize_parameters();
        }

        let n = self.n_states();
        let n_features = self.features().templates.len();

        // Encode all sequences.
        #[cfg(feature = "parallel")]
        let encoded_sequences: Vec<Vec<Vec<Option<usize>>>> = {
            use rayon::prelude::*;
            sequences
                .par_iter()
                .with_min_len(16)
                .map(|seq| self.encode_sequence(seq))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let encoded_sequences: Vec<Vec<Vec<Option<usize>>>> = sequences
            .iter()
            .map(|seq| self.encode_sequence(seq))
            .collect();

        let unknown_lp = self.unknown_log_probs();

        let vocab_sizes: Vec<usize> = (0..n_features)
            .map(|f| self.feature_vocabs()[f].len())
            .collect();

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for _iter in 0..self.n_iter() {
            let make_identity = || EmAccumulator {
                log_initial_acc: vec![f64::NEG_INFINITY; n],
                log_trans_acc: vec![vec![f64::NEG_INFINITY; n]; n],
                log_emit_acc: (0..n_features)
                    .map(|f| vec![vec![f64::NEG_INFINITY; vocab_sizes[f]]; n])
                    .collect(),
                total_log_likelihood: 0.0,
            };

            let process_sequence =
                |acc: &mut EmAccumulator, seq_idx: usize, encoded: &Vec<Vec<Option<usize>>>| {
                    let t_len = sequences[seq_idx].len();

                    // Forward.
                    let mut alpha = vec![vec![f64::NEG_INFINITY; n]; t_len];
                    for i in 0..n {
                        alpha[0][i] = self.log_initial()[i]
                            + self.combined_emission(i, encoded, 0, &unknown_lp);
                    }
                    let mut buf = vec![0.0; n];
                    for t in 1..t_len {
                        for j in 0..n {
                            for i in 0..n {
                                buf[i] = alpha[t - 1][i] + self.log_transition()[i][j];
                            }
                            alpha[t][j] = logsumexp(&buf)
                                + self.combined_emission(j, encoded, t, &unknown_lp);
                        }
                    }
                    let seq_ll = logsumexp(&alpha[t_len - 1]);
                    acc.total_log_likelihood += seq_ll;

                    // Backward.
                    let mut beta = vec![vec![f64::NEG_INFINITY; n]; t_len];
                    for i in 0..n {
                        beta[t_len - 1][i] = 0.0;
                    }
                    for t in (0..t_len - 1).rev() {
                        for i in 0..n {
                            for j in 0..n {
                                buf[j] = self.log_transition()[i][j]
                                    + self.combined_emission(j, encoded, t + 1, &unknown_lp)
                                    + beta[t + 1][j];
                            }
                            beta[t][i] = logsumexp(&buf);
                        }
                    }

                    // Accumulate initial.
                    for i in 0..n {
                        let gamma_0_i = alpha[0][i] + beta[0][i] - seq_ll;
                        acc.log_initial_acc[i] = logsumexp(&[acc.log_initial_acc[i], gamma_0_i]);
                    }

                    // Accumulate transition.
                    for t in 0..t_len - 1 {
                        for i in 0..n {
                            for j in 0..n {
                                let xi = alpha[t][i]
                                    + self.log_transition()[i][j]
                                    + self.combined_emission(j, encoded, t + 1, &unknown_lp)
                                    + beta[t + 1][j]
                                    - seq_ll;
                                acc.log_trans_acc[i][j] = logsumexp(&[acc.log_trans_acc[i][j], xi]);
                            }
                        }
                    }

                    // Accumulate per-feature emission.
                    for t in 0..t_len {
                        for i in 0..n {
                            let gamma_t_i = alpha[t][i] + beta[t][i] - seq_ll;
                            for f in 0..n_features {
                                if let Some(k) = encoded[f][t] {
                                    acc.log_emit_acc[f][i][k] =
                                        logsumexp(&[acc.log_emit_acc[f][i][k], gamma_t_i]);
                                }
                            }
                        }
                    }
                };

            #[cfg(feature = "parallel")]
            let acc = {
                use rayon::prelude::*;
                let merge = |mut a: EmAccumulator, b: EmAccumulator| -> EmAccumulator {
                    for i in 0..n {
                        a.log_initial_acc[i] =
                            logsumexp(&[a.log_initial_acc[i], b.log_initial_acc[i]]);
                    }
                    for i in 0..n {
                        for j in 0..n {
                            a.log_trans_acc[i][j] =
                                logsumexp(&[a.log_trans_acc[i][j], b.log_trans_acc[i][j]]);
                        }
                    }
                    for f in 0..n_features {
                        for i in 0..n {
                            for k in 0..vocab_sizes[f] {
                                a.log_emit_acc[f][i][k] =
                                    logsumexp(&[a.log_emit_acc[f][i][k], b.log_emit_acc[f][i][k]]);
                            }
                        }
                    }
                    a.total_log_likelihood += b.total_log_likelihood;
                    a
                };
                encoded_sequences
                    .par_iter()
                    .enumerate()
                    .with_min_len(16)
                    .fold(make_identity, |mut acc, (seq_idx, encoded)| {
                        process_sequence(&mut acc, seq_idx, encoded);
                        acc
                    })
                    .reduce(make_identity, merge)
            };
            #[cfg(not(feature = "parallel"))]
            let acc = {
                let mut acc = make_identity();
                for (seq_idx, encoded) in encoded_sequences.iter().enumerate() {
                    process_sequence(&mut acc, seq_idx, encoded);
                }
                acc
            };

            // M-step: normalize.
            let total = logsumexp(&acc.log_initial_acc);
            for i in 0..n {
                self.log_initial_mut()[i] = acc.log_initial_acc[i] - total;
            }

            for i in 0..n {
                let row_total = logsumexp(&acc.log_trans_acc[i]);
                for j in 0..n {
                    self.log_transition_mut()[i][j] = acc.log_trans_acc[i][j] - row_total;
                }
            }

            for f in 0..n_features {
                for i in 0..n {
                    let row_total = logsumexp(&acc.log_emit_acc[f][i]);
                    for k in 0..vocab_sizes[f] {
                        self.feature_log_emissions_mut()[f][i][k] =
                            acc.log_emit_acc[f][i][k] - row_total;
                    }
                }
            }

            if (acc.total_log_likelihood - prev_log_likelihood).abs() < self.tolerance() {
                break;
            }
            prev_log_likelihood = acc.total_log_likelihood;
        }

        self.prune_vocab(1e-6);

        if !already_fitted {
            *self.state_labels_mut() = None;
        }
        self.set_fitted(true);
    }

    /// Prune vocabulary entries whose emission values equal the row minimum
    /// across all states (within `tolerance`). These entries behave identically
    /// to the unknown-observation fallback and add no discriminative power.
    ///
    /// After pruning, vocab indices are compacted to be contiguous.
    fn prune_vocab(&mut self, tolerance: f64) {
        let n_features = self.features().templates.len();
        let n_states = self.n_states();

        for f in 0..n_features {
            let vocab_size = self.feature_log_emissions()[f]
                .first()
                .map(|r| r.len())
                .unwrap_or(0);
            if vocab_size == 0 {
                continue;
            }

            // Find the row minimum for each state.
            let row_mins: Vec<f64> = (0..n_states)
                .map(|i| {
                    self.feature_log_emissions()[f][i]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            // Identify which vocab indices to keep: an entry is kept if
            // for at least one state its emission differs from the row min.
            let keep: Vec<bool> = (0..vocab_size)
                .map(|k| {
                    (0..n_states).any(|i| {
                        (self.feature_log_emissions()[f][i][k] - row_mins[i]).abs() > tolerance
                    })
                })
                .collect();

            let kept_count = keep.iter().filter(|&&b| b).count();
            if kept_count == vocab_size {
                continue; // Nothing to prune.
            }

            // Build old-index → new-index mapping.
            let mut old_to_new = vec![0usize; vocab_size];
            let mut new_idx = 0usize;
            for (old_idx, &kept) in keep.iter().enumerate() {
                if kept {
                    old_to_new[old_idx] = new_idx;
                    new_idx += 1;
                }
            }

            // Compact emission matrix.
            let new_emissions: Vec<Vec<f64>> = (0..n_states)
                .map(|i| {
                    keep.iter()
                        .enumerate()
                        .filter(|&(_, kept)| *kept)
                        .map(|(k, _)| self.feature_log_emissions()[f][i][k])
                        .collect()
                })
                .collect();
            self.feature_log_emissions_mut()[f] = new_emissions;

            // Compact vocabulary: remove pruned keys, reindex remaining ones.
            let mut new_vocab: FxHashMap<String, usize> = FxHashMap::default();
            for (key, &old_idx) in self.feature_vocabs()[f].iter() {
                if keep[old_idx] {
                    new_vocab.insert(key.clone(), old_to_new[old_idx]);
                }
            }
            self.feature_vocabs_mut()[f] = new_vocab;
        }
    }

    /// Train the model from labeled (supervised) data.
    ///
    /// Discovers unique labels, auto-sets n_states, counts initial/transition/
    /// emission frequencies with Laplace smoothing, and normalizes to
    /// log-probabilities.
    fn fit_labeled(
        &mut self,
        sequences: Vec<Vec<String>>,
        labels: Vec<Vec<String>>,
    ) -> Result<(), String> {
        if sequences.len() != labels.len() {
            return Err(format!(
                "sequences and labels must have the same length: {} vs {}",
                sequences.len(),
                labels.len()
            ));
        }

        // Filter empty pairs.
        let pairs: Vec<(Vec<String>, Vec<String>)> = sequences
            .into_iter()
            .zip(labels)
            .filter(|(seq, lab)| !seq.is_empty() && !lab.is_empty())
            .collect();

        if pairs.is_empty() {
            self.set_fitted(true);
            return Ok(());
        }

        // Validate each sequence-label pair has matching lengths.
        for (i, (seq, lab)) in pairs.iter().enumerate() {
            if seq.len() != lab.len() {
                return Err(format!(
                    "Sequence {} has length {} but labels have length {}",
                    i,
                    seq.len(),
                    lab.len()
                ));
            }
        }

        // Discover unique labels → sorted → build mapping.
        let mut label_set: BTreeSet<String> = BTreeSet::new();
        for (_, lab) in &pairs {
            for l in lab {
                label_set.insert(l.clone());
            }
        }
        let label_list: Vec<String> = label_set.into_iter().collect();
        let n_states = label_list.len();
        let label_to_idx: FxHashMap<String, usize> = label_list
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i))
            .collect();

        // Auto-set n_states and store state labels.
        self.set_n_states(n_states);
        *self.state_labels_mut() = Some(label_list);

        // Build observation vocab.
        let sequences_only: Vec<Vec<String>> = pairs.iter().map(|(s, _)| s.clone()).collect();
        self.build_vocab(&sequences_only);

        let n_features = self.features().templates.len();

        // Count initial frequencies (Lidstone smoothing: start all at gamma).
        let mut initial_counts = vec![self.gamma(); n_states];
        for (_, lab) in &pairs {
            let state = label_to_idx[&lab[0]];
            initial_counts[state] += 1.0;
        }

        // Count transition frequencies (Lidstone smoothing: start all at gamma).
        let mut transition_counts = vec![vec![self.gamma(); n_states]; n_states];
        for (_, lab) in &pairs {
            for t in 0..lab.len() - 1 {
                let from = label_to_idx[&lab[t]];
                let to = label_to_idx[&lab[t + 1]];
                transition_counts[from][to] += 1.0;
            }
        }

        // Count emission frequencies (Lidstone smoothing: start all at gamma).
        let vocab_sizes: Vec<usize> = (0..n_features)
            .map(|f| self.feature_vocabs()[f].len())
            .collect();
        let mut emission_counts: Vec<Vec<Vec<f64>>> = (0..n_features)
            .map(|f| vec![vec![self.gamma(); vocab_sizes[f]]; n_states])
            .collect();

        for (seq, lab) in &pairs {
            let obs: Vec<&str> = seq.iter().map(|s| s.as_str()).collect();
            for t in 0..obs.len() {
                let state = label_to_idx[&lab[t]];
                for (f, template) in self.features().templates.iter().enumerate() {
                    let feat_val = extract_observation(template, &obs, t);
                    if let Some(&feat_idx) = self.feature_vocabs()[f].get(&feat_val) {
                        emission_counts[f][state][feat_idx] += 1.0;
                    }
                }
            }
        }

        // Normalize initial to log-probabilities.
        let initial_total: f64 = initial_counts.iter().sum();
        *self.log_initial_mut() = initial_counts
            .iter()
            .map(|c| (c / initial_total).ln())
            .collect();

        // Normalize transition to log-probabilities.
        let mut log_transition = vec![vec![0.0; n_states]; n_states];
        for i in 0..n_states {
            let row_total: f64 = transition_counts[i].iter().sum();
            for j in 0..n_states {
                log_transition[i][j] = (transition_counts[i][j] / row_total).ln();
            }
        }
        *self.log_transition_mut() = log_transition;

        // Normalize emissions to log-probabilities.
        let mut feature_log_emissions: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let mut log_emit = vec![vec![0.0; vocab_sizes[f]]; n_states];
            for i in 0..n_states {
                let row_total: f64 = emission_counts[f][i].iter().sum();
                for k in 0..vocab_sizes[f] {
                    log_emit[i][k] = (emission_counts[f][i][k] / row_total).ln();
                }
            }
            feature_log_emissions.push(log_emit);
        }
        *self.feature_log_emissions_mut() = feature_log_emissions;

        self.set_fitted(true);
        Ok(())
    }

    /// Encode a sequence into a flat feature-major buffer.
    ///
    /// Layout: `encoded[f * t_len + t]` for feature `f` at time `t`.
    /// Reuses `buf` to avoid per-call allocation.
    fn encode_sequence_flat(&self, observations: &[String], buf: &mut Vec<Option<usize>>) {
        let t_len = observations.len();
        let templates = &self.features().templates;
        let n_features = templates.len();
        buf.clear();
        buf.reserve(n_features * t_len);
        for (f, template) in templates.iter().enumerate() {
            let vocab = &self.feature_vocabs()[f];
            // Fast path: single position at offset 0 with identity transform.
            // Avoids Vec<&str> allocation and extract_observation_cow overhead.
            if template.positions.len() == 1
                && template.positions[0] == 0
                && template.transform == SeqTransform::Identity
            {
                for obs in observations {
                    buf.push(vocab.get(obs.as_str()).copied());
                }
            } else {
                let obs: Vec<&str> = observations.iter().map(|s| s.as_str()).collect();
                for t in 0..t_len {
                    let feat_val = extract_observation_cow(template, &obs, t);
                    buf.push(vocab.get(feat_val.as_ref()).copied());
                }
            }
        }
    }

    /// Decode with Viterbi (batch).
    fn predict(&self, sequences: Vec<Vec<String>>) -> Result<Vec<Vec<usize>>, String> {
        if !self.fitted() {
            return Err("Model has not been fitted yet.".to_string());
        }
        let n_states = self.n_states();
        let n_features = self.features().templates.len();
        let unknown_lp = self.unknown_log_probs();
        let flat_trans = flatten_transition(self.log_transition(), n_states);
        let feat_emit = self.feature_log_emissions();
        let log_init = self.log_initial();

        let predict_one = |observations: &Vec<String>,
                           enc_buf: &mut Vec<Option<usize>>,
                           vbufs: &mut ViterbiBuffers|
         -> Vec<usize> {
            if observations.is_empty() {
                return Vec::new();
            }
            let t_len = observations.len();
            self.encode_sequence_flat(observations, enc_buf);
            viterbi_decode(
                n_states,
                log_init,
                &flat_trans,
                enc_buf,
                t_len,
                n_features,
                feat_emit,
                &unknown_lp,
                vbufs,
            )
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            Ok(sequences
                .par_iter()
                .with_min_len(128)
                .map_init(
                    || (Vec::new(), ViterbiBuffers::new()),
                    |(enc_buf, vbufs), obs| predict_one(obs, enc_buf, vbufs),
                )
                .collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut enc_buf = Vec::new();
            let mut vbufs = ViterbiBuffers::new();
            Ok(sequences
                .iter()
                .map(|obs| predict_one(obs, &mut enc_buf, &mut vbufs))
                .collect())
        }
    }

    /// Compute log-likelihood with Forward algorithm (batch).
    fn score(&self, sequences: Vec<Vec<String>>) -> Result<Vec<f64>, String> {
        if !self.fitted() {
            return Err("Model has not been fitted yet.".to_string());
        }
        let n = self.n_states();
        let unknown_lp = self.unknown_log_probs();
        let score_one = |observations: &Vec<String>| -> f64 {
            if observations.is_empty() {
                return 0.0;
            }
            let t_len = observations.len();
            let encoded = self.encode_sequence(observations);

            let mut alpha = vec![vec![f64::NEG_INFINITY; n]; t_len];
            for i in 0..n {
                alpha[0][i] =
                    self.log_initial()[i] + self.combined_emission(i, &encoded, 0, &unknown_lp);
            }

            let mut buf = vec![0.0; n];
            for t in 1..t_len {
                for j in 0..n {
                    for i in 0..n {
                        buf[i] = alpha[t - 1][i] + self.log_transition()[i][j];
                    }
                    alpha[t][j] =
                        logsumexp(&buf) + self.combined_emission(j, &encoded, t, &unknown_lp);
                }
            }

            logsumexp(&alpha[t_len - 1])
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            Ok(sequences
                .par_iter()
                .with_min_len(16)
                .map(score_one)
                .collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(sequences.iter().map(score_one).collect())
        }
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    fn save_to_path(&self, path: &str) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        self.save_to_writer(&mut buf)?;
        crate::persistence::save_zstd(path, &buf)
    }

    fn load_from_path(&mut self, path: &str) -> Result<(), ModelError> {
        let bytes = crate::persistence::load_zstd(path, "HMM model")?;
        self.load_from_reader(bytes.as_slice())
    }

    /// Save the model to a FlatBuffers binary stream.
    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<(), ModelError> {
        save_hmm_flatbuffers(
            writer,
            &HmmSaveData {
                log_initial: self.log_initial(),
                log_transition: self.log_transition(),
                feature_vocabs: self.feature_vocabs(),
                feature_log_emissions: self.feature_log_emissions(),
                templates: &self.features().templates,
                n_states: Some(self.n_states()),
                state_labels: self.state_labels().as_deref(),
            },
        )
    }

    /// Load the model from a FlatBuffers byte slice.
    fn load_from_reader<R: Read>(&mut self, reader: R) -> Result<(), ModelError> {
        let bytes = crate::persistence::read_all_bytes(reader)
            .map_err(|e| ModelError::Io(format!("Failed to read FlatBuffers data: {e}")))?;
        let n_features = self.features().templates.len();
        let model_data = load_hmm_flatbuffers(&bytes, n_features, Some(self.n_states()))?;

        *self.log_initial_mut() = model_data.log_initial;
        *self.log_transition_mut() = model_data.log_transition;
        *self.feature_vocabs_mut() = model_data.feature_vocabs;
        *self.feature_log_emissions_mut() = model_data.feature_log_emissions;
        *self.state_labels_mut() = model_data.state_labels;
        self.set_fitted(true);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure Rust struct
// ---------------------------------------------------------------------------

/// A Hidden Markov Model trained with the Baum-Welch algorithm.
///
/// For Python, use [`PyHiddenMarkovModel`].
#[derive(Clone, Debug)]
pub struct HiddenMarkovModel {
    n_states: usize,
    n_iter: usize,
    tolerance: f64,
    gamma: f64,
    random_seed: Option<u64>,
    fitted: bool,
    features: SeqFeatureConfig,
    log_initial: Vec<f64>,
    log_transition: Vec<Vec<f64>>,
    feature_vocabs: Vec<FxHashMap<String, usize>>,
    feature_log_emissions: Vec<Vec<Vec<f64>>>,
    state_labels: Option<Vec<String>>,
}

impl BaseHiddenMarkovModel for HiddenMarkovModel {
    fn n_states(&self) -> usize {
        self.n_states
    }
    fn set_n_states(&mut self, n: usize) {
        self.n_states = n;
    }
    fn n_iter(&self) -> usize {
        self.n_iter
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
    fn gamma(&self) -> f64 {
        self.gamma
    }
    fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }
    fn fitted(&self) -> bool {
        self.fitted
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.fitted = fitted;
    }
    fn log_initial(&self) -> &Vec<f64> {
        &self.log_initial
    }
    fn log_initial_mut(&mut self) -> &mut Vec<f64> {
        &mut self.log_initial
    }
    fn log_transition(&self) -> &Vec<Vec<f64>> {
        &self.log_transition
    }
    fn log_transition_mut(&mut self) -> &mut Vec<Vec<f64>> {
        &mut self.log_transition
    }
    fn features(&self) -> &SeqFeatureConfig {
        &self.features
    }
    fn feature_vocabs(&self) -> &Vec<FxHashMap<String, usize>> {
        &self.feature_vocabs
    }
    fn feature_vocabs_mut(&mut self) -> &mut Vec<FxHashMap<String, usize>> {
        &mut self.feature_vocabs
    }
    fn feature_log_emissions(&self) -> &Vec<Vec<Vec<f64>>> {
        &self.feature_log_emissions
    }
    fn feature_log_emissions_mut(&mut self) -> &mut Vec<Vec<Vec<f64>>> {
        &mut self.feature_log_emissions
    }
    fn state_labels(&self) -> &Option<Vec<String>> {
        &self.state_labels
    }
    fn state_labels_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.state_labels
    }
}

impl HiddenMarkovModel {
    /// Create a new Hidden Markov Model.
    pub fn new(
        n_states: usize,
        n_iter: usize,
        tolerance: f64,
        gamma: f64,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Result<Self, String> {
        if n_states < 1 {
            return Err(format!("n_states must be >= 1: {}", n_states));
        }
        if n_iter < 1 {
            return Err(format!("n_iter must be >= 1: {}", n_iter));
        }
        if tolerance < 0.0 {
            return Err(format!("tolerance must be >= 0: {}", tolerance));
        }
        if gamma <= 0.0 {
            return Err(format!("gamma must be > 0: {}", gamma));
        }
        let templates = features.unwrap_or_else(default_tagger_hmm_features);
        validate_templates(&templates, false)?;
        let n_features = templates.len();
        Ok(Self {
            n_states,
            n_iter,
            tolerance,
            gamma,
            random_seed,
            fitted: false,
            features: SeqFeatureConfig::new(templates),
            log_initial: Vec::new(),
            log_transition: Vec::new(),
            feature_vocabs: (0..n_features).map(|_| FxHashMap::default()).collect(),
            feature_log_emissions: (0..n_features).map(|_| Vec::new()).collect(),
            state_labels: None,
        })
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-exposed wrapper. Python users see this as `HiddenMarkovModel`.
#[pyclass(name = "HiddenMarkovModel", from_py_object)]
#[derive(Clone)]
pub struct PyHiddenMarkovModel {
    pub inner: HiddenMarkovModel,
}

impl BaseHiddenMarkovModel for PyHiddenMarkovModel {
    fn n_states(&self) -> usize {
        self.inner.n_states()
    }
    fn set_n_states(&mut self, n: usize) {
        self.inner.set_n_states(n);
    }
    fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
    fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }
    fn random_seed(&self) -> Option<u64> {
        self.inner.random_seed()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
    fn set_fitted(&mut self, fitted: bool) {
        self.inner.set_fitted(fitted);
    }
    fn log_initial(&self) -> &Vec<f64> {
        self.inner.log_initial()
    }
    fn log_initial_mut(&mut self) -> &mut Vec<f64> {
        self.inner.log_initial_mut()
    }
    fn log_transition(&self) -> &Vec<Vec<f64>> {
        self.inner.log_transition()
    }
    fn log_transition_mut(&mut self) -> &mut Vec<Vec<f64>> {
        self.inner.log_transition_mut()
    }
    fn features(&self) -> &SeqFeatureConfig {
        self.inner.features()
    }
    fn feature_vocabs(&self) -> &Vec<FxHashMap<String, usize>> {
        self.inner.feature_vocabs()
    }
    fn feature_vocabs_mut(&mut self) -> &mut Vec<FxHashMap<String, usize>> {
        self.inner.feature_vocabs_mut()
    }
    fn feature_log_emissions(&self) -> &Vec<Vec<Vec<f64>>> {
        self.inner.feature_log_emissions()
    }
    fn feature_log_emissions_mut(&mut self) -> &mut Vec<Vec<Vec<f64>>> {
        self.inner.feature_log_emissions_mut()
    }
    fn state_labels(&self) -> &Option<Vec<String>> {
        self.inner.state_labels()
    }
    fn state_labels_mut(&mut self) -> &mut Option<Vec<String>> {
        self.inner.state_labels_mut()
    }
}

#[pymethods]
impl PyHiddenMarkovModel {
    /// Initialize a Hidden Markov Model.
    #[new]
    #[pyo3(signature = (*, n_states, n_iter=100, tolerance=1e-6, gamma=1.0, random_seed=None, features=None))]
    fn new(
        n_states: usize,
        n_iter: usize,
        tolerance: f64,
        gamma: f64,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> PyResult<Self> {
        if let Some(ref f) = features {
            validate_templates(f, false).map_err(pyo3::exceptions::PyValueError::new_err)?;
        }
        let inner =
            HiddenMarkovModel::new(n_states, n_iter, tolerance, gamma, random_seed, features)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (sequences, labels=None))]
    fn fit(
        &mut self,
        sequences: Vec<Vec<String>>,
        labels: Option<Vec<Vec<String>>>,
    ) -> PyResult<()> {
        BaseHiddenMarkovModel::fit(self, sequences, labels)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict(&self, sequences: &Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<Vec<usize>>> {
        use pyo3::types::{PyList, PyString};

        if !self.inner.fitted() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Model has not been fitted yet.",
            ));
        }
        let n_states = self.inner.n_states;
        let n_features = self.inner.features.templates.len();
        let unknown_lp = BaseHiddenMarkovModel::unknown_log_probs(self);
        let flat_trans = flatten_transition(&self.inner.log_transition, n_states);
        let feat_emit = &self.inner.feature_log_emissions;
        let log_init = &self.inner.log_initial;
        let templates = &self.inner.features.templates;

        // Phase 1: encode all sequences into a single contiguous array.
        // Borrows Python strings directly — no String allocation.
        let n_seqs = sequences.len();
        let mut flat_encoded: Vec<Option<usize>> = Vec::new();
        let mut offsets: Vec<usize> = Vec::with_capacity(n_seqs + 1);
        let mut seq_lens: Vec<usize> = Vec::with_capacity(n_seqs);
        offsets.push(0);

        for seq_obj in sequences.iter() {
            let seq_list = seq_obj.cast::<PyList>()?;
            let t_len = seq_list.len();
            seq_lens.push(t_len);
            if t_len == 0 {
                offsets.push(flat_encoded.len());
                continue;
            }
            for (f, template) in templates.iter().enumerate() {
                let vocab = &self.inner.feature_vocabs[f];
                // Fast path: identity feature at position 0.
                if template.positions.len() == 1
                    && template.positions[0] == 0
                    && template.transform == SeqTransform::Identity
                {
                    for item in seq_list.iter() {
                        let s = item.cast::<PyString>()?.to_str()?;
                        flat_encoded.push(vocab.get(s).copied());
                    }
                } else {
                    // Slow path: extract to Strings for general templates.
                    let obs: Vec<String> = seq_list.extract()?;
                    let obs_refs: Vec<&str> = obs.iter().map(|s| s.as_str()).collect();
                    for t in 0..t_len {
                        let feat_val = extract_observation_cow(template, &obs_refs, t);
                        flat_encoded.push(vocab.get(feat_val.as_ref()).copied());
                    }
                }
            }
            offsets.push(flat_encoded.len());
        }

        // Phase 2: Viterbi decode on pre-encoded data.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            Ok((0..n_seqs)
                .into_par_iter()
                .with_min_len(128)
                .map_init(ViterbiBuffers::new, |vbufs, i| {
                    let t_len = seq_lens[i];
                    if t_len == 0 {
                        return Vec::new();
                    }
                    let enc = &flat_encoded[offsets[i]..offsets[i + 1]];
                    viterbi_decode(
                        n_states,
                        log_init,
                        &flat_trans,
                        enc,
                        t_len,
                        n_features,
                        feat_emit,
                        &unknown_lp,
                        vbufs,
                    )
                })
                .collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut vbufs = ViterbiBuffers::new();
            Ok((0..n_seqs)
                .map(|i| {
                    let t_len = seq_lens[i];
                    if t_len == 0 {
                        return Vec::new();
                    }
                    let enc = &flat_encoded[offsets[i]..offsets[i + 1]];
                    viterbi_decode(
                        n_states,
                        log_init,
                        &flat_trans,
                        enc,
                        t_len,
                        n_features,
                        feat_emit,
                        &unknown_lp,
                        &mut vbufs,
                    )
                })
                .collect())
        }
    }

    fn score(&self, sequences: Vec<Vec<String>>) -> PyResult<Vec<f64>> {
        BaseHiddenMarkovModel::score(self, sequences)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn n_states(&self) -> usize {
        BaseHiddenMarkovModel::n_states(self)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        Ok(self.save_to_path(&path)?)
    }

    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        Ok(self.load_from_path(&path)?)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the hmm submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let hmm_module = PyModule::new(parent_module.py(), "hmm")?;
    hmm_module.add_class::<PyHiddenMarkovModel>()?;
    parent_module.add_submodule(&hmm_module)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_sequences() -> Vec<Vec<String>> {
        vec![
            vec!["a".into(), "b".into(), "a".into(), "a".into(), "b".into()],
            vec!["b".into(), "a".into(), "b".into(), "b".into()],
            vec!["a".into(), "a".into(), "a".into(), "b".into()],
        ]
    }

    // --- logsumexp tests ---

    #[test]
    fn test_logsumexp_basic() {
        let result = logsumexp(&[0.0, 0.0]);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_neg_infinity() {
        let result = logsumexp(&[f64::NEG_INFINITY, 0.0]);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_empty() {
        assert!(logsumexp(&[]) == f64::NEG_INFINITY);
    }

    #[test]
    fn test_logsumexp_all_neg_infinity() {
        assert!(logsumexp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]) == f64::NEG_INFINITY);
    }

    // --- Constructor tests ---

    #[test]
    fn test_new_valid() {
        let hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None);
        assert!(hmm.is_ok());
        let hmm = hmm.unwrap();
        assert_eq!(hmm.n_states, 2);
        assert!(!hmm.fitted);
    }

    #[test]
    fn test_new_invalid_n_states() {
        let hmm = HiddenMarkovModel::new(0, 10, 1e-6, 1.0, None, None);
        assert!(hmm.is_err());
    }

    #[test]
    fn test_new_invalid_n_iter() {
        let hmm = HiddenMarkovModel::new(2, 0, 1e-6, 1.0, None, None);
        assert!(hmm.is_err());
    }

    #[test]
    fn test_new_invalid_tolerance() {
        let hmm = HiddenMarkovModel::new(2, 10, -1.0, 1.0, None, None);
        assert!(hmm.is_err());
    }

    #[test]
    fn test_new_invalid_gamma_zero() {
        let hmm = HiddenMarkovModel::new(2, 10, 1e-6, 0.0, None, None);
        assert!(hmm.is_err());
    }

    #[test]
    fn test_new_invalid_gamma_negative() {
        let hmm = HiddenMarkovModel::new(2, 10, 1e-6, -0.5, None, None);
        assert!(hmm.is_err());
    }

    #[test]
    fn test_gamma_affects_supervised_scores() {
        let sequences = vec![
            vec!["a".into(), "b".into(), "a".into(), "a".into(), "b".into()],
            vec!["b".into(), "a".into(), "b".into(), "b".into()],
        ];
        let labels = vec![
            vec!["X".into(), "Y".into(), "X".into(), "X".into(), "Y".into()],
            vec!["Y".into(), "X".into(), "Y".into(), "Y".into()],
        ];

        let mut hmm_laplace = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm_laplace
            .fit(sequences.clone(), Some(labels.clone()))
            .unwrap();

        let mut hmm_small = HiddenMarkovModel::new(1, 10, 1e-6, 0.01, Some(42), None).unwrap();
        hmm_small.fit(sequences, Some(labels)).unwrap();

        let obs = vec![vec!["a".into(), "b".into()]];
        let score_laplace = hmm_laplace.score(obs.clone()).unwrap()[0];
        let score_small = hmm_small.score(obs).unwrap()[0];
        assert!((score_laplace - score_small).abs() > 1e-6);
    }

    // --- Unsupervised fit tests ---

    #[test]
    fn test_predict_before_fit() {
        let hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, None, None).unwrap();
        assert!(hmm.predict(vec![vec!["a".into()]]).is_err());
    }

    #[test]
    fn test_score_before_fit() {
        let hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, None, None).unwrap();
        assert!(hmm.score(vec![vec!["a".into()]]).is_err());
    }

    #[test]
    fn test_fit_builds_vocab() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        assert_eq!(hmm.feature_vocabs[0].len(), 2);
        assert!(hmm.fitted);
    }

    #[test]
    fn test_fit_and_predict() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let paths = hmm
            .predict(vec![vec!["a".into(), "b".into(), "a".into()]])
            .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 3);
        assert!(paths[0].iter().all(|&s| s < 2));
    }

    #[test]
    fn test_predict_empty() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let paths = hmm.predict(vec![vec![]]).unwrap();
        assert!(paths[0].is_empty());
    }

    #[test]
    fn test_score_returns_finite() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let scores = hmm
            .score(vec![vec!["a".into(), "b".into(), "a".into()]])
            .unwrap();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite());
        assert!(scores[0] < 0.0);
    }

    #[test]
    fn test_score_empty() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let scores = hmm.score(vec![vec![]]).unwrap();
        assert!((scores[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mut hmm1 = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm1.fit(toy_sequences(), None).unwrap();
        let mut hmm2 = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm2.fit(toy_sequences(), None).unwrap();
        let obs: Vec<Vec<String>> = vec![vec!["a".into(), "b".into()]];
        assert_eq!(
            hmm1.predict(obs.clone()).unwrap(),
            hmm2.predict(obs.clone()).unwrap()
        );
        assert_eq!(hmm1.score(obs.clone()).unwrap(), hmm2.score(obs).unwrap());
    }

    #[test]
    fn test_predict_unknown_obs() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let result = hmm.predict(vec![vec!["a".into(), "c".into(), "b".into()]]);
        assert_eq!(result.unwrap()[0].len(), 3);
    }

    #[test]
    fn test_score_unknown_obs() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        assert!(
            hmm.score(vec![vec!["a".into(), "c".into(), "b".into()]])
                .unwrap()[0]
                .is_finite()
        );
    }

    #[test]
    fn test_fit_filters_empty_sequences() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        let mut seqs = toy_sequences();
        seqs.push(vec![]);
        hmm.fit(seqs, None).unwrap();
        assert!(hmm.fitted);
        assert_eq!(hmm.feature_vocabs[0].len(), 2);
    }

    #[test]
    fn test_convergence() {
        let mut hmm_1 = HiddenMarkovModel::new(2, 1, 0.0, 1.0, Some(42), None).unwrap();
        hmm_1.fit(toy_sequences(), None).unwrap();
        let score_1: f64 = toy_sequences()
            .iter()
            .map(|seq| hmm_1.score(vec![seq.clone()]).unwrap()[0])
            .sum();

        let mut hmm_many = HiddenMarkovModel::new(2, 50, 0.0, 1.0, Some(42), None).unwrap();
        hmm_many.fit(toy_sequences(), None).unwrap();
        let score_many: f64 = toy_sequences()
            .iter()
            .map(|seq| hmm_many.score(vec![seq.clone()]).unwrap()[0])
            .sum();

        assert!(score_many >= score_1 - 1e-6);
    }

    #[test]
    fn test_single_state() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        assert_eq!(
            hmm.predict(vec![vec!["a".into(), "b".into()]]).unwrap()[0],
            vec![0, 0]
        );
    }

    #[test]
    fn test_save_and_load() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hmm_model.bin");
        let path_str = path.to_str().unwrap();
        hmm.save_to_path(path_str).unwrap();

        let mut loaded = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, None, None).unwrap();
        loaded.load_from_path(path_str).unwrap();
        assert!(loaded.fitted);

        let obs: Vec<Vec<String>> = vec![vec!["a".into(), "b".into(), "a".into()]];
        assert_eq!(
            hmm.predict(obs.clone()).unwrap(),
            loaded.predict(obs).unwrap()
        );
    }

    #[test]
    fn test_load_nonexistent_file() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, None, None).unwrap();
        assert!(hmm.load_from_path("/nonexistent/path/model.bin").is_err());
    }

    #[test]
    fn test_load_n_states_mismatch() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hmm_model.bin");
        let path_str = path.to_str().unwrap();
        hmm.save_to_path(path_str).unwrap();

        let mut mismatched = HiddenMarkovModel::new(3, 10, 1e-6, 1.0, None, None).unwrap();
        assert!(mismatched.load_from_path(path_str).is_err());
    }

    // --- Supervised fit (fit_labeled) tests ---

    #[test]
    fn test_fit_labeled_basic() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        let sequences = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "a".into()],
        ];
        let labels = vec![
            vec!["X".into(), "Y".into(), "X".into()],
            vec!["Y".into(), "X".into()],
        ];
        hmm.fit(sequences, Some(labels)).unwrap();
        assert!(hmm.fitted());
        assert_eq!(hmm.n_states(), 2);
        assert_eq!(
            hmm.state_labels(),
            &Some(vec!["X".to_string(), "Y".to_string()])
        );
    }

    #[test]
    fn test_fit_labeled_mismatched_lengths() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, None, None).unwrap();
        let sequences = vec![vec!["a".into()]];
        let labels = vec![vec!["X".into()], vec!["Y".into()]];
        assert!(hmm.fit(sequences, Some(labels)).is_err());
    }

    #[test]
    fn test_fit_labeled_mismatched_inner_lengths() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, None, None).unwrap();
        let sequences = vec![vec!["a".into(), "b".into()]];
        let labels = vec![vec!["X".into()]];
        assert!(hmm.fit(sequences, Some(labels)).is_err());
    }

    #[test]
    fn test_fit_labeled_predict() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        let sequences = vec![
            vec!["a".into(), "b".into(), "a".into(), "a".into(), "b".into()],
            vec!["b".into(), "a".into(), "b".into(), "b".into()],
        ];
        let labels = vec![
            vec!["X".into(), "Y".into(), "X".into(), "X".into(), "Y".into()],
            vec!["Y".into(), "X".into(), "Y".into(), "Y".into()],
        ];
        hmm.fit(sequences, Some(labels)).unwrap();
        let paths = hmm
            .predict(vec![vec!["a".into(), "b".into(), "a".into()]])
            .unwrap();
        assert_eq!(paths[0].len(), 3);
        assert!(paths[0].iter().all(|&s| s < 2));
    }

    #[test]
    fn test_fit_labeled_score() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        let sequences = vec![vec!["a".into(), "b".into(), "a".into()]];
        let labels = vec![vec!["X".into(), "Y".into(), "X".into()]];
        hmm.fit(sequences, Some(labels)).unwrap();
        let scores = hmm.score(vec![vec!["a".into(), "b".into()]]).unwrap();
        assert!(scores[0].is_finite());
        assert!(scores[0] < 0.0);
    }

    #[test]
    fn test_fit_labeled_save_and_load() {
        let mut hmm = HiddenMarkovModel::new(1, 10, 1e-6, 1.0, Some(42), None).unwrap();
        let sequences = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "a".into()],
        ];
        let labels = vec![
            vec!["X".into(), "Y".into(), "X".into()],
            vec!["Y".into(), "X".into()],
        ];
        hmm.fit(sequences, Some(labels)).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hmm_labeled.bin");
        let path_str = path.to_str().unwrap();
        hmm.save_to_path(path_str).unwrap();

        let mut loaded = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, None, None).unwrap();
        loaded.load_from_path(path_str).unwrap();
        assert!(loaded.fitted());
        assert_eq!(
            loaded.state_labels(),
            &Some(vec!["X".to_string(), "Y".to_string()])
        );

        let obs = vec![vec!["a".into(), "b".into(), "a".into()]];
        assert_eq!(
            hmm.predict(obs.clone()).unwrap(),
            loaded.predict(obs).unwrap()
        );
    }

    // --- Batch predict/score tests ---

    #[test]
    fn test_predict_batch() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let paths = hmm
            .predict(vec![
                vec!["a".into(), "b".into()],
                vec!["b".into(), "a".into(), "b".into()],
            ])
            .unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].len(), 2);
        assert_eq!(paths[1].len(), 3);
    }

    #[test]
    fn test_score_batch() {
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let scores = hmm
            .score(vec![
                vec!["a".into(), "b".into()],
                vec!["b".into(), "a".into(), "b".into()],
            ])
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
    }

    // --- Semi-supervised tests ---

    fn supervised_fit_hmm() -> HiddenMarkovModel {
        let mut hmm = HiddenMarkovModel::new(1, 50, 1e-6, 1.0, Some(42), None).unwrap();
        let sequences = vec![
            vec!["a".into(), "b".into(), "a".into(), "a".into(), "b".into()],
            vec!["b".into(), "a".into(), "b".into(), "b".into()],
            vec!["a".into(), "a".into(), "b".into(), "a".into()],
        ];
        let labels = vec![
            vec!["X".into(), "Y".into(), "X".into(), "X".into(), "Y".into()],
            vec!["Y".into(), "X".into(), "Y".into(), "Y".into()],
            vec!["X".into(), "X".into(), "Y".into(), "X".into()],
        ];
        hmm.fit(sequences, Some(labels)).unwrap();
        hmm
    }

    #[test]
    fn test_semi_supervised_preserves_state_labels() {
        let mut hmm = supervised_fit_hmm();
        assert_eq!(
            hmm.state_labels(),
            &Some(vec!["X".to_string(), "Y".to_string()])
        );
        let unlabeled = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "b".into(), "a".into()],
        ];
        hmm.fit(unlabeled, None).unwrap();
        assert_eq!(
            hmm.state_labels(),
            &Some(vec!["X".to_string(), "Y".to_string()])
        );
    }

    #[test]
    fn test_semi_supervised_extends_vocab() {
        let mut hmm = supervised_fit_hmm();
        let vocab_before: usize = hmm.feature_vocabs[0].len();
        assert_eq!(vocab_before, 2); // "a" and "b"

        let unlabeled = vec![vec!["a".into(), "c".into(), "b".into()]];
        hmm.fit(unlabeled, None).unwrap();

        assert_eq!(hmm.feature_vocabs[0].len(), 3); // "a", "b", "c"
        assert!(hmm.feature_vocabs[0].contains_key("c"));

        // Predict works on sequences with the new vocab item.
        let paths = hmm.predict(vec![vec!["c".into(), "a".into()]]).unwrap();
        assert_eq!(paths[0].len(), 2);
        assert!(paths[0].iter().all(|&s| s < 2));
    }

    #[test]
    fn test_semi_supervised_predict_works() {
        let mut hmm = supervised_fit_hmm();
        let unlabeled = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "a".into(), "b".into()],
        ];
        hmm.fit(unlabeled, None).unwrap();

        let paths = hmm
            .predict(vec![vec!["a".into(), "b".into(), "a".into()]])
            .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 3);
        assert!(paths[0].iter().all(|&s| s < 2));
    }

    #[test]
    fn test_semi_supervised_improves_or_maintains_likelihood() {
        let mut hmm = supervised_fit_hmm();
        let test_data = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "a".into(), "b".into()],
        ];
        let score_before = hmm.score(test_data.clone()).unwrap();

        // Unsupervised refinement on the same data.
        hmm.fit(test_data.clone(), None).unwrap();
        let score_after = hmm.score(test_data).unwrap();

        // EM should not decrease total log-likelihood.
        let total_before: f64 = score_before.iter().sum();
        let total_after: f64 = score_after.iter().sum();
        assert!(
            total_after >= total_before - 1e-6,
            "EM decreased likelihood: {} -> {}",
            total_before,
            total_after
        );
    }

    #[test]
    fn test_cold_unsupervised_unchanged() {
        // A fresh (not fitted) model should behave exactly as before.
        let mut hmm = HiddenMarkovModel::new(2, 10, 1e-6, 1.0, Some(42), None).unwrap();
        hmm.fit(toy_sequences(), None).unwrap();
        let paths = hmm
            .predict(vec![vec!["a".into(), "b".into(), "a".into()]])
            .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 3);
        assert!(hmm.state_labels().is_none());
    }

    #[test]
    fn test_semi_supervised_no_new_vocab() {
        let mut hmm = supervised_fit_hmm();
        let vocab_before: usize = hmm.feature_vocabs[0].len();
        let emission_cols_before: usize = hmm.feature_log_emissions[0][0].len();

        // Unsupervised fit with same vocab (no new items).
        let unlabeled = vec![
            vec!["a".into(), "b".into(), "a".into()],
            vec!["b".into(), "b".into()],
        ];
        hmm.fit(unlabeled, None).unwrap();

        assert_eq!(hmm.feature_vocabs[0].len(), vocab_before);
        assert_eq!(hmm.feature_log_emissions[0][0].len(), emission_cols_before);
    }
}
