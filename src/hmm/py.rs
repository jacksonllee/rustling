use pyo3::prelude::*;
use std::path::PathBuf;

use super::{
    BaseHiddenMarkovModel, HiddenMarkovModel, ViterbiBuffers, flatten_transition, viterbi_decode,
};
use crate::persistence::pathbuf_to_string;
use crate::seq_feature::{
    SeqFeatureConfig, SeqFeatureTemplate, SeqTransform, extract_observation_cow, validate_templates,
};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-exposed wrapper. Python users see this as `HiddenMarkovModel`.
#[pyclass(name = "HiddenMarkovModel", subclass, from_py_object)]
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
            validate_templates(f, false).map_err(PyErr::from)?;
        }
        let inner =
            HiddenMarkovModel::new(n_states, n_iter, tolerance, gamma, random_seed, features)
                .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (sequences, labels=None))]
    fn fit(
        &mut self,
        sequences: Vec<Vec<String>>,
        labels: Option<Vec<Vec<String>>>,
    ) -> PyResult<()> {
        BaseHiddenMarkovModel::fit(self, sequences, labels).map_err(PyErr::from)
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
        BaseHiddenMarkovModel::score(self, sequences).map_err(PyErr::from)
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
