use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::path::PathBuf;

use rustc_hash::{FxHashMap, FxHashSet};

use super::{AveragedPerceptron, BaseTagger, PerceptronModel};
use crate::persistence::pathbuf_to_string;
use crate::seq_feature::{SeqFeatureConfig, SeqFeatureTemplate, validate_templates};

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-exposed wrapper. Python users see this as `AveragedPerceptron`.
#[pyclass(name = "AveragedPerceptron", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyAveragedPerceptron {
    pub inner: AveragedPerceptron,
}

impl BaseTagger for PyAveragedPerceptron {
    fn frequency_threshold(&self) -> u32 {
        self.inner.frequency_threshold()
    }
    fn ambiguity_threshold(&self) -> f64 {
        self.inner.ambiguity_threshold()
    }
    fn n_iter(&self) -> u32 {
        self.inner.n_iter()
    }
    fn random_seed(&self) -> Option<u64> {
        self.inner.random_seed()
    }
    fn model(&self) -> &PerceptronModel {
        self.inner.model()
    }
    fn model_mut(&mut self) -> &mut PerceptronModel {
        self.inner.model_mut()
    }
    fn tagdict_ref(&self) -> &FxHashMap<String, String> {
        self.inner.tagdict_ref()
    }
    fn tagdict_mut(&mut self) -> &mut FxHashMap<String, String> {
        self.inner.tagdict_mut()
    }
    fn classes_ref(&self) -> &FxHashSet<String> {
        self.inner.classes_ref()
    }
    fn classes_mut(&mut self) -> &mut FxHashSet<String> {
        self.inner.classes_mut()
    }
    fn features(&self) -> &SeqFeatureConfig {
        self.inner.features()
    }
}

#[pymethods]
impl PyAveragedPerceptron {
    /// Initialize a part-of-speech tagger.
    #[new]
    #[pyo3(signature = (*, frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5, random_seed=None, features=None))]
    fn new(
        frequency_threshold: u32,
        ambiguity_threshold: f64,
        n_iter: u32,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> PyResult<Self> {
        if let Some(ref f) = features {
            validate_templates(f, true).map_err(PyErr::from)?;
        }
        Ok(Self {
            inner: AveragedPerceptron::new(
                frequency_threshold,
                ambiguity_threshold,
                n_iter,
                random_seed,
                features,
            ),
        })
    }

    /// Predict tags for the sequences.
    fn predict(&self, sequences: Vec<Vec<String>>) -> Vec<Vec<String>> {
        BaseTagger::predict(self, sequences)
    }

    /// Fit a model.
    fn fit(&mut self, sequences: Vec<Vec<String>>, tags: Vec<Vec<String>>) {
        BaseTagger::fit(self, sequences, tags);
    }

    /// Save the model to a gzip-compressed JSON file.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.save_to_path(&path).map_err(PyErr::from)
    }

    /// Load a model from a gzip-compressed JSON file.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.load_from_path(&path).map_err(PyErr::from)
    }

    /// Get the model's weights dictionary.
    #[getter]
    fn weights(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (feat, weights) in &self.model().weights {
            let inner_dict = PyDict::new(py);
            for (class, weight) in weights {
                inner_dict.set_item(class, weight)?;
            }
            dict.set_item(feat, inner_dict)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Get the tag dictionary.
    #[getter]
    fn tagdict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (word, tag) in self.tagdict_ref() {
            dict.set_item(word, tag)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Get the set of POS tag classes.
    #[getter]
    fn classes(&self) -> HashSet<String> {
        self.classes_ref().iter().cloned().collect()
    }
}

/// Register the perceptron_pos_tagger submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(parent_module.py(), "perceptron_pos_tagger")?;
    submodule.add_class::<PyAveragedPerceptron>()?;
    parent_module.add_submodule(&submodule)?;
    Ok(())
}
