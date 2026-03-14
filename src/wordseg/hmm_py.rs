use pyo3::prelude::*;
use std::path::PathBuf;

use super::hmm::{BaseHiddenMarkovModelSegmenter, HiddenMarkovModelSegmenter};
use crate::hmm::HiddenMarkovModel;
use crate::persistence::pathbuf_to_string;
use crate::seq_feature::{SeqFeatureTemplate, validate_templates};

/// Python-exposed wrapper. Python users see this as `HiddenMarkovModelSegmenter`.
#[pyclass(name = "HiddenMarkovModelSegmenter", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyHiddenMarkovModelSegmenter {
    pub inner: HiddenMarkovModelSegmenter,
}

impl BaseHiddenMarkovModelSegmenter for PyHiddenMarkovModelSegmenter {
    fn hmm(&self) -> &HiddenMarkovModel {
        self.inner.hmm()
    }
    fn hmm_mut(&mut self) -> &mut HiddenMarkovModel {
        self.inner.hmm_mut()
    }
    fn from_hmm(hmm: HiddenMarkovModel) -> Self {
        Self {
            inner: HiddenMarkovModelSegmenter::from_hmm(hmm),
        }
    }
}

#[pymethods]
impl PyHiddenMarkovModelSegmenter {
    /// Initialize an HMM-based word segmenter.
    #[new]
    #[pyo3(signature = (*, n_iter=None, tolerance=None, gamma=None, random_seed=None, features=None))]
    fn new(
        n_iter: Option<usize>,
        tolerance: Option<f64>,
        gamma: Option<f64>,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> PyResult<Self> {
        if let Some(g) = gamma
            && g <= 0.0
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gamma must be > 0: {}",
                g
            )));
        }
        if let Some(ref f) = features {
            validate_templates(f, false).map_err(PyErr::from)?;
        }
        Ok(Self {
            inner: HiddenMarkovModelSegmenter::new(n_iter, tolerance, gamma, random_seed, features),
        })
    }

    /// Train the model with supervised segmented sentences.
    fn fit_segmented(&mut self, sents: Vec<Vec<String>>) {
        BaseHiddenMarkovModelSegmenter::fit_segmented(self, sents);
    }

    /// Train the model with unsupervised unsegmented sentences (Baum-Welch EM).
    fn fit_unsegmented(&mut self, sent_strs: Vec<String>) {
        BaseHiddenMarkovModelSegmenter::fit_unsegmented(self, sent_strs);
    }

    /// Compute log-likelihood of segmented sentences.
    fn score(&self, sents: Vec<Vec<String>>) -> PyResult<Vec<f64>> {
        BaseHiddenMarkovModelSegmenter::score(self, sents).map_err(PyErr::from)
    }

    /// Segment the given unsegmented sentences.
    #[pyo3(signature = (sent_strs, *, offsets=false))]
    fn predict(
        &self,
        py: Python<'_>,
        sent_strs: Vec<String>,
        offsets: bool,
    ) -> PyResult<Py<PyAny>> {
        let words = BaseHiddenMarkovModelSegmenter::predict(self, sent_strs);
        if offsets {
            let with_offsets = super::attach_offsets(words);
            Ok(with_offsets.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(words.into_pyobject(py)?.into_any().unbind())
        }
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
