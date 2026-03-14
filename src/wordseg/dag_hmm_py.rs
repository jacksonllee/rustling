use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;

use super::dag_hmm::DagHmmSegmenter;
use crate::seq_feature::SeqFeatureTemplate;

/// Python-exposed DAG + HMM hybrid segmenter.
///
/// Python users see this as `DAGHMMSegmenter`.
#[pyclass(name = "DAGHMMSegmenter", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyDagHmmSegmenter {
    pub inner: DagHmmSegmenter,
}

#[pymethods]
impl PyDagHmmSegmenter {
    /// Create a new DAGHMMSegmenter.
    #[new]
    #[pyo3(signature = (*, n_iter=None, tolerance=None, gamma=None, random_seed=None, features=None))]
    fn new(
        n_iter: Option<usize>,
        tolerance: Option<f64>,
        gamma: Option<f64>,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Self {
        Self {
            inner: DagHmmSegmenter::new(n_iter, tolerance, gamma, random_seed, features),
        }
    }

    fn fit_segmented(&mut self, sents: Vec<Vec<String>>) {
        self.inner.fit_segmented(sents);
    }

    fn fit_unsegmented(&mut self, sent_strs: Vec<String>) {
        self.inner.fit_unsegmented(sent_strs);
    }

    fn score(&self, sents: Vec<Vec<String>>) -> PyResult<Vec<f64>> {
        self.inner.score(sents).map_err(PyErr::from)
    }

    #[pyo3(signature = (sent_strs, *, offsets=false))]
    fn predict(
        &self,
        py: Python<'_>,
        sent_strs: Vec<String>,
        offsets: bool,
    ) -> PyResult<Py<PyAny>> {
        let words = self.inner.predict(sent_strs);
        if offsets {
            let with_offsets = super::attach_offsets(words);
            Ok(with_offsets.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(words.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn save(&self, path: PathBuf, metadata: HashMap<String, String>) -> PyResult<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Path is not valid UTF-8"))?;
        Ok(self.inner.save(path_str, &metadata)?)
    }

    fn load(&mut self, path: PathBuf) -> PyResult<HashMap<String, String>> {
        let path_str = path
            .to_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Path is not valid UTF-8"))?;
        Ok(self.inner.load(path_str)?)
    }
}
