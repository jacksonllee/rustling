use pyo3::prelude::*;

use super::random_segmenter::{BaseRandomSegmenter, RandomSegmenter};

/// Python-exposed wrapper. Python users see this as `RandomSegmenter`.
#[pyclass(name = "RandomSegmenter", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyRandomSegmenter {
    pub inner: RandomSegmenter,
}

impl BaseRandomSegmenter for PyRandomSegmenter {
    fn prob(&self) -> f64 {
        self.inner.prob()
    }
    fn from_prob(prob: f64) -> Self {
        Self {
            inner: RandomSegmenter::from_prob(prob),
        }
    }
}

#[pymethods]
impl PyRandomSegmenter {
    /// Initialize a random segmenter.
    ///
    /// # Arguments
    ///
    /// * `prob` - The probability from [0, 1) that segmentation occurs between
    ///            two symbols.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If prob is outside [0, 1).
    #[new]
    #[pyo3(signature = (*, prob))]
    fn new(prob: f64) -> PyResult<Self> {
        RandomSegmenter::new(prob)
            .map(|inner| Self { inner })
            .map_err(PyErr::from)
    }

    /// Segment the given unsegmented sentences.
    ///
    /// # Arguments
    ///
    /// * `sent_strs` - An iterable of unsegmented sentences.
    /// * `offsets` - If true, return each word as (word, (start, end)).
    ///
    /// # Returns
    ///
    /// A list of segmented sentences.
    #[pyo3(signature = (sent_strs, *, offsets=false))]
    fn predict(
        &self,
        py: Python<'_>,
        sent_strs: Vec<String>,
        offsets: bool,
    ) -> PyResult<Py<PyAny>> {
        let words = BaseRandomSegmenter::predict(self, sent_strs);
        if offsets {
            let with_offsets = super::attach_offsets(words);
            Ok(with_offsets.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(words.into_pyobject(py)?.into_any().unbind())
        }
    }
}
