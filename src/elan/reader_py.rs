use pyo3::prelude::*;
use pyo3::types::{PySlice, PyType};

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use super::reader::{Annotation, BaseElan, Elan, ElanError, ElanFile, Tier, WriteError};
use crate::persistence::pathbuf_to_string;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn elan_error_to_pyerr(e: ElanError) -> PyErr {
    match e {
        ElanError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        ElanError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(e),
        ElanError::Zip(e) => pyo3::exceptions::PyIOError::new_err(e),
        ElanError::Xml(e) => pyo3::exceptions::PyValueError::new_err(e),
        ElanError::Source(e) => e.into(),
    }
}

// ---------------------------------------------------------------------------
// PyAnnotation
// ---------------------------------------------------------------------------

/// A single annotation within an ELAN tier.
#[pyclass(name = "Annotation", from_py_object)]
#[derive(Clone)]
pub struct PyAnnotation(pub(crate) Annotation);

#[pymethods]
impl PyAnnotation {
    /// Annotation ID (e.g. "a1").
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }

    /// Start time in milliseconds, or None if unresolvable.
    #[getter]
    fn start_time(&self) -> Option<i64> {
        self.0.start_time
    }

    /// End time in milliseconds, or None if unresolvable.
    #[getter]
    fn end_time(&self) -> Option<i64> {
        self.0.end_time
    }

    /// The annotation text content.
    #[getter]
    fn value(&self) -> &str {
        &self.0.value
    }

    /// Parent annotation ID (from ANNOTATION_REF in REF_ANNOTATION),
    /// or None for alignable annotations.
    #[getter]
    fn parent_id(&self) -> Option<&str> {
        self.0.parent_id.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "Annotation(id={:?}, start_time={:?}, end_time={:?}, value={:?}, parent_id={:?})",
            self.0.id, self.0.start_time, self.0.end_time, self.0.value, self.0.parent_id
        )
    }

    fn __eq__(&self, other: &PyAnnotation) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.id.hash(&mut hasher);
        self.0.start_time.hash(&mut hasher);
        self.0.end_time.hash(&mut hasher);
        self.0.value.hash(&mut hasher);
        self.0.parent_id.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyTier
// ---------------------------------------------------------------------------

/// An annotation tier (layer) within an ELAN file.
#[pyclass(name = "Tier", from_py_object)]
#[derive(Clone)]
pub struct PyTier(pub(crate) Tier);

#[pymethods]
impl PyTier {
    /// Tier ID (e.g. "G-jyutping").
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }

    /// Participant name.
    #[getter]
    fn participant(&self) -> &str {
        &self.0.participant
    }

    /// Annotator name.
    #[getter]
    fn annotator(&self) -> &str {
        &self.0.annotator
    }

    /// Linguistic type reference.
    #[getter]
    fn linguistic_type_ref(&self) -> &str {
        &self.0.linguistic_type_ref
    }

    /// Parent tier ID, or None for root tiers.
    #[getter]
    fn parent_id(&self) -> Option<&str> {
        self.0.parent_id.as_deref()
    }

    /// Child tier IDs, or None if no children.
    #[getter]
    fn child_ids(&self) -> Option<Vec<String>> {
        self.0.child_ids.clone()
    }

    /// Annotations in this tier.
    #[getter]
    fn annotations(&self) -> Vec<PyAnnotation> {
        self.0
            .annotations
            .iter()
            .map(|a| PyAnnotation(a.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tier(id={:?}, participant={:?}, annotations={})",
            self.0.id,
            self.0.participant,
            self.0.annotations.len()
        )
    }

    fn __eq__(&self, other: &PyTier) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.id.hash(&mut hasher);
        self.0.participant.hash(&mut hasher);
        self.0.annotator.hash(&mut hasher);
        self.0.linguistic_type_ref.hash(&mut hasher);
        self.0.parent_id.hash(&mut hasher);
        self.0.child_ids.hash(&mut hasher);
        self.0.annotations.len().hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyElan
// ---------------------------------------------------------------------------

/// ELAN (.eaf) data reader.
#[pyclass(name = "ELAN", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyElan {
    pub inner: Elan,
}

impl BaseElan for PyElan {
    fn files(&self) -> &VecDeque<ElanFile> {
        self.inner.files()
    }
    fn files_mut(&mut self) -> &mut VecDeque<ElanFile> {
        self.inner.files_mut()
    }
    fn from_files(files: VecDeque<ElanFile>) -> Self {
        Self {
            inner: Elan::from_files(files),
        }
    }
}

#[pymethods]
impl PyElan {
    #[new]
    fn new() -> Self {
        Self::from_files(VecDeque::new())
    }

    /// Parse ELAN data from in-memory strings.
    #[classmethod]
    #[pyo3(signature = (strs, ids=None, parallel=true))]
    fn from_strs(
        _cls: &Bound<'_, PyType>,
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
    ) -> PyResult<Self> {
        if let Some(ref ids) = ids
            && strs.len() != ids.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "strs and ids must have the same length: {} vs {}",
                strs.len(),
                ids.len()
            )));
        }
        let elan = Elan::from_strs(strs, ids, parallel).map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Load ELAN data from file paths.
    #[classmethod]
    #[pyo3(name = "from_files")]
    #[pyo3(signature = (paths, *, parallel=true))]
    fn read_files(_cls: &Bound<'_, PyType>, paths: Vec<PathBuf>, parallel: bool) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let elan = Elan::read_files(&paths, parallel).map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Recursively load ELAN data from a directory.
    #[classmethod]
    #[pyo3(name = "from_dir")]
    #[pyo3(signature = (path, *, r#match=None, extension=".eaf", parallel=true))]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let elan =
            Elan::read_dir(&path, r#match, extension, parallel).map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Load ELAN data from a ZIP archive.
    #[classmethod]
    #[pyo3(name = "from_zip")]
    #[pyo3(signature = (path, *, r#match=None, extension=".eaf", parallel=true))]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let elan =
            Elan::read_zip(&path, r#match, extension, parallel).map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Load ELAN data from a git repository.
    #[classmethod]
    #[pyo3(name = "from_git")]
    #[pyo3(signature = (url, *, rev=None, depth=None, r#match=None, extension=".eaf", cache_dir=None, force_download=false, parallel=true))]
    #[allow(clippy::too_many_arguments)]
    fn from_git(
        _cls: &Bound<'_, PyType>,
        url: &str,
        rev: Option<&str>,
        depth: Option<u32>,
        r#match: Option<&str>,
        extension: &str,
        cache_dir: Option<PathBuf>,
        force_download: bool,
        parallel: bool,
    ) -> PyResult<Self> {
        let elan = Elan::from_git(
            url,
            rev,
            depth,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
        )
        .map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Load ELAN data from a URL.
    #[classmethod]
    #[pyo3(name = "from_url")]
    #[pyo3(signature = (url, *, r#match=None, extension=".eaf", cache_dir=None, force_download=false, parallel=true))]
    #[allow(clippy::too_many_arguments)]
    fn from_url(
        _cls: &Bound<'_, PyType>,
        url: &str,
        r#match: Option<&str>,
        extension: &str,
        cache_dir: Option<PathBuf>,
        force_download: bool,
        parallel: bool,
    ) -> PyResult<Self> {
        let elan = Elan::from_url(url, r#match, extension, cache_dir, force_download, parallel)
            .map_err(elan_error_to_pyerr)?;
        Ok(Self { inner: elan })
    }

    /// Return the list of file paths.
    #[getter]
    #[pyo3(name = "file_paths")]
    fn py_file_paths(&self) -> Vec<String> {
        self.file_paths()
    }

    /// Return the number of files.
    #[getter]
    fn n_files(&self) -> usize {
        self.num_files()
    }

    /// Return tiers as a list of OrderedDicts (one per file), each keyed by tier ID.
    fn tiers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let collections = py.import("collections")?;
        let ordered_dict_type = collections.getattr("OrderedDict")?;
        let list = pyo3::types::PyList::empty(py);
        for f in self.inner.files() {
            let dict = pyo3::types::PyDict::new(py);
            for t in &f.tiers {
                dict.set_item(&t.id, PyTier(t.clone()))?;
            }
            let od = ordered_dict_type.call1((dict,))?;
            list.append(od)?;
        }
        Ok(list.unbind().into())
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return EAF XML strings, one per file.
    #[pyo3(name = "to_strs")]
    fn py_to_strings(&self) -> Vec<String> {
        self.to_strings()
    }

    /// Return CHAT format strings, one per file.
    #[pyo3(name = "to_chat_strs")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_chat_strings(&self, participants: Option<Vec<String>>) -> Vec<String> {
        self.to_chat_strings(participants.as_deref())
    }

    /// Convert to a CHAT object.
    #[pyo3(name = "to_chat")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_chat(&self, participants: Option<Vec<String>>) -> crate::chat::PyChat {
        crate::chat::PyChat {
            inner: self.to_chat_obj(participants.as_deref()),
        }
    }

    /// Write CHAT (.cha) files to a directory.
    #[pyo3(name = "to_chat_files")]
    #[pyo3(signature = (dir_path, /, *, participants=None, filenames=None))]
    fn write_chat(
        &self,
        dir_path: PathBuf,
        participants: Option<Vec<String>>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_chat_files(&dir_path, participants.as_deref(), filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Write ELAN (.eaf) files to a directory.
    #[pyo3(name = "to_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_files(&dir_path, filenames).map_err(|e| match e {
            WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
        })
    }

    /// Return SRT format strings, one per file.
    #[pyo3(name = "to_srt_strs")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_srt_strings(&self, participants: Option<Vec<String>>) -> Vec<String> {
        self.to_srt_strings(participants.as_deref())
    }

    /// Convert to an SRT object.
    #[pyo3(name = "to_srt")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_srt(&self, participants: Option<Vec<String>>) -> crate::srt::PySrt {
        crate::srt::PySrt {
            inner: self.to_srt(participants.as_deref()),
        }
    }

    /// Write SRT (.srt) files to a directory.
    #[pyo3(name = "to_srt_files")]
    #[pyo3(signature = (dir_path, /, *, participants=None, filenames=None))]
    fn write_srt(
        &self,
        dir_path: PathBuf,
        participants: Option<Vec<String>>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_srt_files(&dir_path, participants.as_deref(), filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Return TextGrid format strings, one per file.
    #[pyo3(name = "to_textgrid_strs")]
    fn py_to_textgrid_strings(&self) -> Vec<String> {
        self.to_textgrid_strings()
    }

    /// Convert to a TextGrid object.
    #[pyo3(name = "to_textgrid")]
    fn py_to_textgrid(&self) -> crate::textgrid::PyTextGrid {
        crate::textgrid::PyTextGrid {
            inner: self.to_textgrid(),
        }
    }

    /// Write TextGrid (.TextGrid) files to a directory.
    #[pyo3(name = "to_textgrid_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write_textgrid(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_textgrid_files(&dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    // -----------------------------------------------------------------------
    // Collection operations
    // -----------------------------------------------------------------------

    /// Append data from another ELAN reader.
    #[pyo3(name = "append", signature = (other, /))]
    fn py_push_back(&mut self, other: &PyElan) {
        self.inner.push_back(&other.inner);
    }

    /// Left-append data from another ELAN reader, preserving order.
    #[pyo3(name = "append_left", signature = (other, /))]
    fn py_push_front(&mut self, other: &PyElan) {
        self.inner.push_front(&other.inner);
    }

    /// Extend data from multiple ELAN readers.
    #[pyo3(name = "extend", signature = (others, /))]
    fn extend_back(&mut self, others: Vec<PyRef<'_, PyElan>>) {
        for other in &others {
            self.files_mut().extend(other.files().iter().cloned());
        }
    }

    /// Remove and return the last file as a new ELAN reader.
    #[pyo3(name = "pop")]
    fn pop_back(&mut self) -> PyResult<PyElan> {
        match self.files_mut().pop_back() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty ELAN reader",
            )),
        }
    }

    /// Remove and return the first file as a new ELAN reader.
    #[pyo3(name = "pop_left")]
    fn pop_front(&mut self) -> PyResult<PyElan> {
        match self.files_mut().pop_front() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty ELAN reader",
            )),
        }
    }

    /// Remove all data from this reader.
    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.files_mut().clear();
    }

    fn __add__(&self, other: &PyElan) -> PyElan {
        let mut result = self.clone();
        result.files_mut().extend(other.files().iter().cloned());
        result
    }

    fn __iadd__(&mut self, other: &PyElan) {
        self.files_mut().extend(other.files().iter().cloned());
    }

    fn __iter__(slf: PyRef<'_, Self>) -> ElanIter {
        ElanIter {
            inner: slf.files().clone(),
            index: 0,
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyElan> {
        if let Ok(i) = index.extract::<isize>() {
            let len = self.files().len() as isize;
            let idx = if i < 0 { len + i } else { i };
            if idx < 0 || idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "index out of range",
                ));
            }
            return Ok(Self::from_files(VecDeque::from(vec![
                self.files()[idx as usize].clone(),
            ])));
        }
        if let Ok(slice) = index.cast::<PySlice>() {
            let indices = slice.indices(self.files().len() as isize)?;
            let mut result = VecDeque::with_capacity(indices.slicelength);
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                result.push_back(self.files()[i as usize].clone());
                i += indices.step;
            }
            return Ok(Self::from_files(result));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers or slices",
        ))
    }

    fn __bool__(&self) -> bool {
        !self.is_empty()
    }

    fn __len__(&self) -> PyResult<usize> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "__len__ of an ELAN object is intentionally undefined. \
             Intuitively, there are different lengths one may refer to: \
             Number of files? Tiers? Annotations? Something else?",
        ))
    }

    fn __repr__(&self) -> String {
        format!("<ELAN with {} file(s)>", self.num_files())
    }

    fn __eq__(&self, other: &PyElan) -> bool {
        self.files().len() == other.files().len()
            && self
                .files()
                .iter()
                .zip(other.files())
                .all(|(a, b)| a.file_path == b.file_path && a.tiers == b.tiers)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.files().len().hash(&mut hasher);
        for f in self.files() {
            f.file_path.hash(&mut hasher);
            f.tiers.len().hash(&mut hasher);
            for t in &f.tiers {
                t.id.hash(&mut hasher);
                t.annotations.len().hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

#[pyclass]
struct ElanIter {
    inner: VecDeque<ElanFile>,
    index: usize,
}

#[pymethods]
impl ElanIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyElan> {
        if self.index < self.inner.len() {
            let file = self.inner[self.index].clone();
            self.index += 1;
            Some(PyElan {
                inner: Elan::from_files(VecDeque::from(vec![file])),
            })
        } else {
            None
        }
    }
}
