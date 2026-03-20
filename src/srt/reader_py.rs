use pyo3::prelude::*;
use pyo3::types::{PySlice, PyType};

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use super::reader::{BaseSrt, Srt, SrtBlock, SrtError, SrtFile, WriteError};
use crate::persistence::pathbuf_to_string;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn srt_error_to_pyerr(e: SrtError) -> PyErr {
    match e {
        SrtError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        SrtError::Parse(e) => pyo3::exceptions::PyValueError::new_err(e),
        SrtError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(e),
        SrtError::Zip(e) => pyo3::exceptions::PyIOError::new_err(e),
        SrtError::Source(e) => e.into(),
    }
}

// ---------------------------------------------------------------------------
// PySrtBlock
// ---------------------------------------------------------------------------

/// A single subtitle block within an SRT file.
#[pyclass(name = "Utterance", from_py_object)]
#[derive(Clone)]
pub struct PySrtBlock(pub(crate) SrtBlock);

#[pymethods]
impl PySrtBlock {
    #[new]
    #[pyo3(signature = (*, index, line, time_marks))]
    fn new(index: usize, line: String, time_marks: (i64, i64)) -> Self {
        Self(SrtBlock {
            index,
            text: line,
            start_ms: time_marks.0,
            end_ms: time_marks.1,
        })
    }

    /// 1-based sequence number from the SRT file.
    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }

    /// The subtitle text.
    #[getter]
    fn line(&self) -> &str {
        &self.0.text
    }

    /// Start and end time in milliseconds as a tuple.
    #[getter]
    fn time_marks(&self) -> (i64, i64) {
        (self.0.start_ms, self.0.end_ms)
    }

    fn __repr__(&self) -> String {
        format!(
            "Utterance(index={}, line={:?}, time_marks=({}, {}))",
            self.0.index, self.0.text, self.0.start_ms, self.0.end_ms,
        )
    }

    fn __eq__(&self, other: &PySrtBlock) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.index.hash(&mut hasher);
        self.0.text.hash(&mut hasher);
        self.0.start_ms.hash(&mut hasher);
        self.0.end_ms.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PySrt
// ---------------------------------------------------------------------------

/// SRT (SubRip Subtitle) data reader.
#[pyclass(name = "SRT", subclass, from_py_object)]
#[derive(Clone)]
pub struct PySrt {
    pub inner: Srt,
}

impl BaseSrt for PySrt {
    fn files(&self) -> &VecDeque<SrtFile> {
        self.inner.files()
    }
    fn files_mut(&mut self) -> &mut VecDeque<SrtFile> {
        self.inner.files_mut()
    }
    fn from_files(files: VecDeque<SrtFile>) -> Self {
        Self {
            inner: Srt::from_files(files),
        }
    }
}

#[pymethods]
impl PySrt {
    #[new]
    fn new() -> Self {
        Self::from_files(VecDeque::new())
    }

    /// Parse SRT data from in-memory strings.
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
        let srt = Srt::from_strs(strs, ids, parallel).map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
    }

    /// Load SRT data from file paths.
    #[classmethod]
    #[pyo3(name = "from_files")]
    #[pyo3(signature = (paths, *, parallel=true))]
    fn read_files(_cls: &Bound<'_, PyType>, paths: Vec<PathBuf>, parallel: bool) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let srt = Srt::read_files(&paths, parallel).map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
    }

    /// Recursively load SRT data from a directory.
    #[classmethod]
    #[pyo3(name = "from_dir")]
    #[pyo3(signature = (path, *, r#match=None, extension=".srt", parallel=true))]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let srt = Srt::read_dir(&path, r#match, extension, parallel).map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
    }

    /// Load SRT data from a ZIP archive.
    #[classmethod]
    #[pyo3(name = "from_zip")]
    #[pyo3(signature = (path, *, r#match=None, extension=".srt", parallel=true))]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let srt = Srt::read_zip(&path, r#match, extension, parallel).map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
    }

    /// Load SRT data from a git repository.
    #[classmethod]
    #[pyo3(name = "from_git")]
    #[pyo3(signature = (url, *, rev=None, depth=None, r#match=None, extension=".srt", cache_dir=None, force_download=false, parallel=true))]
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
        let srt = Srt::from_git(
            url,
            rev,
            depth,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
        )
        .map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
    }

    /// Load SRT data from a URL.
    #[classmethod]
    #[pyo3(name = "from_url")]
    #[pyo3(signature = (url, *, r#match=None, extension=".srt", cache_dir=None, force_download=false, parallel=true))]
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
        let srt = Srt::from_url(url, r#match, extension, cache_dir, force_download, parallel)
            .map_err(srt_error_to_pyerr)?;
        Ok(Self { inner: srt })
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

    /// Return all subtitle blocks across all files as a flat list.
    fn utterances(&self) -> Vec<PySrtBlock> {
        self.files()
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| PySrtBlock(b.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return SRT strings, one per file.
    #[pyo3(name = "to_strs")]
    fn py_to_strings(&self) -> Vec<String> {
        self.to_strings()
    }

    /// Return CHAT format strings, one per file.
    #[pyo3(name = "to_chat_strs")]
    fn py_to_chat_strings(&self) -> Vec<String> {
        self.to_chat_strings()
    }

    /// Convert to a CHAT object.
    #[pyo3(name = "to_chat")]
    fn py_to_chat(&self) -> crate::chat::PyChat {
        crate::chat::PyChat {
            inner: self.to_chat_obj(),
        }
    }

    /// Write CHAT (.cha) files to a directory.
    #[pyo3(name = "to_chat_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write_chat(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_chat_files(&dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Return EAF XML strings, one per file.
    #[pyo3(name = "to_elan_strs")]
    fn py_to_elan_strings(&self) -> Vec<String> {
        self.to_elan_strings()
    }

    /// Convert to an ELAN object.
    #[pyo3(name = "to_elan")]
    fn py_to_elan(&self) -> crate::elan::PyElan {
        crate::elan::PyElan {
            inner: self.to_elan(),
        }
    }

    /// Write ELAN (.eaf) files to a directory.
    #[pyo3(name = "to_elan_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write_elan(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_elan_files(&dir_path, filenames)
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

    /// Write SRT files to a directory.
    #[pyo3(name = "to_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_srt_files(&dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    // -----------------------------------------------------------------------
    // Collection operations
    // -----------------------------------------------------------------------

    /// Append data from another SRT reader.
    #[pyo3(name = "append", signature = (other, /))]
    fn py_push_back(&mut self, other: &PySrt) {
        self.inner.push_back(&other.inner);
    }

    /// Left-append data from another SRT reader, preserving order.
    #[pyo3(name = "append_left", signature = (other, /))]
    fn py_push_front(&mut self, other: &PySrt) {
        self.inner.push_front(&other.inner);
    }

    /// Extend data from multiple SRT readers.
    #[pyo3(name = "extend", signature = (others, /))]
    fn extend_back(&mut self, others: Vec<PyRef<'_, PySrt>>) {
        for other in &others {
            self.files_mut().extend(other.files().iter().cloned());
        }
    }

    /// Remove and return the last file as a new SRT reader.
    #[pyo3(name = "pop")]
    fn pop_back(&mut self) -> PyResult<PySrt> {
        match self.files_mut().pop_back() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty SRT reader",
            )),
        }
    }

    /// Remove and return the first file as a new SRT reader.
    #[pyo3(name = "pop_left")]
    fn pop_front(&mut self) -> PyResult<PySrt> {
        match self.files_mut().pop_front() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty SRT reader",
            )),
        }
    }

    /// Remove all data from this reader.
    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.files_mut().clear();
    }

    fn __add__(&self, other: &PySrt) -> PySrt {
        let mut result = self.clone();
        result.files_mut().extend(other.files().iter().cloned());
        result
    }

    fn __iadd__(&mut self, other: &PySrt) {
        self.files_mut().extend(other.files().iter().cloned());
    }

    fn __iter__(slf: PyRef<'_, Self>) -> SrtIter {
        SrtIter {
            inner: slf.files().clone(),
            index: 0,
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PySrt> {
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
            "__len__ of an SRT object is intentionally undefined. \
             Intuitively, there are different lengths one may refer to: \
             Number of files? Subtitle blocks? Something else?",
        ))
    }

    fn __repr__(&self) -> String {
        format!("<SRT with {} file(s)>", self.num_files())
    }

    fn __eq__(&self, other: &PySrt) -> bool {
        self.files().len() == other.files().len()
            && self
                .files()
                .iter()
                .zip(other.files())
                .all(|(a, b)| a.file_path == b.file_path && a.blocks == b.blocks)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.files().len().hash(&mut hasher);
        for f in self.files() {
            f.file_path.hash(&mut hasher);
            f.blocks.len().hash(&mut hasher);
            for b in &f.blocks {
                b.index.hash(&mut hasher);
                b.text.hash(&mut hasher);
                b.start_ms.hash(&mut hasher);
                b.end_ms.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

#[pyclass]
struct SrtIter {
    inner: VecDeque<SrtFile>,
    index: usize,
}

#[pymethods]
impl SrtIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PySrt> {
        if self.index < self.inner.len() {
            let file = self.inner[self.index].clone();
            self.index += 1;
            Some(PySrt {
                inner: Srt::from_files(VecDeque::from(vec![file])),
            })
        } else {
            None
        }
    }
}
