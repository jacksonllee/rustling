use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyType};

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use super::reader::{
    BaseTextGrid, Interval, Point, TextGrid, TextGridError, TextGridFile, TextGridTier, WriteError,
};
use crate::persistence::pathbuf_to_string;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn textgrid_error_to_pyerr(e: TextGridError) -> PyErr {
    match e {
        TextGridError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        TextGridError::Parse(e) => pyo3::exceptions::PyValueError::new_err(e),
        TextGridError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(e),
        TextGridError::Zip(e) => pyo3::exceptions::PyIOError::new_err(e),
        TextGridError::Source(e) => e.into(),
    }
}

// ---------------------------------------------------------------------------
// PyInterval
// ---------------------------------------------------------------------------

/// A single interval within an IntervalTier.
#[pyclass(name = "Interval", from_py_object)]
#[derive(Clone)]
pub struct PyInterval(pub(crate) Interval);

#[pymethods]
impl PyInterval {
    /// Start time in seconds.
    #[getter]
    fn xmin(&self) -> f64 {
        self.0.xmin
    }

    /// End time in seconds.
    #[getter]
    fn xmax(&self) -> f64 {
        self.0.xmax
    }

    /// The annotation text.
    #[getter]
    fn text(&self) -> &str {
        &self.0.text
    }

    fn __repr__(&self) -> String {
        format!(
            "Interval(xmin={}, xmax={}, text={:?})",
            self.0.xmin, self.0.xmax, self.0.text,
        )
    }

    fn __eq__(&self, other: &PyInterval) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.xmin.to_bits().hash(&mut hasher);
        self.0.xmax.to_bits().hash(&mut hasher);
        self.0.text.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyPoint
// ---------------------------------------------------------------------------

/// A single point within a TextTier (PointTier).
#[pyclass(name = "Point", from_py_object)]
#[derive(Clone)]
pub struct PyPoint(pub(crate) Point);

#[pymethods]
impl PyPoint {
    /// Time in seconds.
    #[getter]
    fn number(&self) -> f64 {
        self.0.number
    }

    /// The annotation text.
    #[getter]
    fn mark(&self) -> &str {
        &self.0.mark
    }

    fn __repr__(&self) -> String {
        format!("Point(number={}, mark={:?})", self.0.number, self.0.mark)
    }

    fn __eq__(&self, other: &PyPoint) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.number.to_bits().hash(&mut hasher);
        self.0.mark.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyIntervalTier
// ---------------------------------------------------------------------------

/// An interval tier (IntervalTier) within a TextGrid file.
#[pyclass(name = "IntervalTier", from_py_object)]
#[derive(Clone)]
pub struct PyIntervalTier {
    pub(crate) name: String,
    pub(crate) xmin: f64,
    pub(crate) xmax: f64,
    pub(crate) intervals: Vec<Interval>,
}

#[pymethods]
impl PyIntervalTier {
    /// Tier name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Start time in seconds.
    #[getter]
    fn xmin(&self) -> f64 {
        self.xmin
    }

    /// End time in seconds.
    #[getter]
    fn xmax(&self) -> f64 {
        self.xmax
    }

    /// Intervals in this tier.
    #[getter]
    fn intervals(&self) -> Vec<PyInterval> {
        self.intervals
            .iter()
            .map(|i| PyInterval(i.clone()))
            .collect()
    }

    /// Tier class: always ``"IntervalTier"``.
    #[getter]
    fn tier_class(&self) -> &str {
        "IntervalTier"
    }

    fn __repr__(&self) -> String {
        format!(
            "IntervalTier(name={:?}, intervals={})",
            self.name,
            self.intervals.len(),
        )
    }

    fn __eq__(&self, other: &PyIntervalTier) -> bool {
        self.name == other.name
            && self.xmin == other.xmin
            && self.xmax == other.xmax
            && self.intervals == other.intervals
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.xmin.to_bits().hash(&mut hasher);
        self.xmax.to_bits().hash(&mut hasher);
        self.intervals.len().hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyTextTier
// ---------------------------------------------------------------------------

/// A text tier (TextTier / PointTier) within a TextGrid file.
#[pyclass(name = "TextTier", from_py_object)]
#[derive(Clone)]
pub struct PyTextTier {
    pub(crate) name: String,
    pub(crate) xmin: f64,
    pub(crate) xmax: f64,
    pub(crate) points: Vec<Point>,
}

#[pymethods]
impl PyTextTier {
    /// Tier name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Start time in seconds.
    #[getter]
    fn xmin(&self) -> f64 {
        self.xmin
    }

    /// End time in seconds.
    #[getter]
    fn xmax(&self) -> f64 {
        self.xmax
    }

    /// Points in this tier.
    #[getter]
    fn points(&self) -> Vec<PyPoint> {
        self.points.iter().map(|p| PyPoint(p.clone())).collect()
    }

    /// Tier class: always ``"TextTier"``.
    #[getter]
    fn tier_class(&self) -> &str {
        "TextTier"
    }

    fn __repr__(&self) -> String {
        format!(
            "TextTier(name={:?}, points={})",
            self.name,
            self.points.len(),
        )
    }

    fn __eq__(&self, other: &PyTextTier) -> bool {
        self.name == other.name
            && self.xmin == other.xmin
            && self.xmax == other.xmax
            && self.points == other.points
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.xmin.to_bits().hash(&mut hasher);
        self.xmax.to_bits().hash(&mut hasher);
        self.points.len().hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Helper: convert TextGridTier to PyObject
// ---------------------------------------------------------------------------

fn tier_to_py<'py>(py: Python<'py>, tier: &TextGridTier) -> Bound<'py, PyAny> {
    match tier {
        TextGridTier::IntervalTier {
            name,
            xmin,
            xmax,
            intervals,
        } => {
            let obj = PyIntervalTier {
                name: name.clone(),
                xmin: *xmin,
                xmax: *xmax,
                intervals: intervals.clone(),
            };
            Bound::new(py, obj).unwrap().into_any()
        }
        TextGridTier::TextTier {
            name,
            xmin,
            xmax,
            points,
        } => {
            let obj = PyTextTier {
                name: name.clone(),
                xmin: *xmin,
                xmax: *xmax,
                points: points.clone(),
            };
            Bound::new(py, obj).unwrap().into_any()
        }
    }
}

// ---------------------------------------------------------------------------
// PyTextGrid
// ---------------------------------------------------------------------------

/// TextGrid (Praat) data reader.
#[pyclass(name = "TextGrid", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyTextGrid {
    pub inner: TextGrid,
}

impl BaseTextGrid for PyTextGrid {
    fn files(&self) -> &VecDeque<TextGridFile> {
        self.inner.files()
    }
    fn files_mut(&mut self) -> &mut VecDeque<TextGridFile> {
        self.inner.files_mut()
    }
    fn from_files(files: VecDeque<TextGridFile>) -> Self {
        Self {
            inner: TextGrid::from_files(files),
        }
    }
}

#[pymethods]
impl PyTextGrid {
    #[new]
    fn new() -> Self {
        Self::from_files(VecDeque::new())
    }

    /// Parse TextGrid data from in-memory strings.
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
        let tg = TextGrid::from_strs(strs, ids, parallel).map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
    }

    /// Load TextGrid data from file paths.
    #[classmethod]
    #[pyo3(name = "from_files")]
    #[pyo3(signature = (paths, *, parallel=true))]
    fn read_files(_cls: &Bound<'_, PyType>, paths: Vec<PathBuf>, parallel: bool) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let tg = TextGrid::read_files(&paths, parallel).map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
    }

    /// Recursively load TextGrid data from a directory.
    #[classmethod]
    #[pyo3(name = "from_dir")]
    #[pyo3(signature = (path, *, r#match=None, extension=".TextGrid", parallel=true))]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let tg = TextGrid::read_dir(&path, r#match, extension, parallel)
            .map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
    }

    /// Load TextGrid data from a ZIP archive.
    #[classmethod]
    #[pyo3(name = "from_zip")]
    #[pyo3(signature = (path, *, r#match=None, extension=".TextGrid", parallel=true))]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let tg = TextGrid::read_zip(&path, r#match, extension, parallel)
            .map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
    }

    /// Load TextGrid data from a git repository.
    #[classmethod]
    #[pyo3(name = "from_git")]
    #[pyo3(signature = (url, *, rev=None, depth=None, r#match=None, extension=".TextGrid", cache_dir=None, force_download=false, parallel=true))]
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
        let tg = TextGrid::from_git(
            url,
            rev,
            depth,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
        )
        .map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
    }

    /// Load TextGrid data from a URL.
    #[classmethod]
    #[pyo3(name = "from_url")]
    #[pyo3(signature = (url, *, r#match=None, extension=".TextGrid", cache_dir=None, force_download=false, parallel=true))]
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
        let tg = TextGrid::from_url(url, r#match, extension, cache_dir, force_download, parallel)
            .map_err(textgrid_error_to_pyerr)?;
        Ok(Self { inner: tg })
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

    /// Return tiers as a list of lists, one list per file.
    ///
    /// Each inner list contains :class:`IntervalTier` and/or :class:`TextTier` objects.
    fn tiers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let outer: Vec<Bound<'py, PyList>> = self
            .files()
            .iter()
            .map(|f| {
                let inner: Vec<Bound<'py, PyAny>> =
                    f.tiers.iter().map(|t| tier_to_py(py, t)).collect();
                PyList::new(py, inner)
            })
            .collect::<Result<_, _>>()?;
        PyList::new(py, outer)
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return TextGrid strings, one per file.
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

    /// Write TextGrid files to a directory.
    #[pyo3(name = "to_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_files(&dir_path, filenames).map_err(|e| match e {
            WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
        })
    }

    // -----------------------------------------------------------------------
    // Collection operations
    // -----------------------------------------------------------------------

    /// Append data from another TextGrid reader.
    #[pyo3(name = "append", signature = (other, /))]
    fn py_push_back(&mut self, other: &PyTextGrid) {
        self.inner.push_back(&other.inner);
    }

    /// Left-append data from another TextGrid reader, preserving order.
    #[pyo3(name = "append_left", signature = (other, /))]
    fn py_push_front(&mut self, other: &PyTextGrid) {
        self.inner.push_front(&other.inner);
    }

    /// Extend data from multiple TextGrid readers.
    #[pyo3(name = "extend", signature = (others, /))]
    fn extend_back(&mut self, others: Vec<PyRef<'_, PyTextGrid>>) {
        for other in &others {
            self.files_mut().extend(other.files().iter().cloned());
        }
    }

    /// Remove and return the last file as a new TextGrid reader.
    #[pyo3(name = "pop")]
    fn pop_back(&mut self) -> PyResult<PyTextGrid> {
        match self.files_mut().pop_back() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty TextGrid reader",
            )),
        }
    }

    /// Remove and return the first file as a new TextGrid reader.
    #[pyo3(name = "pop_left")]
    fn pop_front(&mut self) -> PyResult<PyTextGrid> {
        match self.files_mut().pop_front() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty TextGrid reader",
            )),
        }
    }

    /// Remove all data from this reader.
    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.files_mut().clear();
    }

    fn __add__(&self, other: &PyTextGrid) -> PyTextGrid {
        let mut result = self.clone();
        result.files_mut().extend(other.files().iter().cloned());
        result
    }

    fn __iadd__(&mut self, other: &PyTextGrid) {
        self.files_mut().extend(other.files().iter().cloned());
    }

    fn __iter__(slf: PyRef<'_, Self>) -> TextGridIter {
        TextGridIter {
            inner: slf.files().clone(),
            index: 0,
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyTextGrid> {
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
            "__len__ of a TextGrid object is intentionally undefined. \
             Intuitively, there are different lengths one may refer to: \
             Number of files? Tiers? Something else?",
        ))
    }

    fn __repr__(&self) -> String {
        format!("<TextGrid with {} file(s)>", self.num_files())
    }

    fn __eq__(&self, other: &PyTextGrid) -> bool {
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
        }
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

#[pyclass]
struct TextGridIter {
    inner: VecDeque<TextGridFile>,
    index: usize,
}

#[pymethods]
impl TextGridIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyTextGrid> {
        if self.index < self.inner.len() {
            let file = self.inner[self.index].clone();
            self.index += 1;
            Some(PyTextGrid {
                inner: TextGrid::from_files(VecDeque::from(vec![file])),
            })
        } else {
            None
        }
    }
}
