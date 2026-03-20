use pyo3::prelude::*;
use pyo3::types::{PySlice, PyType};

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use super::reader::{
    BaseConllu, Conllu, ConlluError, ConlluFile, ConlluToken, Sentence, WriteError,
};
use crate::persistence::pathbuf_to_string;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn conllu_error_to_pyerr(e: ConlluError) -> PyErr {
    match e {
        ConlluError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        ConlluError::Parse(e) => pyo3::exceptions::PyValueError::new_err(e),
        ConlluError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(e),
        ConlluError::Zip(e) => pyo3::exceptions::PyIOError::new_err(e),
        ConlluError::Source(e) => e.into(),
    }
}

// ---------------------------------------------------------------------------
// PyConlluToken
// ---------------------------------------------------------------------------

/// A single token from a CoNLL-U file (10 tab-separated fields).
#[pyclass(name = "Token", from_py_object)]
#[derive(Clone)]
pub struct PyConlluToken(pub(crate) ConlluToken);

#[pymethods]
impl PyConlluToken {
    /// Word index (integer, range, or decimal).
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }

    /// Word form or punctuation symbol.
    #[getter]
    fn form(&self) -> &str {
        &self.0.form
    }

    /// Lemma or stem of the word.
    #[getter]
    fn lemma(&self) -> &str {
        &self.0.lemma
    }

    /// Universal POS tag.
    #[getter]
    fn upos(&self) -> &str {
        &self.0.upos
    }

    /// Language-specific POS tag, or ``"_"``.
    #[getter]
    fn xpos(&self) -> &str {
        &self.0.xpos
    }

    /// Morphological features, or ``"_"``.
    #[getter]
    fn feats(&self) -> &str {
        &self.0.feats
    }

    /// Head of the current word (ID or ``"0"`` for root), or ``"_"``.
    #[getter]
    fn head(&self) -> &str {
        &self.0.head
    }

    /// Universal dependency relation to HEAD, or ``"_"``.
    #[getter]
    fn deprel(&self) -> &str {
        &self.0.deprel
    }

    /// Enhanced dependency graph, or ``"_"``.
    #[getter]
    fn deps(&self) -> &str {
        &self.0.deps
    }

    /// Any other annotation, or ``"_"``.
    #[getter]
    fn misc(&self) -> &str {
        &self.0.misc
    }

    fn __repr__(&self) -> String {
        format!(
            "Token(id={:?}, form={:?}, lemma={:?}, upos={:?})",
            self.0.id, self.0.form, self.0.lemma, self.0.upos,
        )
    }

    fn __eq__(&self, other: &PyConlluToken) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PySentence
// ---------------------------------------------------------------------------

/// A single sentence from a CoNLL-U file.
#[pyclass(name = "Sentence", from_py_object)]
#[derive(Clone)]
pub struct PySentence(pub(crate) Sentence);

#[pymethods]
impl PySentence {
    /// Comment lines (without the leading ``# ``), or ``None``.
    #[getter]
    fn comments(&self) -> Option<Vec<String>> {
        self.0.comments.clone()
    }

    /// Tokens in this sentence.
    fn tokens(&self) -> Vec<PyConlluToken> {
        self.0
            .tokens
            .iter()
            .map(|t| PyConlluToken(t.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        let n_tokens = self.0.tokens.len();
        let n_comments = self.0.comments.as_ref().map_or(0, |c| c.len());
        format!("Sentence(tokens={n_tokens}, comments={n_comments})")
    }

    fn __eq__(&self, other: &PySentence) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        if let Some(comments) = &self.0.comments {
            for c in comments {
                c.hash(&mut hasher);
            }
        }
        for t in &self.0.tokens {
            t.hash(&mut hasher);
        }
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyConllu
// ---------------------------------------------------------------------------

/// CoNLL-U (Universal Dependencies) data reader.
#[pyclass(name = "CoNLLU", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyConllu {
    pub inner: Conllu,
}

impl BaseConllu for PyConllu {
    fn files(&self) -> &VecDeque<ConlluFile> {
        self.inner.files()
    }
    fn files_mut(&mut self) -> &mut VecDeque<ConlluFile> {
        self.inner.files_mut()
    }
    fn from_files(files: VecDeque<ConlluFile>) -> Self {
        Self {
            inner: Conllu::from_files(files),
        }
    }
}

#[pymethods]
impl PyConllu {
    #[new]
    fn new() -> Self {
        Self::from_files(VecDeque::new())
    }

    /// Parse CoNLL-U data from in-memory strings.
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
        let conllu = Conllu::from_strs(strs, ids, parallel).map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
    }

    /// Load CoNLL-U data from file paths.
    #[classmethod]
    #[pyo3(name = "from_files")]
    #[pyo3(signature = (paths, *, parallel=true))]
    fn read_files(_cls: &Bound<'_, PyType>, paths: Vec<PathBuf>, parallel: bool) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let conllu = Conllu::read_files(&paths, parallel).map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
    }

    /// Recursively load CoNLL-U data from a directory.
    #[classmethod]
    #[pyo3(name = "from_dir")]
    #[pyo3(signature = (path, *, r#match=None, extension=".conllu", parallel=true))]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let conllu =
            Conllu::read_dir(&path, r#match, extension, parallel).map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
    }

    /// Load CoNLL-U data from a ZIP archive.
    #[classmethod]
    #[pyo3(name = "from_zip")]
    #[pyo3(signature = (path, *, r#match=None, extension=".conllu", parallel=true))]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let conllu =
            Conllu::read_zip(&path, r#match, extension, parallel).map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
    }

    /// Load CoNLL-U data from a git repository.
    #[classmethod]
    #[pyo3(name = "from_git")]
    #[pyo3(signature = (url, *, rev=None, depth=None, r#match=None, extension=".conllu", cache_dir=None, force_download=false, parallel=true))]
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
        let conllu = Conllu::from_git(
            url,
            rev,
            depth,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
        )
        .map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
    }

    /// Load CoNLL-U data from a URL.
    #[classmethod]
    #[pyo3(name = "from_url")]
    #[pyo3(signature = (url, *, r#match=None, extension=".conllu", cache_dir=None, force_download=false, parallel=true))]
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
        let conllu = Conllu::from_url(url, r#match, extension, cache_dir, force_download, parallel)
            .map_err(conllu_error_to_pyerr)?;
        Ok(Self { inner: conllu })
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

    /// Return all sentences across all files as a flat list.
    fn sentences(&self) -> Vec<PySentence> {
        self.files()
            .iter()
            .flat_map(|f| &f.sentences)
            .map(|s| PySentence(s.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return CoNLL-U strings, one per file.
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

    /// Write CoNLL-U files to a directory.
    #[pyo3(name = "to_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_conllu_files(&dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    // -----------------------------------------------------------------------
    // Collection operations
    // -----------------------------------------------------------------------

    /// Append data from another CoNLL-U reader.
    #[pyo3(name = "append", signature = (other, /))]
    fn py_push_back(&mut self, other: &PyConllu) {
        self.inner.push_back(&other.inner);
    }

    /// Left-append data from another CoNLL-U reader, preserving order.
    #[pyo3(name = "append_left", signature = (other, /))]
    fn py_push_front(&mut self, other: &PyConllu) {
        self.inner.push_front(&other.inner);
    }

    /// Extend data from multiple CoNLL-U readers.
    #[pyo3(name = "extend", signature = (others, /))]
    fn extend_back(&mut self, others: Vec<PyRef<'_, PyConllu>>) {
        for other in &others {
            self.files_mut().extend(other.files().iter().cloned());
        }
    }

    /// Remove and return the last file as a new CoNLL-U reader.
    #[pyo3(name = "pop")]
    fn pop_back(&mut self) -> PyResult<PyConllu> {
        match self.files_mut().pop_back() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty CoNLL-U reader",
            )),
        }
    }

    /// Remove and return the first file as a new CoNLL-U reader.
    #[pyo3(name = "pop_left")]
    fn pop_front(&mut self) -> PyResult<PyConllu> {
        match self.files_mut().pop_front() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty CoNLL-U reader",
            )),
        }
    }

    /// Remove all data from this reader.
    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.files_mut().clear();
    }

    fn __add__(&self, other: &PyConllu) -> PyConllu {
        let mut result = self.clone();
        result.files_mut().extend(other.files().iter().cloned());
        result
    }

    fn __iadd__(&mut self, other: &PyConllu) {
        self.files_mut().extend(other.files().iter().cloned());
    }

    fn __iter__(slf: PyRef<'_, Self>) -> ConlluIter {
        ConlluIter {
            inner: slf.files().clone(),
            index: 0,
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyConllu> {
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
            "__len__ of a CoNLL-U object is intentionally undefined. \
             Intuitively, there are different lengths one may refer to: \
             Number of files? Sentences? Something else?",
        ))
    }

    fn __repr__(&self) -> String {
        format!("<CoNLLU with {} file(s)>", self.num_files())
    }

    fn __eq__(&self, other: &PyConllu) -> bool {
        self.files().len() == other.files().len()
            && self
                .files()
                .iter()
                .zip(other.files())
                .all(|(a, b)| a.file_path == b.file_path && a.sentences == b.sentences)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.files().len().hash(&mut hasher);
        for f in self.files() {
            f.file_path.hash(&mut hasher);
            f.sentences.len().hash(&mut hasher);
        }
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

#[pyclass]
struct ConlluIter {
    inner: VecDeque<ConlluFile>,
    index: usize,
}

#[pymethods]
impl ConlluIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyConllu> {
        if self.index < self.inner.len() {
            let file = self.inner[self.index].clone();
            self.index += 1;
            Some(PyConllu {
                inner: Conllu::from_files(VecDeque::from(vec![file])),
            })
        } else {
            None
        }
    }
}
