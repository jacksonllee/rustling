use pyo3::prelude::*;
use pyo3::types::{PySlice, PyType};

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use super::header::{Age, Headers};
use super::reader::{
    BaseChat, Chat, ChatError, ChatFile, MisalignmentInfo, filter_chat_file_by_participants,
};
use super::utterance::Utterance;
use super::utterance_py::{PyToken, PyUtterance, PyUtterances};
use crate::chat::validation::{ValidationError, validate_chat_file};
use crate::ngram::{BaseNgrams, Ngrams, PyNgrams};
use crate::persistence::pathbuf_to_string;

use fancy_regex::Regex as FancyRegex;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compile file-path regex patterns from a Python str or iterable of str.
fn compile_file_patterns(files: &Bound<'_, PyAny>) -> PyResult<Vec<FancyRegex>> {
    let raw_patterns: Vec<String> = if let Ok(s) = files.extract::<String>() {
        vec![s]
    } else if let Ok(v) = files.extract::<Vec<String>>() {
        v
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "files must be a str or iterable of str",
        ));
    };

    if raw_patterns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "files must not be empty",
        ));
    }

    raw_patterns
        .iter()
        .map(|p| {
            FancyRegex::new(p).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid file regex '{p}': {e}"))
            })
        })
        .collect()
}

/// Compile participant regex patterns from a Python str or iterable of str.
/// Each pattern is auto-anchored with ^(?:...)$ for full-match semantics.
fn compile_participant_patterns(participants: &Bound<'_, PyAny>) -> PyResult<Vec<FancyRegex>> {
    let raw_patterns: Vec<String> = if let Ok(s) = participants.extract::<String>() {
        vec![s]
    } else if let Ok(v) = participants.extract::<Vec<String>>() {
        v
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "participants must be a str or iterable of str",
        ));
    };

    if raw_patterns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "participants must not be empty",
        ));
    }

    raw_patterns
        .iter()
        .map(|p| {
            let anchored = if p.starts_with('^') || p.ends_with('$') {
                p.clone()
            } else {
                format!("^(?:{p})$")
            };
            FancyRegex::new(&anchored).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid participant regex '{p}': {e}"
                ))
            })
        })
        .collect()
}

/// Convert user-facing tier names (without `%`) to internal `%`-prefixed keys.
///
/// If either tier is `None`, both are set to `None` (disabling mor+gra parsing).
fn tier_keys(mor_tier: Option<&str>, gra_tier: Option<&str>) -> (Option<String>, Option<String>) {
    match (mor_tier, gra_tier) {
        (Some(m), Some(g)) => {
            let mk = if m.starts_with('%') {
                m.to_string()
            } else {
                format!("%{m}")
            };
            let gk = if g.starts_with('%') {
                g.to_string()
            } else {
                format!("%{g}")
            };
            (Some(mk), Some(gk))
        }
        _ => (None, None),
    }
}

/// Check collected misalignments and either raise or warn.
fn handle_misalignments(
    misalignments: &[MisalignmentInfo],
    strict: bool,
    py: Python<'_>,
) -> PyResult<()> {
    if misalignments.is_empty() {
        return Ok(());
    }

    if strict {
        let mut msg = format!(
            "Found {} utterance(s) with mor/word misalignment:\n",
            misalignments.len()
        );
        for (i, m) in misalignments.iter().enumerate() {
            msg.push_str(&format!(
                "\n  {}. File: {}\n     Participant: {}\n     Main tier: {}\n\
                 \x20    {} tier: {}\n     Words ({}): {}\n\
                 \x20    Non-clitic mor items ({}): {}\n",
                i + 1,
                m.file_path,
                m.participant,
                m.main_tier,
                m.mor_tier_name,
                m.mor_tier_content,
                m.word_count,
                m.words.join(" "),
                m.mor_count,
                m.mor_labels.join(" "),
            ));
        }
        msg.push_str(
            "\nTo suppress this error and parse with empty tokens for \
             misaligned utterances, pass strict=False.",
        );
        return Err(pyo3::exceptions::PyValueError::new_err(msg));
    }

    // strict=False: emit Python warnings.
    let warnings = py.import("warnings")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("stacklevel", 2)?;
    for m in misalignments {
        let msg = format!(
            "mor/word misalignment in file '{}', participant '{}':\n\
             \x20 Main tier: {}\n\
             \x20 {} tier: {}\n\
             \x20 Words ({}): {}\n\
             \x20 Non-clitic mor items ({}): {}\n\
             Tokens set to empty for this utterance; \
             raw tier data is preserved in utterance.tiers.",
            m.file_path,
            m.participant,
            m.main_tier,
            m.mor_tier_name,
            m.mor_tier_content,
            m.word_count,
            m.words.join(" "),
            m.mor_count,
            m.mor_labels.join(" "),
        );
        warnings.call_method("warn", (&msg,), Some(&kwargs))?;
    }
    Ok(())
}

/// Convert a [`ChatError`] to a Python exception.
fn chat_error_to_pyerr(e: ChatError) -> pyo3::PyErr {
    match e {
        ChatError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        ChatError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(e),
        ChatError::Zip(e) => pyo3::exceptions::PyIOError::new_err(e),
        ChatError::Source(e) => e.into(),
    }
}

/// Validate all files in a Chat and return collected errors.
fn validate_chat(chat: &Chat) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    for file in chat.files() {
        errors.extend(validate_chat_file(
            &file.file_path,
            &file.headers,
            &file.events,
            &file.raw_lines,
        ));
    }
    errors
}

/// Check validation errors and either raise or silently pass.
fn handle_validation_errors(errors: &[ValidationError], strict: bool) -> PyResult<()> {
    if errors.is_empty() || !strict {
        return Ok(());
    }
    // Raise the first validation error.
    let msg = &errors[0].message;
    Err(pyo3::exceptions::PyValueError::new_err(msg.clone()))
}

// ---------------------------------------------------------------------------
// Cached PyO3 methods on ChatFile
// ---------------------------------------------------------------------------

impl ChatFile {
    pub(crate) fn cached_py_utterances(&self, py: Python<'_>) -> &[Py<PyUtterance>] {
        self.py_utterances.get_or_init(|| {
            self.utterances()
                .map(|utt| Py::new(py, PyUtterance(utt.clone())).unwrap())
                .collect()
        })
    }

    pub(crate) fn cached_py_tokens(&self, py: Python<'_>) -> &[Vec<Py<PyToken>>] {
        self.py_tokens.get_or_init(|| {
            self.real_utterances()
                .map(|u| {
                    u.tokens
                        .as_ref()
                        .map(|toks| {
                            toks.iter()
                                .map(|t| Py::new(py, PyToken(t.clone())).unwrap())
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
    }
}

// ---------------------------------------------------------------------------
// BasePyChat
// ---------------------------------------------------------------------------

use super::reader::WriteError;

/// Shared Python-boundary methods with default implementations.
///
/// Downstream crates can use these defaults.
/// Since `#[pymethods]` cannot be applied to trait impl blocks, the concrete
/// types have thin `#[pymethods]` wrappers that delegate to these methods.
pub trait BasePyChat: BaseChat {
    /// Return words grouped by utterance and/or file as Python objects.
    fn py_words(&self, py: Python<'_>, by_utterance: bool, by_file: bool) -> PyResult<Py<PyAny>> {
        match (by_utterance, by_file) {
            (false, false) => {
                let words: Vec<String> = self
                    .files()
                    .iter()
                    .flat_map(|f| f.real_utterances())
                    .flat_map(|u| u.tokens.as_deref().unwrap_or(&[]).iter())
                    .filter(|t| !t.word.is_empty())
                    .map(|t| t.word.clone())
                    .collect();
                Ok(words.into_pyobject(py)?.into_any().unbind())
            }
            (true, false) => {
                let words: Vec<Vec<String>> = self
                    .files()
                    .iter()
                    .flat_map(|f| f.real_utterances())
                    .map(|u| {
                        u.tokens
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|t| !t.word.is_empty())
                            .map(|t| t.word.clone())
                            .collect()
                    })
                    .collect();
                Ok(words.into_pyobject(py)?.into_any().unbind())
            }
            (false, true) => {
                let words: Vec<Vec<String>> = self
                    .files()
                    .iter()
                    .map(|f| {
                        f.real_utterances()
                            .flat_map(|u| u.tokens.as_deref().unwrap_or(&[]).iter())
                            .filter(|t| !t.word.is_empty())
                            .map(|t| t.word.clone())
                            .collect()
                    })
                    .collect();
                Ok(words.into_pyobject(py)?.into_any().unbind())
            }
            (true, true) => {
                let words: Vec<Vec<Vec<String>>> = self
                    .files()
                    .iter()
                    .map(|f| {
                        f.real_utterances()
                            .map(|u| {
                                u.tokens
                                    .as_deref()
                                    .unwrap_or(&[])
                                    .iter()
                                    .filter(|t| !t.word.is_empty())
                                    .map(|t| t.word.clone())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                Ok(words.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Write CHAT (.cha) files to a directory with Python error conversion.
    fn py_write_chat(&self, dir_path: &str, filenames: Option<Vec<String>>) -> PyResult<()> {
        self.write_chat_files(dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Write ELAN (.eaf) files to a directory with Python error conversion.
    fn py_write_elan(&self, dir_path: &str, filenames: Option<Vec<String>>) -> PyResult<()> {
        self.write_elan_files(dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Write SRT (.srt) files to a directory with Python error conversion.
    fn py_write_srt(
        &self,
        dir_path: &str,
        participants: Option<&[String]>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.write_srt_files(dir_path, participants, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Write TextGrid (.TextGrid) files to a directory with Python error conversion.
    fn py_write_textgrid(
        &self,
        dir_path: &str,
        participants: Option<&[String]>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.write_textgrid_files(dir_path, participants, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    /// Print a summary of this reader's data.
    fn py_info(&self, py: Python<'_>, verbose: bool) -> PyResult<()> {
        let n_files = self.files().len();

        let total_utterances: usize = self
            .files()
            .iter()
            .map(|f| f.real_utterances().count())
            .sum();

        let total_words: usize = self
            .files()
            .iter()
            .map(|f| {
                f.real_utterances()
                    .map(|u| {
                        u.tokens
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|t| !t.word.is_empty())
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();

        let py_print = py.import("builtins")?.getattr("print")?;

        py_print.call1((format!("{n_files} files"),))?;
        py_print.call1((format!("{total_utterances} utterances"),))?;
        py_print.call1((format!("{total_words} words"),))?;

        if n_files < 2 {
            return Ok(());
        }

        // Collect per-file stats.
        let stats: Vec<(usize, usize, &str)> = self
            .files()
            .iter()
            .map(|f| {
                let utt_count = f.real_utterances().count();
                let word_count: usize = f
                    .real_utterances()
                    .map(|u| {
                        u.tokens
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|t| !t.word.is_empty())
                            .count()
                    })
                    .sum();
                (utt_count, word_count, f.file_path.as_str())
            })
            .collect();

        let max_rows = if verbose { n_files } else { 5.min(n_files) };
        let display_stats = &stats[..max_rows];

        // Column widths.
        let idx_width = format!("#{max_rows}").len().max(2);
        let utt_header = "Utterance Count";
        let word_header = "Word Count";
        let path_header = "File Path";

        let utt_width = display_stats
            .iter()
            .map(|(c, _, _)| format!("{c}").len())
            .max()
            .unwrap_or(0)
            .max(utt_header.len());
        let word_width = display_stats
            .iter()
            .map(|(_, c, _)| format!("{c}").len())
            .max()
            .unwrap_or(0)
            .max(word_header.len());
        let path_width = display_stats
            .iter()
            .map(|(_, _, p)| p.len())
            .max()
            .unwrap_or(0)
            .max(path_header.len());

        // Header.
        py_print.call1((format!(
            "{:>iw$}  {:>uw$}  {:>ww$}  {:<pw$}",
            "",
            utt_header,
            word_header,
            path_header,
            iw = idx_width,
            uw = utt_width,
            ww = word_width,
            pw = path_width,
        ),))?;

        // Separator.
        py_print.call1((format!(
            "{:->iw$}  {:->uw$}  {:->ww$}  {:->pw$}",
            "",
            "",
            "",
            "",
            iw = idx_width,
            uw = utt_width,
            ww = word_width,
            pw = path_width,
        ),))?;

        // Data rows.
        for (i, (utt, word, path)) in display_stats.iter().enumerate() {
            py_print.call1((format!(
                "{:>iw$}  {:>uw$}  {:>ww$}  {:<pw$}",
                format!("#{}", i + 1),
                utt,
                word,
                path,
                iw = idx_width,
                uw = utt_width,
                ww = word_width,
                pw = path_width,
            ),))?;
        }

        if !verbose {
            py_print.call1(("...",))?;
            py_print.call1(("(set `verbose` to True for all the files)",))?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Python-exposed PyChat wrapper
// ---------------------------------------------------------------------------

/// Python-exposed CHAT data reader.
///
/// Wraps the pure Rust [`Chat`] struct and exposes it to Python via PyO3.
#[pyclass(name = "CHAT", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyChat {
    pub inner: Chat,
}

impl BaseChat for PyChat {
    fn files(&self) -> &VecDeque<ChatFile> {
        self.inner.files()
    }
    fn files_mut(&mut self) -> &mut VecDeque<ChatFile> {
        self.inner.files_mut()
    }
    fn from_files(files: VecDeque<ChatFile>) -> Self {
        Self {
            inner: Chat::from_files(files),
        }
    }
}

impl BasePyChat for PyChat {}

#[pymethods]
impl PyChat {
    #[new]
    fn new() -> Self {
        Self::from_files(VecDeque::new())
    }

    /// Parse CHAT data from in-memory strings.
    #[classmethod]
    #[pyo3(signature = (strs, ids=None, parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
    fn from_strs(
        _cls: &Bound<'_, PyType>,
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
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
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) =
            Chat::from_strs(strs, ids, parallel, mor_key.as_deref(), gra_key.as_deref());
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
    }

    /// Construct a CHAT reader from a list of utterances.
    #[classmethod]
    #[pyo3(name = "from_utterances")]
    #[pyo3(signature = (utterances))]
    fn py_from_utterances(_cls: &Bound<'_, PyType>, utterances: Vec<PyUtterance>) -> Self {
        let utts: Vec<Utterance> = utterances.into_iter().map(|pu| pu.0).collect();
        let result = <Self as BaseChat>::from_utterances(utts);
        let py = _cls.py();
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        result
    }

    /// Load CHAT data from file paths.
    #[classmethod]
    #[pyo3(name = "from_files")]
    #[pyo3(signature = (paths, *, parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
    fn read_files(
        _cls: &Bound<'_, PyType>,
        paths: Vec<PathBuf>,
        parallel: bool,
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) =
            Chat::read_files(&paths, parallel, mor_key.as_deref(), gra_key.as_deref())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
    }

    /// Recursively load CHAT data from a directory.
    #[classmethod]
    #[pyo3(name = "from_dir")]
    #[pyo3(signature = (path, *, r#match=None, extension=".cha", parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
    #[allow(clippy::too_many_arguments)]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) = Chat::read_dir(
            &path,
            r#match,
            extension,
            parallel,
            mor_key.as_deref(),
            gra_key.as_deref(),
        )
        .map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
    }

    /// Load CHAT data from a ZIP archive.
    #[classmethod]
    #[pyo3(name = "from_zip")]
    #[pyo3(signature = (path, *, r#match=None, extension=".cha", parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
    #[allow(clippy::too_many_arguments)]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) = Chat::read_zip(
            &path,
            r#match,
            extension,
            parallel,
            mor_key.as_deref(),
            gra_key.as_deref(),
        )
        .map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
    }

    /// Load CHAT data from a git repository.
    #[classmethod]
    #[pyo3(name = "from_git")]
    #[pyo3(signature = (url, *, rev=None, depth=None, r#match=None, extension=".cha", cache_dir=None, force_download=false, parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
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
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> PyResult<Self> {
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) = Chat::from_git(
            url,
            rev,
            depth,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
            mor_key.as_deref(),
            gra_key.as_deref(),
        )
        .map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
    }

    /// Load CHAT data from a URL.
    #[classmethod]
    #[pyo3(name = "from_url")]
    #[pyo3(signature = (url, *, r#match=None, extension=".cha", cache_dir=None, force_download=false, parallel=true, strict=true, mor_tier=Some("%mor"), gra_tier=Some("%gra")))]
    #[allow(clippy::too_many_arguments)]
    fn from_url(
        _cls: &Bound<'_, PyType>,
        url: &str,
        r#match: Option<&str>,
        extension: &str,
        cache_dir: Option<PathBuf>,
        force_download: bool,
        parallel: bool,
        strict: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> PyResult<Self> {
        let (mor_key, gra_key) = tier_keys(mor_tier, gra_tier);
        let py = _cls.py();
        let (chat, misalignments) = Chat::from_url(
            url,
            r#match,
            extension,
            cache_dir,
            force_download,
            parallel,
            mor_key.as_deref(),
            gra_key.as_deref(),
        )
        .map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
        let validation_errors = validate_chat(&chat);
        handle_validation_errors(&validation_errors, strict)?;
        let result = Self { inner: chat };
        for f in result.inner.files() {
            f.cached_py_utterances(py);
            f.cached_py_tokens(py);
        }
        Ok(result)
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

    /// Print a summary of this reader's data.
    #[pyo3(signature = (*, verbose = false))]
    fn info(&self, py: Python<'_>, verbose: bool) -> PyResult<()> {
        self.py_info(py, verbose)
    }

    /// Return a new CHAT filtered by file path and/or participant regex.
    #[pyo3(signature = (*, files=None, participants=None))]
    fn filter(
        &self,
        files: Option<&Bound<'_, PyAny>>,
        participants: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Step 1: Filter by file path.
        let filtered_files: VecDeque<ChatFile> = if let Some(files_arg) = files {
            let patterns = compile_file_patterns(files_arg)?;
            self.files()
                .iter()
                .filter(|f| {
                    patterns
                        .iter()
                        .any(|re| re.is_match(&f.file_path).unwrap_or(false))
                })
                .cloned()
                .collect()
        } else {
            self.files().clone()
        };

        // Step 2: Filter by participant.
        let filtered_files = if let Some(participants_arg) = participants {
            let patterns = compile_participant_patterns(participants_arg)?;
            filtered_files
                .into_iter()
                .map(|f| filter_chat_file_by_participants(f, &patterns))
                .collect()
        } else {
            filtered_files
        };

        Ok(Self::from_files(filtered_files))
    }

    /// Return utterances, optionally grouped by file.
    #[pyo3(signature = (*, by_file=false))]
    fn utterances(&self, py: Python<'_>, by_file: bool) -> PyResult<Py<PyAny>> {
        if by_file {
            let result: Vec<Vec<Py<PyUtterance>>> = self
                .files()
                .iter()
                .map(|f| {
                    f.cached_py_utterances(py)
                        .iter()
                        .map(|p| p.clone_ref(py))
                        .collect()
                })
                .collect();
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else {
            let mut result: Vec<Py<PyUtterance>> = Vec::new();
            for f in self.files() {
                for p in f.cached_py_utterances(py) {
                    result.push(p.clone_ref(py));
                }
            }
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Return the first n utterances with a formatted display.
    #[pyo3(name = "head", signature = (n=5))]
    fn py_head(&self, n: usize) -> PyUtterances {
        PyUtterances(self.head(n))
    }

    /// Return the last n utterances with a formatted display.
    #[pyo3(name = "tail", signature = (n=5))]
    fn py_tail(&self, n: usize) -> PyUtterances {
        PyUtterances(self.tail(n))
    }

    /// Return words, optionally grouped by utterance and/or file.
    #[pyo3(signature = (*, by_utterance=false, by_file=false))]
    fn words(&self, py: Python<'_>, by_utterance: bool, by_file: bool) -> PyResult<Py<PyAny>> {
        self.py_words(py, by_utterance, by_file)
    }

    /// Return tokens, optionally grouped by utterance and/or file.
    #[pyo3(signature = (*, by_utterance=false, by_file=false))]
    fn tokens(&self, py: Python<'_>, by_utterance: bool, by_file: bool) -> PyResult<Py<PyAny>> {
        match (by_utterance, by_file) {
            (false, false) => {
                let tokens: Vec<Py<PyToken>> = self
                    .files()
                    .iter()
                    .flat_map(|f| f.cached_py_tokens(py))
                    .flat_map(|utt_tokens| utt_tokens.iter())
                    .map(|t| t.clone_ref(py))
                    .collect();
                Ok(tokens.into_pyobject(py)?.into_any().unbind())
            }
            (true, false) => {
                let tokens: Vec<Vec<Py<PyToken>>> = self
                    .files()
                    .iter()
                    .flat_map(|f| f.cached_py_tokens(py))
                    .map(|utt_tokens| utt_tokens.iter().map(|t| t.clone_ref(py)).collect())
                    .collect();
                Ok(tokens.into_pyobject(py)?.into_any().unbind())
            }
            (false, true) => {
                let tokens: Vec<Vec<Py<PyToken>>> = self
                    .files()
                    .iter()
                    .map(|f| {
                        f.cached_py_tokens(py)
                            .iter()
                            .flat_map(|utt_tokens| utt_tokens.iter())
                            .map(|t| t.clone_ref(py))
                            .collect()
                    })
                    .collect();
                Ok(tokens.into_pyobject(py)?.into_any().unbind())
            }
            (true, true) => {
                let tokens: Vec<Vec<Vec<Py<PyToken>>>> = self
                    .files()
                    .iter()
                    .map(|f| {
                        f.cached_py_tokens(py)
                            .iter()
                            .map(|utt_tokens| utt_tokens.iter().map(|t| t.clone_ref(py)).collect())
                            .collect()
                    })
                    .collect();
                Ok(tokens.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Developmental measures
    // -----------------------------------------------------------------------

    /// Mean length of utterance in morphemes, one value per file.
    #[pyo3(name = "mlum", signature = (*, participant="CHI", n=Some(100)))]
    fn py_mlum(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
        self.mlum(participant, n)
    }

    /// Mean length of utterance in morphemes, one value per file.
    ///
    /// Alias for [`mlum`][Chat::mlum].
    #[pyo3(signature = (*, participant="CHI", n=Some(100)))]
    fn mlu(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
        self.mlum(participant, n)
    }

    /// Mean length of utterance in words, one value per file.
    #[pyo3(name = "mluw", signature = (*, participant="CHI", n=Some(100)))]
    fn py_mluw(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
        self.mluw(participant, n)
    }

    /// Type-token ratio, one value per file.
    #[pyo3(name = "ttr", signature = (*, participant="CHI", n=Some(350)))]
    fn py_ttr(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
        self.ttr(participant, n)
    }

    /// Index of Productive Syntax, one value per file.
    #[pyo3(signature = (*, participant="CHI", n=Some(100)))]
    fn ipsyn(&self, participant: &str, n: Option<usize>) -> Vec<usize> {
        self.files()
            .iter()
            .map(|f| {
                let utterances: Vec<_> = f
                    .real_utterances()
                    .filter(|u| u.participant.as_deref() == Some(participant))
                    .collect();
                let utterances = if let Some(n) = n {
                    &utterances[..utterances.len().min(n)]
                } else {
                    &utterances[..]
                };
                super::ipsyn::ipsyn_for_file(utterances)
            })
            .collect()
    }

    /// Return the age of the target child (CHI) in each file.
    #[pyo3(name = "ages")]
    fn py_ages(&self) -> Vec<Option<Age>> {
        self.ages()
    }

    /// Return an Ngrams for word n-grams across all utterances.
    ///
    /// N-grams do not cross utterance boundaries.
    ///
    /// # Arguments
    ///
    /// * `n` - The n-gram order (1 for unigrams, 2 for bigrams, etc.).
    #[pyo3(signature = (n))]
    fn word_ngrams(&self, n: usize) -> PyResult<PyNgrams> {
        let mut counter = Ngrams::new(n, None).map_err(PyErr::from)?;
        for file in self.files() {
            for utt in file.real_utterances() {
                let words: Vec<String> = utt
                    .tokens
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .filter(|t| !t.word.is_empty())
                    .map(|t| t.word.clone())
                    .collect();
                counter.count(words);
            }
        }
        Ok(PyNgrams { inner: counter })
    }

    // -----------------------------------------------------------------------
    // Header access
    // -----------------------------------------------------------------------

    /// Return file-level headers.
    #[pyo3(name = "headers")]
    fn py_headers(&self) -> Vec<Headers> {
        self.headers()
    }

    /// Return participants, optionally grouped by file.
    #[pyo3(name = "participants")]
    #[pyo3(signature = (*, by_file=false))]
    fn py_participants(&self, py: Python<'_>, by_file: bool) -> PyResult<Py<PyAny>> {
        if by_file {
            Ok(self.participants().into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(self
                .unique_participants()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }

    /// Return languages, optionally grouped by file.
    #[pyo3(name = "languages")]
    #[pyo3(signature = (*, by_file=false))]
    fn py_languages(&self, py: Python<'_>, by_file: bool) -> PyResult<Py<PyAny>> {
        if by_file {
            Ok(self.languages().into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(self
                .unique_languages()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }

    // -----------------------------------------------------------------------
    // Stitching / unstitching
    // -----------------------------------------------------------------------

    /// Append data from another CHAT reader.
    #[pyo3(name = "append", signature = (other, /))]
    fn py_push_back(&mut self, other: &PyChat) {
        self.inner.push_back(&other.inner);
    }

    /// Left-append data from another CHAT reader, preserving order.
    #[pyo3(name = "append_left", signature = (other, /))]
    fn py_push_front(&mut self, other: &PyChat) {
        self.inner.push_front(&other.inner);
    }

    /// Extend data from multiple CHAT readers.
    #[pyo3(name = "extend", signature = (others, /))]
    fn extend_back(&mut self, others: Vec<PyRef<'_, PyChat>>) {
        for other in &others {
            self.files_mut().extend(other.files().iter().cloned());
        }
    }

    /// Left-extend data from multiple CHAT readers, preserving order.
    #[pyo3(name = "extend_left", signature = (others, /))]
    fn extend_front(&mut self, others: Vec<PyRef<'_, PyChat>>) {
        let mut new_files: VecDeque<ChatFile> = VecDeque::new();
        for other in &others {
            new_files.extend(other.files().iter().cloned());
        }
        new_files.extend(std::mem::take(self.files_mut()));
        *self.files_mut() = new_files;
    }

    /// Remove and return the last file as a new CHAT reader.
    #[pyo3(name = "pop")]
    fn pop_back(&mut self) -> PyResult<PyChat> {
        match self.files_mut().pop_back() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty CHAT reader",
            )),
        }
    }

    /// Remove and return the first file as a new CHAT reader.
    #[pyo3(name = "pop_left")]
    fn pop_front(&mut self) -> PyResult<PyChat> {
        match self.files_mut().pop_front() {
            Some(file) => Ok(Self::from_files(VecDeque::from(vec![file]))),
            None => Err(pyo3::exceptions::PyIndexError::new_err(
                "pop from an empty CHAT reader",
            )),
        }
    }

    /// Remove all data from this reader.
    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.clear();
    }

    fn __add__(&self, other: &PyChat) -> PyChat {
        let mut result = self.clone();
        result.files_mut().extend(other.files().iter().cloned());
        result
    }

    fn __iadd__(&mut self, other: &PyChat) {
        self.files_mut().extend(other.files().iter().cloned());
    }

    fn __iter__(slf: PyRef<'_, Self>) -> ChatIter {
        ChatIter {
            inner: slf.files().clone(),
            index: 0,
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyChat> {
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
            for _ in 0..indices.slicelength {
                result.push_back(self.files()[i as usize].clone());
                i += indices.step;
            }
            return Ok(Self::from_files(result));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers or slices",
        ))
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return CHAT data strings, one per file.
    #[pyo3(name = "to_strs")]
    fn py_to_strings(&self) -> Vec<String> {
        self.to_strings()
    }

    /// Write CHAT (.cha) files to a directory.
    #[pyo3(name = "to_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write_chat(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.py_write_chat(&dir_path, filenames)
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
        self.py_write_elan(&dir_path, filenames)
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
        self.py_write_srt(&dir_path, participants.as_deref(), filenames)
    }

    /// Return TextGrid format strings, one per file.
    #[pyo3(name = "to_textgrid_strs")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_textgrid_strings(&self, participants: Option<Vec<String>>) -> Vec<String> {
        self.to_textgrid_strings(participants.as_deref())
    }

    /// Convert to a TextGrid object.
    #[pyo3(name = "to_textgrid")]
    #[pyo3(signature = (*, participants=None))]
    fn py_to_textgrid(&self, participants: Option<Vec<String>>) -> crate::textgrid::PyTextGrid {
        crate::textgrid::PyTextGrid {
            inner: self.to_textgrid(participants.as_deref()),
        }
    }

    /// Write TextGrid (.TextGrid) files to a directory.
    #[pyo3(name = "to_textgrid_files")]
    #[pyo3(signature = (dir_path, /, *, participants=None, filenames=None))]
    fn write_textgrid(
        &self,
        dir_path: PathBuf,
        participants: Option<Vec<String>>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.py_write_textgrid(&dir_path, participants.as_deref(), filenames)
    }

    /// Return CoNLL-U format strings, one per file.
    #[pyo3(name = "to_conllu_strs")]
    fn py_to_conllu_strings(&self) -> Vec<String> {
        self.to_conllu_strings()
    }

    /// Convert to a CoNLL-U object.
    #[pyo3(name = "to_conllu")]
    fn py_to_conllu(&self) -> crate::conllu::PyConllu {
        crate::conllu::PyConllu {
            inner: self.to_conllu(),
        }
    }

    /// Write CoNLL-U (.conllu) files to a directory.
    #[pyo3(name = "to_conllu_files")]
    #[pyo3(signature = (dir_path, /, *, filenames=None))]
    fn write_conllu(&self, dir_path: PathBuf, filenames: Option<Vec<String>>) -> PyResult<()> {
        let dir_path = pathbuf_to_string(dir_path)?;
        self.write_conllu_files(&dir_path, filenames)
            .map_err(|e| match e {
                WriteError::Validation(msg) => pyo3::exceptions::PyValueError::new_err(msg),
                WriteError::Io(err) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            })
    }

    fn __bool__(&self) -> bool {
        !self.is_empty()
    }

    fn __len__(&self) -> PyResult<usize> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "__len__ of a CHAT object is intentionally undefined. \
             Intuitively, there are different lengths one may refer to: \
             Number of files? Utterances? Words? Something else?",
        ))
    }

    fn __repr__(&self) -> String {
        format!("<CHAT with {} file(s)>", self.num_files())
    }

    fn __eq__(&self, other: &PyChat) -> bool {
        self.files().len() == other.files().len()
            && self
                .files()
                .iter()
                .zip(other.files())
                .all(|(a, b)| a.eq_data(b))
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.files().len().hash(&mut hasher);
        for f in self.files() {
            f.file_path.hash(&mut hasher);
            f.headers.hash_into(&mut hasher);
            f.events.len().hash(&mut hasher);
            for u in &f.events {
                u.hash_into(&mut hasher);
            }
            f.raw_lines.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Iterator for [`Chat`], yielding single-file `Chat` objects.
#[pyclass]
struct ChatIter {
    inner: VecDeque<ChatFile>,
    index: usize,
}

#[pymethods]
impl ChatIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyChat> {
        if self.index < self.inner.len() {
            let file = self.inner[self.index].clone();
            self.index += 1;
            Some(PyChat {
                inner: Chat::from_files(VecDeque::from(vec![file])),
            })
        } else {
            None
        }
    }
}
