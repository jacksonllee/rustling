use std::io::{self, Write};
use std::path::PathBuf;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// ModelError
// ---------------------------------------------------------------------------

/// Unified error type for model persistence and related operations.
#[derive(Debug)]
pub enum ModelError {
    /// I/O error from the filesystem.
    Io(String),
    /// File not found.
    FileNotFound(String),
    /// Parse error (corrupted model file).
    ParseError(String),
    /// Validation error (invalid parameters).
    ValidationError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(msg) => write!(f, "I/O error: {msg}"),
            ModelError::FileNotFound(msg) => write!(f, "File not found: {msg}"),
            ModelError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            ModelError::ValidationError(msg) => write!(f, "Validation error: {msg}"),
        }
    }
}

impl std::error::Error for ModelError {}

#[cfg(feature = "pyo3")]
impl From<ModelError> for PyErr {
    fn from(e: ModelError) -> PyErr {
        match e {
            ModelError::Io(msg) => pyo3::exceptions::PyIOError::new_err(msg),
            ModelError::FileNotFound(msg) => pyo3::exceptions::PyFileNotFoundError::new_err(msg),
            ModelError::ParseError(msg) => pyo3::exceptions::PyEnvironmentError::new_err(msg),
            ModelError::ValidationError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read all bytes from a reader into a `Vec<u8>`.
pub(crate) fn read_all_bytes<R: io::Read>(mut reader: R) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// FlatBuffers [`VerifierOptions`] with elevated `max_tables`.
///
/// The HMM model has ~1.6 million [`VocabEntry`] tables; 20 million provides
/// a safe margin across all models in this crate.
///
/// [`VerifierOptions`]: flatbuffers::VerifierOptions
/// [`VocabEntry`]: crate::hmm::generated::rustling::hmm_fbs::VocabEntry
pub(crate) fn flatbuffers_verifier_opts() -> flatbuffers::VerifierOptions {
    flatbuffers::VerifierOptions {
        max_tables: 20_000_000,
        ..Default::default()
    }
}

/// Convert a [`PathBuf`] to a UTF-8 [`String`], or return a Python error.
#[cfg(feature = "pyo3")]
pub(crate) fn pathbuf_to_string(path: PathBuf) -> PyResult<String> {
    path.into_os_string().into_string().map_err(|os_str| {
        pyo3::exceptions::PyValueError::new_err(format!("Path is not valid UTF-8: {:?}", os_str))
    })
}

/// Convert a [`PathBuf`] to a UTF-8 [`String`], or return a [`ModelError`].
pub fn pathbuf_to_string_result(path: PathBuf) -> Result<String, ModelError> {
    path.into_os_string().into_string().map_err(|os_str| {
        ModelError::ValidationError(format!("Path is not valid UTF-8: {:?}", os_str))
    })
}

// ---------------------------------------------------------------------------
// Zstd save / load
// ---------------------------------------------------------------------------

/// Write `data` to `path` with zstd compression (level 19).
pub fn save_zstd(path: &str, data: &[u8]) -> Result<(), ModelError> {
    let file = std::fs::File::create(path)
        .map_err(|e| ModelError::Io(format!("Failed to create file: {e}")))?;
    let mut encoder = zstd::Encoder::new(file, 19)
        .map_err(|e| ModelError::Io(format!("Failed to create zstd encoder: {e}")))?;
    encoder
        .write_all(data)
        .map_err(|e| ModelError::Io(format!("Failed to write data: {e}")))?;
    encoder
        .finish()
        .map_err(|e| ModelError::Io(format!("Failed to finish zstd compression: {e}")))?;
    Ok(())
}

/// Read and decompress a zstd-compressed file, returning the raw bytes.
pub fn load_zstd(path: &str, model_desc: &str) -> Result<Vec<u8>, ModelError> {
    let file = std::fs::File::open(path)
        .map_err(|_| ModelError::FileNotFound(format!("Can't locate {model_desc} {path}")))?;
    let decoder = zstd::Decoder::new(file)
        .map_err(|e| ModelError::Io(format!("Failed to create zstd decoder: {e}")))?;
    read_all_bytes(decoder).map_err(|e| ModelError::Io(format!("Failed to read model: {e}")))
}
