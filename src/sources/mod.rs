//! Remote data source support (git repositories and URLs).
//!
//! This module provides shared utilities for downloading data from remote
//! sources. Format-specific classes (CHAT, ELAN, etc.) call [`resolve_git`]
//! and [`resolve_url`] to obtain a local path, then delegate to their own
//! `from_dir` / `from_zip` / `from_strs` methods.

pub mod cache;
pub mod git;
pub mod url;

pub use git::resolve_git;
pub use url::resolve_url;

/// Error type for remote source operations.
#[derive(Debug)]
pub enum SourceError {
    /// I/O error.
    Io(String),
    /// Git command not found on the system.
    GitNotFound,
    /// Git command failed.
    Git(String),
    /// HTTP request failed.
    Http(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::Io(msg) => write!(f, "I/O error: {msg}"),
            SourceError::GitNotFound => write!(
                f,
                "git is not installed or not found in PATH. \
                 Install git to use from_git()."
            ),
            SourceError::Git(msg) => write!(f, "Git error: {msg}"),
            SourceError::Http(msg) => write!(f, "HTTP error: {msg}"),
        }
    }
}

impl std::error::Error for SourceError {}

#[cfg(feature = "pyo3")]
impl From<SourceError> for pyo3::PyErr {
    fn from(e: SourceError) -> pyo3::PyErr {
        match e {
            SourceError::Io(msg) => pyo3::exceptions::PyIOError::new_err(msg),
            SourceError::GitNotFound => {
                pyo3::exceptions::PyEnvironmentError::new_err(e.to_string())
            }
            SourceError::Git(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
            SourceError::Http(msg) => pyo3::exceptions::PyConnectionError::new_err(msg),
        }
    }
}
