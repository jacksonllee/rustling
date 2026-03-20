use std::fs;
use std::path::PathBuf;
use std::process::Command;

use super::SourceError;
use super::cache::{default_cache_dir, url_cache_path};

/// The magic bytes at the start of a ZIP file.
const ZIP_MAGIC: &[u8] = &[0x50, 0x4B, 0x03, 0x04];

/// Download a file from a URL using curl.
fn download(url: &str, dest: &PathBuf) -> Result<(), SourceError> {
    let output = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(dest)
        .arg(url)
        .output()
        .map_err(|e| SourceError::Http(format!("Failed to run curl (is it installed?): {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SourceError::Http(format!(
            "curl failed for {url}: {stderr}"
        )));
    }
    Ok(())
}

/// Download a file from a URL and return the local path.
///
/// - `cache_dir`: override cache directory. If `None`, uses `~/.rustling/cache/`.
/// - `force`: if `true`, re-download even if cached.
///
/// Returns `(local_path, is_zip)` where `is_zip` indicates whether the
/// downloaded file is a ZIP archive (detected by URL suffix or magic bytes).
pub fn resolve_url(
    url: &str,
    cache_dir: Option<PathBuf>,
    force: bool,
) -> Result<(PathBuf, bool), SourceError> {
    let cache_base = match cache_dir {
        Some(dir) => dir,
        None => default_cache_dir()?,
    };
    let local_path = url_cache_path(url, &cache_base)?;

    // Return cached path if it exists and force is false.
    if !force && local_path.exists() {
        let is_zip = detect_zip(&local_path, url)?;
        return Ok((local_path, is_zip));
    }

    // Download the file.
    download(url, &local_path)?;

    let is_zip = detect_zip(&local_path, url)?;
    Ok((local_path, is_zip))
}

/// Detect whether a cached file is a ZIP archive.
fn detect_zip(path: &PathBuf, url: &str) -> Result<bool, SourceError> {
    if url.to_lowercase().ends_with(".zip") {
        return Ok(true);
    }
    let bytes =
        fs::read(path).map_err(|e| SourceError::Io(format!("Failed to read cached file: {e}")))?;
    Ok(bytes.len() >= 4 && bytes[..4] == *ZIP_MAGIC)
}
