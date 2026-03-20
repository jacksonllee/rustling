use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use super::SourceError;

/// Return the default cache directory: `~/.rustling/cache/`.
pub fn default_cache_dir() -> Result<PathBuf, SourceError> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| SourceError::Io("Cannot determine home directory".to_string()))?;
    Ok(PathBuf::from(home).join(".rustling").join("cache"))
}

/// Compute a deterministic cache key from the given components.
fn cache_key(components: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    for c in components {
        c.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Return the cache path for a git clone, creating parent dirs as needed.
///
/// Layout: `<cache_dir>/git/<hash>/`
pub fn git_cache_path(
    url: &str,
    rev: Option<&str>,
    cache_dir: &Path,
) -> Result<PathBuf, SourceError> {
    let rev_str = rev.unwrap_or("");
    let key = cache_key(&[url, rev_str]);
    let path = cache_dir.join("git").join(key);
    fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| SourceError::Io(format!("Failed to create cache directory: {e}")))?;
    Ok(path)
}

/// Return the cache path for a URL download, creating parent dirs as needed.
///
/// Layout: `<cache_dir>/url/<hash>`
pub fn url_cache_path(url: &str, cache_dir: &Path) -> Result<PathBuf, SourceError> {
    let key = cache_key(&[url]);
    let path = cache_dir.join("url").join(key);
    fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| SourceError::Io(format!("Failed to create cache directory: {e}")))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = cache_key(&["https://github.com/foo/bar.git", "main"]);
        let k2 = cache_key(&["https://github.com/foo/bar.git", "main"]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_differs_by_rev() {
        let k1 = cache_key(&["https://github.com/foo/bar.git", "main"]);
        let k2 = cache_key(&["https://github.com/foo/bar.git", "dev"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_differs_by_url() {
        let k1 = cache_key(&["https://github.com/foo/bar.git", ""]);
        let k2 = cache_key(&["https://github.com/foo/baz.git", ""]);
        assert_ne!(k1, k2);
    }
}
