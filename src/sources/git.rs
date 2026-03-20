use std::path::PathBuf;
use std::process::Command;

use super::SourceError;
use super::cache::{default_cache_dir, git_cache_path};

/// Clone a git repository and return the local path to the clone.
///
/// - `rev`: optional branch, tag, or commit hash.
/// - `depth`: clone depth (default 1 for shallow clone).
/// - `cache_dir`: override cache directory. If `None`, uses `~/.rustling/cache/`.
/// - `force`: if `true`, re-clone even if cached.
pub fn resolve_git(
    url: &str,
    rev: Option<&str>,
    depth: Option<u32>,
    cache_dir: Option<PathBuf>,
    force: bool,
) -> Result<PathBuf, SourceError> {
    let cache_base = match cache_dir {
        Some(dir) => dir,
        None => default_cache_dir()?,
    };
    let local_path = git_cache_path(url, rev, &cache_base)?;

    // Return cached path if it exists and force is false.
    if !force && local_path.exists() {
        return Ok(local_path);
    }

    // Remove stale cache entry if forcing.
    if local_path.exists() {
        std::fs::remove_dir_all(&local_path)
            .map_err(|e| SourceError::Io(format!("Failed to remove stale cache: {e}")))?;
    }

    // Check that git is available.
    Command::new("git")
        .arg("--version")
        .output()
        .map_err(|_| SourceError::GitNotFound)?;

    let depth = depth.unwrap_or(1);

    // Determine if rev looks like a commit hash (40 hex chars or 7+ hex prefix).
    let is_commit_hash = rev
        .is_some_and(|r| r.len() >= 7 && r.len() <= 40 && r.chars().all(|c| c.is_ascii_hexdigit()));

    if is_commit_hash {
        // For commit hashes: full clone (no depth limit), then checkout.
        let output = Command::new("git")
            .args(["clone", url])
            .arg(&local_path)
            .output()
            .map_err(|e| SourceError::Io(format!("Failed to run git clone: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SourceError::Git(format!("git clone failed: {stderr}")));
        }

        let rev = rev.unwrap();
        let output = Command::new("git")
            .args(["-C"])
            .arg(&local_path)
            .args(["checkout", rev])
            .output()
            .map_err(|e| SourceError::Io(format!("Failed to run git checkout: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SourceError::Git(format!(
                "git checkout {rev} failed: {stderr}"
            )));
        }
    } else {
        // For branches/tags or default: shallow clone with --branch if specified.
        let mut cmd = Command::new("git");
        cmd.args(["clone", "--depth", &depth.to_string()]);
        if let Some(rev) = rev {
            cmd.args(["--branch", rev]);
        }
        cmd.arg(url).arg(&local_path);

        let output = cmd
            .output()
            .map_err(|e| SourceError::Io(format!("Failed to run git clone: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SourceError::Git(format!("git clone failed: {stderr}")));
        }
    }

    Ok(local_path)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_is_commit_hash_detection() {
        // 7-char hex prefix should be treated as commit hash
        let rev = Some("abc1234");
        assert!(rev.is_some_and(|r| r.len() >= 7
            && r.len() <= 40
            && r.chars().all(|c| c.is_ascii_hexdigit())));

        // Full 40-char SHA
        let rev = Some("abc1234567890abc1234567890abc1234567890a");
        assert!(rev.is_some_and(|r| r.len() >= 7
            && r.len() <= 40
            && r.chars().all(|c| c.is_ascii_hexdigit())));

        // Branch name should not be treated as commit hash
        let rev = Some("main");
        assert!(!rev.is_some_and(|r| r.len() >= 7
            && r.len() <= 40
            && r.chars().all(|c| c.is_ascii_hexdigit())));

        // Short hex (< 7) should not be treated as commit hash
        let rev = Some("abc123");
        assert!(!rev.is_some_and(|r| r.len() >= 7
            && r.len() <= 40
            && r.chars().all(|c| c.is_ascii_hexdigit())));
    }
}
