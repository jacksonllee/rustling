//! SRT (SubRip Subtitle) parsing.
//!
//! This module provides a parser for SRT subtitle files
//! and data structures for accessing subtitle blocks.

mod chat_writer;
mod elan_writer;
#[cfg(feature = "pyo3")]
mod reader_py;
mod textgrid_writer;

pub(crate) use reader::format_srt_time;
pub use reader::{BaseSrt, Srt, SrtBlock, SrtError, SrtFile, WriteError};
#[cfg(feature = "pyo3")]
pub use reader_py::{PySrt, PySrtBlock};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Register the srt submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let srt_module = PyModule::new(parent_module.py(), "srt")?;
    srt_module.add_class::<PySrt>()?;
    srt_module.add_class::<PySrtBlock>()?;
    parent_module.add_submodule(&srt_module)?;
    Ok(())
}

mod reader {
    use crate::chat::filter_file_paths;

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use std::collections::VecDeque;

    // -----------------------------------------------------------------------
    // Error types
    // -----------------------------------------------------------------------

    /// Errors that can occur when reading or parsing SRT data.
    #[derive(Debug)]
    pub enum SrtError {
        /// An I/O error occurred.
        Io(std::io::Error),
        /// A parse error occurred.
        Parse(String),
        /// An invalid regex pattern was provided.
        InvalidPattern(String),
        /// An error occurred reading a ZIP archive.
        Zip(String),
        /// A remote source error occurred (git clone, HTTP download, etc.).
        Source(crate::sources::SourceError),
    }

    impl std::fmt::Display for SrtError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SrtError::Io(e) => write!(f, "{e}"),
                SrtError::Parse(e) => write!(f, "SRT parse error: {e}"),
                SrtError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
                SrtError::Zip(e) => write!(f, "{e}"),
                SrtError::Source(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for SrtError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                SrtError::Io(e) => Some(e),
                SrtError::Source(e) => Some(e),
                _ => None,
            }
        }
    }

    impl From<std::io::Error> for SrtError {
        fn from(e: std::io::Error) -> Self {
            SrtError::Io(e)
        }
    }

    impl From<crate::sources::SourceError> for SrtError {
        fn from(e: crate::sources::SourceError) -> Self {
            SrtError::Source(e)
        }
    }

    /// Error type for [`BaseSrt::write_files`].
    #[derive(Debug)]
    pub enum WriteError {
        /// Validation error (e.g., wrong number of filenames).
        Validation(String),
        /// I/O error from the filesystem.
        Io(std::io::Error),
    }

    // -----------------------------------------------------------------------
    // Domain types
    // -----------------------------------------------------------------------

    /// A single subtitle block within an SRT file.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SrtBlock {
        /// 1-based sequence number from the SRT file.
        pub index: usize,
        /// The subtitle text (multiline text joined with `\n`).
        pub text: String,
        /// Start time in milliseconds.
        pub start_ms: i64,
        /// End time in milliseconds.
        pub end_ms: i64,
    }

    /// A single parsed SRT file.
    #[derive(Debug, Clone)]
    pub struct SrtFile {
        /// File path or identifier.
        pub file_path: String,
        /// Subtitle blocks in this file.
        pub blocks: Vec<SrtBlock>,
    }

    // -----------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------

    /// Parse an SRT time string (`HH:MM:SS,mmm` or `HH:MM:SS.mmm`) to milliseconds.
    fn parse_srt_time(s: &str) -> Result<i64, SrtError> {
        let s = s.trim();
        // Accept both comma and period as millisecond separator.
        let s = s.replace(',', ".");
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(SrtError::Parse(format!("Invalid time format: {s}")));
        }
        let hours: i64 = parts[0]
            .parse()
            .map_err(|_| SrtError::Parse(format!("Invalid hours in time: {s}")))?;
        let minutes: i64 = parts[1]
            .parse()
            .map_err(|_| SrtError::Parse(format!("Invalid minutes in time: {s}")))?;
        let sec_parts: Vec<&str> = parts[2].split('.').collect();
        let seconds: i64 = sec_parts[0]
            .parse()
            .map_err(|_| SrtError::Parse(format!("Invalid seconds in time: {s}")))?;
        let millis: i64 = if sec_parts.len() > 1 {
            let ms_str = sec_parts[1];
            // Pad or truncate to 3 digits.
            let padded = format!("{ms_str:0<3}");
            padded[..3]
                .parse()
                .map_err(|_| SrtError::Parse(format!("Invalid milliseconds in time: {s}")))?
        } else {
            0
        };
        Ok(hours * 3_600_000 + minutes * 60_000 + seconds * 1_000 + millis)
    }

    /// Format milliseconds as an SRT time string (`HH:MM:SS,mmm`).
    pub(crate) fn format_srt_time(ms: i64) -> String {
        let total_seconds = ms / 1000;
        let millis = ms % 1000;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
    }

    /// Parse a single SRT string into an [`SrtFile`].
    pub fn parse_srt_str(content: &str, file_path: String) -> Result<SrtFile, SrtError> {
        // Strip BOM if present.
        let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);
        // Normalize line endings.
        let content = content.replace("\r\n", "\n").replace('\r', "\n");

        let mut blocks = Vec::new();

        // Split on blank lines (two or more newlines).
        for chunk in content.split("\n\n") {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }

            let mut lines = chunk.lines();

            // First line: sequence number.
            let index_line = match lines.next() {
                Some(line) => line.trim(),
                None => continue,
            };
            let index: usize = match index_line.parse() {
                Ok(n) => n,
                Err(_) => {
                    return Err(SrtError::Parse(format!(
                        "Expected sequence number, got: {index_line:?}"
                    )));
                }
            };

            // Second line: time range.
            let time_line = match lines.next() {
                Some(line) => line.trim(),
                None => {
                    return Err(SrtError::Parse(format!(
                        "Missing time range for subtitle {index}"
                    )));
                }
            };
            let arrow_pos = time_line.find("-->").ok_or_else(|| {
                SrtError::Parse(format!("Missing '-->' in time range: {time_line:?}"))
            })?;
            let start_str = &time_line[..arrow_pos];
            let end_str = &time_line[arrow_pos + 3..];
            let start_ms = parse_srt_time(start_str)?;
            let end_ms = parse_srt_time(end_str)?;

            // Remaining lines: subtitle text.
            let text: String = lines.collect::<Vec<_>>().join("\n");
            if text.is_empty() {
                continue;
            }

            blocks.push(SrtBlock {
                index,
                text,
                start_ms,
                end_ms,
            });
        }

        Ok(SrtFile { file_path, blocks })
    }

    // -----------------------------------------------------------------------
    // Batch parsing helpers
    // -----------------------------------------------------------------------

    fn parse_srt_strs(
        pairs: Vec<(String, String)>,
        parallel: bool,
    ) -> Result<Vec<SrtFile>, SrtError> {
        let parse_one = |(content, id): (String, String)| -> Result<SrtFile, SrtError> {
            parse_srt_str(&content, id)
        };

        if parallel {
            #[cfg(feature = "parallel")]
            {
                pairs
                    .into_par_iter()
                    .map(parse_one)
                    .collect::<Result<Vec<_>, _>>()
            }
            #[cfg(not(feature = "parallel"))]
            {
                pairs.into_iter().map(parse_one).collect()
            }
        } else {
            pairs.into_iter().map(parse_one).collect()
        }
    }

    fn load_srt_files(paths: &[String], parallel: bool) -> Result<Vec<SrtFile>, SrtError> {
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(paths.len());
        for path in paths {
            let content = std::fs::read_to_string(path)?;
            pairs.push((content, path.clone()));
        }
        parse_srt_strs(pairs, parallel)
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize an [`SrtFile`] back to an SRT string.
    pub fn serialize_srt_file(file: &SrtFile) -> String {
        let mut output = String::with_capacity(1024);
        for (i, block) in file.blocks.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            output.push_str(&format!("{}\n", block.index));
            output.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(block.start_ms),
                format_srt_time(block.end_ms),
            ));
            output.push_str(&block.text);
            output.push('\n');
        }
        output
    }

    // -----------------------------------------------------------------------
    // BaseSrt trait
    // -----------------------------------------------------------------------

    /// Core SRT reader behavior with default implementations.
    pub trait BaseSrt: Sized {
        fn files(&self) -> &VecDeque<SrtFile>;
        fn files_mut(&mut self) -> &mut VecDeque<SrtFile>;
        fn from_files(files: VecDeque<SrtFile>) -> Self;

        /// Number of loaded files.
        fn num_files(&self) -> usize {
            self.files().len()
        }

        /// Whether the reader contains no files.
        fn is_empty(&self) -> bool {
            self.files().is_empty()
        }

        /// Return the file paths.
        fn file_paths(&self) -> Vec<String> {
            self.files().iter().map(|f| f.file_path.clone()).collect()
        }

        /// Return all subtitle blocks across all files (flat).
        fn utterances(&self) -> Vec<&SrtBlock> {
            self.files().iter().flat_map(|f| &f.blocks).collect()
        }

        /// Return SRT strings, one per file.
        fn to_strings(&self) -> Vec<String> {
            self.files().iter().map(serialize_srt_file).collect()
        }

        /// Derive default output filenames from existing file paths.
        fn default_output_filenames(&self, target_ext: &str) -> Vec<String> {
            let derived: Vec<Option<String>> = self
                .files()
                .iter()
                .map(|f| {
                    let path = std::path::Path::new(&f.file_path);
                    let stem = path.file_stem()?.to_str()?;
                    if uuid::Uuid::try_parse(stem).is_ok() {
                        return None;
                    }
                    Some(format!("{stem}{target_ext}"))
                })
                .collect();

            if derived.iter().all(|d| d.is_some()) {
                let names: Vec<String> = derived.into_iter().map(|d| d.unwrap()).collect();
                let unique: std::collections::HashSet<&String> = names.iter().collect();
                if unique.len() == names.len() {
                    return names;
                }
            }

            (0..self.files().len())
                .map(|i| format!("{:04}{target_ext}", i + 1))
                .collect()
        }

        /// Write SRT files to a directory.
        fn write_srt_files(
            &self,
            dir_path: &str,
            filenames: Option<Vec<String>>,
        ) -> Result<(), WriteError> {
            let strs = self.to_strings();
            let dir = std::path::Path::new(dir_path);
            std::fs::create_dir_all(dir).map_err(WriteError::Io)?;

            let names: Vec<String> = match filenames {
                Some(names) => {
                    if names.len() != self.files().len() {
                        return Err(WriteError::Validation(format!(
                            "There are {} SRT files to create, \
                             but {} filenames were provided.",
                            self.files().len(),
                            names.len()
                        )));
                    }
                    names
                }
                None => self.default_output_filenames(".srt"),
            };

            for (name, content) in names.iter().zip(strs.iter()) {
                let file_path = dir.join(name);
                std::fs::write(&file_path, content).map_err(WriteError::Io)?;
            }
            Ok(())
        }

        // -------------------------------------------------------------------
        // CHAT conversion
        // -------------------------------------------------------------------

        /// Return CHAT format strings (one per file) for CHAT export.
        fn to_chat_strings(&self) -> Vec<String> {
            self.files()
                .iter()
                .map(super::chat_writer::srt_file_to_chat_str)
                .collect()
        }

        /// Convert to a [`Chat`](crate::chat::Chat) object.
        fn to_chat_obj(&self) -> crate::chat::Chat {
            let strs = self.to_chat_strings();
            let ids: Vec<String> = self
                .files()
                .iter()
                .map(|f| {
                    let path = std::path::Path::new(&f.file_path);
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if uuid::Uuid::try_parse(stem).is_ok() {
                        f.file_path.clone()
                    } else {
                        format!("{stem}.cha")
                    }
                })
                .collect();
            let (chat, _) = crate::chat::Chat::from_strs(strs, Some(ids), false, None, None);
            chat
        }

        /// Write CHAT (.cha) files to a directory.
        fn write_chat_files(
            &self,
            dir_path: &str,
            filenames: Option<Vec<String>>,
        ) -> Result<(), WriteError> {
            let strs = self.to_chat_strings();
            let dir = std::path::Path::new(dir_path);
            std::fs::create_dir_all(dir).map_err(WriteError::Io)?;

            let names: Vec<String> = match filenames {
                Some(names) => {
                    if names.len() != self.files().len() {
                        return Err(WriteError::Validation(format!(
                            "There are {} CHAT files to create, \
                             but {} filenames were provided.",
                            self.files().len(),
                            names.len()
                        )));
                    }
                    names
                }
                None => self.default_output_filenames(".cha"),
            };

            for (name, content) in names.iter().zip(strs.iter()) {
                let file_path = dir.join(name);
                std::fs::write(&file_path, content).map_err(WriteError::Io)?;
            }
            Ok(())
        }

        // -------------------------------------------------------------------
        // ELAN conversion
        // -------------------------------------------------------------------

        /// Return EAF XML strings (one per file) for ELAN export.
        fn to_elan_strings(&self) -> Vec<String> {
            self.files()
                .iter()
                .map(super::elan_writer::srt_file_to_eaf_xml)
                .collect()
        }

        /// Convert to an [`Elan`](crate::elan::Elan) object.
        fn to_elan(&self) -> crate::elan::Elan {
            let strs = self.to_elan_strings();
            let ids: Vec<String> = self
                .files()
                .iter()
                .map(|f| {
                    let path = std::path::Path::new(&f.file_path);
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if uuid::Uuid::try_parse(stem).is_ok() {
                        f.file_path.clone()
                    } else {
                        format!("{stem}.eaf")
                    }
                })
                .collect();
            crate::elan::Elan::from_strs(strs, Some(ids), false).unwrap()
        }

        /// Write ELAN (.eaf) files to a directory.
        fn write_elan_files(
            &self,
            dir_path: &str,
            filenames: Option<Vec<String>>,
        ) -> Result<(), WriteError> {
            let strs = self.to_elan_strings();
            let dir = std::path::Path::new(dir_path);
            std::fs::create_dir_all(dir).map_err(WriteError::Io)?;

            let names: Vec<String> = match filenames {
                Some(names) => {
                    if names.len() != self.files().len() {
                        return Err(WriteError::Validation(format!(
                            "There are {} ELAN files to create, \
                             but {} filenames were provided.",
                            self.files().len(),
                            names.len()
                        )));
                    }
                    names
                }
                None => self.default_output_filenames(".eaf"),
            };

            for (name, content) in names.iter().zip(strs.iter()) {
                let file_path = dir.join(name);
                std::fs::write(&file_path, content).map_err(WriteError::Io)?;
            }
            Ok(())
        }

        // -------------------------------------------------------------------
        // TextGrid conversion
        // -------------------------------------------------------------------

        /// Return TextGrid format strings (one per file) for TextGrid export.
        fn to_textgrid_strings(&self) -> Vec<String> {
            self.files()
                .iter()
                .map(super::textgrid_writer::srt_file_to_textgrid_str)
                .collect()
        }

        /// Convert to a [`TextGrid`](crate::textgrid::TextGrid) object.
        fn to_textgrid(&self) -> crate::textgrid::TextGrid {
            let strs = self.to_textgrid_strings();
            let ids: Vec<String> = self
                .files()
                .iter()
                .map(|f| {
                    let path = std::path::Path::new(&f.file_path);
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if uuid::Uuid::try_parse(stem).is_ok() {
                        f.file_path.clone()
                    } else {
                        format!("{stem}.TextGrid")
                    }
                })
                .collect();
            crate::textgrid::TextGrid::from_strs(strs, Some(ids), false).unwrap()
        }

        /// Write TextGrid (.TextGrid) files to a directory.
        fn write_textgrid_files(
            &self,
            dir_path: &str,
            filenames: Option<Vec<String>>,
        ) -> Result<(), WriteError> {
            let strs = self.to_textgrid_strings();
            let dir = std::path::Path::new(dir_path);
            std::fs::create_dir_all(dir).map_err(WriteError::Io)?;

            let names: Vec<String> = match filenames {
                Some(names) => {
                    if names.len() != self.files().len() {
                        return Err(WriteError::Validation(format!(
                            "There are {} TextGrid files to create, \
                             but {} filenames were provided.",
                            self.files().len(),
                            names.len()
                        )));
                    }
                    names
                }
                None => self.default_output_filenames(".TextGrid"),
            };

            for (name, content) in names.iter().zip(strs.iter()) {
                let file_path = dir.join(name);
                std::fs::write(&file_path, content).map_err(WriteError::Io)?;
            }
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Srt struct
    // -----------------------------------------------------------------------

    /// SRT data reader.
    ///
    /// This is a pure Rust struct. For the Python-exposed wrapper, see `PySrt`.
    #[derive(Clone, Debug)]
    pub struct Srt {
        pub(crate) files: VecDeque<SrtFile>,
    }

    impl BaseSrt for Srt {
        fn files(&self) -> &VecDeque<SrtFile> {
            &self.files
        }
        fn files_mut(&mut self) -> &mut VecDeque<SrtFile> {
            &mut self.files
        }
        fn from_files(files: VecDeque<SrtFile>) -> Self {
            Self { files }
        }
    }

    impl Srt {
        /// Construct from a Vec of [`SrtFile`] entries.
        pub fn from_srt_files(files: Vec<SrtFile>) -> Self {
            Self {
                files: VecDeque::from(files),
            }
        }

        /// Append data from another Srt.
        pub fn push_back(&mut self, other: &Srt) {
            self.files.extend(other.files.iter().cloned());
        }

        /// Prepend data from another Srt.
        pub fn push_front(&mut self, other: &Srt) {
            let mut new_files = other.files.clone();
            new_files.extend(std::mem::take(&mut self.files));
            self.files = new_files;
        }

        /// Remove and return the last file as a new Srt.
        pub fn pop_back(&mut self) -> Option<Srt> {
            self.files
                .pop_back()
                .map(|f| Srt::from_files(VecDeque::from(vec![f])))
        }

        /// Remove and return the first file as a new Srt.
        pub fn pop_front(&mut self) -> Option<Srt> {
            self.files
                .pop_front()
                .map(|f| Srt::from_files(VecDeque::from(vec![f])))
        }

        /// Parse SRT data from in-memory strings.
        pub fn from_strs(
            strs: Vec<String>,
            ids: Option<Vec<String>>,
            parallel: bool,
        ) -> Result<Self, SrtError> {
            let ids = ids.unwrap_or_else(|| {
                strs.iter()
                    .map(|_| uuid::Uuid::new_v4().to_string())
                    .collect()
            });
            assert_eq!(
                strs.len(),
                ids.len(),
                "strs and ids must have the same length: {} vs {}",
                strs.len(),
                ids.len()
            );
            let pairs: Vec<(String, String)> = strs.into_iter().zip(ids).collect();
            let files = parse_srt_strs(pairs, parallel)?;
            Ok(Self::from_srt_files(files))
        }

        /// Load and parse SRT data from file paths.
        pub fn read_files(paths: &[String], parallel: bool) -> Result<Self, SrtError> {
            let files = load_srt_files(paths, parallel)?;
            Ok(Self::from_srt_files(files))
        }

        /// Recursively load SRT data from a directory.
        pub fn read_dir(
            path: &str,
            match_pattern: Option<&str>,
            extension: &str,
            parallel: bool,
        ) -> Result<Self, SrtError> {
            let mut paths: Vec<String> = Vec::new();
            for entry in walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    let file_path = entry.path().to_string_lossy().to_string();
                    if file_path.ends_with(extension) {
                        paths.push(file_path);
                    }
                }
            }
            paths.sort();

            let filtered = filter_file_paths(&paths, match_pattern)
                .map_err(|e| SrtError::InvalidPattern(e.to_string()))?;
            let files = load_srt_files(&filtered, parallel)?;
            Ok(Self::from_srt_files(files))
        }

        /// Load SRT data from a ZIP archive.
        pub fn read_zip(
            path: &str,
            match_pattern: Option<&str>,
            extension: &str,
            parallel: bool,
        ) -> Result<Self, SrtError> {
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file)
                .map_err(|e| SrtError::Zip(format!("Invalid zip file: {e}")))?;

            let mut entry_names: Vec<String> = (0..archive.len())
                .filter_map(|i| {
                    let entry = archive.by_index(i).ok()?;
                    let name = entry.name().to_string();
                    if name.ends_with(extension) && !entry.is_dir() {
                        Some(name)
                    } else {
                        None
                    }
                })
                .collect();
            entry_names.sort();

            let filtered = filter_file_paths(&entry_names, match_pattern)
                .map_err(|e| SrtError::InvalidPattern(e.to_string()))?;

            let mut pairs: Vec<(String, String)> = Vec::new();
            for name in &filtered {
                let mut entry = archive
                    .by_name(name)
                    .map_err(|e| SrtError::Zip(format!("Zip entry error: {e}")))?;
                let mut content = String::new();
                std::io::Read::read_to_string(&mut entry, &mut content)
                    .map_err(|e| SrtError::Zip(format!("Read error: {e}")))?;
                pairs.push((content, name.clone()));
            }

            let files = parse_srt_strs(pairs, parallel)?;
            Ok(Self::from_srt_files(files))
        }

        /// Load SRT data from a git repository.
        ///
        /// Clones the repository (or uses a cached clone) and parses all
        /// matching files from the resulting directory.
        #[allow(clippy::too_many_arguments)]
        pub fn from_git(
            url: &str,
            rev: Option<&str>,
            depth: Option<u32>,
            match_pattern: Option<&str>,
            extension: &str,
            cache_dir: Option<std::path::PathBuf>,
            force_download: bool,
            parallel: bool,
        ) -> Result<Self, SrtError> {
            let local_path =
                crate::sources::resolve_git(url, rev, depth, cache_dir, force_download)?;
            let path = local_path.to_string_lossy();
            Self::read_dir(&path, match_pattern, extension, parallel)
        }

        /// Load SRT data from a URL.
        ///
        /// Downloads the file (or uses a cached copy) and parses it.
        /// ZIP files are automatically detected by URL suffix or magic bytes.
        pub fn from_url(
            url: &str,
            match_pattern: Option<&str>,
            extension: &str,
            cache_dir: Option<std::path::PathBuf>,
            force_download: bool,
            parallel: bool,
        ) -> Result<Self, SrtError> {
            let (local_path, is_zip) = crate::sources::resolve_url(url, cache_dir, force_download)?;
            let path = local_path.to_string_lossy();
            if is_zip {
                Self::read_zip(&path, match_pattern, extension, parallel)
            } else {
                let content = std::fs::read_to_string(local_path)?;
                Self::from_strs(vec![content], None, parallel)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sample_srt() -> &'static str {
            "1\n\
             00:02:16,612 --> 00:02:19,376\n\
             Senator, we're making\n\
             our final approach into Coruscant.\n\
             \n\
             2\n\
             00:02:19,482 --> 00:02:21,609\n\
             Very good, Lieutenant.\n"
        }

        #[test]
        fn test_parse_basic() {
            let file = parse_srt_str(sample_srt(), "test.srt".to_string()).unwrap();
            assert_eq!(file.file_path, "test.srt");
            assert_eq!(file.blocks.len(), 2);

            let b1 = &file.blocks[0];
            assert_eq!(b1.index, 1);
            assert_eq!(
                b1.text,
                "Senator, we're making\nour final approach into Coruscant."
            );
            assert_eq!(b1.start_ms, 136612); // 2*60000 + 16*1000 + 612
            assert_eq!(b1.end_ms, 139376); // 2*60000 + 19*1000 + 376

            let b2 = &file.blocks[1];
            assert_eq!(b2.index, 2);
            assert_eq!(b2.text, "Very good, Lieutenant.");
            assert_eq!(b2.start_ms, 139482);
            assert_eq!(b2.end_ms, 141609);
        }

        #[test]
        fn test_parse_with_bom() {
            let srt = format!("\u{FEFF}{}", sample_srt());
            let file = parse_srt_str(&srt, "bom.srt".to_string()).unwrap();
            assert_eq!(file.blocks.len(), 2);
            assert_eq!(file.blocks[0].index, 1);
        }

        #[test]
        fn test_parse_windows_line_endings() {
            let srt = "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello world.\r\n\r\n\
                        2\r\n00:00:03,000 --> 00:00:04,000\r\nGoodbye.\r\n";
            let file = parse_srt_str(srt, "win.srt".to_string()).unwrap();
            assert_eq!(file.blocks.len(), 2);
            assert_eq!(file.blocks[0].text, "Hello world.");
            assert_eq!(file.blocks[1].text, "Goodbye.");
        }

        #[test]
        fn test_parse_period_separator() {
            let srt = "1\n00:00:01.500 --> 00:00:02.750\nHello.\n";
            let file = parse_srt_str(srt, "period.srt".to_string()).unwrap();
            assert_eq!(file.blocks.len(), 1);
            assert_eq!(file.blocks[0].start_ms, 1500);
            assert_eq!(file.blocks[0].end_ms, 2750);
        }

        #[test]
        fn test_parse_empty() {
            let file = parse_srt_str("", "empty.srt".to_string()).unwrap();
            assert_eq!(file.blocks.len(), 0);
        }

        #[test]
        fn test_parse_error_bad_index() {
            let srt = "abc\n00:00:01,000 --> 00:00:02,000\nHello.\n";
            let result = parse_srt_str(srt, "bad.srt".to_string());
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), SrtError::Parse(_)));
        }

        #[test]
        fn test_parse_error_missing_arrow() {
            let srt = "1\n00:00:01,000 00:00:02,000\nHello.\n";
            let result = parse_srt_str(srt, "bad.srt".to_string());
            assert!(result.is_err());
        }

        #[test]
        fn test_format_srt_time() {
            assert_eq!(format_srt_time(0), "00:00:00,000");
            assert_eq!(format_srt_time(1500), "00:00:01,500");
            assert_eq!(format_srt_time(136612), "00:02:16,612");
            assert_eq!(format_srt_time(3661001), "01:01:01,001");
        }

        #[test]
        fn test_serialize_round_trip() {
            let file = parse_srt_str(sample_srt(), "test.srt".to_string()).unwrap();
            let output = serialize_srt_file(&file);
            let file2 = parse_srt_str(&output, "test.srt".to_string()).unwrap();
            assert_eq!(file.blocks, file2.blocks);
        }

        #[test]
        fn test_srt_from_strs() {
            let srt = Srt::from_strs(vec![sample_srt().to_string()], None, false).unwrap();
            assert_eq!(srt.num_files(), 1);
            assert_eq!(srt.utterances().len(), 2);
        }

        #[test]
        fn test_srt_base_trait() {
            let srt = Srt::from_strs(
                vec![sample_srt().to_string(), sample_srt().to_string()],
                Some(vec!["file1.srt".to_string(), "file2.srt".to_string()]),
                false,
            )
            .unwrap();
            assert_eq!(srt.num_files(), 2);
            assert_eq!(srt.file_paths(), vec!["file1.srt", "file2.srt"]);
            assert!(!srt.is_empty());
            assert_eq!(srt.utterances().len(), 4);
        }

        #[test]
        fn test_srt_push_pop() {
            let mut srt = Srt::from_strs(
                vec![sample_srt().to_string()],
                Some(vec!["file1.srt".to_string()]),
                false,
            )
            .unwrap();
            let srt2 = Srt::from_strs(
                vec![sample_srt().to_string()],
                Some(vec!["file2.srt".to_string()]),
                false,
            )
            .unwrap();

            srt.push_back(&srt2);
            assert_eq!(srt.num_files(), 2);
            assert_eq!(srt.file_paths(), vec!["file1.srt", "file2.srt"]);

            let popped = srt.pop_back().unwrap();
            assert_eq!(popped.file_paths(), vec!["file2.srt"]);
            assert_eq!(srt.num_files(), 1);

            srt.push_front(&srt2);
            assert_eq!(srt.file_paths(), vec!["file2.srt", "file1.srt"]);

            let popped = srt.pop_front().unwrap();
            assert_eq!(popped.file_paths(), vec!["file2.srt"]);
        }

        #[test]
        fn test_write_srt_files() {
            let srt = Srt::from_strs(
                vec![sample_srt().to_string()],
                Some(vec!["test.srt".to_string()]),
                false,
            )
            .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let out_dir = dir.path().join("output");
            srt.write_srt_files(out_dir.to_str().unwrap(), None)
                .unwrap();
            let content = std::fs::read_to_string(out_dir.join("test.srt")).unwrap();
            let file2 = parse_srt_str(&content, "test.srt".to_string()).unwrap();
            assert_eq!(srt.files()[0].blocks, file2.blocks);
        }

        #[test]
        fn test_write_srt_files_validation() {
            let srt = Srt::from_strs(
                vec![sample_srt().to_string(), sample_srt().to_string()],
                Some(vec!["f1.srt".to_string(), "f2.srt".to_string()]),
                false,
            )
            .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let result = srt.write_srt_files(
                dir.path().to_str().unwrap(),
                Some(vec!["only_one.srt".to_string()]),
            );
            assert!(matches!(result, Err(WriteError::Validation(_))));
        }

        #[test]
        fn test_to_strings_round_trip() {
            let srt = Srt::from_strs(
                vec![sample_srt().to_string()],
                Some(vec!["test.srt".to_string()]),
                false,
            )
            .unwrap();
            let strs = srt.to_strings();
            let srt2 = Srt::from_strs(strs, Some(vec!["test.srt".to_string()]), false).unwrap();
            assert_eq!(srt.files()[0].blocks, srt2.files()[0].blocks);
        }
    }
}
