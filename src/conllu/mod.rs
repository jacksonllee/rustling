//! CoNLL-U (Universal Dependencies) parsing.
//!
//! This module provides a parser for CoNLL-U format files
//! and data structures for accessing sentences and tokens.

mod chat_writer;
#[cfg(feature = "pyo3")]
mod reader_py;

pub use reader::{BaseConllu, Conllu, ConlluError, ConlluFile, ConlluToken, Sentence, WriteError};
#[cfg(feature = "pyo3")]
pub use reader_py::{PyConllu, PyConlluToken, PySentence};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Register the conllu submodule with Python.
#[cfg(feature = "pyo3")]
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let conllu_module = PyModule::new(parent_module.py(), "conllu")?;
    conllu_module.add_class::<PyConllu>()?;
    conllu_module.add_class::<PySentence>()?;
    conllu_module.add_class::<PyConlluToken>()?;
    parent_module.add_submodule(&conllu_module)?;
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

    /// Errors that can occur when reading or parsing CoNLL-U data.
    #[derive(Debug)]
    pub enum ConlluError {
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

    impl std::fmt::Display for ConlluError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ConlluError::Io(e) => write!(f, "{e}"),
                ConlluError::Parse(e) => write!(f, "CoNLL-U parse error: {e}"),
                ConlluError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
                ConlluError::Zip(e) => write!(f, "{e}"),
                ConlluError::Source(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for ConlluError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                ConlluError::Io(e) => Some(e),
                ConlluError::Source(e) => Some(e),
                _ => None,
            }
        }
    }

    impl From<std::io::Error> for ConlluError {
        fn from(e: std::io::Error) -> Self {
            ConlluError::Io(e)
        }
    }

    impl From<crate::sources::SourceError> for ConlluError {
        fn from(e: crate::sources::SourceError) -> Self {
            ConlluError::Source(e)
        }
    }

    /// Error type for write operations.
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

    /// A single token from a CoNLL-U file (one line with 10 tab-separated fields).
    #[derive(Debug, Clone, PartialEq, Hash)]
    pub struct ConlluToken {
        /// Word index (integer, range like "1-2", or decimal like "1.1").
        pub id: String,
        /// Word form or punctuation symbol.
        pub form: String,
        /// Lemma or stem of the word.
        pub lemma: String,
        /// Universal POS tag.
        pub upos: String,
        /// Language-specific POS tag, or `_`.
        pub xpos: String,
        /// Morphological features, or `_`.
        pub feats: String,
        /// Head of the current word (ID or `0` for root), or `_`.
        pub head: String,
        /// Universal dependency relation to HEAD, or `_`.
        pub deprel: String,
        /// Enhanced dependency graph, or `_`.
        pub deps: String,
        /// Any other annotation, or `_`.
        pub misc: String,
    }

    impl ConlluToken {
        /// Whether this token has a multiword range ID (e.g., "1-2").
        pub fn is_multiword(&self) -> bool {
            self.id.contains('-')
        }

        /// Whether this token is an empty node (e.g., "1.1").
        pub fn is_empty_node(&self) -> bool {
            self.id.contains('.')
        }
    }

    /// A single sentence from a CoNLL-U file.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Sentence {
        /// Comment lines (without the leading `# `), or `None` if there are none.
        pub comments: Option<Vec<String>>,
        /// Tokens in this sentence.
        pub tokens: Vec<ConlluToken>,
    }

    /// A single parsed CoNLL-U file.
    #[derive(Debug, Clone)]
    pub struct ConlluFile {
        /// File path or identifier.
        pub file_path: String,
        /// Sentences in this file.
        pub sentences: Vec<Sentence>,
    }

    // -----------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------

    /// Parse a single CoNLL-U string into a [`ConlluFile`].
    pub fn parse_conllu_str(content: &str, file_path: String) -> Result<ConlluFile, ConlluError> {
        // Strip BOM if present.
        let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);
        // Normalize line endings.
        let content = content.replace("\r\n", "\n").replace('\r', "\n");

        let mut sentences = Vec::new();
        let mut comments: Vec<String> = Vec::new();
        let mut tokens: Vec<ConlluToken> = Vec::new();

        for line in content.lines() {
            let line = line.trim_end();

            if line.is_empty() {
                // Blank line: end of sentence.
                if !tokens.is_empty() {
                    sentences.push(Sentence {
                        comments: if comments.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut comments))
                        },
                        tokens: std::mem::take(&mut tokens),
                    });
                } else if !comments.is_empty() {
                    // Comments with no tokens (unusual but possible).
                    sentences.push(Sentence {
                        comments: Some(std::mem::take(&mut comments)),
                        tokens: Vec::new(),
                    });
                }
                continue;
            }

            if let Some(comment) = line.strip_prefix('#') {
                // Comment line: strip the leading `# ` or `#`.
                let comment = comment.strip_prefix(' ').unwrap_or(comment);
                comments.push(comment.to_string());
                continue;
            }

            // Token line: 10 tab-separated fields.
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 10 {
                return Err(ConlluError::Parse(format!(
                    "Expected 10 tab-separated fields, got {} in line: {line:?}",
                    fields.len()
                )));
            }
            tokens.push(ConlluToken {
                id: fields[0].to_string(),
                form: fields[1].to_string(),
                lemma: fields[2].to_string(),
                upos: fields[3].to_string(),
                xpos: fields[4].to_string(),
                feats: fields[5].to_string(),
                head: fields[6].to_string(),
                deprel: fields[7].to_string(),
                deps: fields[8].to_string(),
                misc: fields[9].to_string(),
            });
        }

        // Handle final sentence (no trailing blank line).
        if !tokens.is_empty() || !comments.is_empty() {
            sentences.push(Sentence {
                comments: if comments.is_empty() {
                    None
                } else {
                    Some(comments)
                },
                tokens,
            });
        }

        Ok(ConlluFile {
            file_path,
            sentences,
        })
    }

    /// Serialize a [`ConlluFile`] back to a CoNLL-U string.
    pub fn serialize_conllu_file(file: &ConlluFile) -> String {
        let mut output = String::with_capacity(4096);
        for (i, sentence) in file.sentences.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            if let Some(comments) = &sentence.comments {
                for comment in comments {
                    output.push_str("# ");
                    output.push_str(comment);
                    output.push('\n');
                }
            }
            for token in &sentence.tokens {
                output.push_str(&token.id);
                output.push('\t');
                output.push_str(&token.form);
                output.push('\t');
                output.push_str(&token.lemma);
                output.push('\t');
                output.push_str(&token.upos);
                output.push('\t');
                output.push_str(&token.xpos);
                output.push('\t');
                output.push_str(&token.feats);
                output.push('\t');
                output.push_str(&token.head);
                output.push('\t');
                output.push_str(&token.deprel);
                output.push('\t');
                output.push_str(&token.deps);
                output.push('\t');
                output.push_str(&token.misc);
                output.push('\n');
            }
        }
        output
    }

    // -----------------------------------------------------------------------
    // Batch parsing helpers
    // -----------------------------------------------------------------------

    fn parse_conllu_strs(
        pairs: Vec<(String, String)>,
        parallel: bool,
    ) -> Result<Vec<ConlluFile>, ConlluError> {
        let parse_one = |(content, id): (String, String)| -> Result<ConlluFile, ConlluError> {
            parse_conllu_str(&content, id)
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

    fn load_conllu_files(paths: &[String], parallel: bool) -> Result<Vec<ConlluFile>, ConlluError> {
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(paths.len());
        for path in paths {
            let content = std::fs::read_to_string(path)?;
            pairs.push((content, path.clone()));
        }
        parse_conllu_strs(pairs, parallel)
    }

    // -----------------------------------------------------------------------
    // BaseConllu trait
    // -----------------------------------------------------------------------

    /// Core CoNLL-U reader behavior with default implementations.
    pub trait BaseConllu: Sized {
        fn files(&self) -> &VecDeque<ConlluFile>;
        fn files_mut(&mut self) -> &mut VecDeque<ConlluFile>;
        fn from_files(files: VecDeque<ConlluFile>) -> Self;

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

        /// Return all sentences across all files (flat).
        fn sentences(&self) -> Vec<&Sentence> {
            self.files().iter().flat_map(|f| &f.sentences).collect()
        }

        /// Return CoNLL-U strings, one per file.
        fn to_strings(&self) -> Vec<String> {
            self.files().iter().map(serialize_conllu_file).collect()
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

        /// Write CoNLL-U files to a directory.
        fn write_conllu_files(
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
                            "There are {} CoNLL-U files to create, \
                             but {} filenames were provided.",
                            self.files().len(),
                            names.len()
                        )));
                    }
                    names
                }
                None => self.default_output_filenames(".conllu"),
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
                .map(super::chat_writer::conllu_file_to_chat_str)
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
    }

    // -----------------------------------------------------------------------
    // Conllu struct
    // -----------------------------------------------------------------------

    /// CoNLL-U data reader.
    ///
    /// This is a pure Rust struct. For the Python-exposed wrapper, see `PyConllu`.
    #[derive(Clone, Debug)]
    pub struct Conllu {
        pub(crate) files: VecDeque<ConlluFile>,
    }

    impl BaseConllu for Conllu {
        fn files(&self) -> &VecDeque<ConlluFile> {
            &self.files
        }
        fn files_mut(&mut self) -> &mut VecDeque<ConlluFile> {
            &mut self.files
        }
        fn from_files(files: VecDeque<ConlluFile>) -> Self {
            Self { files }
        }
    }

    impl Conllu {
        /// Construct from a Vec of [`ConlluFile`] entries.
        pub fn from_conllu_files(files: Vec<ConlluFile>) -> Self {
            Self {
                files: VecDeque::from(files),
            }
        }

        /// Append data from another Conllu.
        pub fn push_back(&mut self, other: &Conllu) {
            self.files.extend(other.files.iter().cloned());
        }

        /// Prepend data from another Conllu.
        pub fn push_front(&mut self, other: &Conllu) {
            let mut new_files = other.files.clone();
            new_files.extend(std::mem::take(&mut self.files));
            self.files = new_files;
        }

        /// Remove and return the last file as a new Conllu.
        pub fn pop_back(&mut self) -> Option<Conllu> {
            self.files
                .pop_back()
                .map(|f| Conllu::from_files(VecDeque::from(vec![f])))
        }

        /// Remove and return the first file as a new Conllu.
        pub fn pop_front(&mut self) -> Option<Conllu> {
            self.files
                .pop_front()
                .map(|f| Conllu::from_files(VecDeque::from(vec![f])))
        }

        /// Parse CoNLL-U data from in-memory strings.
        pub fn from_strs(
            strs: Vec<String>,
            ids: Option<Vec<String>>,
            parallel: bool,
        ) -> Result<Self, ConlluError> {
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
            let files = parse_conllu_strs(pairs, parallel)?;
            Ok(Self::from_conllu_files(files))
        }

        /// Load and parse CoNLL-U data from file paths.
        pub fn read_files(paths: &[String], parallel: bool) -> Result<Self, ConlluError> {
            let files = load_conllu_files(paths, parallel)?;
            Ok(Self::from_conllu_files(files))
        }

        /// Recursively load CoNLL-U data from a directory.
        pub fn read_dir(
            path: &str,
            match_pattern: Option<&str>,
            extension: &str,
            parallel: bool,
        ) -> Result<Self, ConlluError> {
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
                .map_err(|e| ConlluError::InvalidPattern(e.to_string()))?;
            let files = load_conllu_files(&filtered, parallel)?;
            Ok(Self::from_conllu_files(files))
        }

        /// Load CoNLL-U data from a ZIP archive.
        pub fn read_zip(
            path: &str,
            match_pattern: Option<&str>,
            extension: &str,
            parallel: bool,
        ) -> Result<Self, ConlluError> {
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file)
                .map_err(|e| ConlluError::Zip(format!("Invalid zip file: {e}")))?;

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
                .map_err(|e| ConlluError::InvalidPattern(e.to_string()))?;

            let mut pairs: Vec<(String, String)> = Vec::new();
            for name in &filtered {
                let mut entry = archive
                    .by_name(name)
                    .map_err(|e| ConlluError::Zip(format!("Zip entry error: {e}")))?;
                let mut content = String::new();
                std::io::Read::read_to_string(&mut entry, &mut content)
                    .map_err(|e| ConlluError::Zip(format!("Read error: {e}")))?;
                pairs.push((content, name.clone()));
            }

            let files = parse_conllu_strs(pairs, parallel)?;
            Ok(Self::from_conllu_files(files))
        }

        /// Load CoNLL-U data from a git repository.
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
        ) -> Result<Self, ConlluError> {
            let local_path =
                crate::sources::resolve_git(url, rev, depth, cache_dir, force_download)?;
            let path = local_path.to_string_lossy();
            Self::read_dir(&path, match_pattern, extension, parallel)
        }

        /// Load CoNLL-U data from a URL.
        pub fn from_url(
            url: &str,
            match_pattern: Option<&str>,
            extension: &str,
            cache_dir: Option<std::path::PathBuf>,
            force_download: bool,
            parallel: bool,
        ) -> Result<Self, ConlluError> {
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

        fn sample_conllu() -> &'static str {
            "# sent_id = 1\n\
             # text = The cat sat on the mat.\n\
             1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_\n\
             2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_\n\
             3\tsat\tsit\tVERB\tVBD\tMood=Ind|Tense=Past\t0\troot\t_\t_\n\
             4\ton\ton\tADP\tIN\t_\t6\tcase\t_\t_\n\
             5\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t6\tdet\t_\t_\n\
             6\tmat\tmat\tNOUN\tNN\tNumber=Sing\t3\tnmod\t_\t_\n\
             7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tSpaceAfter=No\n\
             \n\
             # sent_id = 2\n\
             # text = I like it.\n\
             1\tI\tI\tPRON\tPRP\tCase=Nom|Number=Sing|Person=1|PronType=Prs\t2\tnsubj\t_\t_\n\
             2\tlike\tlike\tVERB\tVBP\tMood=Ind|Number=Sing|Person=1|Tense=Pres\t0\troot\t_\t_\n\
             3\tit\tit\tPRON\tPRP\tCase=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs\t2\tobj\t_\tSpaceAfter=No\n\
             4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No\n"
        }

        fn sample_with_multiword() -> &'static str {
            "# sent_id = 1\n\
             # text = Vámonos al mar.\n\
             1\tVámonos\tir\tVERB\t_\t_\t0\troot\t_\t_\n\
             2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_\n\
             2\ta\ta\tADP\t_\t_\t4\tcase\t_\t_\n\
             3\tel\tel\tDET\t_\t_\t4\tdet\t_\t_\n\
             4\tmar\tmar\tNOUN\t_\t_\t1\tobl\t_\tSpaceAfter=No\n\
             5\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\tSpaceAfter=No\n"
        }

        #[test]
        fn test_parse_basic() {
            let file = parse_conllu_str(sample_conllu(), "test.conllu".to_string()).unwrap();
            assert_eq!(file.file_path, "test.conllu");
            assert_eq!(file.sentences.len(), 2);

            let s1 = &file.sentences[0];
            assert_eq!(
                s1.comments.as_ref().unwrap(),
                &vec!["sent_id = 1", "text = The cat sat on the mat."]
            );
            assert_eq!(s1.tokens.len(), 7);
            assert_eq!(s1.tokens[0].form, "The");
            assert_eq!(s1.tokens[0].upos, "DET");
            assert_eq!(s1.tokens[0].head, "2");
            assert_eq!(s1.tokens[0].deprel, "det");
            assert_eq!(s1.tokens[2].lemma, "sit");
            assert_eq!(s1.tokens[2].feats, "Mood=Ind|Tense=Past");
            assert_eq!(s1.tokens[6].misc, "SpaceAfter=No");

            let s2 = &file.sentences[1];
            assert_eq!(s2.tokens.len(), 4);
            assert_eq!(s2.tokens[0].form, "I");
        }

        #[test]
        fn test_parse_multiword() {
            let file = parse_conllu_str(sample_with_multiword(), "mw.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 1);
            let s = &file.sentences[0];
            assert_eq!(s.tokens.len(), 6); // includes multiword token
            assert!(s.tokens[1].is_multiword());
            assert_eq!(s.tokens[1].id, "2-3");
            assert!(!s.tokens[0].is_multiword());
            assert!(!s.tokens[0].is_empty_node());
        }

        #[test]
        fn test_parse_with_bom() {
            let conllu = format!("\u{FEFF}{}", sample_conllu());
            let file = parse_conllu_str(&conllu, "bom.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 2);
        }

        #[test]
        fn test_parse_windows_line_endings() {
            let conllu = "# sent_id = 1\r\n1\tHello\thello\tNOUN\t_\t_\t0\troot\t_\t_\r\n\r\n";
            let file = parse_conllu_str(conllu, "win.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 1);
            assert_eq!(file.sentences[0].tokens[0].form, "Hello");
        }

        #[test]
        fn test_parse_empty() {
            let file = parse_conllu_str("", "empty.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 0);
        }

        #[test]
        fn test_parse_no_comments() {
            let conllu = "1\tHello\thello\tNOUN\t_\t_\t0\troot\t_\t_\n";
            let file = parse_conllu_str(conllu, "test.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 1);
            assert!(file.sentences[0].comments.is_none());
        }

        #[test]
        fn test_parse_no_trailing_newline() {
            let conllu = "# text = Hello\n1\tHello\thello\tNOUN\t_\t_\t0\troot\t_\t_";
            let file = parse_conllu_str(conllu, "test.conllu".to_string()).unwrap();
            assert_eq!(file.sentences.len(), 1);
            assert_eq!(file.sentences[0].tokens.len(), 1);
        }

        #[test]
        fn test_parse_error_wrong_field_count() {
            let conllu = "1\tHello\thello\n";
            let result = parse_conllu_str(conllu, "bad.conllu".to_string());
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ConlluError::Parse(_)));
        }

        #[test]
        fn test_serialize_round_trip() {
            let file = parse_conllu_str(sample_conllu(), "test.conllu".to_string()).unwrap();
            let output = serialize_conllu_file(&file);
            let file2 = parse_conllu_str(&output, "test.conllu".to_string()).unwrap();
            assert_eq!(file.sentences, file2.sentences);
        }

        #[test]
        fn test_conllu_from_strs() {
            let conllu = Conllu::from_strs(vec![sample_conllu().to_string()], None, false).unwrap();
            assert_eq!(conllu.num_files(), 1);
            assert_eq!(conllu.sentences().len(), 2);
        }

        #[test]
        fn test_conllu_base_trait() {
            let conllu = Conllu::from_strs(
                vec![sample_conllu().to_string(), sample_conllu().to_string()],
                Some(vec!["file1.conllu".to_string(), "file2.conllu".to_string()]),
                false,
            )
            .unwrap();
            assert_eq!(conllu.num_files(), 2);
            assert_eq!(conllu.file_paths(), vec!["file1.conllu", "file2.conllu"]);
            assert!(!conllu.is_empty());
            assert_eq!(conllu.sentences().len(), 4);
        }

        #[test]
        fn test_conllu_push_pop() {
            let mut conllu = Conllu::from_strs(
                vec![sample_conllu().to_string()],
                Some(vec!["file1.conllu".to_string()]),
                false,
            )
            .unwrap();
            let conllu2 = Conllu::from_strs(
                vec![sample_conllu().to_string()],
                Some(vec!["file2.conllu".to_string()]),
                false,
            )
            .unwrap();

            conllu.push_back(&conllu2);
            assert_eq!(conllu.num_files(), 2);
            assert_eq!(conllu.file_paths(), vec!["file1.conllu", "file2.conllu"]);

            let popped = conllu.pop_back().unwrap();
            assert_eq!(popped.file_paths(), vec!["file2.conllu"]);
            assert_eq!(conllu.num_files(), 1);

            conllu.push_front(&conllu2);
            assert_eq!(conllu.file_paths(), vec!["file2.conllu", "file1.conllu"]);

            let popped = conllu.pop_front().unwrap();
            assert_eq!(popped.file_paths(), vec!["file2.conllu"]);
        }

        #[test]
        fn test_write_conllu_files() {
            let conllu = Conllu::from_strs(
                vec![sample_conllu().to_string()],
                Some(vec!["test.conllu".to_string()]),
                false,
            )
            .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let out_dir = dir.path().join("output");
            conllu
                .write_conllu_files(out_dir.to_str().unwrap(), None)
                .unwrap();
            let content = std::fs::read_to_string(out_dir.join("test.conllu")).unwrap();
            let file2 = parse_conllu_str(&content, "test.conllu".to_string()).unwrap();
            assert_eq!(conllu.files()[0].sentences, file2.sentences);
        }

        #[test]
        fn test_to_strings_round_trip() {
            let conllu = Conllu::from_strs(
                vec![sample_conllu().to_string()],
                Some(vec!["test.conllu".to_string()]),
                false,
            )
            .unwrap();
            let strs = conllu.to_strings();
            let conllu2 =
                Conllu::from_strs(strs, Some(vec!["test.conllu".to_string()]), false).unwrap();
            assert_eq!(conllu.files()[0].sentences, conllu2.files()[0].sentences);
        }
    }
}
