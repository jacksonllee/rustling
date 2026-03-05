//! CHAT data reader for CHILDES/TalkBank transcripts.

use crate::chat::clean_utterance::clean_utterance;
use crate::chat::header::{
    Age, ChangeableHeader, Headers, Participant, parse_changeable, parse_file_headers,
    split_header_line,
};
use crate::chat::utterance::{
    BaseUtterance, Gra, PyToken, PyUtterance, PyUtterances, Token, Utterance, Utterances,
};
use crate::ngram::{BaseNgrams, Ngrams, PyNgrams};

use fancy_regex::Regex as FancyRegex;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyType};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use regex::Regex;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, OnceLock};

static TIME_MARKS_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x15-?(\d+)_(\d+)-?\x15").unwrap());

/// Representation of a single parsed CHAT file.
///
/// Contains the file path, headers, utterances (events), and raw lines.
/// Available to external Rust crates for building language-specific extensions.
#[derive(Debug)]
pub struct ChatFile {
    pub file_path: String,
    pub headers: Headers,
    pub events: Vec<Utterance>,
    /// Raw joined lines (headers, utterances, dependent tiers) for serialization.
    pub raw_lines: Vec<String>,
    /// Lazy cache of Python Utterance objects.
    py_utterances: Arc<OnceLock<Vec<Py<PyUtterance>>>>,
    /// Lazy cache of Python Token objects, grouped per real utterance.
    py_tokens: Arc<OnceLock<Vec<Vec<Py<PyToken>>>>>,
}

impl Clone for ChatFile {
    fn clone(&self) -> Self {
        Self {
            file_path: self.file_path.clone(),
            headers: self.headers.clone(),
            events: self.events.clone(),
            raw_lines: self.raw_lines.clone(),
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        }
    }
}

impl ChatFile {
    /// Construct a new ChatFile with the given fields and fresh caches.
    pub fn new(
        file_path: String,
        headers: Headers,
        events: Vec<Utterance>,
        raw_lines: Vec<String>,
    ) -> Self {
        Self {
            file_path,
            headers,
            events,
            raw_lines,
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        }
    }

    /// Iterate over all events (utterances and changeable headers) in file order.
    pub fn utterances(&self) -> impl Iterator<Item = &Utterance> {
        self.events.iter()
    }

    /// Iterate over only real utterances (excluding changeable headers).
    pub fn real_utterances(&self) -> impl Iterator<Item = &Utterance> {
        self.events.iter().filter(|u| u.changeable_header.is_none())
    }

    fn eq_data(&self, other: &ChatFile) -> bool {
        self.file_path == other.file_path
            && self.headers == other.headers
            && self.events == other.events
            && self.raw_lines == other.raw_lines
    }

    /// Whether this file contains no utterances or headers.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn cached_py_utterances(&self, py: Python<'_>) -> &[Py<PyUtterance>] {
        self.py_utterances.get_or_init(|| {
            self.utterances()
                .map(|utt| Py::new(py, PyUtterance(utt.clone())).unwrap())
                .collect()
        })
    }

    fn cached_py_tokens(&self, py: Python<'_>) -> &[Vec<Py<PyToken>>] {
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

    /// Reset all cached Python objects.
    ///
    /// Call this after mutating `events` (e.g. via participant filtering)
    /// to avoid returning stale cached data.
    pub fn reset_caches(&mut self) {
        self.py_utterances = Arc::new(OnceLock::new());
        self.py_tokens = Arc::new(OnceLock::new());
    }
}

/// A tier group: one utterance line with its dependent tiers.
struct TierGroup {
    participant: String,
    main_tier: String,
    dependent_tiers: HashMap<String, String>,
}

/// An intermediate morphology item from %mor parsing.
struct MorItem {
    pos: String,
    mor: String,
    is_clitic: bool,
}

/// Word/mor count mismatch info from `build_tokens`.
struct MisalignmentCounts {
    word_count: usize,
    mor_count: usize,
    words: Vec<String>,
    mor_labels: Vec<String>,
}

/// Full misalignment diagnostic for error/warning reporting.
pub struct MisalignmentInfo {
    pub file_path: String,
    pub participant: String,
    pub main_tier: String,
    pub mor_tier: String,
    pub word_count: usize,
    pub mor_count: usize,
    pub words: Vec<String>,
    pub mor_labels: Vec<String>,
}

/// Error type for CHAT reading operations.
#[derive(Debug)]
pub enum ChatError {
    /// An I/O error occurred.
    Io(std::io::Error),
    /// An invalid regex pattern was provided.
    InvalidPattern(String),
    /// An error occurred reading a ZIP archive.
    Zip(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Io(e) => write!(f, "{e}"),
            ChatError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
            ChatError::Zip(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ChatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ChatError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ChatError {
    fn from(e: std::io::Error) -> Self {
        ChatError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Join continuation lines and return all lines.
fn get_lines(chat_str: &str) -> Vec<String> {
    let mut lines = Vec::new();
    for raw_line in chat_str.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('*') || line.starts_with('%') || line.starts_with('@') {
            lines.push(line.to_string());
        } else if let Some(last) = lines.last_mut() {
            // Continuation line: append to previous line.
            last.push(' ');
            last.push_str(line.trim());
        }
    }
    lines
}

/// Intermediate result from scanning lines after the file-level headers.
enum EventOrTierGroup {
    TierGroup(TierGroup),
    Header(ChangeableHeader),
}

/// Scan lines starting from `start_idx`, grouping utterance tiers and
/// recognizing changeable headers that appear mid-file.
fn get_all_events(lines: &[String], start_idx: usize) -> Vec<EventOrTierGroup> {
    let mut results = Vec::new();
    let mut current: Option<TierGroup> = None;

    for line in &lines[start_idx..] {
        if line.starts_with('@') {
            // Changeable header mid-file.
            let (name, value) = split_header_line(line);
            if name == "End" {
                continue;
            }
            if let Some(ch) = parse_changeable(name, value) {
                // Flush any pending tier group before emitting the header.
                if let Some(group) = current.take() {
                    results.push(EventOrTierGroup::TierGroup(group));
                }
                results.push(EventOrTierGroup::Header(ch));
            }
            continue;
        }
        if line.starts_with('*') {
            if let Some(group) = current.take() {
                results.push(EventOrTierGroup::TierGroup(group));
            }
            // Parse *CODE:\t content  or  *CODE: content
            if let Some(colon_pos) = line.find(':') {
                let participant = line[1..colon_pos].to_string();
                let content = line[colon_pos + 1..]
                    .trim_start_matches('\t')
                    .trim()
                    .to_string();
                current = Some(TierGroup {
                    participant,
                    main_tier: content,
                    dependent_tiers: HashMap::new(),
                });
            }
        } else if line.starts_with('%')
            && let Some(ref mut group) = current
            && let Some(colon_pos) = line.find(':')
        {
            let tier_name = line[..colon_pos].to_string();
            let content = line[colon_pos + 1..]
                .trim_start_matches('\t')
                .trim()
                .to_string();
            group.dependent_tiers.insert(tier_name, content);
        }
    }
    if let Some(group) = current {
        results.push(EventOrTierGroup::TierGroup(group));
    }
    results
}

/// Split a POS|morphology item at the first pipe.
fn split_pos_mor(item: &str) -> (String, String) {
    if let Some(pipe_pos) = item.find('|') {
        (
            item[..pipe_pos].to_string(),
            item[pipe_pos + 1..].to_string(),
        )
    } else {
        // Punctuation items (like ".") have no pipe.
        (String::new(), item.to_string())
    }
}

/// Parse the %mor tier into a list of morphology items.
///
/// Handles preclitics (marked with `$`) and postclitics (marked with `~`).
/// For example: `pro:dem|that~cop|be&3S` produces two items.
fn parse_mor_tier(mor_str: &str) -> Vec<MorItem> {
    let mut items = Vec::new();

    for mor_token in mor_str.split_whitespace() {
        // Split by ~ to get main + postclitics.
        let tilde_parts: Vec<&str> = mor_token.split('~').collect();

        for (tilde_idx, tilde_part) in tilde_parts.iter().enumerate() {
            // Split by $ to get preclitics + main.
            let dollar_parts: Vec<&str> = tilde_part.split('$').collect();

            for (dollar_idx, dollar_part) in dollar_parts.iter().enumerate() {
                let (pos, mor) = split_pos_mor(dollar_part);
                let is_clitic = tilde_idx > 0 || dollar_idx < dollar_parts.len() - 1;
                items.push(MorItem {
                    pos,
                    mor,
                    is_clitic,
                });
            }
        }
    }

    // Split trailing sentence-final punctuation from the last item.
    // Handles cases like "n|cookie-PL." where the period is attached
    // without a preceding space.
    if let Some(last) = items.last_mut()
        && !last.pos.is_empty()
        && last.mor.len() > 1
    {
        let final_byte = last.mor.as_bytes()[last.mor.len() - 1];
        if matches!(final_byte, b'.' | b'?' | b'!') {
            let punct = last.mor[last.mor.len() - 1..].to_string();
            last.mor.truncate(last.mor.len() - 1);
            items.push(MorItem {
                pos: String::new(),
                mor: punct,
                is_clitic: false,
            });
        }
    }

    items
}

/// Parse the %gra tier into a list of grammatical relations.
fn parse_gra_tier(gra_str: &str) -> Vec<Gra> {
    gra_str
        .split_whitespace()
        .filter_map(|item| {
            let parts: Vec<&str> = item.split('|').collect();
            if parts.len() >= 3 {
                Some(Gra {
                    dep: parts[0].parse().unwrap_or(0),
                    head: parts[1].parse().unwrap_or(0),
                    rel: parts[2].to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Build tokens by aligning words with morphology and grammar data.
///
/// Returns `(tokens, misalignment)`. On misalignment, tokens is empty and
/// the caller should use `MisalignmentCounts` for diagnostics.
fn build_tokens(
    words: &[&str],
    mor_items: Option<&[MorItem]>,
    gra_items: Option<&[Gra]>,
) -> (Vec<Token>, Option<MisalignmentCounts>) {
    if words.is_empty() {
        return (Vec::new(), None);
    }

    let Some(mor_items) = mor_items else {
        // No mor data: return tokens with words only.
        return (
            words
                .iter()
                .map(|w| Token {
                    word: w.to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                })
                .collect(),
            None,
        );
    };

    // Count non-clitic mor items — should equal word count.
    let non_clitic_count = mor_items.iter().filter(|m| !m.is_clitic).count();

    if non_clitic_count != words.len() {
        // Misalignment: return empty tokens with diagnostic info.
        let word_list = words.iter().map(|w| w.to_string()).collect();
        let mor_list = mor_items
            .iter()
            .filter(|m| !m.is_clitic)
            .map(|m| format!("{}|{}", m.pos, m.mor))
            .collect();
        return (
            Vec::new(),
            Some(MisalignmentCounts {
                word_count: words.len(),
                mor_count: non_clitic_count,
                words: word_list,
                mor_labels: mor_list,
            }),
        );
    }

    let mut tokens = Vec::new();
    let mut mor_idx = 0;
    let mut word_idx = 0;

    while mor_idx < mor_items.len() {
        let item = &mor_items[mor_idx];

        if item.is_clitic {
            // Clitic: empty word, but has pos/mor/gra.
            let gra = gra_items.and_then(|g| g.get(mor_idx)).cloned();
            tokens.push(Token {
                word: String::new(),
                pos: Some(item.pos.clone()),
                mor: Some(item.mor.clone()),
                gra,
            });
        } else {
            // Regular word.
            let word = if word_idx < words.len() {
                words[word_idx]
            } else {
                ""
            };
            let gra = gra_items.and_then(|g| g.get(mor_idx)).cloned();
            tokens.push(Token {
                word: word.to_string(),
                pos: Some(item.pos.clone()),
                mor: Some(item.mor.clone()),
                gra,
            });
            word_idx += 1;
        }

        mor_idx += 1;
    }

    (tokens, None)
}

/// Parse a CHAT string into headers, ordered events, raw lines, and misalignments.
///
/// Only mid-file changeable headers are included in the returned events;
/// file-level headers are stored in the `Headers` struct.
#[allow(unused_variables)]
fn parse_chat_str(
    chat_str: &str,
    parallel: bool,
) -> (Headers, Vec<Utterance>, Vec<String>, Vec<MisalignmentInfo>) {
    let lines = get_lines(chat_str);
    let (headers, start_idx, _initial_events) = parse_file_headers(&lines);
    let event_or_groups = get_all_events(&lines, start_idx);

    // Separate tier groups (need building) from headers (pass through).
    let tier_groups: Vec<&TierGroup> = event_or_groups
        .iter()
        .filter_map(|e| match e {
            EventOrTierGroup::TierGroup(tg) => Some(tg),
            EventOrTierGroup::Header(_) => None,
        })
        .collect();

    #[cfg(feature = "parallel")]
    let results: Vec<(Utterance, Option<MisalignmentInfo>)> = if parallel {
        tier_groups
            .par_iter()
            .with_min_len(16)
            .map(|tg| build_utterance(tg))
            .collect()
    } else {
        tier_groups.iter().map(|tg| build_utterance(tg)).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(Utterance, Option<MisalignmentInfo>)> =
        tier_groups.iter().map(|tg| build_utterance(tg)).collect();

    // Split results into utterances and misalignment info.
    let mut utterances = Vec::with_capacity(results.len());
    let mut misalignments = Vec::new();
    for (utt, mis) in results {
        utterances.push(utt);
        if let Some(m) = mis {
            misalignments.push(m);
        }
    }

    // Reassemble in order: mid-file interleaved utterances and changeable headers.
    let mut events: Vec<Utterance> = Vec::new();
    let mut utt_iter = utterances.into_iter();
    for eg in event_or_groups {
        match eg {
            EventOrTierGroup::TierGroup(_) => {
                events.push(utt_iter.next().unwrap());
            }
            EventOrTierGroup::Header(h) => {
                events.push(Utterance {
                    participant: None,
                    tokens: None,
                    time_marks: None,
                    tiers: None,
                    changeable_header: Some(h),
                });
            }
        }
    }

    (headers, events, lines, misalignments)
}

/// Build an Utterance from a TierGroup.
///
/// Returns `(utterance, misalignment)`. `misalignment` is `Some` when
/// the word count doesn't match the non-clitic mor item count.
fn build_utterance(group: &TierGroup) -> (Utterance, Option<MisalignmentInfo>) {
    // Extract time marks.
    let time_marks = TIME_MARKS_REGEX
        .captures(&group.main_tier)
        .and_then(|caps| {
            let start: i64 = caps.get(1)?.as_str().parse().ok()?;
            let end: i64 = caps.get(2)?.as_str().parse().ok()?;
            Some((start, end))
        });

    // Clean the utterance text.
    let cleaned = clean_utterance(&group.main_tier);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    // Parse %mor tier.
    let mor_items = group.dependent_tiers.get("%mor").map(|s| parse_mor_tier(s));

    // Parse %gra tier.
    let gra_items = group.dependent_tiers.get("%gra").map(|s| parse_gra_tier(s));

    // Build tokens.
    let (tokens, misalignment_counts) =
        build_tokens(&words, mor_items.as_deref(), gra_items.as_deref());

    // Build misalignment info if detected.
    let misalignment = misalignment_counts.map(|counts| MisalignmentInfo {
        file_path: String::new(), // Populated later by the caller.
        participant: group.participant.clone(),
        main_tier: group.main_tier.clone(),
        mor_tier: group
            .dependent_tiers
            .get("%mor")
            .cloned()
            .unwrap_or_default(),
        word_count: counts.word_count,
        mor_count: counts.mor_count,
        words: counts.words,
        mor_labels: counts.mor_labels,
    });

    // Build tiers map.
    let mut tiers = group.dependent_tiers.clone();
    tiers.insert(group.participant.clone(), group.main_tier.clone());

    (
        Utterance {
            participant: Some(group.participant.clone()),
            tokens: Some(tokens),
            time_marks,
            tiers: Some(tiers),
            changeable_header: None,
        },
        misalignment,
    )
}

/// Filter file paths by match regex pattern.
pub fn filter_file_paths(
    paths: &[String],
    match_pattern: Option<&str>,
) -> Result<Vec<String>, String> {
    let match_re = match_pattern
        .map(FancyRegex::new)
        .transpose()
        .map_err(|e| format!("Invalid match regex: {e}"))?;

    Ok(paths
        .iter()
        .filter(|p| {
            if let Some(ref re) = match_re
                && !re.is_match(p).unwrap_or(false)
            {
                return false;
            }
            true
        })
        .cloned()
        .collect())
}

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

/// Filter a ChatFile's events and header participants by regex patterns.
pub(crate) fn filter_chat_file_by_participants(
    mut file: ChatFile,
    patterns: &[FancyRegex],
) -> ChatFile {
    file.events.retain(|u| {
        if u.changeable_header.is_some() {
            false
        } else {
            patterns.iter().any(|re| {
                re.is_match(u.participant.as_deref().unwrap_or(""))
                    .unwrap_or(false)
            })
        }
    });

    file.headers.participants.retain(|p| {
        patterns
            .iter()
            .any(|re| re.is_match(&p.code).unwrap_or(false))
    });

    // Reset cached Python objects (stale after filtering).
    file.reset_caches();

    file
}

/// Parse CHAT data from in-memory string pairs (content, id).
///
/// Returns `(files, misalignments)` with file_path set on each misalignment.
pub(crate) fn parse_chat_strs(
    pairs: Vec<(String, String)>,
    parallel: bool,
) -> (Vec<ChatFile>, Vec<MisalignmentInfo>) {
    let build = |content: &str, id: &str| {
        let (headers, events, raw_lines, mut mis) = parse_chat_str(content, parallel);
        for m in &mut mis {
            m.file_path = id.to_string();
        }
        (
            ChatFile {
                file_path: id.to_string(),
                headers,
                events,
                raw_lines,
                py_utterances: Arc::new(OnceLock::new()),
                py_tokens: Arc::new(OnceLock::new()),
            },
            mis,
        )
    };

    #[cfg(feature = "parallel")]
    if parallel {
        let results: Vec<(ChatFile, Vec<MisalignmentInfo>)> = pairs
            .par_iter()
            .with_min_len(16)
            .map(|(content, id)| build(content, id))
            .collect();
        let (files, nested): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        return (files, nested.into_iter().flatten().collect());
    }

    let results: Vec<(ChatFile, Vec<MisalignmentInfo>)> = pairs
        .iter()
        .map(|(content, id)| build(content, id))
        .collect();
    let (files, nested): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    (files, nested.into_iter().flatten().collect())
}

/// Load and parse CHAT files from paths.
///
/// Returns `(files, misalignments)` with file_path set on each misalignment.
pub(crate) fn load_chat_files(
    paths: &[String],
    parallel: bool,
) -> Result<(Vec<ChatFile>, Vec<MisalignmentInfo>), std::io::Error> {
    let build = |path: &str| -> Result<(ChatFile, Vec<MisalignmentInfo>), std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let (headers, events, raw_lines, mut mis) = parse_chat_str(&content, parallel);
        for m in &mut mis {
            m.file_path = path.to_string();
        }
        Ok((
            ChatFile {
                file_path: path.to_string(),
                headers,
                events,
                raw_lines,
                py_utterances: Arc::new(OnceLock::new()),
                py_tokens: Arc::new(OnceLock::new()),
            },
            mis,
        ))
    };

    #[cfg(feature = "parallel")]
    if parallel {
        let results: Vec<(ChatFile, Vec<MisalignmentInfo>)> = paths
            .par_iter()
            .with_min_len(16)
            .map(|path| build(path))
            .collect::<Result<Vec<_>, _>>()?;
        let (files, nested): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        return Ok((files, nested.into_iter().flatten().collect()));
    }

    let results: Vec<(ChatFile, Vec<MisalignmentInfo>)> = paths
        .iter()
        .map(|path| build(path))
        .collect::<Result<Vec<_>, _>>()?;
    let (files, nested): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    Ok((files, nested.into_iter().flatten().collect()))
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a ChatFile back to a CHAT format string.
pub fn serialize_chat_file(file: &ChatFile) -> String {
    let mut output = String::new();
    for line in &file.raw_lines {
        if line == "@End" {
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    output.push_str("@End\n");
    output
}

// ---------------------------------------------------------------------------
// Misalignment handling
// ---------------------------------------------------------------------------

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
                 \x20    %mor tier: {}\n     Words ({}): {}\n\
                 \x20    Non-clitic mor items ({}): {}\n",
                i + 1,
                m.file_path,
                m.participant,
                m.main_tier,
                m.mor_tier,
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
             \x20 %mor tier: {}\n\
             \x20 Words ({}): {}\n\
             \x20 Non-clitic mor items ({}): {}\n\
             Tokens set to empty for this utterance; \
             raw tier data is preserved in utterance.tiers.",
            m.file_path,
            m.participant,
            m.main_tier,
            m.mor_tier,
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
    }
}

use crate::persistence::pathbuf_to_string;

// ---------------------------------------------------------------------------
// WriteError
// ---------------------------------------------------------------------------

/// Error type for [`BaseChat::write_files`].
pub enum WriteError {
    /// Validation error (e.g., wrong number of files or filenames).
    Validation(String),
    /// I/O error from the filesystem.
    Io(std::io::Error),
}

// ---------------------------------------------------------------------------
// BaseChat
// ---------------------------------------------------------------------------

/// Core CHAT reader behavior with default implementations.
///
/// Implementors provide three required methods that grant access to the
/// underlying `VecDeque<ChatFile>`. All other methods are provided as defaults.
///
/// # Required methods
///
/// - [`files`](BaseChat::files) — immutable access to the file collection
/// - [`files_mut`](BaseChat::files_mut) — mutable access to the file collection
/// - [`from_files`](BaseChat::from_files) — construct a new instance from files
pub trait BaseChat: Sized {
    fn files(&self) -> &VecDeque<ChatFile>;
    fn files_mut(&mut self) -> &mut VecDeque<ChatFile>;
    fn from_files(files: VecDeque<ChatFile>) -> Self;

    // -----------------------------------------------------------------------
    // Construction from utterances
    // -----------------------------------------------------------------------

    /// Construct from a list of utterances.
    ///
    /// Creates a single virtual file with default headers and raw lines
    /// synthesized from the utterances' tier data.
    fn from_utterances<U: BaseUtterance>(utterances: Vec<U>) -> Self {
        let mut raw_lines = Vec::new();
        let mut events = Vec::new();
        for utt in &utterances {
            raw_lines.extend(utt.to_chat_lines());
            events.push(utt.to_utterance());
        }
        let file = ChatFile::new(
            uuid::Uuid::new_v4().to_string(),
            Headers::default(),
            events,
            raw_lines,
        );
        Self::from_files(VecDeque::from(vec![file]))
    }

    // -----------------------------------------------------------------------
    // Basic queries
    // -----------------------------------------------------------------------

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

    /// Return file-level headers.
    fn headers(&self) -> Vec<Headers> {
        self.files().iter().map(|f| f.headers.clone()).collect()
    }

    /// Return the age of the target child (CHI) in each file.
    fn ages(&self) -> Vec<Option<Age>> {
        self.files()
            .iter()
            .map(|f| {
                f.headers
                    .participants
                    .iter()
                    .find(|p| p.code == "CHI")
                    .and_then(|p| p.age.clone())
            })
            .collect()
    }

    /// Return participants per file.
    fn participants(&self) -> Vec<Vec<Participant>> {
        self.files()
            .iter()
            .map(|f| f.headers.participants.clone())
            .collect()
    }

    /// Return unique participants across all files.
    fn unique_participants(&self) -> Vec<Participant> {
        let mut seen = HashSet::new();
        self.files()
            .iter()
            .flat_map(|f| f.headers.participants.clone())
            .filter(|p| seen.insert(p.clone()))
            .collect()
    }

    /// Return languages per file.
    fn languages(&self) -> Vec<Vec<String>> {
        self.files()
            .iter()
            .map(|f| f.headers.languages.clone())
            .collect()
    }

    /// Return unique languages across all files.
    fn unique_languages(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        self.files()
            .iter()
            .flat_map(|f| f.headers.languages.clone())
            .filter(|lang| seen.insert(lang.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Return CHAT data strings, one per file.
    fn to_strings(&self) -> Vec<String> {
        self.files().iter().map(serialize_chat_file).collect()
    }

    /// Write CHAT data to disk.
    fn write_files(
        &self,
        path: &str,
        is_dir: bool,
        filenames: Option<Vec<String>>,
    ) -> Result<(), WriteError> {
        let strs = self.to_strings();

        if !is_dir {
            if self.files().len() > 1 {
                return Err(WriteError::Validation(
                    "The CHAT data in this reader exists in more than one file. \
                     Set is_dir=True and pass a directory path."
                        .into(),
                ));
            }
            if let Some(content) = strs.first() {
                if let Some(parent) = std::path::Path::new(path).parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent).map_err(WriteError::Io)?;
                }
                std::fs::write(path, content).map_err(WriteError::Io)?;
            }
        } else {
            let dir = std::path::Path::new(path);
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
                None => (0..self.files().len())
                    .map(|i| format!("{:04}.cha", i + 1))
                    .collect(),
            };

            for (name, content) in names.iter().zip(strs.iter()) {
                let file_path = dir.join(name);
                std::fs::write(&file_path, content).map_err(WriteError::Io)?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Filtering
    // -----------------------------------------------------------------------

    /// Filter by file path and/or participant regex patterns (pure Rust).
    fn filter_by(&self, files: Option<&str>, participants: Option<&str>) -> Result<Self, String> {
        let mut filtered: VecDeque<ChatFile> = if let Some(pattern) = files {
            let re = FancyRegex::new(pattern).map_err(|e| format!("Invalid file regex: {e}"))?;
            self.files()
                .iter()
                .filter(|f| re.is_match(&f.file_path).unwrap_or(false))
                .cloned()
                .collect()
        } else {
            self.files().clone()
        };

        if let Some(pattern) = participants {
            let anchored = if pattern.starts_with('^') || pattern.ends_with('$') {
                pattern.to_string()
            } else {
                format!("^(?:{pattern})$")
            };
            let re = FancyRegex::new(&anchored)
                .map_err(|e| format!("Invalid participant regex: {e}"))?;
            filtered = filtered
                .into_iter()
                .map(|f| filter_chat_file_by_participants(f, std::slice::from_ref(&re)))
                .collect();
        }

        Ok(Self::from_files(filtered))
    }

    // -----------------------------------------------------------------------
    // Info
    // -----------------------------------------------------------------------

    /// Return a formatted info string.
    fn info_string(&self, verbose: bool) -> String {
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

        let mut output =
            format!("{n_files} files\n{total_utterances} utterances\n{total_words} words\n");

        if n_files >= 2 {
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
            for (i, (utts, words, path)) in stats[..max_rows].iter().enumerate() {
                output.push_str(&format!(
                    "  #{}: {} utterances, {} words — {}\n",
                    i + 1,
                    utts,
                    words,
                    path
                ));
            }
            if !verbose && max_rows < n_files {
                output.push_str("...\n(set `verbose` to True for all the files)\n");
            }
        }

        output
    }

    // -----------------------------------------------------------------------
    // Developmental measures
    // -----------------------------------------------------------------------

    /// Mean length of utterance in morphemes, one value per file.
    fn mlum(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
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
                if utterances.is_empty() {
                    return 0.0;
                }
                let total: usize = utterances
                    .iter()
                    .map(|u| {
                        u.tokens
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|t| t.pos.as_ref().is_some_and(|p| !p.is_empty()))
                            .count()
                    })
                    .sum();
                total as f64 / utterances.len() as f64
            })
            .collect()
    }

    /// Mean length of utterance in words, one value per file.
    fn mluw(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
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
                if utterances.is_empty() {
                    return 0.0;
                }
                let total: usize = utterances
                    .iter()
                    .map(|u| {
                        u.tokens
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|t| !t.word.is_empty() && t.pos.as_deref() != Some(""))
                            .count()
                    })
                    .sum();
                total as f64 / utterances.len() as f64
            })
            .collect()
    }

    /// Type-token ratio, one value per file.
    fn ttr(&self, participant: &str, n: Option<usize>) -> Vec<f64> {
        self.files()
            .iter()
            .map(|f| {
                let words: Vec<&str> = f
                    .real_utterances()
                    .filter(|u| u.participant.as_deref() == Some(participant))
                    .flat_map(|u| u.tokens.as_deref().unwrap_or(&[]))
                    .filter(|t| !t.word.is_empty() && t.pos.as_deref() != Some(""))
                    .map(|t| t.word.as_str())
                    .collect();
                let words = if let Some(n) = n {
                    &words[..words.len().min(n)]
                } else {
                    &words[..]
                };
                if words.is_empty() {
                    0.0
                } else {
                    let types: HashSet<&str> = words.iter().copied().collect();
                    types.len() as f64 / words.len() as f64
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Remove all data.
    fn clear(&mut self) {
        self.files_mut().clear();
    }

    // -----------------------------------------------------------------------
    // Head / tail
    // -----------------------------------------------------------------------

    /// Return the first n utterances.
    fn head(&self, n: usize) -> Utterances {
        let utterances: Vec<Utterance> = self
            .files()
            .iter()
            .flat_map(|f| f.utterances())
            .take(n)
            .cloned()
            .collect();
        Utterances::new(utterances)
    }

    /// Return the last n utterances.
    fn tail(&self, n: usize) -> Utterances {
        let all: Vec<&Utterance> = self.files().iter().flat_map(|f| f.utterances()).collect();
        let start = all.len().saturating_sub(n);
        let utterances: Vec<Utterance> = all[start..].iter().map(|u| (*u).clone()).collect();
        Utterances::new(utterances)
    }
}

// ---------------------------------------------------------------------------
// BasePyChat
// ---------------------------------------------------------------------------

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

    /// Write CHAT data to disk with Python error conversion.
    fn py_write(&self, path: &str, is_dir: bool, filenames: Option<Vec<String>>) -> PyResult<()> {
        self.write_files(path, is_dir, filenames)
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
// Pure Rust Chat struct
// ---------------------------------------------------------------------------

/// CHAT data reader for CHILDES/TalkBank transcripts.
///
/// This is a pure Rust struct. For the Python-exposed wrapper, see [`PyChat`].
#[derive(Clone, Debug)]
pub struct Chat {
    pub(crate) files: VecDeque<ChatFile>,
}

impl BaseChat for Chat {
    fn files(&self) -> &VecDeque<ChatFile> {
        &self.files
    }
    fn files_mut(&mut self) -> &mut VecDeque<ChatFile> {
        &mut self.files
    }
    fn from_files(files: VecDeque<ChatFile>) -> Self {
        Self { files }
    }
}

impl Chat {
    /// Construct from a Vec of [`ChatFile`] entries.
    pub fn from_chat_files(files: Vec<ChatFile>) -> Self {
        Self {
            files: VecDeque::from(files),
        }
    }

    /// Append data from another Chat.
    pub fn push_back(&mut self, other: &Chat) {
        self.files.extend(other.files.iter().cloned());
    }

    /// Prepend data from another Chat.
    pub fn push_front(&mut self, other: &Chat) {
        let mut new_files = other.files.clone();
        new_files.extend(std::mem::take(&mut self.files));
        self.files = new_files;
    }

    /// Remove and return the last file as a new Chat.
    pub fn pop_back(&mut self) -> Option<Chat> {
        self.files
            .pop_back()
            .map(|f| Chat::from_files(VecDeque::from(vec![f])))
    }

    /// Remove and return the first file as a new Chat.
    pub fn pop_front(&mut self) -> Option<Chat> {
        self.files
            .pop_front()
            .map(|f| Chat::from_files(VecDeque::from(vec![f])))
    }

    /// Parse CHAT data from in-memory strings.
    ///
    /// Returns `(Chat, misalignments)`. The caller decides how to handle
    /// any misalignment diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is `Some` and its length differs from `strs`.
    pub fn from_strs(
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
    ) -> (Self, Vec<MisalignmentInfo>) {
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
        let (files, misalignments) = parse_chat_strs(pairs, parallel);
        (Self::from_chat_files(files), misalignments)
    }

    /// Load and parse CHAT data from file paths.
    ///
    /// Returns `(Chat, misalignments)` on success.
    pub fn read_files(
        paths: &[String],
        parallel: bool,
    ) -> Result<(Self, Vec<MisalignmentInfo>), std::io::Error> {
        let (files, misalignments) = load_chat_files(paths, parallel)?;
        Ok((Self::from_chat_files(files), misalignments))
    }

    /// Recursively load CHAT data from a directory.
    ///
    /// Walks `path` for files ending with `extension` (e.g. `".cha"`),
    /// optionally filtering by a regex `match_pattern` on the full path.
    pub fn read_dir(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<(Self, Vec<MisalignmentInfo>), ChatError> {
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

        let filtered =
            filter_file_paths(&paths, match_pattern).map_err(ChatError::InvalidPattern)?;
        let (files, misalignments) = load_chat_files(&filtered, parallel)?;
        Ok((Self::from_chat_files(files), misalignments))
    }

    /// Load CHAT data from a ZIP archive.
    ///
    /// Reads entries ending with `extension` (e.g. `".cha"`),
    /// optionally filtering by a regex `match_pattern` on entry names.
    pub fn read_zip(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<(Self, Vec<MisalignmentInfo>), ChatError> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| ChatError::Zip(format!("Invalid zip file: {e}")))?;

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

        let filtered =
            filter_file_paths(&entry_names, match_pattern).map_err(ChatError::InvalidPattern)?;

        let mut pairs: Vec<(String, String)> = Vec::new();
        for name in &filtered {
            let mut entry = archive
                .by_name(name)
                .map_err(|e| ChatError::Zip(format!("Zip entry error: {e}")))?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut entry, &mut content)
                .map_err(|e| ChatError::Zip(format!("Read error: {e}")))?;
            pairs.push((content, name.clone()));
        }

        let (files, misalignments) = parse_chat_strs(pairs, parallel);
        Ok((Self::from_chat_files(files), misalignments))
    }
}

// ---------------------------------------------------------------------------
// Python-exposed PyChat wrapper
// ---------------------------------------------------------------------------

/// Python-exposed CHAT data reader.
///
/// Wraps the pure Rust [`Chat`] struct and exposes it to Python via PyO3.
#[pyclass(name = "CHAT", from_py_object)]
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
    #[pyo3(signature = (strs, ids=None, parallel=true, strict=true))]
    fn from_strs(
        _cls: &Bound<'_, PyType>,
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
        strict: bool,
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
        let py = _cls.py();
        let (chat, misalignments) = Chat::from_strs(strs, ids, parallel);
        handle_misalignments(&misalignments, strict, py)?;
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
    #[pyo3(signature = (paths, *, parallel=true, strict=true))]
    fn read_files(
        _cls: &Bound<'_, PyType>,
        paths: Vec<PathBuf>,
        parallel: bool,
        strict: bool,
    ) -> PyResult<Self> {
        let paths: Vec<String> = paths
            .into_iter()
            .map(pathbuf_to_string)
            .collect::<PyResult<_>>()?;
        let py = _cls.py();
        let (chat, misalignments) = Chat::read_files(&paths, parallel)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        handle_misalignments(&misalignments, strict, py)?;
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
    #[pyo3(signature = (path, *, r#match=None, extension=".cha", parallel=true, strict=true))]
    fn read_dir(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
        strict: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let py = _cls.py();
        let (chat, misalignments) =
            Chat::read_dir(&path, r#match, extension, parallel).map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
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
    #[pyo3(signature = (path, *, r#match=None, extension=".cha", parallel=true, strict=true))]
    fn open_zip(
        _cls: &Bound<'_, PyType>,
        path: PathBuf,
        r#match: Option<&str>,
        extension: &str,
        parallel: bool,
        strict: bool,
    ) -> PyResult<Self> {
        let path = pathbuf_to_string(path)?;
        let py = _cls.py();
        let (chat, misalignments) =
            Chat::read_zip(&path, r#match, extension, parallel).map_err(chat_error_to_pyerr)?;
        handle_misalignments(&misalignments, strict, py)?;
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
        let mut counter = Ngrams::new(n, None).map_err(pyo3::exceptions::PyValueError::new_err)?;
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

    /// Write CHAT data to disk.
    #[pyo3(name = "to_chat")]
    #[pyo3(signature = (path, *, is_dir=false, filenames=None))]
    fn write(&self, path: PathBuf, is_dir: bool, filenames: Option<Vec<String>>) -> PyResult<()> {
        let path = pathbuf_to_string(path)?;
        self.py_write(&path, is_dir, filenames)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::utterance::Utterances;

    fn make_basic_chat() -> &'static str {
        "@UTF8\n@Begin\n@Participants:\tCHI Child, MOT Mother\n*CHI:\tI want cookie .\n%mor:\tpro|I v|want n|cookie .\n%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n*MOT:\tno .\n%mor:\tco|no .\n%gra:\t1|0|ROOT 2|1|PUNCT\n@End\n"
    }

    #[test]
    fn test_chat_file_is_empty() {
        let empty_file = ChatFile {
            file_path: String::new(),
            headers: Headers::default(),
            events: vec![],
            raw_lines: vec![],
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        };
        assert!(empty_file.is_empty());

        let (headers, events, raw_lines, _) = parse_chat_str(make_basic_chat(), true);
        let non_empty_file = ChatFile {
            file_path: "test".to_string(),
            headers,
            events,
            raw_lines,
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        };
        assert!(!non_empty_file.is_empty());
    }

    #[test]
    fn test_get_lines_joins_continuations() {
        let input = "@Begin\n*CHI:\tI want\n\tcookie .\n@End\n";
        let lines = get_lines(input);
        assert!(lines.iter().any(|l| l.contains("I want cookie .")));
    }

    #[test]
    fn test_get_lines_trims_leading_whitespace() {
        let input = "  @Begin\n  *CHI:\tI want cookie .\n  *MOT:\tno .\n  @End\n";
        let lines = get_lines(input);
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("@Begin"));
        assert!(lines[1].starts_with("*CHI:"));
        assert!(lines[2].starts_with("*MOT:"));
        assert!(lines[3].starts_with("@End"));
    }

    #[test]
    fn test_parse_chat_str_leading_whitespace() {
        let input = "  @UTF8\n  @Begin\n  @Participants:\tCHI Child, MOT Mother\n  *CHI:\tI want cookie .\n  %mor:\tpro|I v|want n|cookie .\n  %gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n  @End\n";
        let (_, events, _, _) = parse_chat_str(input, true);
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].participant.as_deref(), Some("CHI"));
        let tokens = utterances[0].tokens.as_ref().unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].word, "I");
        assert_eq!(tokens[2].word, "cookie");
    }

    #[test]
    fn test_get_all_events_extracts_tiers() {
        let lines = get_lines(make_basic_chat());
        let (_, start_idx, _) = parse_file_headers(&lines);
        let events = get_all_events(&lines, start_idx);
        let tier_groups: Vec<&TierGroup> = events
            .iter()
            .filter_map(|e| match e {
                EventOrTierGroup::TierGroup(tg) => Some(tg),
                _ => None,
            })
            .collect();
        assert_eq!(tier_groups.len(), 2);
        assert_eq!(tier_groups[0].participant, "CHI");
        assert_eq!(tier_groups[1].participant, "MOT");
    }

    #[test]
    fn test_parse_mor_tier_basic() {
        let items = parse_mor_tier("pro|I v|want n|cookie .");
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].pos, "pro");
        assert_eq!(items[0].mor, "I");
        assert_eq!(items[1].pos, "v");
        assert_eq!(items[1].mor, "want");
        assert_eq!(items[3].pos, "");
        assert_eq!(items[3].mor, ".");
    }

    #[test]
    fn test_parse_mor_tier_postclitic() {
        // "that's" -> pro:dem|that~cop|be&3S
        let items = parse_mor_tier("pro:dem|that~cop|be&3S adj|good .");
        assert_eq!(items.len(), 4); // that, be&3S(clitic), good, .
        assert_eq!(items[0].pos, "pro:dem");
        assert!(!items[0].is_clitic);
        assert_eq!(items[1].pos, "cop");
        assert!(items[1].is_clitic);
        assert_eq!(items[2].pos, "adj");
        assert!(!items[2].is_clitic);
    }

    #[test]
    fn test_parse_mor_tier_preclitic() {
        // "won't" -> aux|will$neg|not
        let items = parse_mor_tier("aux|will$neg|not");
        assert_eq!(items.len(), 2);
        assert!(items[0].is_clitic); // preclitic
        assert!(!items[1].is_clitic); // main
    }

    #[test]
    fn test_parse_mor_tier_preclitic_and_postclitic() {
        // "da~me~lo" -> v|da-give$pro|me&dat-me~pro|lo&acc-it
        let items = parse_mor_tier("v|da-give$pro|me&dat-me~pro|lo&acc-it");
        assert_eq!(items.len(), 3);
        assert!(items[0].is_clitic); // v|da-give (preclitic)
        assert!(!items[1].is_clitic); // pro|me&dat-me (main)
        assert!(items[2].is_clitic); // pro|lo&acc-it (postclitic)
    }

    #[test]
    fn test_parse_mor_tier_attached_period() {
        let items = parse_mor_tier("pro:sub|she v|say&PAST pro:sub|I v|want n|cookie-PL.");
        assert_eq!(items.len(), 6);
        assert_eq!(items[4].pos, "n");
        assert_eq!(items[4].mor, "cookie-PL");
        assert!(!items[4].is_clitic);
        assert_eq!(items[5].pos, "");
        assert_eq!(items[5].mor, ".");
        assert!(!items[5].is_clitic);
    }

    #[test]
    fn test_parse_mor_tier_attached_question_mark() {
        let items = parse_mor_tier("pro|what v|be&3S n|that?");
        assert_eq!(items.len(), 4);
        assert_eq!(items[2].pos, "n");
        assert_eq!(items[2].mor, "that");
        assert_eq!(items[3].pos, "");
        assert_eq!(items[3].mor, "?");
    }

    #[test]
    fn test_parse_mor_tier_attached_exclamation() {
        let items = parse_mor_tier("co|yes!");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].pos, "co");
        assert_eq!(items[0].mor, "yes");
        assert_eq!(items[1].pos, "");
        assert_eq!(items[1].mor, "!");
    }

    #[test]
    fn test_parse_mor_tier_standalone_punct_unchanged() {
        let items = parse_mor_tier("pro|I v|want n|cookie .");
        assert_eq!(items.len(), 4);
        assert_eq!(items[2].pos, "n");
        assert_eq!(items[2].mor, "cookie");
        assert_eq!(items[3].pos, "");
        assert_eq!(items[3].mor, ".");
    }

    #[test]
    fn test_parse_mor_tier_postclitic_attached_period() {
        let items = parse_mor_tier("pro:dem|that~cop|be&3S.");
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].pos, "pro:dem");
        assert!(!items[0].is_clitic);
        assert_eq!(items[1].pos, "cop");
        assert_eq!(items[1].mor, "be&3S");
        assert!(items[1].is_clitic);
        assert_eq!(items[2].pos, "");
        assert_eq!(items[2].mor, ".");
        assert!(!items[2].is_clitic);
    }

    #[test]
    fn test_parse_gra_tier() {
        let items = parse_gra_tier("1|2|SUBJ 2|0|ROOT 3|2|OBJ");
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].dep, 1);
        assert_eq!(items[0].head, 2);
        assert_eq!(items[0].rel, "SUBJ");
    }

    #[test]
    fn test_parse_chat_str_basic() {
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true);
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].participant.as_deref(), Some("CHI"));
        let tokens0 = utterances[0].tokens.as_ref().unwrap();
        assert_eq!(tokens0.len(), 4); // I, want, cookie, .
        assert_eq!(tokens0[0].word, "I");
        assert_eq!(tokens0[0].pos.as_deref(), Some("pro"));
        assert_eq!(tokens0[0].mor.as_deref(), Some("I"));
        assert!(tokens0[0].gra.is_some());
        assert_eq!(tokens0[0].gra.as_ref().unwrap().rel, "SUBJ");
    }

    #[test]
    fn test_parse_chat_str_attached_mor_period() {
        let input = "@UTF8\n@Begin\n@Participants:\tCHI Child\n\
                     *CHI:\tshe said \u{201c}I want cookies\u{201d} .\n\
                     %mor:\tpro:sub|she v|say&PAST pro:sub|I v|want n|cookie-PL.\n\
                     @End\n";
        let (_, events, _, misalignments) = parse_chat_str(input, false);
        assert!(misalignments.is_empty());
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        assert_eq!(utterances.len(), 1);
        let tokens = utterances[0].tokens.as_ref().unwrap();
        assert_eq!(tokens.len(), 6); // she, said, I, want, cookies, .
        assert_eq!(tokens[4].word, "cookies");
        assert_eq!(tokens[4].pos.as_deref(), Some("n"));
        assert_eq!(tokens[4].mor.as_deref(), Some("cookie-PL"));
        assert_eq!(tokens[5].word, ".");
        assert_eq!(tokens[5].pos.as_deref(), Some(""));
        assert_eq!(tokens[5].mor.as_deref(), Some("."));
    }

    #[test]
    fn test_parse_chat_str_time_marks() {
        let input = "@UTF8\n@Begin\n*CHI:\thello . \x15123_456\x15\n@End\n";
        let (_, events, _, _) = parse_chat_str(input, true);
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].time_marks, Some((123, 456)));
    }

    #[test]
    fn test_parse_chat_str_no_mor() {
        let input = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n";
        let (_, events, _, _) = parse_chat_str(input, true);
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        assert_eq!(utterances.len(), 1);
        let tokens0 = utterances[0].tokens.as_ref().unwrap();
        assert_eq!(tokens0.len(), 3);
        assert_eq!(tokens0[0].word, "hello");
        assert!(tokens0[0].pos.is_none());
    }

    #[test]
    fn test_build_tokens_alignment_with_clitics() {
        // "that's good ." -> words: ["that's", "good", "."]
        // mor: pro:dem|that~cop|be&3S adj|good .
        // items: [that(non-clitic), be&3S(clitic), good(non-clitic), .(non-clitic)]
        let mor_items = parse_mor_tier("pro:dem|that~cop|be&3S adj|good .");
        let words = vec!["that's", "good", "."];
        let (tokens, misalignment) = build_tokens(&words, Some(&mor_items), None);
        assert!(misalignment.is_none());
        // non-clitic count = 3, words = 3, so alignment should work
        assert_eq!(tokens.len(), 4); // 3 words + 1 clitic
        assert_eq!(tokens[0].word, "that's");
        assert_eq!(tokens[0].pos.as_deref(), Some("pro:dem"));
        assert_eq!(tokens[1].word, ""); // clitic
        assert_eq!(tokens[1].pos.as_deref(), Some("cop"));
        assert_eq!(tokens[2].word, "good");
    }

    #[test]
    fn test_build_tokens_misalignment_returns_empty() {
        // mor: pro|I, v|want, . → 3 non-clitic items vs 4 words → misalignment
        let mor_items = parse_mor_tier("pro|I v|want .");
        let words = vec!["I", "want", "cookie", "."];
        let (tokens, misalignment) = build_tokens(&words, Some(&mor_items), None);
        assert!(tokens.is_empty());
        assert!(misalignment.is_some());
        let counts = misalignment.unwrap();
        assert_eq!(counts.word_count, 4);
        assert_eq!(counts.mor_count, 3);
    }

    #[test]
    fn test_build_tokens_no_mor() {
        let words = vec!["hello", "world", "."];
        let (tokens, misalignment) = build_tokens(&words, None, None);
        assert!(misalignment.is_none());
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].word, "hello");
        assert!(tokens[0].pos.is_none());
    }

    #[test]
    fn test_parse_chat_str_collects_misalignments() {
        let input = "@UTF8\n@Begin\n@Participants:\tCHI Child\n\
                     *CHI:\tI want cookie .\n\
                     %mor:\tpro|I v|want .\n\
                     @End\n";
        let (_, _, _, misalignments) = parse_chat_str(input, false);
        assert!(!misalignments.is_empty());
        assert_eq!(misalignments[0].participant, "CHI");
    }

    #[test]
    fn test_parse_chat_str_no_misalignment() {
        let (_, _, _, misalignments) = parse_chat_str(make_basic_chat(), true);
        assert!(misalignments.is_empty());
    }

    #[test]
    fn test_filter_file_paths() {
        let paths = vec![
            "a/action.cha".to_string(),
            "a/codes.cha".to_string(),
            "a/phono.cha".to_string(),
        ];
        let filtered = filter_file_paths(&paths, Some("action")).unwrap();
        assert_eq!(filtered, vec!["a/action.cha"]);

        let filtered = filter_file_paths(&paths, None).unwrap();
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_filter_negative_lookahead_drops_chi_and_headers() {
        let chat_str = "@UTF8\n@Begin\n@Participants:\tCHI Child, MOT Mother\n\
                         *CHI:\thello .\n\
                         @New Episode\n\
                         *MOT:\thi .\n\
                         @End\n";
        let file = make_chat_file("test", chat_str);
        // Sanity: unfiltered file has 2 real utterances + 1 changeable header.
        assert_eq!(file.events.len(), 3);

        let pattern = FancyRegex::new("^(?!CHI$)").unwrap();
        let filtered = filter_chat_file_by_participants(file, &[pattern]);

        // Only MOT utterance remains (no CHI, no changeable header).
        assert_eq!(filtered.events.len(), 1);
        assert_eq!(filtered.events[0].participant.as_deref(), Some("MOT"));
        assert!(filtered.events[0].changeable_header.is_none());

        // Header participants list filtered to MOT only.
        assert_eq!(filtered.headers.participants.len(), 1);
        assert_eq!(filtered.headers.participants[0].code, "MOT");
    }

    #[test]
    fn test_tiers_in_utterance() {
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true);
        let utterances: Vec<&Utterance> = events
            .iter()
            .filter(|u| u.changeable_header.is_none())
            .collect();
        let tiers = utterances[0].tiers.as_ref().unwrap();
        assert!(tiers.contains_key("CHI"));
        assert!(tiers.contains_key("%mor"));
        assert!(tiers.contains_key("%gra"));
    }

    #[test]
    fn test_raw_lines_captured() {
        let (_, _, raw_lines, _) = parse_chat_str(make_basic_chat(), true);
        assert!(raw_lines.iter().any(|l| l == "@UTF8"));
        assert!(raw_lines.iter().any(|l| l == "@Begin"));
        assert!(raw_lines.iter().any(|l| l.starts_with("@Participants:")));
        assert!(raw_lines.iter().any(|l| l.starts_with("*CHI:")));
        assert!(raw_lines.iter().any(|l| l.starts_with("%mor:")));
        assert!(raw_lines.iter().any(|l| l == "@End"));
    }

    #[test]
    fn test_serialize_round_trip() {
        let input = make_basic_chat();
        let (_, _, raw_lines, _) = parse_chat_str(input, true);
        let file = ChatFile {
            file_path: "test".to_string(),
            headers: Headers::default(),
            events: vec![],
            raw_lines,
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        };
        let output = serialize_chat_file(&file);
        // Re-parse and verify the lines match.
        let (_, _, raw_lines2, _) = parse_chat_str(&output, true);
        let (_, _, raw_lines_orig, _) = parse_chat_str(input, true);
        assert_eq!(raw_lines2, raw_lines_orig);
    }

    fn make_chat_file(id: &str, chat_str: &str) -> ChatFile {
        let (headers, events, raw_lines, _) = parse_chat_str(chat_str, false);
        ChatFile {
            file_path: id.to_string(),
            headers,
            events,
            raw_lines,
            py_utterances: Arc::new(OnceLock::new()),
            py_tokens: Arc::new(OnceLock::new()),
        }
    }

    fn make_chat(files: Vec<ChatFile>) -> Chat {
        Chat {
            files: VecDeque::from(files),
        }
    }

    #[test]
    fn test_push_back() {
        let mut chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let other = make_chat(vec![make_chat_file("b", make_basic_chat())]);
        chat.push_back(&other);
        assert_eq!(chat.files.len(), 2);
        assert_eq!(chat.files[0].file_path, "a");
        assert_eq!(chat.files[1].file_path, "b");
    }

    #[test]
    fn test_push_front() {
        let mut chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let other = make_chat(vec![
            make_chat_file("b", make_basic_chat()),
            make_chat_file("c", make_basic_chat()),
        ]);
        chat.push_front(&other);
        assert_eq!(chat.files.len(), 3);
        assert_eq!(chat.files[0].file_path, "b");
        assert_eq!(chat.files[1].file_path, "c");
        assert_eq!(chat.files[2].file_path, "a");
    }

    #[test]
    fn test_pop_back() {
        let mut chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        let popped = chat.pop_back().unwrap();
        assert_eq!(chat.files.len(), 1);
        assert_eq!(chat.files[0].file_path, "a");
        assert_eq!(popped.files.len(), 1);
        assert_eq!(popped.files[0].file_path, "b");
    }

    #[test]
    fn test_pop_front() {
        let mut chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        let popped = chat.pop_front().unwrap();
        assert_eq!(chat.files.len(), 1);
        assert_eq!(chat.files[0].file_path, "b");
        assert_eq!(popped.files.len(), 1);
        assert_eq!(popped.files[0].file_path, "a");
    }

    #[test]
    fn test_pop_empty() {
        let mut chat = make_chat(vec![]);
        assert!(chat.pop_back().is_none());
        assert!(chat.pop_front().is_none());
    }

    #[test]
    fn test_from_utterances() {
        let utts = vec![
            Utterance {
                participant: Some("CHI".to_string()),
                tokens: Some(vec![Token {
                    word: "hello".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                }]),
                time_marks: None,
                tiers: None,
                changeable_header: None,
            },
            Utterance {
                participant: Some("MOT".to_string()),
                tokens: Some(vec![Token {
                    word: "hi".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                }]),
                time_marks: None,
                tiers: None,
                changeable_header: None,
            },
        ];
        let chat = Chat::from_utterances(utts.clone());
        assert_eq!(chat.files.len(), 1);
        assert_eq!(chat.files[0].events.len(), 2);
        assert_eq!(chat.files[0].events, utts);
        assert_eq!(chat.files[0].headers, Headers::default());
        // No tiers → raw_lines is empty.
        assert!(chat.files[0].raw_lines.is_empty());
    }

    #[test]
    fn test_from_utterances_empty() {
        let chat = Chat::from_utterances(Vec::<Utterance>::new());
        assert_eq!(chat.files.len(), 1);
        assert!(chat.files[0].events.is_empty());
    }

    #[test]
    fn test_from_utterances_with_tiers() {
        let mut tiers = HashMap::new();
        tiers.insert("CHI".to_string(), "hello .".to_string());
        tiers.insert("%mor".to_string(), "co|hello .".to_string());
        let utts = vec![Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![Token {
                word: "hello".to_string(),
                pos: Some("co".to_string()),
                mor: Some("hello".to_string()),
                gra: None,
            }]),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
        }];
        let chat = Chat::from_utterances(utts);
        assert_eq!(chat.files[0].raw_lines.len(), 2);
        assert_eq!(chat.files[0].raw_lines[0], "*CHI:\thello .");
        assert_eq!(chat.files[0].raw_lines[1], "%mor:\tco|hello .");
    }

    #[test]
    fn test_from_utterances_serialization_round_trip() {
        // Parse CHAT, extract utterances, reconstruct, serialize, re-parse.
        let (original, _) = Chat::from_strs(vec![make_basic_chat().to_string()], None, false);
        let utts: Vec<Utterance> = original
            .files
            .iter()
            .flat_map(|f| f.utterances().cloned())
            .collect();
        let rebuilt = Chat::from_utterances(utts);
        let serialized = rebuilt.to_strings();
        assert_eq!(serialized.len(), 1);
        let output = &serialized[0];
        // The serialized output should contain the utterance content.
        assert!(output.contains("*CHI:"));
        assert!(output.contains("%mor:"));
        assert!(output.ends_with("@End\n"));
    }

    #[test]
    fn test_clear() {
        let mut chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        chat.clear();
        assert_eq!(chat.files.len(), 0);
    }

    #[test]
    fn test_serialize_chat_file() {
        let file = make_chat_file("test", make_basic_chat());
        let output = serialize_chat_file(&file);
        assert!(output.starts_with("@UTF8\n"));
        assert!(output.contains("*CHI:"));
        assert!(output.contains("%mor:"));
        assert!(output.ends_with("@End\n"));
        // Ensure only one @End.
        assert_eq!(output.matches("@End").count(), 1);
    }

    #[test]
    fn test_serialize_ensures_at_end() {
        // Input without @End.
        let input = "@UTF8\n@Begin\n*CHI:\thello .\n";
        let file = make_chat_file("test", input);
        let output = serialize_chat_file(&file);
        assert!(output.ends_with("@End\n"));
        assert_eq!(output.matches("@End").count(), 1);
    }

    #[test]
    fn test_to_strings() {
        let chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        let strs = chat.to_strings();
        assert_eq!(strs.len(), 2);
        assert!(strs[0].contains("@UTF8"));
        assert!(strs[0].contains("@End"));
        assert!(strs[1].contains("*CHI:"));
    }

    // -----------------------------------------------------------------------
    // Developmental measures tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mlum_basic() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.mlum("CHI", Some(100));
        assert_eq!(result.len(), 1);
        // CHI: 3 morphemes (I, want, cookie)
        assert!((result[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mlu_aliases_mlum() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        assert_eq!(chat.mlum("CHI", Some(100)), chat.mlum("CHI", Some(100)));
    }

    #[test]
    fn test_mluw_basic() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.mluw("CHI", Some(100));
        assert_eq!(result.len(), 1);
        // CHI: 3 words (I, want, cookie)
        assert!((result[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ttr_basic() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.ttr("CHI", Some(350));
        assert_eq!(result.len(), 1);
        // Words: I, want, cookie, no -> 4 unique / 4 total = 1.0
        assert!((result[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mlum_empty() {
        let chat = make_chat(vec![]);
        assert!(chat.mlum("CHI", Some(100)).is_empty());
    }

    #[test]
    fn test_mluw_empty() {
        let chat = make_chat(vec![]);
        assert!(chat.mluw("CHI", Some(100)).is_empty());
    }

    #[test]
    fn test_ttr_empty() {
        let chat = make_chat(vec![]);
        assert!(chat.ttr("CHI", Some(350)).is_empty());
    }

    #[test]
    fn test_measures_multiple_files() {
        let chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        let mlum = chat.mlum("CHI", Some(100));
        let mluw = chat.mluw("CHI", Some(100));
        let ttr = chat.ttr("CHI", Some(350));
        assert_eq!(mlum.len(), 2);
        assert_eq!(mluw.len(), 2);
        assert_eq!(ttr.len(), 2);
        assert!((mlum[0] - 3.0).abs() < f64::EPSILON);
        assert!((mlum[1] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mlum_with_clitics() {
        let input =
            "@UTF8\n@Begin\n*CHI:\tthat's good .\n%mor:\tpro:dem|that~cop|be&3S adj|good .\n@End\n";
        let chat = make_chat(vec![make_chat_file("a", input)]);
        let result = chat.mlum("CHI", Some(100));
        // Morphemes: that(pro:dem), be&3S(cop clitic), good(adj) = 3
        assert!((result[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mluw_with_clitics() {
        let input =
            "@UTF8\n@Begin\n*CHI:\tthat's good .\n%mor:\tpro:dem|that~cop|be&3S adj|good .\n@End\n";
        let chat = make_chat(vec![make_chat_file("a", input)]);
        let result = chat.mluw("CHI", Some(100));
        // Words: "that's"(non-empty, pos non-empty), ""(clitic excluded), "good", "."(pos="" excluded)
        // = 2 words
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ttr_with_repeated_words() {
        let input = "@UTF8\n@Begin\n*CHI:\tno no no .\n%mor:\tco|no co|no co|no .\n@End\n";
        let chat = make_chat(vec![make_chat_file("a", input)]);
        let result = chat.ttr("CHI", Some(350));
        // 1 unique word / 3 total = 0.333...
        assert!((result[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_measures_no_mor_tier() {
        let input = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n";
        let chat = make_chat(vec![make_chat_file("a", input)]);
        // mlum: pos is None for all tokens -> 0 morphemes -> 0.0
        let mlum = chat.mlum("CHI", Some(100));
        assert!((mlum[0] - 0.0).abs() < f64::EPSILON);
        // mluw: word non-empty AND pos != Some("") -> all 3 tokens counted
        let mluw = chat.mluw("CHI", Some(100));
        assert!((mluw[0] - 3.0).abs() < f64::EPSILON);
        // ttr: 3 unique / 3 total = 1.0
        let ttr = chat.ttr("CHI", Some(350));
        assert!((ttr[0] - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // head / tail tests
    // -----------------------------------------------------------------------

    /// Format an Utterances as text (same logic as __repr__, for testing).
    fn utterances_text(us: &Utterances) -> String {
        us.utterances
            .iter()
            .map(|u| u.to_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    #[test]
    fn test_head_first_utterance() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.head(1);
        assert_eq!(result.utterances.len(), 1);
        let text = utterances_text(&result);
        assert!(text.contains("*CHI:"));
        assert!(!text.contains("*MOT:"));
    }

    #[test]
    fn test_head_all_utterances() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.head(5);
        assert_eq!(result.utterances.len(), 2);
        let text = utterances_text(&result);
        assert!(text.contains("*CHI:"));
        assert!(text.contains("*MOT:"));
        // Two utterances separated by blank line.
        assert!(text.contains("\n\n"));
    }

    #[test]
    fn test_tail_last_utterance() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let result = chat.tail(1);
        assert_eq!(result.utterances.len(), 1);
        let text = utterances_text(&result);
        assert!(text.contains("*MOT:"));
        assert!(!text.contains("*CHI:"));
    }

    #[test]
    fn test_head_empty() {
        let chat = make_chat(vec![]);
        let result = chat.head(5);
        assert_eq!(result.utterances.len(), 0);
        assert_eq!(utterances_text(&result), "");
    }

    #[test]
    fn test_tail_empty() {
        let chat = make_chat(vec![]);
        let result = chat.tail(5);
        assert_eq!(result.utterances.len(), 0);
        assert_eq!(utterances_text(&result), "");
    }

    #[test]
    fn test_head_across_files() {
        let chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        // 2 utts per file, head(3) = CHI + MOT from file a, CHI from file b
        let result = chat.head(3);
        assert_eq!(result.utterances.len(), 3);
        let text = utterances_text(&result);
        assert_eq!(text.matches("*CHI:").count(), 2);
        assert_eq!(text.matches("*MOT:").count(), 1);
    }

    #[test]
    fn test_tail_across_files() {
        let chat = make_chat(vec![
            make_chat_file("a", make_basic_chat()),
            make_chat_file("b", make_basic_chat()),
        ]);
        // 4 utts total, tail(3) = MOT from file a, CHI + MOT from file b
        let result = chat.tail(3);
        assert_eq!(result.utterances.len(), 3);
        let text = utterances_text(&result);
        assert_eq!(text.matches("*CHI:").count(), 1);
        assert_eq!(text.matches("*MOT:").count(), 2);
    }

    #[test]
    fn test_head_contains_mor_and_gra() {
        let chat = make_chat(vec![make_chat_file("a", make_basic_chat())]);
        let text = utterances_text(&chat.head(1));
        assert!(text.contains("%mor:"));
        assert!(text.contains("%gra:"));
        assert!(text.contains("pro|I"));
        assert!(text.contains("1|2|SUBJ"));
    }

    // -----------------------------------------------------------------------
    // Chat reading methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_from_strs() {
        let (chat, misalignments) = Chat::from_strs(
            vec![make_basic_chat().to_string()],
            Some(vec!["test-id".to_string()]),
            false,
        );
        assert!(misalignments.is_empty());
        assert_eq!(chat.num_files(), 1);
        assert_eq!(chat.file_paths(), vec!["test-id"]);
        let utts: Vec<&Utterance> = chat
            .files()
            .iter()
            .flat_map(|f| f.real_utterances())
            .collect();
        assert_eq!(utts.len(), 2);
        assert_eq!(utts[0].participant.as_deref(), Some("CHI"));
        assert_eq!(utts[1].participant.as_deref(), Some("MOT"));
    }

    #[test]
    fn test_chat_from_strs_auto_ids() {
        let (chat, _) = Chat::from_strs(
            vec![make_basic_chat().to_string(), make_basic_chat().to_string()],
            None,
            false,
        );
        assert_eq!(chat.num_files(), 2);
        // Auto-generated UUIDs should be unique.
        let paths = chat.file_paths();
        assert_ne!(paths[0], paths[1]);
    }

    #[test]
    #[should_panic(expected = "strs and ids must have the same length")]
    fn test_chat_from_strs_length_mismatch() {
        Chat::from_strs(
            vec![make_basic_chat().to_string()],
            Some(vec!["a".to_string(), "b".to_string()]),
            false,
        );
    }

    #[test]
    fn test_chat_read_files() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.cha");
        std::fs::write(&file_path, make_basic_chat()).unwrap();

        let (chat, misalignments) =
            Chat::read_files(&[file_path.to_string_lossy().to_string()], false).unwrap();
        assert!(misalignments.is_empty());
        assert_eq!(chat.num_files(), 1);
        let utts: Vec<&Utterance> = chat
            .files()
            .iter()
            .flat_map(|f| f.real_utterances())
            .collect();
        assert_eq!(utts.len(), 2);
    }

    #[test]
    fn test_chat_read_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.cha"), make_basic_chat()).unwrap();
        std::fs::write(dir.path().join("b.cha"), make_basic_chat()).unwrap();
        std::fs::write(dir.path().join("c.txt"), "not a chat file").unwrap();

        let (chat, _) = Chat::read_dir(&dir.path().to_string_lossy(), None, ".cha", false).unwrap();
        assert_eq!(chat.num_files(), 2);
    }

    #[test]
    fn test_chat_read_dir_with_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("alpha.cha"), make_basic_chat()).unwrap();
        std::fs::write(dir.path().join("beta.cha"), make_basic_chat()).unwrap();

        let (chat, _) =
            Chat::read_dir(&dir.path().to_string_lossy(), Some("alpha"), ".cha", false).unwrap();
        assert_eq!(chat.num_files(), 1);
    }

    #[test]
    fn test_chat_read_zip() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("test.zip");
        let file = std::fs::File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("a.cha", options).unwrap();
        std::io::Write::write_all(&mut zip, make_basic_chat().as_bytes()).unwrap();
        zip.start_file("b.cha", options).unwrap();
        std::io::Write::write_all(&mut zip, make_basic_chat().as_bytes()).unwrap();
        zip.start_file("c.txt", options).unwrap();
        std::io::Write::write_all(&mut zip, b"not a chat file").unwrap();
        zip.finish().unwrap();

        let (chat, _) = Chat::read_zip(&zip_path.to_string_lossy(), None, ".cha", false).unwrap();
        assert_eq!(chat.num_files(), 2);
    }

    #[test]
    fn test_chat_read_zip_with_match() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("test.zip");
        let file = std::fs::File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("alpha.cha", options).unwrap();
        std::io::Write::write_all(&mut zip, make_basic_chat().as_bytes()).unwrap();
        zip.start_file("beta.cha", options).unwrap();
        std::io::Write::write_all(&mut zip, make_basic_chat().as_bytes()).unwrap();
        zip.finish().unwrap();

        let (chat, _) =
            Chat::read_zip(&zip_path.to_string_lossy(), Some("alpha"), ".cha", false).unwrap();
        assert_eq!(chat.num_files(), 1);
    }
}
