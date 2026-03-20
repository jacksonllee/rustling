//! CHAT data reader for CHILDES/TalkBank transcripts.

#[cfg(feature = "pyo3")]
use super::utterance_py::{PyToken, PyUtterance};
use crate::chat::clean_utterance::clean_utterance;
use crate::chat::header::{
    Age, ChangeableHeader, Headers, Participant, parse_changeable, parse_file_headers,
    split_header_line,
};
use crate::chat::utterance::{BaseUtterance, Gra, Token, Utterance, Utterances};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use fancy_regex::Regex as FancyRegex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::LazyLock;
#[cfg(feature = "pyo3")]
use std::sync::{Arc, OnceLock};

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
    #[cfg(feature = "pyo3")]
    pub(crate) py_utterances: Arc<OnceLock<Vec<Py<PyUtterance>>>>,
    /// Lazy cache of Python Token objects, grouped per real utterance.
    #[cfg(feature = "pyo3")]
    pub(crate) py_tokens: Arc<OnceLock<Vec<Vec<Py<PyToken>>>>>,
}

impl Clone for ChatFile {
    fn clone(&self) -> Self {
        Self {
            file_path: self.file_path.clone(),
            headers: self.headers.clone(),
            events: self.events.clone(),
            raw_lines: self.raw_lines.clone(),
            #[cfg(feature = "pyo3")]
            py_utterances: Arc::new(OnceLock::new()),
            #[cfg(feature = "pyo3")]
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
            #[cfg(feature = "pyo3")]
            py_utterances: Arc::new(OnceLock::new()),
            #[cfg(feature = "pyo3")]
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

    pub(crate) fn eq_data(&self, other: &ChatFile) -> bool {
        self.file_path == other.file_path
            && self.headers == other.headers
            && self.events == other.events
            && self.raw_lines == other.raw_lines
    }

    /// Whether this file contains no utterances or headers.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Reset all cached Python objects.
    ///
    /// Call this after mutating `events` (e.g. via participant filtering)
    /// to avoid returning stale cached data.
    pub fn reset_caches(&mut self) {
        #[cfg(feature = "pyo3")]
        {
            self.py_utterances = Arc::new(OnceLock::new());
            self.py_tokens = Arc::new(OnceLock::new());
        }
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
    /// The `%`-prefixed name of the mor tier (e.g., `"%mor"`, `"%xmor"`).
    pub mor_tier_name: String,
    /// The raw content of the mor tier.
    pub mor_tier_content: String,
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
    /// A remote source error occurred (git clone, HTTP download, etc.).
    Source(crate::sources::SourceError),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Io(e) => write!(f, "{e}"),
            ChatError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
            ChatError::Zip(e) => write!(f, "{e}"),
            ChatError::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ChatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ChatError::Io(e) => Some(e),
            ChatError::Source(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::sources::SourceError> for ChatError {
    fn from(e: crate::sources::SourceError) -> Self {
        ChatError::Source(e)
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
    mor_tier: Option<&str>,
    gra_tier: Option<&str>,
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
            .map(|tg| build_utterance(tg, mor_tier, gra_tier))
            .collect()
    } else {
        tier_groups
            .iter()
            .map(|tg| build_utterance(tg, mor_tier, gra_tier))
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(Utterance, Option<MisalignmentInfo>)> = tier_groups
        .iter()
        .map(|tg| build_utterance(tg, mor_tier, gra_tier))
        .collect();

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
                    mor_tier_name: None,
                    gra_tier_name: None,
                });
            }
        }
    }

    (headers, events, lines, misalignments)
}

/// Build an Utterance from a TierGroup.
///
/// `mor_tier` and `gra_tier` are the `%`-prefixed tier names to use for
/// morphology and grammatical relations (e.g., `Some("%mor")`, `Some("%xmor")`).
/// If either is `None`, all mor+gra parsing is disabled.
///
/// Returns `(utterance, misalignment)`. `misalignment` is `Some` when
/// the word count doesn't match the non-clitic mor item count.
fn build_utterance(
    group: &TierGroup,
    mor_tier: Option<&str>,
    gra_tier: Option<&str>,
) -> (Utterance, Option<MisalignmentInfo>) {
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

    // If either tier is None, disable both mor+gra parsing.
    let (mor_items, gra_items) = if let (Some(mt), Some(gt)) = (mor_tier, gra_tier) {
        (
            group.dependent_tiers.get(mt).map(|s| parse_mor_tier(s)),
            group.dependent_tiers.get(gt).map(|s| parse_gra_tier(s)),
        )
    } else {
        (None, None)
    };

    // Build tokens.
    let (tokens, misalignment_counts) =
        build_tokens(&words, mor_items.as_deref(), gra_items.as_deref());

    // Build misalignment info if detected.
    let misalignment = misalignment_counts.map(|counts| MisalignmentInfo {
        file_path: String::new(), // Populated later by the caller.
        participant: group.participant.clone(),
        main_tier: group.main_tier.clone(),
        mor_tier_name: mor_tier.unwrap_or("%mor").to_string(),
        mor_tier_content: mor_tier
            .and_then(|mt| group.dependent_tiers.get(mt))
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
            mor_tier_name: mor_tier.map(|s| s.to_string()),
            gra_tier_name: gra_tier.map(|s| s.to_string()),
        },
        misalignment,
    )
}

/// Filter file paths by match regex pattern.
pub fn filter_file_paths(
    paths: &[String],
    match_pattern: Option<&str>,
) -> Result<Vec<String>, ModelError> {
    let match_re = match_pattern
        .map(FancyRegex::new)
        .transpose()
        .map_err(|e| ModelError::ValidationError(format!("Invalid match regex: {e}")))?;

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
    mor_tier: Option<&str>,
    gra_tier: Option<&str>,
) -> (Vec<ChatFile>, Vec<MisalignmentInfo>) {
    let build = |content: &str, id: &str| {
        let (headers, events, raw_lines, mut mis) =
            parse_chat_str(content, parallel, mor_tier, gra_tier);
        for m in &mut mis {
            m.file_path = id.to_string();
        }
        (
            ChatFile::new(id.to_string(), headers, events, raw_lines),
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
    mor_tier: Option<&str>,
    gra_tier: Option<&str>,
) -> Result<(Vec<ChatFile>, Vec<MisalignmentInfo>), std::io::Error> {
    let build = |path: &str| -> Result<(ChatFile, Vec<MisalignmentInfo>), std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let (headers, events, raw_lines, mut mis) =
            parse_chat_str(&content, parallel, mor_tier, gra_tier);
        for m in &mut mis {
            m.file_path = path.to_string();
        }
        Ok((
            ChatFile::new(path.to_string(), headers, events, raw_lines),
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

use crate::persistence::ModelError;

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

    /// Derive default output filenames from existing file paths.
    ///
    /// If all files have real (non-UUID) filenames, use those with the
    /// extension replaced by `target_ext` (e.g., `".cha"` or `".eaf"`).
    /// Otherwise fall back to numbered names (`0001.cha`, etc.).
    fn default_output_filenames(&self, target_ext: &str) -> Vec<String> {
        let derived: Vec<Option<String>> = self
            .files()
            .iter()
            .map(|f| {
                let path = std::path::Path::new(&f.file_path);
                let stem = path.file_stem()?.to_str()?;
                // If the stem is a UUID, treat as in-memory / unnamed.
                if uuid::Uuid::try_parse(stem).is_ok() {
                    return None;
                }
                Some(format!("{stem}{target_ext}"))
            })
            .collect();

        // Use derived names only if ALL files have them and there are no duplicates.
        if derived.iter().all(|d| d.is_some()) {
            let names: Vec<String> = derived.into_iter().map(|d| d.unwrap()).collect();
            let unique: HashSet<&String> = names.iter().collect();
            if unique.len() == names.len() {
                return names;
            }
        }

        // Fallback: numbered.
        (0..self.files().len())
            .map(|i| format!("{:04}{target_ext}", i + 1))
            .collect()
    }

    /// Return CHAT data strings, one per file.
    fn to_strings(&self) -> Vec<String> {
        self.files().iter().map(serialize_chat_file).collect()
    }

    /// Return EAF XML strings (one per file) for ELAN export.
    fn to_elan_strings(&self) -> Vec<String> {
        self.files()
            .iter()
            .map(super::elan_writer::chat_file_to_eaf_xml)
            .collect()
    }

    /// Convert to an [`Elan`] object.
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
        // The generated XML is always valid, so unwrap is safe.
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

    /// Write CHAT (.cha) files to a directory.
    fn write_chat_files(
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

    // -----------------------------------------------------------------------
    // SRT conversion
    // -----------------------------------------------------------------------

    /// Return SRT format strings (one per file) for SRT export.
    fn to_srt_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::srt_writer::chat_file_to_srt_str(f, participants))
            .collect()
    }

    /// Convert to an [`Srt`](crate::srt::Srt) object.
    fn to_srt(&self, participants: Option<&[String]>) -> crate::srt::Srt {
        let strs = self.to_srt_strings(participants);
        let ids: Vec<String> = self
            .files()
            .iter()
            .map(|f| {
                let path = std::path::Path::new(&f.file_path);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if uuid::Uuid::try_parse(stem).is_ok() {
                    f.file_path.clone()
                } else {
                    format!("{stem}.srt")
                }
            })
            .collect();
        crate::srt::Srt::from_strs(strs, Some(ids), false).unwrap()
    }

    /// Write SRT (.srt) files to a directory.
    fn write_srt_files(
        &self,
        dir_path: &str,
        participants: Option<&[String]>,
        filenames: Option<Vec<String>>,
    ) -> Result<(), WriteError> {
        let strs = self.to_srt_strings(participants);
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

    // -----------------------------------------------------------------------
    // CoNLL-U conversion
    // -----------------------------------------------------------------------

    /// Return CoNLL-U format strings (one per file) for CoNLL-U export.
    fn to_conllu_strings(&self) -> Vec<String> {
        self.files()
            .iter()
            .map(super::conllu_writer::chat_file_to_conllu_str)
            .collect()
    }

    /// Convert to a [`Conllu`](crate::conllu::Conllu) object.
    fn to_conllu(&self) -> crate::conllu::Conllu {
        let strs = self.to_conllu_strings();
        let ids: Vec<String> = self
            .files()
            .iter()
            .map(|f| {
                let path = std::path::Path::new(&f.file_path);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if uuid::Uuid::try_parse(stem).is_ok() {
                    f.file_path.clone()
                } else {
                    format!("{stem}.conllu")
                }
            })
            .collect();
        crate::conllu::Conllu::from_strs(strs, Some(ids), false).unwrap()
    }

    /// Write CoNLL-U (.conllu) files to a directory.
    fn write_conllu_files(
        &self,
        dir_path: &str,
        filenames: Option<Vec<String>>,
    ) -> Result<(), WriteError> {
        let strs = self.to_conllu_strings();
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

    // -----------------------------------------------------------------------
    // TextGrid conversion
    // -----------------------------------------------------------------------

    /// Return TextGrid format strings (one per file) for TextGrid export.
    fn to_textgrid_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::textgrid_writer::chat_file_to_textgrid_str(f, participants))
            .collect()
    }

    /// Convert to a [`TextGrid`](crate::textgrid::TextGrid) object.
    fn to_textgrid(&self, participants: Option<&[String]>) -> crate::textgrid::TextGrid {
        let strs = self.to_textgrid_strings(participants);
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
        participants: Option<&[String]>,
        filenames: Option<Vec<String>>,
    ) -> Result<(), WriteError> {
        let strs = self.to_textgrid_strings(participants);
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

    // -----------------------------------------------------------------------
    // Filtering
    // -----------------------------------------------------------------------

    /// Filter by file path and/or participant regex patterns (pure Rust).
    fn filter_by(
        &self,
        files: Option<&str>,
        participants: Option<&str>,
    ) -> Result<Self, ModelError> {
        let mut filtered: VecDeque<ChatFile> = if let Some(pattern) = files {
            let re = FancyRegex::new(pattern)
                .map_err(|e| ModelError::ValidationError(format!("Invalid file regex: {e}")))?;
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
            let re = FancyRegex::new(&anchored).map_err(|e| {
                ModelError::ValidationError(format!("Invalid participant regex: {e}"))
            })?;
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
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
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
        let (files, misalignments) = parse_chat_strs(pairs, parallel, mor_tier, gra_tier);
        (Self::from_chat_files(files), misalignments)
    }

    /// Load and parse CHAT data from file paths.
    ///
    /// Returns `(Chat, misalignments)` on success.
    pub fn read_files(
        paths: &[String],
        parallel: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> Result<(Self, Vec<MisalignmentInfo>), std::io::Error> {
        let (files, misalignments) = load_chat_files(paths, parallel, mor_tier, gra_tier)?;
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
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
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

        let filtered = filter_file_paths(&paths, match_pattern)
            .map_err(|e| ChatError::InvalidPattern(e.to_string()))?;
        let (files, misalignments) = load_chat_files(&filtered, parallel, mor_tier, gra_tier)?;
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
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
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

        let filtered = filter_file_paths(&entry_names, match_pattern)
            .map_err(|e| ChatError::InvalidPattern(e.to_string()))?;

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

        let (files, misalignments) = parse_chat_strs(pairs, parallel, mor_tier, gra_tier);
        Ok((Self::from_chat_files(files), misalignments))
    }

    /// Load CHAT data from a git repository.
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
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> Result<(Self, Vec<MisalignmentInfo>), ChatError> {
        let local_path = crate::sources::resolve_git(url, rev, depth, cache_dir, force_download)?;
        let path = local_path.to_string_lossy();
        Self::read_dir(
            &path,
            match_pattern,
            extension,
            parallel,
            mor_tier,
            gra_tier,
        )
    }

    /// Load CHAT data from a URL.
    ///
    /// Downloads the file (or uses a cached copy) and parses it.
    /// ZIP files are automatically detected by URL suffix or magic bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_url(
        url: &str,
        match_pattern: Option<&str>,
        extension: &str,
        cache_dir: Option<std::path::PathBuf>,
        force_download: bool,
        parallel: bool,
        mor_tier: Option<&str>,
        gra_tier: Option<&str>,
    ) -> Result<(Self, Vec<MisalignmentInfo>), ChatError> {
        let (local_path, is_zip) = crate::sources::resolve_url(url, cache_dir, force_download)?;
        let path = local_path.to_string_lossy();
        if is_zip {
            Self::read_zip(
                &path,
                match_pattern,
                extension,
                parallel,
                mor_tier,
                gra_tier,
            )
        } else {
            let content = std::fs::read_to_string(local_path)?;
            Ok(Self::from_strs(
                vec![content],
                None,
                parallel,
                mor_tier,
                gra_tier,
            ))
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
        let empty_file = ChatFile::new(String::new(), Headers::default(), vec![], vec![]);
        assert!(empty_file.is_empty());

        let (headers, events, raw_lines, _) =
            parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, DEFAULT_GRA);
        let non_empty_file = ChatFile::new("test".to_string(), headers, events, raw_lines);
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
        let (_, events, _, _) = parse_chat_str(input, true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, events, _, misalignments) = parse_chat_str(input, false, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, events, _, _) = parse_chat_str(input, true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, events, _, _) = parse_chat_str(input, true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, _, _, misalignments) = parse_chat_str(input, false, DEFAULT_MOR, DEFAULT_GRA);
        assert!(!misalignments.is_empty());
        assert_eq!(misalignments[0].participant, "CHI");
    }

    #[test]
    fn test_parse_chat_str_no_misalignment() {
        let (_, _, _, misalignments) =
            parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, _, raw_lines, _) =
            parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, DEFAULT_GRA);
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
        let (_, _, raw_lines, _) = parse_chat_str(input, true, DEFAULT_MOR, DEFAULT_GRA);
        let file = ChatFile::new("test".to_string(), Headers::default(), vec![], raw_lines);
        let output = serialize_chat_file(&file);
        // Re-parse and verify the lines match.
        let (_, _, raw_lines2, _) = parse_chat_str(&output, true, DEFAULT_MOR, DEFAULT_GRA);
        let (_, _, raw_lines_orig, _) = parse_chat_str(input, true, DEFAULT_MOR, DEFAULT_GRA);
        assert_eq!(raw_lines2, raw_lines_orig);
    }

    /// Default mor/gra tier names for tests.
    const DEFAULT_MOR: Option<&str> = Some("%mor");
    const DEFAULT_GRA: Option<&str> = Some("%gra");

    fn make_chat_file(id: &str, chat_str: &str) -> ChatFile {
        let (headers, events, raw_lines, _) =
            parse_chat_str(chat_str, false, DEFAULT_MOR, DEFAULT_GRA);
        ChatFile::new(id.to_string(), headers, events, raw_lines)
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
                mor_tier_name: Some("%mor".to_string()),
                gra_tier_name: Some("%gra".to_string()),
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
                mor_tier_name: Some("%mor".to_string()),
                gra_tier_name: Some("%gra".to_string()),
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
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        }];
        let chat = Chat::from_utterances(utts);
        assert_eq!(chat.files[0].raw_lines.len(), 2);
        assert_eq!(chat.files[0].raw_lines[0], "*CHI:\thello .");
        assert_eq!(chat.files[0].raw_lines[1], "%mor:\tco|hello .");
    }

    #[test]
    fn test_from_utterances_serialization_round_trip() {
        // Parse CHAT, extract utterances, reconstruct, serialize, re-parse.
        let (original, _) = Chat::from_strs(
            vec![make_basic_chat().to_string()],
            None,
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        );
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
            DEFAULT_MOR,
            DEFAULT_GRA,
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
            DEFAULT_MOR,
            DEFAULT_GRA,
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
            DEFAULT_MOR,
            DEFAULT_GRA,
        );
    }

    #[test]
    fn test_chat_read_files() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.cha");
        std::fs::write(&file_path, make_basic_chat()).unwrap();

        let (chat, misalignments) = Chat::read_files(
            &[file_path.to_string_lossy().to_string()],
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        )
        .unwrap();
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

        let (chat, _) = Chat::read_dir(
            &dir.path().to_string_lossy(),
            None,
            ".cha",
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        )
        .unwrap();
        assert_eq!(chat.num_files(), 2);
    }

    #[test]
    fn test_chat_read_dir_with_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("alpha.cha"), make_basic_chat()).unwrap();
        std::fs::write(dir.path().join("beta.cha"), make_basic_chat()).unwrap();

        let (chat, _) = Chat::read_dir(
            &dir.path().to_string_lossy(),
            Some("alpha"),
            ".cha",
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        )
        .unwrap();
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

        let (chat, _) = Chat::read_zip(
            &zip_path.to_string_lossy(),
            None,
            ".cha",
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        )
        .unwrap();
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

        let (chat, _) = Chat::read_zip(
            &zip_path.to_string_lossy(),
            Some("alpha"),
            ".cha",
            false,
            DEFAULT_MOR,
            DEFAULT_GRA,
        )
        .unwrap();
        assert_eq!(chat.num_files(), 1);
    }

    // -----------------------------------------------------------------------
    // Tests for custom mor_tier / gra_tier
    // -----------------------------------------------------------------------

    fn make_chat_with_custom_tiers() -> &'static str {
        "@UTF8\n@Begin\n@Participants:\tCHI Child\n*CHI:\tI want cookie .\n%xmor:\tpro|I v|want n|cookie .\n%xgra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n@End\n"
    }

    #[test]
    fn test_custom_tier_names_parsed() {
        let (_, events, _, misalignments) = parse_chat_str(
            make_chat_with_custom_tiers(),
            true,
            Some("%xmor"),
            Some("%xgra"),
        );
        assert!(misalignments.is_empty());
        assert_eq!(events.len(), 1);
        let utt = &events[0];
        let tokens = utt.tokens.as_ref().unwrap();
        // mor should be parsed from %xmor: pos="pro", mor="I"
        assert_eq!(tokens[0].pos.as_deref(), Some("pro"));
        assert_eq!(tokens[0].mor.as_deref(), Some("I"));
        assert_eq!(tokens[0].gra.as_ref().unwrap().rel, "SUBJ");
        // tier names should be stored
        assert_eq!(utt.mor_tier_name.as_deref(), Some("%xmor"));
        assert_eq!(utt.gra_tier_name.as_deref(), Some("%xgra"));
    }

    #[test]
    fn test_default_tiers_ignore_custom_tier_data() {
        // If data has %xmor but we look for %mor, no mor/gra should be parsed.
        let (_, events, _, _) = parse_chat_str(
            make_chat_with_custom_tiers(),
            true,
            DEFAULT_MOR,
            DEFAULT_GRA,
        );
        let tokens = events[0].tokens.as_ref().unwrap();
        assert!(tokens[0].mor.is_none());
        assert!(tokens[0].gra.is_none());
    }

    #[test]
    fn test_none_tiers_disable_mor_gra() {
        let (_, events, _, misalignments) = parse_chat_str(make_basic_chat(), true, None, None);
        assert!(misalignments.is_empty());
        let tokens = events[0].tokens.as_ref().unwrap();
        // mor and gra should not be parsed even though %mor/%gra tiers exist
        assert!(tokens[0].mor.is_none());
        assert!(tokens[0].gra.is_none());
        // tier names should be None
        assert!(events[0].mor_tier_name.is_none());
        assert!(events[0].gra_tier_name.is_none());
    }

    #[test]
    fn test_none_mor_disables_both() {
        // If only mor_tier is None, both should be disabled
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true, None, DEFAULT_GRA);
        let tokens = events[0].tokens.as_ref().unwrap();
        assert!(tokens[0].mor.is_none());
        assert!(tokens[0].gra.is_none());
    }

    #[test]
    fn test_none_gra_disables_both() {
        // If only gra_tier is None, both should be disabled
        let (_, events, _, _) = parse_chat_str(make_basic_chat(), true, DEFAULT_MOR, None);
        let tokens = events[0].tokens.as_ref().unwrap();
        assert!(tokens[0].mor.is_none());
        assert!(tokens[0].gra.is_none());
    }

    #[test]
    fn test_custom_tiers_from_strs() {
        let (chat, misalignments) = Chat::from_strs(
            vec![make_chat_with_custom_tiers().to_string()],
            None,
            true,
            Some("%xmor"),
            Some("%xgra"),
        );
        assert!(misalignments.is_empty());
        let files = chat.files();
        let utt = files[0].utterances().next().unwrap();
        let tokens = utt.tokens.as_ref().unwrap();
        assert_eq!(tokens[0].pos.as_deref(), Some("pro"));
        assert_eq!(tokens[0].mor.as_deref(), Some("I"));
    }

    #[test]
    fn test_disabled_tiers_from_strs() {
        let (chat, _) =
            Chat::from_strs(vec![make_basic_chat().to_string()], None, true, None, None);
        let files = chat.files();
        let utt = files[0].utterances().next().unwrap();
        let tokens = utt.tokens.as_ref().unwrap();
        assert!(tokens[0].mor.is_none());
    }

    #[test]
    fn test_custom_tiers_to_chat_lines() {
        let (_, events, _, _) = parse_chat_str(
            make_chat_with_custom_tiers(),
            true,
            Some("%xmor"),
            Some("%xgra"),
        );
        let lines = events[0].to_chat_lines();
        let joined = lines.join("\n");
        // Serialized output should use %xmor and %xgra, not %mor/%gra
        assert!(joined.contains("%xmor:"));
        assert!(joined.contains("%xgra:"));
        assert!(!joined.contains("%mor:"));
        assert!(!joined.contains("%gra:"));
    }
}
