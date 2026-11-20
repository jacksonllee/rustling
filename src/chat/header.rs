//! Data structures for CHAT file headers.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

/// Age in the CHAT format: years;months.days (e.g., "2;10.05").
#[cfg_attr(feature = "pyo3", pyclass(from_py_object))]
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Age {
    pub years: u32,
    pub months: Option<u32>,
    pub days: Option<u32>,
}

/// A single participant from @Participants + @ID fields merged.
#[cfg_attr(feature = "pyo3", pyclass(from_py_object))]
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Participant {
    pub code: String,
    pub name: String,
    pub role: String,
    // From @ID (pipe-delimited fields):
    pub language: Option<String>,
    pub corpus: Option<String>,
    pub age: Option<Age>,
    pub sex: Option<String>,
    pub group: Option<String>,
    pub ses: Option<String>,
    pub education: Option<String>,
    pub custom: Option<String>,
    // From participant-specific headers:
    pub birth: Option<String>,
    pub birthplace: Option<String>,
    pub l1: Option<String>,
}

/// Media descriptor from @Media header (internal only; exposed to Python as a dict).
#[derive(Clone, Debug, Default, Hash, PartialEq)]
pub(crate) struct Media {
    pub filename: String,
    pub format: String,
    pub status: Option<String>,
}

// ---------------------------------------------------------------------------
// Date parsing
// ---------------------------------------------------------------------------

/// Parse a CHAT date string into (year, month, day).
///
/// Tries DD-MMM-YYYY first (e.g., "25-JAN-1983"), then ISO YYYY-MM-DD.
/// Returns `None` if neither format matches.
pub(crate) fn parse_chat_date(s: &str) -> Option<(i32, u32, u32)> {
    // Try DD-MMM-YYYY
    if let Some(result) = parse_dmy(s) {
        return Some(result);
    }
    // Try YYYY-MM-DD (ISO)
    parse_iso(s)
}

fn parse_dmy(s: &str) -> Option<(i32, u32, u32)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let day: u32 = parts[0].parse().ok()?;
    let month = match parts[1].to_ascii_uppercase().as_str() {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };
    let year: i32 = parts[2].parse().ok()?;
    Some((year, month, day))
}

fn parse_iso(s: &str) -> Option<(i32, u32, u32)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some((year, month, day))
}

// ---------------------------------------------------------------------------
// File-level headers
// ---------------------------------------------------------------------------

/// All file-level (non-changeable) headers from a CHAT file.
#[cfg_attr(feature = "pyo3", pyclass(from_py_object))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Headers {
    // Hidden
    pub pid: Option<String>,
    // Initial
    pub languages: Vec<String>,
    pub participants: Vec<Participant>,
    pub options: Option<String>,
    pub(crate) media_data: Option<Media>,
    // Constant + initial changeable stored at file level
    pub date: Option<String>,
    pub location: Option<String>,
    pub number: Option<String>,
    pub recording_quality: Option<String>,
    pub room_layout: Option<String>,
    pub tape_location: Option<String>,
    pub time_duration: Option<String>,
    pub time_start: Option<String>,
    pub transcriber: Option<String>,
    pub transcription: Option<String>,
    pub types: Option<String>,
    pub videos: Option<String>,
    pub warning: Option<String>,
    pub situation: Option<String>,
    // Comments (preserves all @Comment lines in order; None if no @Comment lines)
    pub comments: Option<Vec<String>>,
    // Catch-all for unrecognized headers
    pub other: HashMap<String, String>,
}

impl Headers {
    pub(crate) fn hash_into(&self, hasher: &mut impl Hasher) {
        self.pid.hash(hasher);
        self.languages.hash(hasher);
        self.participants.hash(hasher);
        self.options.hash(hasher);
        self.media_data.hash(hasher);
        self.date.hash(hasher);
        self.location.hash(hasher);
        self.number.hash(hasher);
        self.recording_quality.hash(hasher);
        self.room_layout.hash(hasher);
        self.tape_location.hash(hasher);
        self.time_duration.hash(hasher);
        self.time_start.hash(hasher);
        self.transcriber.hash(hasher);
        self.transcription.hash(hasher);
        self.types.hash(hasher);
        self.videos.hash(hasher);
        self.warning.hash(hasher);
        self.situation.hash(hasher);
        self.comments.hash(hasher);
        hash_hashmap(&self.other, hasher);
    }
}

// ---------------------------------------------------------------------------
// Changeable headers (can appear mid-file)
// ---------------------------------------------------------------------------

/// A changeable header that can appear mid-file in CHAT transcripts.
#[cfg_attr(feature = "pyo3", pyclass(eq, hash, from_py_object))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum ChangeableHeader {
    Activities { value: String },
    Bck { value: String },
    Bg { value: Option<String> },
    Blank {},
    Comment { value: String },
    Date { value: String },
    Eg { value: Option<String> },
    G { value: Option<String> },
    NewEpisode {},
    Page { value: String },
    Situation { value: String },
}

/// Hash a `HashMap<String, String>` deterministically by sorting entries.
pub(crate) fn hash_hashmap(map: &HashMap<String, String>, hasher: &mut impl Hasher) {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by_key(|(k, _)| k.as_str());
    entries.len().hash(hasher);
    for (k, v) in &entries {
        k.hash(hasher);
        v.hash(hasher);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chat_date_dmy() {
        assert_eq!(parse_chat_date("25-JAN-1983"), Some((1983, 1, 25)));
        assert_eq!(parse_chat_date("12-NOV-1962"), Some((1962, 11, 12)));
        assert_eq!(parse_chat_date("01-feb-2020"), Some((2020, 2, 1)));
        assert_eq!(parse_chat_date("31-Dec-1999"), Some((1999, 12, 31)));
    }

    #[test]
    fn test_parse_chat_date_iso() {
        assert_eq!(parse_chat_date("1983-01-25"), Some((1983, 1, 25)));
        assert_eq!(parse_chat_date("2020-12-31"), Some((2020, 12, 31)));
    }

    #[test]
    fn test_parse_chat_date_invalid() {
        assert_eq!(parse_chat_date("not-a-date"), None);
        assert_eq!(parse_chat_date("25/JAN/1983"), None);
        assert_eq!(parse_chat_date(""), None);
        assert_eq!(parse_chat_date("2020-13-01"), None);
        assert_eq!(parse_chat_date("2020-00-01"), None);
    }
}
