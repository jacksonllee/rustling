//! Data structures for CHAT file headers.

use crate::chat::utterance::Utterance;

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
// Ordered event model
// ---------------------------------------------------------------------------

/// An ordered event in a CHAT file: either an utterance or a changeable header.
///
/// Used as an intermediate type during parsing; not stored in `ChatFile`.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum ChatEvent {
    Utterance(Utterance),
    Header(ChangeableHeader),
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Split a header line into (name, optional value).
///
/// Examples:
///   "@Languages:\teng, zho" -> ("Languages", Some("eng, zho"))
///   "@Begin" -> ("Begin", None)
///   "@Birth of CHI:\t28-JUN-2001" -> ("Birth of CHI", Some("28-JUN-2001"))
pub(crate) fn split_header_line(line: &str) -> (&str, Option<&str>) {
    // Strip the leading '@'.
    let rest = &line[1..];
    if let Some(colon_pos) = rest.find(':') {
        let name = &rest[..colon_pos];
        let value = rest[colon_pos + 1..].trim_start_matches('\t').trim();
        if value.is_empty() {
            (name, None)
        } else {
            (name, Some(value))
        }
    } else {
        (rest.trim(), None)
    }
}

/// Parse an age string in CHAT format: "years;months.days".
///
/// Examples: "2;10.05", "6;04.", "6;", "2;10"
pub(crate) fn parse_age(s: &str) -> Option<Age> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (years_str, rest) = s.split_once(';')?;
    let years: u32 = years_str.parse().ok()?;

    if rest.is_empty() {
        return Some(Age {
            years,
            months: None,
            days: None,
        });
    }

    let (months_str, days_str) = if let Some((m, d)) = rest.split_once('.') {
        (m, d)
    } else {
        (rest, "")
    };

    let months = if months_str.is_empty() {
        None
    } else {
        Some(months_str.parse().ok()?)
    };

    let days = if days_str.is_empty() {
        None
    } else {
        Some(days_str.parse().ok()?)
    };

    Some(Age {
        years,
        months,
        days,
    })
}

/// Parse @Languages line value: "eng, zho" -> vec!["eng", "zho"]
pub(crate) fn parse_languages(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse @Participants line value into (code, name, role) tuples.
///
/// Format: "CHI Mark Target_Child, MOT Mary Mother"
/// Name may be omitted: "CHI Target_Child" means code=CHI, name="", role=Target_Child.
pub(crate) fn parse_participants(value: &str) -> Vec<(String, String, String)> {
    value
        .split(',')
        .filter_map(|segment| {
            let parts: Vec<&str> = segment.split_whitespace().collect();
            match parts.len() {
                0 => None,
                1 => Some((parts[0].to_string(), String::new(), String::new())),
                2 => {
                    // CODE Role (no name)
                    Some((parts[0].to_string(), String::new(), parts[1].to_string()))
                }
                _ => {
                    // CODE Name Role (name may have underscores)
                    Some((
                        parts[0].to_string(),
                        parts[1].to_string(),
                        parts[2..].join(" "),
                    ))
                }
            }
        })
        .collect()
}

/// Intermediate @ID fields (excluding code, which is used for matching).
pub(crate) struct IdFields {
    pub language: Option<String>,
    pub corpus: Option<String>,
    pub age: Option<Age>,
    pub sex: Option<String>,
    pub group: Option<String>,
    pub ses: Option<String>,
    pub role: Option<String>,
    pub education: Option<String>,
    pub custom: Option<String>,
}

/// Parse @ID line value: "lang|corpus|code|age|sex|group|eth,SES|role|education|custom|"
///
/// Returns (participant_code, IdFields).
pub(crate) fn parse_id(value: &str) -> Option<(String, IdFields)> {
    let fields: Vec<&str> = value.split('|').collect();
    if fields.len() < 3 {
        return None;
    }

    let code = fields[2].trim().to_string();
    if code.is_empty() {
        return None;
    }

    let opt = |idx: usize| -> Option<String> {
        fields.get(idx).and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
    };

    let age = fields.get(3).and_then(|s| parse_age(s));

    Some((
        code,
        IdFields {
            language: opt(0),
            corpus: opt(1),
            age,
            sex: opt(4),
            group: opt(5),
            ses: opt(6),
            role: opt(7),
            education: opt(8),
            custom: opt(9),
        },
    ))
}

/// Parse @Media line value: "filename, format[, status]"
pub(crate) fn parse_media(value: &str) -> Media {
    let parts: Vec<&str> = value.split(',').map(|s| s.trim()).collect();
    Media {
        filename: parts.first().unwrap_or(&"").to_string(),
        format: parts.get(1).unwrap_or(&"").to_string(),
        status: parts.get(2).map(|s| s.to_string()),
    }
}

/// Classify and parse a changeable header into a ChangeableHeader variant.
pub(crate) fn parse_changeable(name: &str, value: Option<&str>) -> Option<ChangeableHeader> {
    match name {
        "Activities" => Some(ChangeableHeader::Activities {
            value: value.unwrap_or("").to_string(),
        }),
        "Bck" => Some(ChangeableHeader::Bck {
            value: value.unwrap_or("").to_string(),
        }),
        "Bg" => Some(ChangeableHeader::Bg {
            value: value.map(|v| v.to_string()),
        }),
        "Blank" => Some(ChangeableHeader::Blank {}),
        "Comment" => Some(ChangeableHeader::Comment {
            value: value.unwrap_or("").to_string(),
        }),
        "Date" => Some(ChangeableHeader::Date {
            value: value.unwrap_or("").to_string(),
        }),
        "Eg" => Some(ChangeableHeader::Eg {
            value: value.map(|v| v.to_string()),
        }),
        "G" => Some(ChangeableHeader::G {
            value: value.map(|v| v.to_string()),
        }),
        "New Episode" => Some(ChangeableHeader::NewEpisode {}),
        "Page" => Some(ChangeableHeader::Page {
            value: value.unwrap_or("").to_string(),
        }),
        "Situation" => Some(ChangeableHeader::Situation {
            value: value.unwrap_or("").to_string(),
        }),
        _ => None,
    }
}

/// Check if a header name is a changeable header.
pub(crate) fn is_changeable(name: &str) -> bool {
    matches!(
        name,
        "Activities"
            | "Bck"
            | "Bg"
            | "Blank"
            | "Comment"
            | "Date"
            | "Eg"
            | "G"
            | "New Episode"
            | "Page"
            | "Situation"
    )
}

/// Parse all file-level headers from the initial @ lines of a CHAT file.
///
/// Returns (Headers, index of first non-header line, initial changeable events).
pub(crate) fn parse_file_headers(lines: &[String]) -> (Headers, usize, Vec<ChatEvent>) {
    let mut headers = Headers::default();
    let mut initial_events = Vec::new();
    let mut participant_map: HashMap<String, usize> = HashMap::new();
    let mut i = 0;

    while i < lines.len() {
        let line = &lines[i];
        // Stop at the first utterance or dependent tier line.
        if line.starts_with('*') || line.starts_with('%') {
            break;
        }
        if !line.starts_with('@') {
            i += 1;
            continue;
        }

        let (name, value) = split_header_line(line);

        match name {
            "UTF8" | "Begin" | "End" | "Window" | "Font" | "Color Words" | "ColorWords" => {
                // Skip: implicit or editor-specific.
            }
            "PID" => {
                headers.pid = value.map(|v| v.to_string());
            }
            "Languages" => {
                if let Some(v) = value {
                    headers.languages = parse_languages(v);
                }
            }
            "Participants" => {
                if let Some(v) = value {
                    let parsed = parse_participants(v);
                    headers.participants = parsed
                        .iter()
                        .enumerate()
                        .map(|(idx, (code, name, role))| {
                            participant_map.insert(code.clone(), idx);
                            Participant {
                                code: code.clone(),
                                name: name.clone(),
                                role: role.clone(),
                                ..Default::default()
                            }
                        })
                        .collect();
                }
            }
            "Options" => {
                headers.options = value.map(|v| v.to_string());
            }
            "ID" => {
                if let Some(v) = value
                    && let Some((code, fields)) = parse_id(v)
                    && let Some(&idx) = participant_map.get(&code)
                {
                    let p = &mut headers.participants[idx];
                    p.language = fields.language;
                    p.corpus = fields.corpus;
                    p.age = fields.age;
                    p.sex = fields.sex;
                    p.group = fields.group;
                    p.ses = fields.ses;
                    p.education = fields.education;
                    p.custom = fields.custom;
                    // Also update role from @ID if available and current role is empty.
                    if let Some(role) = fields.role
                        && p.role.is_empty()
                    {
                        p.role = role;
                    }
                }
            }
            "Media" => {
                if let Some(v) = value {
                    headers.media_data = Some(parse_media(v));
                }
            }
            // Participant-specific headers
            _ if name.starts_with("Birth of ") => {
                let code = &name["Birth of ".len()..];
                if let Some(&idx) = participant_map.get(code) {
                    headers.participants[idx].birth = value.map(|v| v.to_string());
                }
            }
            _ if name.starts_with("Birthplace of ") => {
                let code = &name["Birthplace of ".len()..];
                if let Some(&idx) = participant_map.get(code) {
                    headers.participants[idx].birthplace = value.map(|v| v.to_string());
                }
            }
            _ if name.starts_with("L1 of ") => {
                let code = &name["L1 of ".len()..];
                if let Some(&idx) = participant_map.get(code) {
                    headers.participants[idx].l1 = value.map(|v| v.to_string());
                }
            }
            // Constant headers
            "Location" => headers.location = value.map(|v| v.to_string()),
            "Number" => headers.number = value.map(|v| v.to_string()),
            "Recording Quality" => headers.recording_quality = value.map(|v| v.to_string()),
            "Room Layout" => headers.room_layout = value.map(|v| v.to_string()),
            "Tape Location" => headers.tape_location = value.map(|v| v.to_string()),
            "Time Duration" => headers.time_duration = value.map(|v| v.to_string()),
            "Time Start" => headers.time_start = value.map(|v| v.to_string()),
            "Transcriber" => headers.transcriber = value.map(|v| v.to_string()),
            "Transcription" => headers.transcription = value.map(|v| v.to_string()),
            "Types" => headers.types = value.map(|v| v.to_string()),
            "Videos" => headers.videos = value.map(|v| v.to_string()),
            "Warning" => headers.warning = value.map(|v| v.to_string()),
            // Changeable headers in the initial position: store at file level + as event
            "Date" => {
                headers.date = value.map(|v| v.to_string());
            }
            "Situation" => {
                headers.situation = value.map(|v| v.to_string());
            }
            "Comment" => {
                let v = value.unwrap_or("").to_string();
                headers
                    .comments
                    .get_or_insert_with(Vec::new)
                    .push(v.clone());
                initial_events.push(ChatEvent::Header(ChangeableHeader::Comment { value: v }));
            }
            _ if is_changeable(name) => {
                if let Some(ch) = parse_changeable(name, value) {
                    initial_events.push(ChatEvent::Header(ch));
                }
            }
            // Unknown headers
            _ => {
                if let Some(v) = value {
                    headers.other.insert(name.to_string(), v.to_string());
                }
            }
        }

        i += 1;
    }

    (headers, i, initial_events)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_header_line_with_value() {
        let (name, value) = split_header_line("@Languages:\teng, zho");
        assert_eq!(name, "Languages");
        assert_eq!(value, Some("eng, zho"));
    }

    #[test]
    fn test_split_header_line_bare() {
        let (name, value) = split_header_line("@Begin");
        assert_eq!(name, "Begin");
        assert_eq!(value, None);
    }

    #[test]
    fn test_split_header_line_bare_with_space() {
        let (name, value) = split_header_line("@New Episode");
        assert_eq!(name, "New Episode");
        assert_eq!(value, None);
    }

    #[test]
    fn test_split_header_line_participant_specific() {
        let (name, value) = split_header_line("@Birth of CHI:\t28-JUN-2001");
        assert_eq!(name, "Birth of CHI");
        assert_eq!(value, Some("28-JUN-2001"));
    }

    #[test]
    fn test_split_header_line_empty_entry() {
        let (name, value) = split_header_line("@Languages:");
        assert_eq!(name, "Languages");
        assert_eq!(value, None);
    }

    #[test]
    fn test_parse_age_full() {
        let age = parse_age("2;10.05").unwrap();
        assert_eq!(age.years, 2);
        assert_eq!(age.months, Some(10));
        assert_eq!(age.days, Some(5));
    }

    #[test]
    fn test_parse_age_no_days() {
        let age = parse_age("6;04.").unwrap();
        assert_eq!(age.years, 6);
        assert_eq!(age.months, Some(4));
        assert_eq!(age.days, None);
    }

    #[test]
    fn test_parse_age_no_months() {
        let age = parse_age("6;").unwrap();
        assert_eq!(age.years, 6);
        assert_eq!(age.months, None);
        assert_eq!(age.days, None);
    }

    #[test]
    fn test_parse_age_months_only() {
        let age = parse_age("2;10").unwrap();
        assert_eq!(age.years, 2);
        assert_eq!(age.months, Some(10));
        assert_eq!(age.days, None);
    }

    #[test]
    fn test_parse_age_empty() {
        assert!(parse_age("").is_none());
    }

    #[test]
    fn test_parse_languages() {
        let langs = parse_languages("eng, zho");
        assert_eq!(langs, vec!["eng", "zho"]);
    }

    #[test]
    fn test_parse_languages_single() {
        let langs = parse_languages("eng");
        assert_eq!(langs, vec!["eng"]);
    }

    #[test]
    fn test_parse_participants_basic() {
        let parts = parse_participants("CHI Mark Target_Child, MOT Mary Mother");
        assert_eq!(parts.len(), 2);
        assert_eq!(
            parts[0],
            ("CHI".into(), "Mark".into(), "Target_Child".into())
        );
        assert_eq!(parts[1], ("MOT".into(), "Mary".into(), "Mother".into()));
    }

    #[test]
    fn test_parse_participants_no_name() {
        let parts = parse_participants("CHI Target_Child, MOT Mother");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], ("CHI".into(), "".into(), "Target_Child".into()));
        assert_eq!(parts[1], ("MOT".into(), "".into(), "Mother".into()));
    }

    #[test]
    fn test_parse_id_basic() {
        let (code, fields) = parse_id("eng|macwhinney|CHI|2;10.10||||Target_Child|||").unwrap();
        assert_eq!(code, "CHI");
        assert_eq!(fields.language.as_deref(), Some("eng"));
        assert_eq!(fields.corpus.as_deref(), Some("macwhinney"));
        let age = fields.age.unwrap();
        assert_eq!(age.years, 2);
        assert_eq!(age.months, Some(10));
        assert_eq!(age.days, Some(10));
        assert_eq!(fields.role.as_deref(), Some("Target_Child"));
    }

    #[test]
    fn test_parse_id_minimal() {
        let (code, fields) = parse_id("||MOT|||||Mother|||").unwrap();
        assert_eq!(code, "MOT");
        assert!(fields.language.is_none());
        assert!(fields.age.is_none());
        assert_eq!(fields.role.as_deref(), Some("Mother"));
    }

    #[test]
    fn test_parse_media_full() {
        let media = parse_media("abe88, audio, missing");
        assert_eq!(media.filename, "abe88");
        assert_eq!(media.format, "audio");
        assert_eq!(media.status.as_deref(), Some("missing"));
    }

    #[test]
    fn test_parse_media_no_status() {
        let media = parse_media("abe88, video");
        assert_eq!(media.filename, "abe88");
        assert_eq!(media.format, "video");
        assert!(media.status.is_none());
    }

    #[test]
    fn test_parse_changeable_comment() {
        let ch = parse_changeable("Comment", Some("some text")).unwrap();
        match ch {
            ChangeableHeader::Comment { value } => assert_eq!(value, "some text"),
            _ => panic!("Expected Comment"),
        }
    }

    #[test]
    fn test_parse_changeable_new_episode() {
        let ch = parse_changeable("New Episode", None).unwrap();
        assert!(matches!(ch, ChangeableHeader::NewEpisode {}));
    }

    #[test]
    fn test_parse_changeable_unknown() {
        assert!(parse_changeable("Unknown", None).is_none());
    }

    #[test]
    fn test_parse_file_headers_basic() {
        let lines = vec![
            "@UTF8".to_string(),
            "@PID:\t11312/c-00044068-1".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tCHI Child Target_Child, MOT Mary Mother".to_string(),
            "@ID:\teng|brown|CHI|2;10.05|male|||Target_Child|||".to_string(),
            "@ID:\teng|brown|MOT||female|||Mother|||".to_string(),
            "@Date:\t25-JAN-1983".to_string(),
            "@Location:\tBoston, MA, USA".to_string(),
            "@Media:\tabe88, audio".to_string(),
            "*CHI:\thello .".to_string(),
        ];
        let (headers, start_idx, initial_events) = parse_file_headers(&lines);

        // Check that start_idx points to the first utterance.
        assert_eq!(start_idx, 10);
        assert!(initial_events.is_empty());

        // Hidden
        assert_eq!(headers.pid.as_deref(), Some("11312/c-00044068-1"));

        // Languages
        assert_eq!(headers.languages, vec!["eng"]);

        // Participants
        assert_eq!(headers.participants.len(), 2);
        assert_eq!(headers.participants[0].code, "CHI");
        assert_eq!(headers.participants[0].name, "Child");
        assert_eq!(headers.participants[0].role, "Target_Child");
        assert_eq!(headers.participants[0].language.as_deref(), Some("eng"));
        assert_eq!(headers.participants[0].corpus.as_deref(), Some("brown"));
        let age = headers.participants[0].age.as_ref().unwrap();
        assert_eq!(age.years, 2);
        assert_eq!(age.months, Some(10));
        assert_eq!(age.days, Some(5));
        assert_eq!(headers.participants[0].sex.as_deref(), Some("male"));
        assert_eq!(headers.participants[1].code, "MOT");
        assert_eq!(headers.participants[1].sex.as_deref(), Some("female"));

        // Constant
        assert_eq!(headers.date.as_deref(), Some("25-JAN-1983"));
        assert_eq!(parse_chat_date("25-JAN-1983"), Some((1983, 1, 25)));
        assert_eq!(headers.location.as_deref(), Some("Boston, MA, USA"));

        // Media
        let media = headers.media_data.as_ref().unwrap();
        assert_eq!(media.filename, "abe88");
        assert_eq!(media.format, "audio");
    }

    #[test]
    fn test_parse_file_headers_with_initial_comments() {
        let lines = vec![
            "@UTF8".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tCHI Child Target_Child".to_string(),
            "@Comment:\tThis is a comment".to_string(),
            "*CHI:\thello .".to_string(),
        ];
        let (headers, start_idx, initial_events) = parse_file_headers(&lines);

        assert_eq!(start_idx, 5);
        assert_eq!(headers.languages, vec!["eng"]);
        // Comment is a changeable header -> goes into initial_events
        assert_eq!(initial_events.len(), 1);
        match &initial_events[0] {
            ChatEvent::Header(ChangeableHeader::Comment { value }) => {
                assert_eq!(value, "This is a comment");
            }
            _ => panic!("Expected Comment event"),
        }
    }

    #[test]
    fn test_parse_file_headers_participant_specific() {
        let lines = vec![
            "@UTF8".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tCHI Ross Target_Child".to_string(),
            "@ID:\teng|macwhinney|CHI|2;06.||||Target_Child|||".to_string(),
            "@Birth of CHI:\t28-JUN-2001".to_string(),
            "@Birthplace of CHI:\tPittsburgh, PA".to_string(),
            "@L1 of CHI:\teng".to_string(),
            "*CHI:\thello .".to_string(),
        ];
        let (headers, _, _) = parse_file_headers(&lines);

        assert_eq!(
            headers.participants[0].birth.as_deref(),
            Some("28-JUN-2001")
        );
        assert_eq!(
            headers.participants[0].birthplace.as_deref(),
            Some("Pittsburgh, PA")
        );
        assert_eq!(headers.participants[0].l1.as_deref(), Some("eng"));
    }

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
