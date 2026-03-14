//! Data structures for CHAT transcription data.

use crate::chat::clean_utterance::audible_utterance;
use crate::chat::header::{ChangeableHeader, hash_hashmap};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Escape HTML special characters in text content.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ---------------------------------------------------------------------------
// Gra (shared, stays as #[pyclass])
// ---------------------------------------------------------------------------

/// A grammatical relation from the %gra tier.
#[cfg_attr(feature = "pyo3", pyclass(from_py_object))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Gra {
    pub dep: usize,
    pub head: usize,
    pub rel: String,
}

// ---------------------------------------------------------------------------
// BaseToken
// ---------------------------------------------------------------------------

/// Shared read access to token fields.
///
/// Implemented by rustling's [`Token`] and can be implemented by downstream
/// crates to share behavior.
pub trait BaseToken {
    fn word(&self) -> &str;
    fn pos(&self) -> Option<&str>;
    fn mor(&self) -> Option<&str>;
    fn gra(&self) -> Option<&Gra>;
}

// ---------------------------------------------------------------------------
// Token (pure Rust)
// ---------------------------------------------------------------------------

/// A token with word, POS, morphology, and grammatical relation.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Token {
    pub word: String,
    pub pos: Option<String>,
    pub mor: Option<String>,
    pub gra: Option<Gra>,
}

impl BaseToken for Token {
    fn word(&self) -> &str {
        &self.word
    }
    fn pos(&self) -> Option<&str> {
        self.pos.as_deref()
    }
    fn mor(&self) -> Option<&str> {
        self.mor.as_deref()
    }
    fn gra(&self) -> Option<&Gra> {
        self.gra.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Utterance (pure Rust)
// ---------------------------------------------------------------------------

/// A single utterance from a CHAT transcript.
///
/// For changeable headers (e.g., `@Comment`, `@New Episode`), only
/// `changeable_header` is set; all other fields are `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct Utterance {
    pub participant: Option<String>,
    pub tokens: Option<Vec<Token>>,
    pub time_marks: Option<(i64, i64)>,
    pub tiers: Option<HashMap<String, String>>,
    pub changeable_header: Option<ChangeableHeader>,
    /// The `%`-prefixed tier name used as the morphology tier (e.g., `"%mor"`, `"%xmor"`),
    /// or `None` if mor+gra handling was disabled.
    pub mor_tier_name: Option<String>,
    /// The `%`-prefixed tier name used as the grammatical relation tier (e.g., `"%gra"`),
    /// or `None` if mor+gra handling was disabled.
    pub gra_tier_name: Option<String>,
}

impl Utterance {
    /// Audibly faithful transcript of this utterance, or None for headers.
    ///
    /// When tier data is available the result is computed from the original
    /// main-tier text via [`audible_utterance`]; otherwise it falls back to
    /// joining the token words (for manually constructed utterances).
    pub fn audible(&self) -> Option<String> {
        // Primary: compute from original main tier text.
        if let (Some(participant), Some(tiers)) = (&self.participant, &self.tiers)
            && let Some(main_tier) = tiers.get(participant)
        {
            return Some(audible_utterance(main_tier));
        }
        // Fallback: join token words (for manually constructed utterances).
        self.tokens.as_ref().map(|tokens| {
            tokens
                .iter()
                .map(|t| t.word.as_str())
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
    }

    pub(crate) fn hash_into(&self, hasher: &mut impl Hasher) {
        self.participant.hash(hasher);
        self.tokens.hash(hasher);
        self.time_marks.hash(hasher);
        match &self.tiers {
            Some(tiers) => {
                true.hash(hasher);
                hash_hashmap(tiers, hasher);
            }
            None => false.hash(hasher),
        }
        self.changeable_header.hash(hasher);
        self.mor_tier_name.hash(hasher);
        self.gra_tier_name.hash(hasher);
    }

    /// Return an HTML representation of this utterance.
    pub fn repr_html(&self) -> String {
        if let Some(ref ch) = self.changeable_header {
            return format!(
                "<div class=\"rustling-changeable-header\" \
                 style=\"font-family:monospace;font-size:13px;color:#888\">{}</div>",
                html_escape(&changeable_header_to_chat(ch))
            );
        }

        let tokens = self.tokens.as_deref().unwrap_or(&[]);
        let participant = self.participant.as_deref().unwrap_or("");
        let tiers = self.tiers.as_ref();
        let n_tokens = tokens.len();
        let n_cols = n_tokens.max(1);

        let has_mor = tokens.iter().any(|t| t.pos.is_some() || t.mor.is_some());
        let has_gra = tokens.iter().any(|t| t.gra.is_some());

        let empty_map = HashMap::new();
        let tiers_map = tiers.unwrap_or(&empty_map);
        let mor_tier = self.mor_tier_name.as_deref().unwrap_or("%mor");
        let gra_tier = self.gra_tier_name.as_deref().unwrap_or("%gra");

        let mut other_tiers: Vec<(&String, &String)> = tiers_map
            .iter()
            .filter(|(k, _)| {
                k.as_str() != participant && k.as_str() != mor_tier && k.as_str() != gra_tier
            })
            .collect();
        other_tiers.sort_by_key(|(k, _)| k.as_str().to_owned());

        let th_style = "text-align:left;padding:4px 10px 4px 0;\
                         font-weight:bold;color:#555;border:none;\
                         white-space:nowrap;vertical-align:top";
        let td_style = "text-align:left;padding:4px 8px;border:none;white-space:nowrap";

        let mut html = String::with_capacity(512);

        html.push_str(
            "<table class=\"rustling-utterance\" style=\"\
             border-collapse:collapse;border:none;\
             font-family:monospace;font-size:13px\">\n",
        );

        // Row: participant + words
        html.push_str("<tr>");
        html.push_str(&format!(
            "<th style=\"{th_style}\">*{}:</th>",
            html_escape(participant)
        ));
        if n_tokens == 0 {
            html.push_str(&format!(
                "<td style=\"{td_style}\" colspan=\"{n_cols}\"></td>"
            ));
        } else {
            for token in tokens {
                html.push_str(&format!(
                    "<td style=\"{td_style}\">{}</td>",
                    html_escape(&token.word)
                ));
            }
        }
        html.push_str("</tr>\n");

        // Row: %mor (reconstructed from token fields)
        if has_mor {
            html.push_str("<tr>");
            html.push_str(&format!(
                "<th style=\"{th_style}\">{}:</th>",
                html_escape(mor_tier)
            ));
            for token in tokens {
                let cell = match (&token.pos, &token.mor) {
                    (Some(pos), Some(mor)) if !pos.is_empty() => {
                        format!("{}|{}", html_escape(pos), html_escape(mor))
                    }
                    (Some(_pos), Some(mor)) if _pos.is_empty() => html_escape(mor),
                    (Some(pos), None) => html_escape(pos),
                    (None, Some(mor)) => html_escape(mor),
                    _ => String::new(),
                };
                html.push_str(&format!("<td style=\"{td_style}\">{cell}</td>"));
            }
            html.push_str("</tr>\n");
        }

        // Row: %gra (reconstructed from token fields)
        if has_gra {
            html.push_str("<tr>");
            html.push_str(&format!(
                "<th style=\"{th_style}\">{}:</th>",
                html_escape(gra_tier)
            ));
            for token in tokens {
                let cell = match &token.gra {
                    Some(g) => format!("{}|{}|{}", g.dep, g.head, html_escape(&g.rel)),
                    None => String::new(),
                };
                html.push_str(&format!("<td style=\"{td_style}\">{cell}</td>"));
            }
            html.push_str("</tr>\n");
        }

        // Rows: other tiers (sorted alphabetically)
        for (tier_name, tier_value) in &other_tiers {
            html.push_str("<tr>");
            html.push_str(&format!(
                "<th style=\"{th_style}\">{}:</th>",
                html_escape(tier_name)
            ));
            html.push_str(&format!(
                "<td style=\"{td_style}\" colspan=\"{n_cols}\">{}</td>",
                html_escape(tier_value)
            ));
            html.push_str("</tr>\n");
        }

        html.push_str("</table>");

        // Time marks as a footer below the table
        if let Some((start, end)) = self.time_marks {
            let table = html;
            html = format!(
                "<div class=\"rustling-utterance-wrapper\">\
                 {table}\
                 <div style=\"font-family:monospace;font-size:11px;\
                 color:#888;padding-top:2px\">\u{23F1} {start}\u{2013}{end} ms</div>\
                 </div>"
            );
        }

        html
    }

    /// Return a plain text tabular representation of this utterance.
    pub fn to_str(&self) -> String {
        if let Some(ref ch) = self.changeable_header {
            return changeable_header_to_chat(ch);
        }

        let tokens = self.tokens.as_deref().unwrap_or(&[]);
        let participant = self.participant.as_deref().unwrap_or("");
        let tiers = self.tiers.as_ref();
        let n_tokens = tokens.len();

        let has_mor = tokens.iter().any(|t| t.pos.is_some() || t.mor.is_some());
        let has_gra = tokens.iter().any(|t| t.gra.is_some());

        let empty_map = HashMap::new();
        let tiers_map = tiers.unwrap_or(&empty_map);
        let mor_tier = self.mor_tier_name.as_deref().unwrap_or("%mor");
        let gra_tier = self.gra_tier_name.as_deref().unwrap_or("%gra");

        let mut other_tiers: Vec<(&String, &String)> = tiers_map
            .iter()
            .filter(|(k, _)| {
                k.as_str() != participant && k.as_str() != mor_tier && k.as_str() != gra_tier
            })
            .collect();
        other_tiers.sort_by_key(|(k, _)| k.as_str().to_owned());

        // Build label and cell arrays for column-aligned rows.
        let participant_label = format!("*{participant}:");
        let participant_cells: Vec<String> = tokens.iter().map(|t| t.word.clone()).collect();

        let mor_cells: Vec<String> = if has_mor {
            tokens
                .iter()
                .map(|t| match (&t.pos, &t.mor) {
                    (Some(pos), Some(mor)) if !pos.is_empty() => format!("{pos}|{mor}"),
                    (Some(_), Some(mor)) => mor.clone(),
                    (Some(pos), None) => pos.clone(),
                    (None, Some(mor)) => mor.clone(),
                    _ => String::new(),
                })
                .collect()
        } else {
            Vec::new()
        };

        let gra_cells: Vec<String> = if has_gra {
            tokens
                .iter()
                .map(|t| match &t.gra {
                    Some(g) => format!("{}|{}|{}", g.dep, g.head, g.rel),
                    None => String::new(),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compute label column width.
        let mor_label = format!("{mor_tier}:");
        let gra_label = format!("{gra_tier}:");
        let mut label_width = participant_label.len();
        if has_mor {
            label_width = label_width.max(mor_label.len());
        }
        if has_gra {
            label_width = label_width.max(gra_label.len());
        }
        for (tier_name, _) in &other_tiers {
            label_width = label_width.max(tier_name.len() + 1); // "name:"
        }

        // Compute per-column widths.
        let col_widths: Vec<usize> = (0..n_tokens)
            .map(|i| {
                let mut w = participant_cells.get(i).map_or(0, |s| s.len());
                if has_mor {
                    w = w.max(mor_cells.get(i).map_or(0, |s| s.len()));
                }
                if has_gra {
                    w = w.max(gra_cells.get(i).map_or(0, |s| s.len()));
                }
                w
            })
            .collect();

        let mut lines: Vec<String> = Vec::new();

        // Format one row with label and column-aligned cells.
        let format_row = |label: &str, cells: &[String]| -> String {
            let mut row = format!("{:<width$}", label, width = label_width);
            for (i, cell) in cells.iter().enumerate() {
                row.push_str("  ");
                if i < cells.len() - 1 {
                    row.push_str(&format!("{:<width$}", cell, width = col_widths[i]));
                } else {
                    row.push_str(cell);
                }
            }
            row
        };

        // Participant row
        if n_tokens == 0 {
            lines.push(format!(
                "{:<width$}",
                participant_label,
                width = label_width
            ));
        } else {
            lines.push(format_row(&participant_label, &participant_cells));
        }

        // %mor row
        if has_mor {
            lines.push(format_row(&mor_label, &mor_cells));
        }

        // %gra row
        if has_gra {
            lines.push(format_row(&gra_label, &gra_cells));
        }

        // Other tiers (full-width, not column-aligned)
        for (tier_name, tier_value) in &other_tiers {
            let label = format!("{tier_name}:");
            lines.push(format!(
                "{:<width$}  {tier_value}",
                label,
                width = label_width
            ));
        }

        // Time marks footer
        if let Some((start, end)) = self.time_marks {
            lines.push(format!(
                "{:<width$}  \u{23F1} {start}\u{2013}{end} ms",
                "",
                width = label_width
            ));
        }

        lines.join("\n")
    }
}

/// Convert a `ChangeableHeader` to its CHAT-format string (e.g., `@Comment:\tChild laughs`).
pub(crate) fn changeable_header_to_chat(ch: &ChangeableHeader) -> String {
    match ch {
        ChangeableHeader::Activities { value } => format!("@Activities:\t{value}"),
        ChangeableHeader::Bck { value } => format!("@Bck:\t{value}"),
        ChangeableHeader::Bg { value } => match value {
            Some(v) => format!("@Bg:\t{v}"),
            None => "@Bg".to_string(),
        },
        ChangeableHeader::Blank {} => "@Blank".to_string(),
        ChangeableHeader::Comment { value } => format!("@Comment:\t{value}"),
        ChangeableHeader::Date { value } => format!("@Date:\t{value}"),
        ChangeableHeader::Eg { value } => match value {
            Some(v) => format!("@Eg:\t{v}"),
            None => "@Eg".to_string(),
        },
        ChangeableHeader::G { value } => match value {
            Some(v) => format!("@G:\t{v}"),
            None => "@G".to_string(),
        },
        ChangeableHeader::NewEpisode {} => "@New Episode".to_string(),
        ChangeableHeader::Page { value } => format!("@Page:\t{value}"),
        ChangeableHeader::Situation { value } => format!("@Situation:\t{value}"),
    }
}

// ---------------------------------------------------------------------------
// BaseUtterance
// ---------------------------------------------------------------------------

/// Shared behavior for utterance types.
///
/// Implemented by rustling's [`Utterance`] and can be implemented by downstream
/// crates (e.g., pycantonese) so that [`BaseChat::from_utterances`](super::reader::BaseChat::from_utterances) works
/// generically.
pub trait BaseUtterance {
    /// Convert this utterance to CHAT-format raw lines.
    ///
    /// For changeable headers, returns a single line (e.g., `@G`).
    /// For regular utterances, returns the main tier followed by dependent tiers.
    /// Returns an empty Vec if tier data is unavailable.
    fn to_chat_lines(&self) -> Vec<String>;

    /// Convert this utterance to a rustling [`Utterance`] for storage in
    /// [`ChatFile`](super::reader::ChatFile).
    fn to_utterance(&self) -> Utterance;
}

impl BaseUtterance for Utterance {
    fn to_chat_lines(&self) -> Vec<String> {
        if let Some(ref ch) = self.changeable_header {
            return vec![changeable_header_to_chat(ch)];
        }

        let (Some(participant), Some(tiers)) = (&self.participant, &self.tiers) else {
            return Vec::new();
        };

        let mut lines = Vec::new();

        // Main tier: *PARTICIPANT:\t<content>
        if let Some(main_content) = tiers.get(participant) {
            lines.push(format!("*{participant}:\t{main_content}"));
        }

        // Dependent tiers: mor tier first, gra tier second, then others sorted.
        let mor_tier = self.mor_tier_name.as_deref().unwrap_or("%mor");
        let gra_tier = self.gra_tier_name.as_deref().unwrap_or("%gra");
        for key in [mor_tier, gra_tier] {
            if let Some(value) = tiers.get(key) {
                lines.push(format!("{key}:\t{value}"));
            }
        }
        let mut other_keys: Vec<_> = tiers
            .keys()
            .filter(|k| {
                k.as_str() != participant && k.as_str() != mor_tier && k.as_str() != gra_tier
            })
            .collect();
        other_keys.sort();
        for key in other_keys {
            lines.push(format!("{key}:\t{}", tiers[key]));
        }

        lines
    }

    fn to_utterance(&self) -> Utterance {
        self.clone()
    }
}

// ---------------------------------------------------------------------------
// Utterances (pure Rust)
// ---------------------------------------------------------------------------

/// A sequence of utterances with a formatted display for terminal/notebook use.
///
/// Returned by `Chat::head` and `Chat::tail`.
#[derive(Clone)]
pub struct Utterances {
    pub utterances: Vec<Utterance>,
}

impl Utterances {
    pub fn new(utterances: Vec<Utterance>) -> Self {
        Self { utterances }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_str_basic() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![
                Token {
                    word: "I".to_string(),
                    pos: Some("pro".to_string()),
                    mor: Some("I".to_string()),
                    gra: Some(Gra {
                        dep: 1,
                        head: 2,
                        rel: "SUBJ".to_string(),
                    }),
                },
                Token {
                    word: "want".to_string(),
                    pos: Some("v".to_string()),
                    mor: Some("want".to_string()),
                    gra: Some(Gra {
                        dep: 2,
                        head: 0,
                        rel: "ROOT".to_string(),
                    }),
                },
                Token {
                    word: "cookie".to_string(),
                    pos: Some("n".to_string()),
                    mor: Some("cookie".to_string()),
                    gra: Some(Gra {
                        dep: 3,
                        head: 2,
                        rel: "OBJ".to_string(),
                    }),
                },
            ]),
            time_marks: None,
            tiers: Some(HashMap::new()),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        assert!(s.contains("*CHI:"));
        assert!(s.contains("%mor:"));
        assert!(s.contains("%gra:"));
        assert!(s.contains("pro|I"));
        assert!(s.contains("1|2|SUBJ"));
        // Check all lines start at same column for labels
        let line_list: Vec<&str> = s.lines().collect();
        assert_eq!(line_list.len(), 3);
    }

    #[test]
    fn test_to_str_no_mor() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![
                Token {
                    word: "hello".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
                Token {
                    word: "world".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
            ]),
            time_marks: None,
            tiers: Some(HashMap::new()),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        assert!(s.contains("*CHI:"));
        assert!(!s.contains("%mor:"));
        assert!(!s.contains("%gra:"));
        assert!(s.contains("hello"));
        assert!(s.contains("world"));
    }

    #[test]
    fn test_to_str_time_marks() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![Token {
                word: "hi".to_string(),
                pos: None,
                mor: None,
                gra: None,
            }]),
            time_marks: Some((0, 1500)),
            tiers: Some(HashMap::new()),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        assert!(s.contains("0"));
        assert!(s.contains("1500"));
        assert!(s.contains("ms"));
    }

    #[test]
    fn test_to_str_other_tiers() {
        let mut tiers = HashMap::new();
        tiers.insert("CHI".to_string(), "hello .".to_string());
        tiers.insert("%sit".to_string(), "playing with toys".to_string());
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![Token {
                word: "hello".to_string(),
                pos: None,
                mor: None,
                gra: None,
            }]),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        assert!(s.contains("%sit:"));
        assert!(s.contains("playing with toys"));
    }

    #[test]
    fn test_to_str_empty_tokens() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![]),
            time_marks: None,
            tiers: Some(HashMap::new()),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        assert!(s.contains("*CHI:"));
        assert_eq!(s.lines().count(), 1);
    }

    #[test]
    fn test_to_str_column_alignment() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![
                Token {
                    word: "I".to_string(),
                    pos: Some("pro".to_string()),
                    mor: Some("I".to_string()),
                    gra: Some(Gra {
                        dep: 1,
                        head: 2,
                        rel: "SUBJ".to_string(),
                    }),
                },
                Token {
                    word: "go".to_string(),
                    pos: Some("v".to_string()),
                    mor: Some("go".to_string()),
                    gra: Some(Gra {
                        dep: 2,
                        head: 0,
                        rel: "ROOT".to_string(),
                    }),
                },
            ]),
            time_marks: None,
            tiers: Some(HashMap::new()),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let s = utt.to_str();
        let line_list: Vec<&str> = s.lines().collect();
        // All label columns should have the same width
        // Find where the first data column starts (after label + 2 spaces)
        let first_data_positions: Vec<usize> = line_list
            .iter()
            .map(|line| {
                let trimmed = line.trim_start();
                line.len() - trimmed.len() + trimmed.find("  ").map_or(0, |pos| pos + 2)
            })
            .collect();
        // All rows should start data at the same position
        assert!(first_data_positions.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_to_str_changeable_header() {
        let utt = Utterance {
            participant: None,
            tokens: None,
            time_marks: None,
            tiers: None,
            changeable_header: Some(ChangeableHeader::Comment {
                value: "Child laughs".to_string(),
            }),
            mor_tier_name: None,
            gra_tier_name: None,
        };
        assert_eq!(utt.to_str(), "@Comment:\tChild laughs");
    }

    #[test]
    fn test_to_str_changeable_header_new_episode() {
        let utt = Utterance {
            participant: None,
            tokens: None,
            time_marks: None,
            tiers: None,
            changeable_header: Some(ChangeableHeader::NewEpisode {}),
            mor_tier_name: None,
            gra_tier_name: None,
        };
        assert_eq!(utt.to_str(), "@New Episode");
    }

    #[test]
    fn test_audible_with_tokens() {
        // Fallback path: no tiers, so audible is computed from tokens.
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![
                Token {
                    word: "I".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
                Token {
                    word: "want".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
                Token {
                    word: "cookie".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
            ]),
            time_marks: None,
            tiers: None,
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        assert_eq!(utt.audible(), Some("I want cookie".to_string()));
    }

    #[test]
    fn test_audible_with_none_tokens() {
        let utt = Utterance {
            participant: None,
            tokens: None,
            time_marks: None,
            tiers: None,
            changeable_header: Some(ChangeableHeader::NewEpisode {}),
            mor_tier_name: None,
            gra_tier_name: None,
        };
        assert_eq!(utt.audible(), None);
    }

    #[test]
    fn test_audible_with_empty_words() {
        // Fallback path: no tiers, so audible is computed from tokens.
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![
                Token {
                    word: "hello".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
                Token {
                    word: "".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
                Token {
                    word: "world".to_string(),
                    pos: None,
                    mor: None,
                    gra: None,
                },
            ]),
            time_marks: None,
            tiers: None,
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        assert_eq!(utt.audible(), Some("hello world".to_string()));
    }

    #[test]
    fn test_audible_from_tiers() {
        // Primary path: tiers available, audible is computed from main tier.
        let mut tiers = HashMap::new();
        tiers.insert("CHI".to_string(), "I want cookie .".to_string());
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![]),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        assert_eq!(utt.audible(), Some("I want cookie .".to_string()));
    }

    #[test]
    fn test_to_chat_lines_regular() {
        let mut tiers = HashMap::new();
        tiers.insert("CHI".to_string(), "I want cookie .".to_string());
        tiers.insert("%mor".to_string(), "pro|I v|want n|cookie .".to_string());
        tiers.insert(
            "%gra".to_string(),
            "1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT".to_string(),
        );
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![]),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let lines = utt.to_chat_lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "*CHI:\tI want cookie .");
        assert_eq!(lines[1], "%mor:\tpro|I v|want n|cookie .");
        assert_eq!(lines[2], "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT");
    }

    #[test]
    fn test_to_chat_lines_with_other_tiers() {
        let mut tiers = HashMap::new();
        tiers.insert("CHI".to_string(), "hello .".to_string());
        tiers.insert("%sit".to_string(), "playing with toys".to_string());
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![]),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let lines = utt.to_chat_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "*CHI:\thello .");
        assert_eq!(lines[1], "%sit:\tplaying with toys");
    }

    #[test]
    fn test_to_chat_lines_changeable_header() {
        let utt = Utterance {
            participant: None,
            tokens: None,
            time_marks: None,
            tiers: None,
            changeable_header: Some(ChangeableHeader::G { value: None }),
            mor_tier_name: None,
            gra_tier_name: None,
        };
        let lines = utt.to_chat_lines();
        assert_eq!(lines, vec!["@G"]);
    }

    #[test]
    fn test_to_chat_lines_no_tiers() {
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![]),
            time_marks: None,
            tiers: None,
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let lines = utt.to_chat_lines();
        assert!(lines.is_empty());
    }
}
