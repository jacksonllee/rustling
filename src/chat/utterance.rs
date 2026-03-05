//! Data structures for CHAT transcription data.

use crate::chat::header::{ChangeableHeader, hash_hashmap};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
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
#[pyclass(from_py_object)]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Gra {
    #[pyo3(get)]
    pub dep: usize,
    #[pyo3(get)]
    pub head: usize,
    #[pyo3(get)]
    pub rel: String,
}

#[pymethods]
impl Gra {
    #[new]
    fn new(dep: usize, head: usize, rel: String) -> Self {
        Self { dep, head, rel }
    }

    fn __repr__(&self) -> String {
        format!(
            "Gra(dep={}, head={}, rel='{}')",
            self.dep, self.head, self.rel
        )
    }

    fn __eq__(&self, other: &Gra) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
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
// PyToken (Python wrapper)
// ---------------------------------------------------------------------------

/// Python wrapper for [`Token`].
#[pyclass(name = "Token", from_py_object)]
#[derive(Clone)]
pub struct PyToken(pub Token);

#[pymethods]
impl PyToken {
    #[new]
    #[pyo3(signature = (word, pos=None, mor=None, gra=None))]
    fn new(word: String, pos: Option<String>, mor: Option<String>, gra: Option<Gra>) -> Self {
        Self(Token {
            word,
            pos,
            mor,
            gra,
        })
    }

    #[getter]
    fn word(&self) -> &str {
        &self.0.word
    }

    #[getter]
    fn pos(&self) -> Option<&str> {
        self.0.pos.as_deref()
    }

    #[getter]
    fn mor(&self) -> Option<&str> {
        self.0.mor.as_deref()
    }

    #[getter]
    fn gra(&self) -> Option<Gra> {
        self.0.gra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Token(word='{}', pos={}, mor={}, gra={})",
            self.0.word,
            match &self.0.pos {
                Some(p) => format!("'{p}'"),
                None => "None".to_string(),
            },
            match &self.0.mor {
                Some(m) => format!("'{m}'"),
                None => "None".to_string(),
            },
            match &self.0.gra {
                Some(g) => g.__repr__(),
                None => "None".to_string(),
            },
        )
    }

    fn __eq__(&self, other: &PyToken) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
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
}

impl Utterance {
    /// Raw transcript of this utterance, or None for headers.
    pub fn raw(&self) -> Option<String> {
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
        let mut other_tiers: Vec<(&String, &String)> = tiers_map
            .iter()
            .filter(|(k, _)| {
                k.as_str() != participant && k.as_str() != "%mor" && k.as_str() != "%gra"
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
            html.push_str(&format!("<th style=\"{th_style}\">%mor:</th>"));
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
            html.push_str(&format!("<th style=\"{th_style}\">%gra:</th>"));
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
        let mut other_tiers: Vec<(&String, &String)> = tiers_map
            .iter()
            .filter(|(k, _)| {
                k.as_str() != participant && k.as_str() != "%mor" && k.as_str() != "%gra"
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
        let mut label_width = participant_label.len();
        if has_mor {
            label_width = label_width.max(5); // "%mor:"
        }
        if has_gra {
            label_width = label_width.max(5); // "%gra:"
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
            lines.push(format_row("%mor:", &mor_cells));
        }

        // %gra row
        if has_gra {
            lines.push(format_row("%gra:", &gra_cells));
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

// ---------------------------------------------------------------------------
// PyUtterance (Python wrapper)
// ---------------------------------------------------------------------------

/// Python wrapper for [`Utterance`].
#[pyclass(name = "Utterance", from_py_object)]
#[derive(Clone)]
pub struct PyUtterance(pub Utterance);

#[pymethods]
impl PyUtterance {
    #[new]
    #[pyo3(signature = (*, participant=None, tokens=None, time_marks=None, tiers=None, changeable_header=None))]
    fn new(
        participant: Option<String>,
        tokens: Option<Vec<PyToken>>,
        time_marks: Option<(i64, i64)>,
        tiers: Option<HashMap<String, String>>,
        changeable_header: Option<ChangeableHeader>,
    ) -> Self {
        Self(Utterance {
            participant,
            tokens: tokens.map(|ts| ts.into_iter().map(|pt| pt.0).collect()),
            time_marks,
            tiers,
            changeable_header,
        })
    }

    #[getter]
    fn participant(&self) -> Option<&str> {
        self.0.participant.as_deref()
    }

    #[getter]
    fn tokens(&self) -> Option<Vec<PyToken>> {
        self.0
            .tokens
            .as_ref()
            .map(|ts| ts.iter().map(|t| PyToken(t.clone())).collect())
    }

    #[getter]
    fn time_marks(&self) -> Option<(i64, i64)> {
        self.0.time_marks
    }

    #[getter]
    fn tiers(&self) -> Option<HashMap<String, String>> {
        self.0.tiers.clone()
    }

    #[getter]
    fn changeable_header(&self) -> Option<ChangeableHeader> {
        self.0.changeable_header.clone()
    }

    /// Raw transcript of this utterance, or None for headers.
    #[getter]
    fn raw(&self) -> Option<String> {
        self.0.raw()
    }

    fn __repr__(&self) -> String {
        if let Some(ref ch) = self.0.changeable_header {
            return format!("Utterance(changeable_header={ch:?})");
        }
        format!(
            "Utterance(participant='{}', tokens=[...{} tokens], time_marks={:?})",
            self.0.participant.as_deref().unwrap_or(""),
            self.0.tokens.as_ref().map_or(0, |t| t.len()),
            self.0.time_marks,
        )
    }

    fn _repr_html_(&self) -> String {
        self.0.repr_html()
    }

    fn __eq__(&self, other: &PyUtterance) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash_into(&mut hasher);
        hasher.finish()
    }

    /// Return a plain text tabular representation of this utterance.
    pub fn to_str(&self) -> String {
        self.0.to_str()
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

        // Dependent tiers: %mor first, %gra second, then others sorted.
        for key in ["%mor", "%gra"] {
            if let Some(value) = tiers.get(key) {
                lines.push(format!("{key}:\t{value}"));
            }
        }
        let mut other_keys: Vec<_> = tiers
            .keys()
            .filter(|k| k.as_str() != participant && k.as_str() != "%mor" && k.as_str() != "%gra")
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

// ---------------------------------------------------------------------------
// PyUtterances (Python wrapper)
// ---------------------------------------------------------------------------

/// Python wrapper for [`Utterances`].
#[pyclass(name = "Utterances", from_py_object)]
#[derive(Clone)]
pub struct PyUtterances(pub Utterances);

#[pymethods]
impl PyUtterances {
    fn __repr__(&self) -> String {
        self.0
            .utterances
            .iter()
            .map(|u| u.to_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn _repr_html_(&self) -> String {
        self.0
            .utterances
            .iter()
            .map(|u| u.repr_html())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn __len__(&self) -> usize {
        self.0.utterances.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PyUtterance> {
        let len = self.0.utterances.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        Ok(PyUtterance(self.0.utterances[idx as usize].clone()))
    }

    fn __eq__(&self, other: &PyUtterances) -> bool {
        self.0.utterances == other.0.utterances
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.utterances.len().hash(&mut hasher);
        for u in &self.0.utterances {
            u.hash_into(&mut hasher);
        }
        hasher.finish()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyUtterancesIter {
        PyUtterancesIter {
            inner: slf.0.utterances.clone(),
            index: 0,
        }
    }
}

/// Iterator for [`PyUtterances`].
#[pyclass]
struct PyUtterancesIter {
    inner: Vec<Utterance>,
    index: usize,
}

#[pymethods]
impl PyUtterancesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyUtterance> {
        if self.index < self.inner.len() {
            let item = self.inner[self.index].clone();
            self.index += 1;
            Some(PyUtterance(item))
        } else {
            None
        }
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
        };
        assert_eq!(utt.to_str(), "@New Episode");
    }

    #[test]
    fn test_raw_with_tokens() {
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
        };
        assert_eq!(utt.raw(), Some("I want cookie".to_string()));
    }

    #[test]
    fn test_raw_with_none_tokens() {
        let utt = Utterance {
            participant: None,
            tokens: None,
            time_marks: None,
            tiers: None,
            changeable_header: Some(ChangeableHeader::NewEpisode {}),
        };
        assert_eq!(utt.raw(), None);
    }

    #[test]
    fn test_raw_with_empty_words() {
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
        };
        assert_eq!(utt.raw(), Some("hello world".to_string()));
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
        };
        let lines = utt.to_chat_lines();
        assert!(lines.is_empty());
    }
}
