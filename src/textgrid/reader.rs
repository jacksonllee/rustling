//! TextGrid data reader.
//!
//! Praat TextGrid format: <https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html>

use crate::chat::filter_file_paths;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when reading or parsing TextGrid data.
#[derive(Debug)]
pub enum TextGridError {
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

impl std::fmt::Display for TextGridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextGridError::Io(e) => write!(f, "{e}"),
            TextGridError::Parse(e) => write!(f, "TextGrid parse error: {e}"),
            TextGridError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
            TextGridError::Zip(e) => write!(f, "{e}"),
            TextGridError::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TextGridError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TextGridError::Io(e) => Some(e),
            TextGridError::Source(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TextGridError {
    fn from(e: std::io::Error) -> Self {
        TextGridError::Io(e)
    }
}

impl From<crate::sources::SourceError> for TextGridError {
    fn from(e: crate::sources::SourceError) -> Self {
        TextGridError::Source(e)
    }
}

/// Error type for [`BaseTextGrid::write_files`].
#[derive(Debug)]
pub enum WriteError {
    /// Validation error (e.g., wrong number of filenames).
    Validation(String),
    /// I/O error from the filesystem.
    Io(std::io::Error),
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A single interval within an IntervalTier.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    /// Start time in seconds.
    pub xmin: f64,
    /// End time in seconds.
    pub xmax: f64,
    /// The annotation text.
    pub text: String,
}

/// A single point within a TextTier (PointTier).
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Time in seconds.
    pub number: f64,
    /// The annotation text.
    pub mark: String,
}

/// A tier within a TextGrid file.
#[derive(Debug, Clone, PartialEq)]
pub enum TextGridTier {
    /// An interval tier with non-overlapping, gap-free time intervals.
    IntervalTier {
        /// Tier name.
        name: String,
        /// Start time of the tier.
        xmin: f64,
        /// End time of the tier.
        xmax: f64,
        /// Intervals in this tier.
        intervals: Vec<Interval>,
    },
    /// A point (text) tier with time-stamped labels.
    TextTier {
        /// Tier name.
        name: String,
        /// Start time of the tier.
        xmin: f64,
        /// End time of the tier.
        xmax: f64,
        /// Points in this tier.
        points: Vec<Point>,
    },
}

impl TextGridTier {
    /// Return the tier name.
    pub fn name(&self) -> &str {
        match self {
            TextGridTier::IntervalTier { name, .. } => name,
            TextGridTier::TextTier { name, .. } => name,
        }
    }

    /// Return the tier class as a string.
    pub fn tier_class(&self) -> &str {
        match self {
            TextGridTier::IntervalTier { .. } => "IntervalTier",
            TextGridTier::TextTier { .. } => "TextTier",
        }
    }
}

/// A single parsed TextGrid file.
#[derive(Debug, Clone)]
pub struct TextGridFile {
    /// File path or identifier.
    pub file_path: String,
    /// Start time of the file domain in seconds.
    pub xmin: f64,
    /// End time of the file domain in seconds.
    pub xmax: f64,
    /// Tiers in this file.
    pub tiers: Vec<TextGridTier>,
    /// Original raw text content for faithful round-tripping.
    pub raw_text: String,
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Extract a Praat-quoted string value from a line.
///
/// Handles the Praat quoting convention where inner quotes are doubled: `""`.
fn extract_quoted_string(s: &str) -> Result<String, TextGridError> {
    let s = s.trim();
    if !s.starts_with('"') {
        return Err(TextGridError::Parse(format!(
            "Expected quoted string, got: {s:?}"
        )));
    }
    // Find the closing quote. Inner quotes are doubled ("").
    let bytes = s.as_bytes();
    let mut result = String::new();
    let mut i = 1; // skip opening quote
    while i < bytes.len() {
        if bytes[i] == b'"' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                // Escaped quote
                result.push('"');
                i += 2;
            } else {
                // Closing quote
                return Ok(result);
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    Err(TextGridError::Parse(format!(
        "Unterminated quoted string: {s:?}"
    )))
}

/// Parse a floating-point value from a string, trimming whitespace.
fn parse_float(s: &str) -> Result<f64, TextGridError> {
    s.trim()
        .parse::<f64>()
        .map_err(|_| TextGridError::Parse(format!("Invalid number: {s:?}")))
}

/// Parse an integer value from a string, trimming whitespace.
fn parse_int(s: &str) -> Result<usize, TextGridError> {
    s.trim()
        .parse::<usize>()
        .map_err(|_| TextGridError::Parse(format!("Invalid integer: {s:?}")))
}

/// Extract the value part after `=` in a line like `key = value`.
fn value_after_eq(line: &str) -> Option<&str> {
    let pos = line.find('=')?;
    Some(line[pos + 1..].trim())
}

// ---------------------------------------------------------------------------
// Normal text format parser
// ---------------------------------------------------------------------------

/// Parse a TextGrid in the normal "text" format.
fn parse_text_format(
    lines: &[&str],
    file_path: String,
    raw_text: String,
) -> Result<TextGridFile, TextGridError> {
    let mut idx = 0;

    // Skip header lines (File type, Object class, blank lines).
    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with("File type") || line.starts_with("Object class") {
            idx += 1;
        } else {
            break;
        }
    }

    // xmin
    let xmin = parse_float(
        value_after_eq(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing xmin".to_string()))?,
        )
        .ok_or_else(|| TextGridError::Parse("Missing xmin value".to_string()))?,
    )?;
    idx += 1;

    // xmax
    let xmax = parse_float(
        value_after_eq(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing xmax".to_string()))?,
        )
        .ok_or_else(|| TextGridError::Parse("Missing xmax value".to_string()))?,
    )?;
    idx += 1;

    // tiers? <exists>
    idx += 1;

    // size = N
    let size_line = lines
        .get(idx)
        .ok_or_else(|| TextGridError::Parse("Missing size line".to_string()))?;
    let num_tiers = parse_int(
        value_after_eq(size_line)
            .ok_or_else(|| TextGridError::Parse("Missing size value".to_string()))?,
    )?;
    idx += 1;

    // item []:
    idx += 1;

    let mut tiers = Vec::with_capacity(num_tiers);

    for _ in 0..num_tiers {
        // Skip to "item [N]:" line
        while idx < lines.len() {
            let line = lines[idx].trim();
            if line.starts_with("item [") || line.starts_with("item[") {
                idx += 1;
                break;
            }
            idx += 1;
        }

        // class = "IntervalTier" or "TextTier"
        let class_line = lines
            .get(idx)
            .ok_or_else(|| TextGridError::Parse("Missing class line".to_string()))?;
        let class_val = value_after_eq(class_line)
            .ok_or_else(|| TextGridError::Parse("Missing class value".to_string()))?;
        let class_str = extract_quoted_string(class_val)?;
        idx += 1;

        // name = "..."
        let name_line = lines
            .get(idx)
            .ok_or_else(|| TextGridError::Parse("Missing name line".to_string()))?;
        let name_val = value_after_eq(name_line)
            .ok_or_else(|| TextGridError::Parse("Missing name value".to_string()))?;
        let name = extract_quoted_string(name_val)?;
        idx += 1;

        // tier xmin
        let tier_xmin = parse_float(
            value_after_eq(
                lines
                    .get(idx)
                    .ok_or_else(|| TextGridError::Parse("Missing tier xmin".to_string()))?,
            )
            .ok_or_else(|| TextGridError::Parse("Missing tier xmin value".to_string()))?,
        )?;
        idx += 1;

        // tier xmax
        let tier_xmax = parse_float(
            value_after_eq(
                lines
                    .get(idx)
                    .ok_or_else(|| TextGridError::Parse("Missing tier xmax".to_string()))?,
            )
            .ok_or_else(|| TextGridError::Parse("Missing tier xmax value".to_string()))?,
        )?;
        idx += 1;

        match class_str.as_str() {
            "IntervalTier" => {
                // intervals: size = N
                let isize_line = lines.get(idx).ok_or_else(|| {
                    TextGridError::Parse("Missing intervals size line".to_string())
                })?;
                let num_intervals = parse_int(value_after_eq(isize_line).ok_or_else(|| {
                    TextGridError::Parse("Missing intervals size value".to_string())
                })?)?;
                idx += 1;

                let mut intervals = Vec::with_capacity(num_intervals);
                for _ in 0..num_intervals {
                    // Skip to "intervals [N]:" line
                    while idx < lines.len() {
                        let line = lines[idx].trim();
                        if line.starts_with("intervals [") || line.starts_with("intervals[") {
                            idx += 1;
                            break;
                        }
                        idx += 1;
                    }

                    // xmin
                    let ixmin = parse_float(
                        value_after_eq(lines.get(idx).ok_or_else(|| {
                            TextGridError::Parse("Missing interval xmin".to_string())
                        })?)
                        .ok_or_else(|| {
                            TextGridError::Parse("Missing interval xmin value".to_string())
                        })?,
                    )?;
                    idx += 1;

                    // xmax
                    let ixmax = parse_float(
                        value_after_eq(lines.get(idx).ok_or_else(|| {
                            TextGridError::Parse("Missing interval xmax".to_string())
                        })?)
                        .ok_or_else(|| {
                            TextGridError::Parse("Missing interval xmax value".to_string())
                        })?,
                    )?;
                    idx += 1;

                    // text
                    let text_line = lines
                        .get(idx)
                        .ok_or_else(|| TextGridError::Parse("Missing interval text".to_string()))?;
                    let text_val = value_after_eq(text_line).ok_or_else(|| {
                        TextGridError::Parse("Missing interval text value".to_string())
                    })?;
                    let text = extract_quoted_string(text_val)?;
                    idx += 1;

                    intervals.push(Interval {
                        xmin: ixmin,
                        xmax: ixmax,
                        text,
                    });
                }

                tiers.push(TextGridTier::IntervalTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    intervals,
                });
            }
            "TextTier" => {
                // points: size = N
                let psize_line = lines
                    .get(idx)
                    .ok_or_else(|| TextGridError::Parse("Missing points size line".to_string()))?;
                let num_points = parse_int(value_after_eq(psize_line).ok_or_else(|| {
                    TextGridError::Parse("Missing points size value".to_string())
                })?)?;
                idx += 1;

                let mut points = Vec::with_capacity(num_points);
                for _ in 0..num_points {
                    // Skip to "points [N]:" line
                    while idx < lines.len() {
                        let line = lines[idx].trim();
                        if line.starts_with("points [") || line.starts_with("points[") {
                            idx += 1;
                            break;
                        }
                        idx += 1;
                    }

                    // number
                    let number = parse_float(
                        value_after_eq(lines.get(idx).ok_or_else(|| {
                            TextGridError::Parse("Missing point number".to_string())
                        })?)
                        .ok_or_else(|| {
                            TextGridError::Parse("Missing point number value".to_string())
                        })?,
                    )?;
                    idx += 1;

                    // mark
                    let mark_line = lines
                        .get(idx)
                        .ok_or_else(|| TextGridError::Parse("Missing point mark".to_string()))?;
                    let mark_val = value_after_eq(mark_line).ok_or_else(|| {
                        TextGridError::Parse("Missing point mark value".to_string())
                    })?;
                    let mark = extract_quoted_string(mark_val)?;
                    idx += 1;

                    points.push(Point { number, mark });
                }

                tiers.push(TextGridTier::TextTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    points,
                });
            }
            other => {
                return Err(TextGridError::Parse(format!(
                    "Unknown tier class: {other:?}"
                )));
            }
        }
    }

    Ok(TextGridFile {
        file_path,
        xmin,
        xmax,
        tiers,
        raw_text,
    })
}

// ---------------------------------------------------------------------------
// Short text format parser
// ---------------------------------------------------------------------------

/// Parse a TextGrid in the "short text" format.
fn parse_short_text_format(
    lines: &[&str],
    file_path: String,
    raw_text: String,
) -> Result<TextGridFile, TextGridError> {
    let mut idx = 0;

    // Skip header lines (File type, Object class, blank lines).
    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with("File type") || line.starts_with("Object class") {
            idx += 1;
        } else {
            break;
        }
    }

    // xmin
    let xmin = parse_float(
        lines
            .get(idx)
            .ok_or_else(|| TextGridError::Parse("Missing xmin".to_string()))?,
    )?;
    idx += 1;

    // xmax
    let xmax = parse_float(
        lines
            .get(idx)
            .ok_or_else(|| TextGridError::Parse("Missing xmax".to_string()))?,
    )?;
    idx += 1;

    // <exists>
    idx += 1;

    // number of tiers
    let num_tiers = parse_int(
        lines
            .get(idx)
            .ok_or_else(|| TextGridError::Parse("Missing tier count".to_string()))?,
    )?;
    idx += 1;

    let mut tiers = Vec::with_capacity(num_tiers);

    for _ in 0..num_tiers {
        // class
        let class_str = extract_quoted_string(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing tier class".to_string()))?,
        )?;
        idx += 1;

        // name
        let name = extract_quoted_string(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing tier name".to_string()))?,
        )?;
        idx += 1;

        // tier xmin
        let tier_xmin = parse_float(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing tier xmin".to_string()))?,
        )?;
        idx += 1;

        // tier xmax
        let tier_xmax = parse_float(
            lines
                .get(idx)
                .ok_or_else(|| TextGridError::Parse("Missing tier xmax".to_string()))?,
        )?;
        idx += 1;

        match class_str.as_str() {
            "IntervalTier" => {
                let num_intervals =
                    parse_int(lines.get(idx).ok_or_else(|| {
                        TextGridError::Parse("Missing interval count".to_string())
                    })?)?;
                idx += 1;

                let mut intervals = Vec::with_capacity(num_intervals);
                for _ in 0..num_intervals {
                    let ixmin = parse_float(lines.get(idx).ok_or_else(|| {
                        TextGridError::Parse("Missing interval xmin".to_string())
                    })?)?;
                    idx += 1;

                    let ixmax = parse_float(lines.get(idx).ok_or_else(|| {
                        TextGridError::Parse("Missing interval xmax".to_string())
                    })?)?;
                    idx += 1;

                    let text = extract_quoted_string(lines.get(idx).ok_or_else(|| {
                        TextGridError::Parse("Missing interval text".to_string())
                    })?)?;
                    idx += 1;

                    intervals.push(Interval {
                        xmin: ixmin,
                        xmax: ixmax,
                        text,
                    });
                }

                tiers.push(TextGridTier::IntervalTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    intervals,
                });
            }
            "TextTier" => {
                let num_points = parse_int(
                    lines
                        .get(idx)
                        .ok_or_else(|| TextGridError::Parse("Missing point count".to_string()))?,
                )?;
                idx += 1;

                let mut points = Vec::with_capacity(num_points);
                for _ in 0..num_points {
                    let number = parse_float(lines.get(idx).ok_or_else(|| {
                        TextGridError::Parse("Missing point number".to_string())
                    })?)?;
                    idx += 1;

                    let mark =
                        extract_quoted_string(lines.get(idx).ok_or_else(|| {
                            TextGridError::Parse("Missing point mark".to_string())
                        })?)?;
                    idx += 1;

                    points.push(Point { number, mark });
                }

                tiers.push(TextGridTier::TextTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    points,
                });
            }
            other => {
                return Err(TextGridError::Parse(format!(
                    "Unknown tier class: {other:?}"
                )));
            }
        }
    }

    Ok(TextGridFile {
        file_path,
        xmin,
        xmax,
        tiers,
        raw_text,
    })
}

// ---------------------------------------------------------------------------
// Main parse function
// ---------------------------------------------------------------------------

/// Parse a single TextGrid string into a [`TextGridFile`].
///
/// Supports both the normal "text" format and the "short text" format.
pub fn parse_textgrid_str(content: &str, file_path: String) -> Result<TextGridFile, TextGridError> {
    // Strip BOM if present.
    let content_clean = content.strip_prefix('\u{FEFF}').unwrap_or(content);
    // Normalize line endings.
    let content_clean = content_clean.replace("\r\n", "\n").replace('\r', "\n");

    let lines: Vec<&str> = content_clean.lines().collect();

    if lines.is_empty() {
        return Err(TextGridError::Parse("Empty TextGrid file".to_string()));
    }

    // Detect format: after skipping header lines, check if we find `=` signs.
    // The normal text format uses `key = value` pairs.
    // The short text format has bare values on separate lines.
    let is_normal = lines.iter().any(|line| {
        let trimmed = line.trim();
        // Look for characteristic normal-format patterns.
        trimmed.starts_with("xmin =")
            || trimmed.starts_with("xmax =")
            || trimmed.starts_with("size =")
            || trimmed.starts_with("class =")
    });

    if is_normal {
        parse_text_format(&lines, file_path, content.to_string())
    } else {
        parse_short_text_format(&lines, file_path, content.to_string())
    }
}

// ---------------------------------------------------------------------------
// Batch parsing helpers
// ---------------------------------------------------------------------------

fn parse_textgrid_strs(
    pairs: Vec<(String, String)>,
    parallel: bool,
) -> Result<Vec<TextGridFile>, TextGridError> {
    let parse_one = |(content, id): (String, String)| -> Result<TextGridFile, TextGridError> {
        parse_textgrid_str(&content, id)
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

fn load_textgrid_files(
    paths: &[String],
    parallel: bool,
) -> Result<Vec<TextGridFile>, TextGridError> {
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(paths.len());
    for path in paths {
        let content = std::fs::read_to_string(path)?;
        pairs.push((content, path.clone()));
    }
    parse_textgrid_strs(pairs, parallel)
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Escape a string for Praat TextGrid output (double inner quotes).
fn escape_textgrid_string(s: &str) -> String {
    s.replace('"', "\"\"")
}

/// Serialize a [`TextGridFile`] to the normal "text" format string.
pub fn serialize_textgrid_file(file: &TextGridFile) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("File type = \"ooTextFile\"\n");
    out.push_str("Object class = \"TextGrid\"\n");
    out.push('\n');
    out.push_str(&format!("xmin = {}\n", file.xmin));
    out.push_str(&format!("xmax = {}\n", file.xmax));
    out.push_str("tiers? <exists>\n");
    out.push_str(&format!("size = {}\n", file.tiers.len()));
    out.push_str("item []:\n");

    for (i, tier) in file.tiers.iter().enumerate() {
        out.push_str(&format!("    item [{}]:\n", i + 1));
        match tier {
            TextGridTier::IntervalTier {
                name,
                xmin,
                xmax,
                intervals,
            } => {
                out.push_str("        class = \"IntervalTier\"\n");
                out.push_str(&format!(
                    "        name = \"{}\"\n",
                    escape_textgrid_string(name)
                ));
                out.push_str(&format!("        xmin = {xmin}\n"));
                out.push_str(&format!("        xmax = {xmax}\n"));
                out.push_str(&format!("        intervals: size = {}\n", intervals.len()));
                for (j, interval) in intervals.iter().enumerate() {
                    out.push_str(&format!("            intervals [{}]:\n", j + 1));
                    out.push_str(&format!("                xmin = {}\n", interval.xmin));
                    out.push_str(&format!("                xmax = {}\n", interval.xmax));
                    out.push_str(&format!(
                        "                text = \"{}\"\n",
                        escape_textgrid_string(&interval.text)
                    ));
                }
            }
            TextGridTier::TextTier {
                name,
                xmin,
                xmax,
                points,
            } => {
                out.push_str("        class = \"TextTier\"\n");
                out.push_str(&format!(
                    "        name = \"{}\"\n",
                    escape_textgrid_string(name)
                ));
                out.push_str(&format!("        xmin = {xmin}\n"));
                out.push_str(&format!("        xmax = {xmax}\n"));
                out.push_str(&format!("        points: size = {}\n", points.len()));
                for (j, point) in points.iter().enumerate() {
                    out.push_str(&format!("            points [{}]:\n", j + 1));
                    out.push_str(&format!("                number = {}\n", point.number));
                    out.push_str(&format!(
                        "                mark = \"{}\"\n",
                        escape_textgrid_string(&point.mark)
                    ));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// BaseTextGrid trait
// ---------------------------------------------------------------------------

/// Core TextGrid reader behavior with default implementations.
pub trait BaseTextGrid: Sized {
    fn files(&self) -> &VecDeque<TextGridFile>;
    fn files_mut(&mut self) -> &mut VecDeque<TextGridFile>;
    fn from_files(files: VecDeque<TextGridFile>) -> Self;

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

    /// Return all tiers across all files (flat).
    fn tiers_flat(&self) -> Vec<&TextGridTier> {
        self.files().iter().flat_map(|f| &f.tiers).collect()
    }

    /// Return tiers grouped by file.
    fn tiers_by_file(&self) -> Vec<Vec<&TextGridTier>> {
        self.files()
            .iter()
            .map(|f| f.tiers.iter().collect())
            .collect()
    }

    /// Return TextGrid strings, one per file.
    fn to_strings(&self) -> Vec<String> {
        self.files().iter().map(|f| f.raw_text.clone()).collect()
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

    /// Write TextGrid files to a directory.
    fn write_files(
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
    // CHAT conversion
    // -----------------------------------------------------------------------

    /// Return CHAT format strings (one per file) for CHAT export.
    fn to_chat_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::chat_writer::textgrid_file_to_chat_str(f, participants))
            .collect()
    }

    /// Convert to a [`Chat`](crate::chat::Chat) object.
    fn to_chat_obj(&self, participants: Option<&[String]>) -> crate::chat::Chat {
        let strs = self.to_chat_strings(participants);
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
        participants: Option<&[String]>,
        filenames: Option<Vec<String>>,
    ) -> Result<(), WriteError> {
        let strs = self.to_chat_strings(participants);
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
    // ELAN conversion
    // -----------------------------------------------------------------------

    /// Return EAF XML strings (one per file) for ELAN export.
    fn to_elan_strings(&self) -> Vec<String> {
        self.files()
            .iter()
            .map(super::elan_writer::textgrid_file_to_eaf_xml)
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

    // -----------------------------------------------------------------------
    // SRT conversion
    // -----------------------------------------------------------------------

    /// Return SRT format strings (one per file) for SRT export.
    fn to_srt_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::srt_writer::textgrid_file_to_srt_str(f, participants))
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
}

// ---------------------------------------------------------------------------
// TextGrid struct
// ---------------------------------------------------------------------------

/// TextGrid data reader.
///
/// This is a pure Rust struct. For the Python-exposed wrapper, see `PyTextGrid`.
#[derive(Clone, Debug)]
pub struct TextGrid {
    pub(crate) files: VecDeque<TextGridFile>,
}

impl BaseTextGrid for TextGrid {
    fn files(&self) -> &VecDeque<TextGridFile> {
        &self.files
    }
    fn files_mut(&mut self) -> &mut VecDeque<TextGridFile> {
        &mut self.files
    }
    fn from_files(files: VecDeque<TextGridFile>) -> Self {
        Self { files }
    }
}

impl TextGrid {
    /// Construct from a Vec of [`TextGridFile`] entries.
    pub fn from_textgrid_files(files: Vec<TextGridFile>) -> Self {
        Self {
            files: VecDeque::from(files),
        }
    }

    /// Append data from another TextGrid.
    pub fn push_back(&mut self, other: &TextGrid) {
        self.files.extend(other.files.iter().cloned());
    }

    /// Prepend data from another TextGrid.
    pub fn push_front(&mut self, other: &TextGrid) {
        let mut new_files = other.files.clone();
        new_files.extend(std::mem::take(&mut self.files));
        self.files = new_files;
    }

    /// Remove and return the last file as a new TextGrid.
    pub fn pop_back(&mut self) -> Option<TextGrid> {
        self.files
            .pop_back()
            .map(|f| TextGrid::from_files(VecDeque::from(vec![f])))
    }

    /// Remove and return the first file as a new TextGrid.
    pub fn pop_front(&mut self) -> Option<TextGrid> {
        self.files
            .pop_front()
            .map(|f| TextGrid::from_files(VecDeque::from(vec![f])))
    }

    /// Parse TextGrid data from in-memory strings.
    pub fn from_strs(
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
    ) -> Result<Self, TextGridError> {
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
        let files = parse_textgrid_strs(pairs, parallel)?;
        Ok(Self::from_textgrid_files(files))
    }

    /// Load and parse TextGrid data from file paths.
    pub fn read_files(paths: &[String], parallel: bool) -> Result<Self, TextGridError> {
        let files = load_textgrid_files(paths, parallel)?;
        Ok(Self::from_textgrid_files(files))
    }

    /// Recursively load TextGrid data from a directory.
    pub fn read_dir(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<Self, TextGridError> {
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
            .map_err(|e| TextGridError::InvalidPattern(e.to_string()))?;
        let files = load_textgrid_files(&filtered, parallel)?;
        Ok(Self::from_textgrid_files(files))
    }

    /// Load TextGrid data from a ZIP archive.
    pub fn read_zip(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<Self, TextGridError> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| TextGridError::Zip(format!("Invalid zip file: {e}")))?;

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
            .map_err(|e| TextGridError::InvalidPattern(e.to_string()))?;

        let mut pairs: Vec<(String, String)> = Vec::new();
        for name in &filtered {
            let mut entry = archive
                .by_name(name)
                .map_err(|e| TextGridError::Zip(format!("Zip entry error: {e}")))?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut entry, &mut content)
                .map_err(|e| TextGridError::Zip(format!("Read error: {e}")))?;
            pairs.push((content, name.clone()));
        }

        let files = parse_textgrid_strs(pairs, parallel)?;
        Ok(Self::from_textgrid_files(files))
    }

    /// Load TextGrid data from a git repository.
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
    ) -> Result<Self, TextGridError> {
        let local_path = crate::sources::resolve_git(url, rev, depth, cache_dir, force_download)?;
        let path = local_path.to_string_lossy();
        Self::read_dir(&path, match_pattern, extension, parallel)
    }

    /// Load TextGrid data from a URL.
    pub fn from_url(
        url: &str,
        match_pattern: Option<&str>,
        extension: &str,
        cache_dir: Option<std::path::PathBuf>,
        force_download: bool,
        parallel: bool,
    ) -> Result<Self, TextGridError> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text_format() -> &'static str {
        "File type = \"ooTextFile\"\n\
         Object class = \"TextGrid\"\n\
         \n\
         xmin = 0\n\
         xmax = 2.3\n\
         tiers? <exists>\n\
         size = 2\n\
         item []:\n\
             item [1]:\n\
                 class = \"IntervalTier\"\n\
                 name = \"words\"\n\
                 xmin = 0\n\
                 xmax = 2.3\n\
                 intervals: size = 2\n\
                     intervals [1]:\n\
                         xmin = 0\n\
                         xmax = 1.5\n\
                         text = \"hello\"\n\
                     intervals [2]:\n\
                         xmin = 1.5\n\
                         xmax = 2.3\n\
                         text = \"world\"\n\
             item [2]:\n\
                 class = \"TextTier\"\n\
                 name = \"events\"\n\
                 xmin = 0\n\
                 xmax = 2.3\n\
                 points: size = 1\n\
                     points [1]:\n\
                         number = 0.75\n\
                         mark = \"click\"\n"
    }

    fn sample_short_text_format() -> &'static str {
        "File type = \"ooTextFile\"\n\
         Object class = \"TextGrid\"\n\
         \n\
         0\n\
         2.3\n\
         <exists>\n\
         2\n\
         \"IntervalTier\"\n\
         \"words\"\n\
         0\n\
         2.3\n\
         2\n\
         0\n\
         1.5\n\
         \"hello\"\n\
         1.5\n\
         2.3\n\
         \"world\"\n\
         \"TextTier\"\n\
         \"events\"\n\
         0\n\
         2.3\n\
         1\n\
         0.75\n\
         \"click\"\n"
    }

    fn assert_sample_parsed(file: &TextGridFile) {
        assert_eq!(file.xmin, 0.0);
        assert_eq!(file.xmax, 2.3);
        assert_eq!(file.tiers.len(), 2);

        match &file.tiers[0] {
            TextGridTier::IntervalTier {
                name,
                xmin,
                xmax,
                intervals,
            } => {
                assert_eq!(name, "words");
                assert_eq!(*xmin, 0.0);
                assert_eq!(*xmax, 2.3);
                assert_eq!(intervals.len(), 2);
                assert_eq!(intervals[0].xmin, 0.0);
                assert_eq!(intervals[0].xmax, 1.5);
                assert_eq!(intervals[0].text, "hello");
                assert_eq!(intervals[1].xmin, 1.5);
                assert_eq!(intervals[1].xmax, 2.3);
                assert_eq!(intervals[1].text, "world");
            }
            _ => panic!("Expected IntervalTier"),
        }

        match &file.tiers[1] {
            TextGridTier::TextTier {
                name,
                xmin,
                xmax,
                points,
            } => {
                assert_eq!(name, "events");
                assert_eq!(*xmin, 0.0);
                assert_eq!(*xmax, 2.3);
                assert_eq!(points.len(), 1);
                assert_eq!(points[0].number, 0.75);
                assert_eq!(points[0].mark, "click");
            }
            _ => panic!("Expected TextTier"),
        }
    }

    #[test]
    fn test_parse_text_format() {
        let file = parse_textgrid_str(sample_text_format(), "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.file_path, "test.TextGrid");
        assert_sample_parsed(&file);
    }

    #[test]
    fn test_parse_short_text_format() {
        let file =
            parse_textgrid_str(sample_short_text_format(), "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.file_path, "test.TextGrid");
        assert_sample_parsed(&file);
    }

    #[test]
    fn test_parse_interval_only() {
        let content = "\
            File type = \"ooTextFile\"\n\
            Object class = \"TextGrid\"\n\
            \n\
            xmin = 0\n\
            xmax = 1.0\n\
            tiers? <exists>\n\
            size = 1\n\
            item []:\n\
                item [1]:\n\
                    class = \"IntervalTier\"\n\
                    name = \"tier1\"\n\
                    xmin = 0\n\
                    xmax = 1.0\n\
                    intervals: size = 1\n\
                        intervals [1]:\n\
                            xmin = 0\n\
                            xmax = 1.0\n\
                            text = \"test\"\n";
        let file = parse_textgrid_str(content, "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 1);
        assert_eq!(file.tiers[0].tier_class(), "IntervalTier");
    }

    #[test]
    fn test_parse_texttier_only() {
        let content = "\
            File type = \"ooTextFile\"\n\
            Object class = \"TextGrid\"\n\
            \n\
            xmin = 0\n\
            xmax = 1.0\n\
            tiers? <exists>\n\
            size = 1\n\
            item []:\n\
                item [1]:\n\
                    class = \"TextTier\"\n\
                    name = \"points\"\n\
                    xmin = 0\n\
                    xmax = 1.0\n\
                    points: size = 2\n\
                        points [1]:\n\
                            number = 0.3\n\
                            mark = \"a\"\n\
                        points [2]:\n\
                            number = 0.7\n\
                            mark = \"b\"\n";
        let file = parse_textgrid_str(content, "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 1);
        match &file.tiers[0] {
            TextGridTier::TextTier { points, .. } => {
                assert_eq!(points.len(), 2);
                assert_eq!(points[0].mark, "a");
                assert_eq!(points[1].mark, "b");
            }
            _ => panic!("Expected TextTier"),
        }
    }

    #[test]
    fn test_parse_empty_tiers() {
        let content = "\
            File type = \"ooTextFile\"\n\
            Object class = \"TextGrid\"\n\
            \n\
            xmin = 0\n\
            xmax = 1.0\n\
            tiers? <exists>\n\
            size = 1\n\
            item []:\n\
                item [1]:\n\
                    class = \"IntervalTier\"\n\
                    name = \"empty\"\n\
                    xmin = 0\n\
                    xmax = 1.0\n\
                    intervals: size = 0\n";
        let file = parse_textgrid_str(content, "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 1);
        match &file.tiers[0] {
            TextGridTier::IntervalTier { intervals, .. } => {
                assert!(intervals.is_empty());
            }
            _ => panic!("Expected IntervalTier"),
        }
    }

    #[test]
    fn test_parse_empty_file() {
        let result = parse_textgrid_str("", "empty.TextGrid".to_string());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TextGridError::Parse(_)));
    }

    #[test]
    fn test_parse_escaped_quotes() {
        let content = "\
            File type = \"ooTextFile\"\n\
            Object class = \"TextGrid\"\n\
            \n\
            xmin = 0\n\
            xmax = 1.0\n\
            tiers? <exists>\n\
            size = 1\n\
            item []:\n\
                item [1]:\n\
                    class = \"IntervalTier\"\n\
                    name = \"tier1\"\n\
                    xmin = 0\n\
                    xmax = 1.0\n\
                    intervals: size = 1\n\
                        intervals [1]:\n\
                            xmin = 0\n\
                            xmax = 1.0\n\
                            text = \"he said \"\"hello\"\"\"\n";
        let file = parse_textgrid_str(content, "test.TextGrid".to_string()).unwrap();
        match &file.tiers[0] {
            TextGridTier::IntervalTier { intervals, .. } => {
                assert_eq!(intervals[0].text, "he said \"hello\"");
            }
            _ => panic!("Expected IntervalTier"),
        }
    }

    #[test]
    fn test_parse_with_bom() {
        let content = format!("\u{FEFF}{}", sample_text_format());
        let file = parse_textgrid_str(&content, "bom.TextGrid".to_string()).unwrap();
        assert_sample_parsed(&file);
    }

    #[test]
    fn test_raw_text_preserved() {
        let content = sample_text_format();
        let file = parse_textgrid_str(content, "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.raw_text, content);
    }

    #[test]
    fn test_serialize_round_trip() {
        // Parse, serialize, re-parse, and compare domain objects.
        let file = parse_textgrid_str(sample_text_format(), "test.TextGrid".to_string()).unwrap();
        let serialized = serialize_textgrid_file(&file);
        let file2 = parse_textgrid_str(&serialized, "test.TextGrid".to_string()).unwrap();
        assert_eq!(file.xmin, file2.xmin);
        assert_eq!(file.xmax, file2.xmax);
        assert_eq!(file.tiers, file2.tiers);
    }

    #[test]
    fn test_textgrid_from_strs() {
        let tg = TextGrid::from_strs(vec![sample_text_format().to_string()], None, false).unwrap();
        assert_eq!(tg.num_files(), 1);
        assert_eq!(tg.tiers_flat().len(), 2);
    }

    #[test]
    fn test_textgrid_base_trait() {
        let tg = TextGrid::from_strs(
            vec![
                sample_text_format().to_string(),
                sample_short_text_format().to_string(),
            ],
            Some(vec![
                "file1.TextGrid".to_string(),
                "file2.TextGrid".to_string(),
            ]),
            false,
        )
        .unwrap();
        assert_eq!(tg.num_files(), 2);
        assert_eq!(tg.file_paths(), vec!["file1.TextGrid", "file2.TextGrid"]);
        assert!(!tg.is_empty());
        assert_eq!(tg.tiers_flat().len(), 4);

        let by_file = tg.tiers_by_file();
        assert_eq!(by_file.len(), 2);
        assert_eq!(by_file[0].len(), 2);
        assert_eq!(by_file[1].len(), 2);
    }

    #[test]
    fn test_textgrid_push_pop() {
        let mut tg = TextGrid::from_strs(
            vec![sample_text_format().to_string()],
            Some(vec!["file1.TextGrid".to_string()]),
            false,
        )
        .unwrap();
        let tg2 = TextGrid::from_strs(
            vec![sample_short_text_format().to_string()],
            Some(vec!["file2.TextGrid".to_string()]),
            false,
        )
        .unwrap();

        tg.push_back(&tg2);
        assert_eq!(tg.num_files(), 2);
        assert_eq!(tg.file_paths(), vec!["file1.TextGrid", "file2.TextGrid"]);

        let popped = tg.pop_back().unwrap();
        assert_eq!(popped.file_paths(), vec!["file2.TextGrid"]);
        assert_eq!(tg.num_files(), 1);

        tg.push_front(&tg2);
        assert_eq!(tg.file_paths(), vec!["file2.TextGrid", "file1.TextGrid"]);

        let popped = tg.pop_front().unwrap();
        assert_eq!(popped.file_paths(), vec!["file2.TextGrid"]);
    }

    #[test]
    fn test_to_strings_round_trip() {
        let tg = TextGrid::from_strs(
            vec![sample_text_format().to_string()],
            Some(vec!["test.TextGrid".to_string()]),
            false,
        )
        .unwrap();
        let strs = tg.to_strings();
        let tg2 =
            TextGrid::from_strs(strs, Some(vec!["test.TextGrid".to_string()]), false).unwrap();
        assert_eq!(tg.files()[0].tiers, tg2.files()[0].tiers);
    }

    #[test]
    fn test_write_files() {
        let tg = TextGrid::from_strs(
            vec![sample_text_format().to_string()],
            Some(vec!["test.TextGrid".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("output");
        tg.write_files(out_dir.to_str().unwrap(), None).unwrap();
        let content = std::fs::read_to_string(out_dir.join("test.TextGrid")).unwrap();
        assert_eq!(content, sample_text_format());
    }

    #[test]
    fn test_write_files_validation() {
        let tg = TextGrid::from_strs(
            vec![
                sample_text_format().to_string(),
                sample_short_text_format().to_string(),
            ],
            Some(vec!["f1.TextGrid".to_string(), "f2.TextGrid".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let result = tg.write_files(
            dir.path().to_str().unwrap(),
            Some(vec!["only_one.TextGrid".to_string()]),
        );
        assert!(matches!(result, Err(WriteError::Validation(_))));
    }

    #[test]
    fn test_tier_name_and_class() {
        let tier = TextGridTier::IntervalTier {
            name: "words".to_string(),
            xmin: 0.0,
            xmax: 1.0,
            intervals: vec![],
        };
        assert_eq!(tier.name(), "words");
        assert_eq!(tier.tier_class(), "IntervalTier");

        let tier = TextGridTier::TextTier {
            name: "events".to_string(),
            xmin: 0.0,
            xmax: 1.0,
            points: vec![],
        };
        assert_eq!(tier.name(), "events");
        assert_eq!(tier.tier_class(), "TextTier");
    }
}
