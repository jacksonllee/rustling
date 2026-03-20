//! ELAN (.eaf) data reader.
//!
//! ELAN XML schema: <https://www.mpi.nl/tools/elan/EAFv3.0.xsd>

use crate::chat::filter_file_paths;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when reading or parsing ELAN data.
#[derive(Debug)]
pub enum ElanError {
    /// An I/O error occurred.
    Io(std::io::Error),
    /// An invalid regex pattern was provided.
    InvalidPattern(String),
    /// An error occurred reading a ZIP archive.
    Zip(String),
    /// An XML parsing error occurred.
    Xml(String),
    /// A remote source error occurred (git clone, HTTP download, etc.).
    Source(crate::sources::SourceError),
}

impl std::fmt::Display for ElanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElanError::Io(e) => write!(f, "{e}"),
            ElanError::InvalidPattern(e) => write!(f, "Invalid match regex: {e}"),
            ElanError::Zip(e) => write!(f, "{e}"),
            ElanError::Xml(e) => write!(f, "XML parse error: {e}"),
            ElanError::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ElanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ElanError::Io(e) => Some(e),
            ElanError::Source(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ElanError {
    fn from(e: std::io::Error) -> Self {
        ElanError::Io(e)
    }
}

impl From<crate::sources::SourceError> for ElanError {
    fn from(e: crate::sources::SourceError) -> Self {
        ElanError::Source(e)
    }
}

impl From<quick_xml::DeError> for ElanError {
    fn from(e: quick_xml::DeError) -> Self {
        ElanError::Xml(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Serde intermediate types for XML deserialization
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename = "ANNOTATION_DOCUMENT")]
struct XmlAnnotationDocument {
    #[serde(rename = "TIME_ORDER", default)]
    time_order: Option<XmlTimeOrder>,
    #[serde(rename = "TIER", default)]
    tiers: Vec<XmlTier>,
}

#[derive(Debug, Deserialize)]
struct XmlTimeOrder {
    #[serde(rename = "TIME_SLOT", default)]
    time_slots: Vec<XmlTimeSlot>,
}

#[derive(Debug, Deserialize)]
struct XmlTimeSlot {
    #[serde(rename = "@TIME_SLOT_ID")]
    id: String,
    #[serde(rename = "@TIME_VALUE")]
    value: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct XmlTier {
    #[serde(rename = "@TIER_ID")]
    tier_id: String,
    #[serde(rename = "@PARTICIPANT", default)]
    participant: String,
    #[serde(rename = "@ANNOTATOR", default)]
    annotator: String,
    #[serde(rename = "@LINGUISTIC_TYPE_REF", default)]
    linguistic_type_ref: String,
    #[serde(rename = "@PARENT_REF")]
    parent_ref: Option<String>,
    #[serde(rename = "ANNOTATION", default)]
    annotations: Vec<XmlAnnotationWrapper>,
}

#[derive(Debug, Deserialize)]
struct XmlAnnotationWrapper {
    #[serde(rename = "ALIGNABLE_ANNOTATION")]
    alignable: Option<XmlAlignableAnnotation>,
    #[serde(rename = "REF_ANNOTATION")]
    reference: Option<XmlRefAnnotation>,
}

#[derive(Debug, Deserialize)]
struct XmlAlignableAnnotation {
    #[serde(rename = "@ANNOTATION_ID")]
    annotation_id: String,
    #[serde(rename = "@TIME_SLOT_REF1")]
    time_slot_ref1: String,
    #[serde(rename = "@TIME_SLOT_REF2")]
    time_slot_ref2: String,
    #[serde(rename = "ANNOTATION_VALUE", default)]
    annotation_value: String,
}

#[derive(Debug, Deserialize)]
struct XmlRefAnnotation {
    #[serde(rename = "@ANNOTATION_ID")]
    annotation_id: String,
    #[serde(rename = "@ANNOTATION_REF")]
    annotation_ref: String,
    #[serde(rename = "ANNOTATION_VALUE", default)]
    annotation_value: String,
}

// ---------------------------------------------------------------------------
// WriteError
// ---------------------------------------------------------------------------

/// Error type for [`BaseElan::write_files`].
#[derive(Debug)]
pub enum WriteError {
    /// Validation error (e.g., wrong number of files or filenames).
    Validation(String),
    /// I/O error from the filesystem.
    Io(std::io::Error),
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A single annotation within a tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// Annotation ID (e.g. "a1").
    pub id: String,
    /// Start time in milliseconds, or `None` if unresolvable.
    pub start_time: Option<i64>,
    /// End time in milliseconds, or `None` if unresolvable.
    pub end_time: Option<i64>,
    /// The annotation text content.
    pub value: String,
    /// Parent annotation ID (from `ANNOTATION_REF` in `REF_ANNOTATION`),
    /// or `None` for alignable annotations.
    pub parent_id: Option<String>,
}

/// A tier (annotation layer) within an ELAN file.
#[derive(Debug, Clone, PartialEq)]
pub struct Tier {
    /// Tier ID (e.g. "G-jyutping").
    pub id: String,
    /// Participant name.
    pub participant: String,
    /// Annotator name.
    pub annotator: String,
    /// Linguistic type reference.
    pub linguistic_type_ref: String,
    /// Parent tier ID, if this is a dependent tier.
    pub parent_id: Option<String>,
    /// Child tier IDs, or `None` if no children.
    pub child_ids: Option<Vec<String>>,
    /// Annotations in this tier.
    pub annotations: Vec<Annotation>,
}

/// A single parsed ELAN (.eaf) file.
#[derive(Debug, Clone)]
pub struct ElanFile {
    /// File path or identifier.
    pub file_path: String,
    /// Tiers in this file.
    pub tiers: Vec<Tier>,
    /// Original XML content for faithful round-tripping via [`serialize_eaf_file`].
    pub raw_xml: String,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a single EAF XML string into an `ElanFile`.
pub fn parse_eaf_str(content: &str, file_path: String) -> Result<ElanFile, ElanError> {
    let doc: XmlAnnotationDocument = quick_xml::de::from_str(content)?;

    // Build time slot map: TIME_SLOT_ID -> millisecond value.
    let time_slot_map: HashMap<String, i64> = doc
        .time_order
        .map(|to| {
            to.time_slots
                .into_iter()
                .filter_map(|ts| ts.value.map(|v| (ts.id, v)))
                .collect()
        })
        .unwrap_or_default();

    // First pass: collect all alignable annotation IDs -> (start_time, end_time).
    let mut annotation_time_map: HashMap<String, (Option<i64>, Option<i64>)> = HashMap::new();
    // Also track REF_ANNOTATION -> parent annotation ID for chain resolution.
    let mut ref_chain: HashMap<String, String> = HashMap::new();

    for tier in &doc.tiers {
        for wrapper in &tier.annotations {
            if let Some(ref aa) = wrapper.alignable {
                let start = time_slot_map.get(&aa.time_slot_ref1).copied();
                let end = time_slot_map.get(&aa.time_slot_ref2).copied();
                annotation_time_map.insert(aa.annotation_id.clone(), (start, end));
            }
            if let Some(ref ra) = wrapper.reference {
                ref_chain.insert(ra.annotation_id.clone(), ra.annotation_ref.clone());
            }
        }
    }

    // Resolve REF_ANNOTATION times by following the chain.
    fn resolve_time(
        id: &str,
        annotation_time_map: &HashMap<String, (Option<i64>, Option<i64>)>,
        ref_chain: &HashMap<String, String>,
    ) -> (Option<i64>, Option<i64>) {
        if let Some(&times) = annotation_time_map.get(id) {
            return times;
        }
        if let Some(parent_id) = ref_chain.get(id) {
            return resolve_time(parent_id, annotation_time_map, ref_chain);
        }
        (None, None)
    }

    // Second pass: build Tier structs.
    let mut tiers = Vec::new();
    for xml_tier in doc.tiers {
        let mut annotations = Vec::new();
        for wrapper in xml_tier.annotations {
            if let Some(aa) = wrapper.alignable {
                let start = time_slot_map.get(&aa.time_slot_ref1).copied();
                let end = time_slot_map.get(&aa.time_slot_ref2).copied();
                annotations.push(Annotation {
                    id: aa.annotation_id,
                    start_time: start,
                    end_time: end,
                    value: aa.annotation_value,
                    parent_id: None,
                });
            } else if let Some(ra) = wrapper.reference {
                let (start, end) =
                    resolve_time(&ra.annotation_id, &annotation_time_map, &ref_chain);
                annotations.push(Annotation {
                    id: ra.annotation_id,
                    start_time: start,
                    end_time: end,
                    value: ra.annotation_value,
                    parent_id: Some(ra.annotation_ref),
                });
            }
        }
        tiers.push(Tier {
            id: xml_tier.tier_id,
            participant: xml_tier.participant,
            annotator: xml_tier.annotator,
            linguistic_type_ref: xml_tier.linguistic_type_ref,
            parent_id: xml_tier.parent_ref,
            child_ids: None,
            annotations,
        });
    }

    // Post-processing: populate child_ids.
    let mut children_map: HashMap<String, Vec<String>> = HashMap::new();
    for tier in &tiers {
        if let Some(ref parent_id) = tier.parent_id {
            children_map
                .entry(parent_id.clone())
                .or_default()
                .push(tier.id.clone());
        }
    }
    for tier in &mut tiers {
        tier.child_ids = children_map.remove(&tier.id);
    }

    Ok(ElanFile {
        file_path,
        tiers,
        raw_xml: content.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Batch parsing helpers
// ---------------------------------------------------------------------------

/// Parse multiple EAF strings in parallel or sequentially.
fn parse_eaf_strs(
    pairs: Vec<(String, String)>,
    parallel: bool,
) -> Result<Vec<ElanFile>, ElanError> {
    let parse_one = |(content, id): (String, String)| -> Result<ElanFile, ElanError> {
        parse_eaf_str(&content, id)
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

/// Load and parse EAF files from disk paths.
fn load_eaf_files(paths: &[String], parallel: bool) -> Result<Vec<ElanFile>, ElanError> {
    // Read all file contents first.
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(paths.len());
    for path in paths {
        let content = std::fs::read_to_string(path)?;
        pairs.push((content, path.clone()));
    }
    parse_eaf_strs(pairs, parallel)
}

// ---------------------------------------------------------------------------
// BaseElan trait
// ---------------------------------------------------------------------------

/// Core ELAN reader behavior with default implementations.
pub trait BaseElan: Sized {
    fn files(&self) -> &VecDeque<ElanFile>;
    fn files_mut(&mut self) -> &mut VecDeque<ElanFile>;
    fn from_files(files: VecDeque<ElanFile>) -> Self;

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
    fn tiers_flat(&self) -> Vec<&Tier> {
        self.files().iter().flat_map(|f| &f.tiers).collect()
    }

    /// Return tiers grouped by file.
    fn tiers_by_file(&self) -> Vec<Vec<&Tier>> {
        self.files()
            .iter()
            .map(|f| f.tiers.iter().collect())
            .collect()
    }

    /// Return EAF XML strings, one per file.
    fn to_strings(&self) -> Vec<String> {
        self.files().iter().map(|f| f.raw_xml.clone()).collect()
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

    /// Write ELAN (.eaf) files to a directory.
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
    // CHAT conversion
    // -----------------------------------------------------------------------

    /// Return CHAT format strings (one per file) for CHAT export.
    fn to_chat_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::chat_writer::elan_file_to_chat_str(f, participants))
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
        // Disable mor+gra parsing: ELAN-derived CHAT doesn't have parsed tokens.
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
    // SRT conversion
    // -----------------------------------------------------------------------

    /// Return SRT format strings (one per file) for SRT export.
    fn to_srt_strings(&self, participants: Option<&[String]>) -> Vec<String> {
        self.files()
            .iter()
            .map(|f| super::srt_writer::elan_file_to_srt_str(f, participants))
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
    // TextGrid conversion
    // -----------------------------------------------------------------------

    /// Return TextGrid format strings (one per file) for TextGrid export.
    fn to_textgrid_strings(&self) -> Vec<String> {
        self.files()
            .iter()
            .map(super::textgrid_writer::elan_file_to_textgrid_str)
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

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Return the stored XML content of an [`ElanFile`].
///
/// This replays the original XML that was stored at parse time,
/// ensuring faithful round-tripping.
pub fn serialize_eaf_file(file: &ElanFile) -> &str {
    &file.raw_xml
}

// ---------------------------------------------------------------------------
// Elan struct
// ---------------------------------------------------------------------------

/// ELAN data reader.
///
/// This is a pure Rust struct. For the Python-exposed wrapper, see `PyElan`.
#[derive(Clone, Debug)]
pub struct Elan {
    pub(crate) files: VecDeque<ElanFile>,
}

impl BaseElan for Elan {
    fn files(&self) -> &VecDeque<ElanFile> {
        &self.files
    }
    fn files_mut(&mut self) -> &mut VecDeque<ElanFile> {
        &mut self.files
    }
    fn from_files(files: VecDeque<ElanFile>) -> Self {
        Self { files }
    }
}

impl Elan {
    /// Construct from a Vec of [`ElanFile`] entries.
    pub fn from_elan_files(files: Vec<ElanFile>) -> Self {
        Self {
            files: VecDeque::from(files),
        }
    }

    /// Append data from another Elan.
    pub fn push_back(&mut self, other: &Elan) {
        self.files.extend(other.files.iter().cloned());
    }

    /// Prepend data from another Elan.
    pub fn push_front(&mut self, other: &Elan) {
        let mut new_files = other.files.clone();
        new_files.extend(std::mem::take(&mut self.files));
        self.files = new_files;
    }

    /// Remove and return the last file as a new Elan.
    pub fn pop_back(&mut self) -> Option<Elan> {
        self.files
            .pop_back()
            .map(|f| Elan::from_files(VecDeque::from(vec![f])))
    }

    /// Remove and return the first file as a new Elan.
    pub fn pop_front(&mut self) -> Option<Elan> {
        self.files
            .pop_front()
            .map(|f| Elan::from_files(VecDeque::from(vec![f])))
    }

    /// Parse ELAN data from in-memory strings.
    pub fn from_strs(
        strs: Vec<String>,
        ids: Option<Vec<String>>,
        parallel: bool,
    ) -> Result<Self, ElanError> {
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
        let files = parse_eaf_strs(pairs, parallel)?;
        Ok(Self::from_elan_files(files))
    }

    /// Load and parse ELAN data from file paths.
    pub fn read_files(paths: &[String], parallel: bool) -> Result<Self, ElanError> {
        let files = load_eaf_files(paths, parallel)?;
        Ok(Self::from_elan_files(files))
    }

    /// Recursively load ELAN data from a directory.
    pub fn read_dir(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<Self, ElanError> {
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
            .map_err(|e| ElanError::InvalidPattern(e.to_string()))?;
        let files = load_eaf_files(&filtered, parallel)?;
        Ok(Self::from_elan_files(files))
    }

    /// Load ELAN data from a ZIP archive.
    pub fn read_zip(
        path: &str,
        match_pattern: Option<&str>,
        extension: &str,
        parallel: bool,
    ) -> Result<Self, ElanError> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| ElanError::Zip(format!("Invalid zip file: {e}")))?;

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
            .map_err(|e| ElanError::InvalidPattern(e.to_string()))?;

        let mut pairs: Vec<(String, String)> = Vec::new();
        for name in &filtered {
            let mut entry = archive
                .by_name(name)
                .map_err(|e| ElanError::Zip(format!("Zip entry error: {e}")))?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut entry, &mut content)
                .map_err(|e| ElanError::Zip(format!("Read error: {e}")))?;
            pairs.push((content, name.clone()));
        }

        let files = parse_eaf_strs(pairs, parallel)?;
        Ok(Self::from_elan_files(files))
    }

    /// Load ELAN data from a git repository.
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
    ) -> Result<Self, ElanError> {
        let local_path = crate::sources::resolve_git(url, rev, depth, cache_dir, force_download)?;
        let path = local_path.to_string_lossy();
        Self::read_dir(&path, match_pattern, extension, parallel)
    }

    /// Load ELAN data from a URL.
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
    ) -> Result<Self, ElanError> {
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

    fn sample_eaf() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2024-01-01T00:00:00+00:00" FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="2000"/>
        <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="3500"/>
    </TIME_ORDER>
    <TIER TIER_ID="Speaker1" PARTICIPANT="Alice" ANNOTATOR="Ann" LINGUISTIC_TYPE_REF="default-lt">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>hello world</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
                <ANNOTATION_VALUE>goodbye</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>"#
    }

    fn sample_eaf_with_ref() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>
    </TIME_ORDER>
    <TIER TIER_ID="Main" PARTICIPANT="Alice" ANNOTATOR="" LINGUISTIC_TYPE_REF="default-lt">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>hello</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER TIER_ID="Gloss" PARTICIPANT="" ANNOTATOR="" LINGUISTIC_TYPE_REF="gloss-lt" PARENT_REF="Main">
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a2" ANNOTATION_REF="a1">
                <ANNOTATION_VALUE>greeting</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>"#
    }

    #[test]
    fn test_parse_basic_eaf() {
        let file = parse_eaf_str(sample_eaf(), "test.eaf".to_string()).unwrap();
        assert_eq!(file.file_path, "test.eaf");
        assert_eq!(file.tiers.len(), 1);

        let tier = &file.tiers[0];
        assert_eq!(tier.id, "Speaker1");
        assert_eq!(tier.participant, "Alice");
        assert_eq!(tier.annotator, "Ann");
        assert_eq!(tier.linguistic_type_ref, "default-lt");
        assert!(tier.parent_id.is_none());
        assert!(tier.child_ids.is_none());
        assert_eq!(tier.annotations.len(), 2);

        let a1 = &tier.annotations[0];
        assert_eq!(a1.id, "a1");
        assert_eq!(a1.start_time, Some(0));
        assert_eq!(a1.end_time, Some(1500));
        assert_eq!(a1.value, "hello world");
        assert_eq!(a1.parent_id, None);

        let a2 = &tier.annotations[1];
        assert_eq!(a2.id, "a2");
        assert_eq!(a2.start_time, Some(2000));
        assert_eq!(a2.end_time, Some(3500));
        assert_eq!(a2.value, "goodbye");
        assert_eq!(a2.parent_id, None);
    }

    #[test]
    fn test_parse_ref_annotation() {
        let file = parse_eaf_str(sample_eaf_with_ref(), "test.eaf".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 2);

        let main_tier = &file.tiers[0];
        assert_eq!(main_tier.id, "Main");
        assert_eq!(main_tier.child_ids, Some(vec!["Gloss".to_string()]));
        assert_eq!(main_tier.annotations.len(), 1);
        assert_eq!(main_tier.annotations[0].start_time, Some(0));
        assert_eq!(main_tier.annotations[0].end_time, Some(1500));

        let gloss_tier = &file.tiers[1];
        assert_eq!(gloss_tier.id, "Gloss");
        assert_eq!(gloss_tier.parent_id, Some("Main".to_string()));
        assert!(gloss_tier.child_ids.is_none());
        assert_eq!(gloss_tier.annotations.len(), 1);

        let ref_ann = &gloss_tier.annotations[0];
        assert_eq!(ref_ann.id, "a2");
        assert_eq!(ref_ann.value, "greeting");
        // REF_ANNOTATION resolves time from parent
        assert_eq!(ref_ann.start_time, Some(0));
        assert_eq!(ref_ann.end_time, Some(1500));
        assert_eq!(ref_ann.parent_id, Some("a1".to_string()));
    }

    #[test]
    fn test_parse_empty_eaf() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER/>
</ANNOTATION_DOCUMENT>"#;
        let file = parse_eaf_str(xml, "empty.eaf".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 0);
    }

    #[test]
    fn test_parse_malformed_xml() {
        let result = parse_eaf_str("not xml at all", "bad.eaf".to_string());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ElanError::Xml(_)));
    }

    #[test]
    fn test_elan_from_strs() {
        let elan = Elan::from_strs(vec![sample_eaf().to_string()], None, false).unwrap();
        assert_eq!(elan.num_files(), 1);
        assert_eq!(elan.tiers_flat().len(), 1);
    }

    #[test]
    fn test_elan_base_trait() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string(), sample_eaf_with_ref().to_string()],
            Some(vec!["file1.eaf".to_string(), "file2.eaf".to_string()]),
            false,
        )
        .unwrap();
        assert_eq!(elan.num_files(), 2);
        assert_eq!(elan.file_paths(), vec!["file1.eaf", "file2.eaf"]);
        assert!(!elan.is_empty());

        // tiers_flat: 1 tier from file1 + 2 tiers from file2
        assert_eq!(elan.tiers_flat().len(), 3);

        // tiers_by_file: grouped
        let by_file = elan.tiers_by_file();
        assert_eq!(by_file.len(), 2);
        assert_eq!(by_file[0].len(), 1);
        assert_eq!(by_file[1].len(), 2);
    }

    #[test]
    fn test_elan_push_pop() {
        let mut elan = Elan::from_strs(
            vec![sample_eaf().to_string()],
            Some(vec!["file1.eaf".to_string()]),
            false,
        )
        .unwrap();
        let elan2 = Elan::from_strs(
            vec![sample_eaf_with_ref().to_string()],
            Some(vec!["file2.eaf".to_string()]),
            false,
        )
        .unwrap();

        elan.push_back(&elan2);
        assert_eq!(elan.num_files(), 2);
        assert_eq!(elan.file_paths(), vec!["file1.eaf", "file2.eaf"]);

        let popped = elan.pop_back().unwrap();
        assert_eq!(popped.file_paths(), vec!["file2.eaf"]);
        assert_eq!(elan.num_files(), 1);

        elan.push_front(&elan2);
        assert_eq!(elan.file_paths(), vec!["file2.eaf", "file1.eaf"]);

        let popped = elan.pop_front().unwrap();
        assert_eq!(popped.file_paths(), vec!["file2.eaf"]);
    }

    #[test]
    fn test_parse_tier_without_annotations() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER/>
    <TIER TIER_ID="empty" LINGUISTIC_TYPE_REF="default-lt"/>
</ANNOTATION_DOCUMENT>"#;
        let file = parse_eaf_str(xml, "test.eaf".to_string()).unwrap();
        assert_eq!(file.tiers.len(), 1);
        assert_eq!(file.tiers[0].id, "empty");
        assert!(file.tiers[0].child_ids.is_none());
        assert_eq!(file.tiers[0].annotations.len(), 0);
    }

    #[test]
    fn test_raw_xml_preserved() {
        let xml = sample_eaf();
        let file = parse_eaf_str(xml, "test.eaf".to_string()).unwrap();
        assert_eq!(file.raw_xml, xml);
    }

    #[test]
    fn test_serialize_eaf_file() {
        let xml = sample_eaf();
        let file = parse_eaf_str(xml, "test.eaf".to_string()).unwrap();
        assert_eq!(serialize_eaf_file(&file), xml);
    }

    #[test]
    fn test_to_strings() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string(), sample_eaf_with_ref().to_string()],
            Some(vec!["f1.eaf".to_string(), "f2.eaf".to_string()]),
            false,
        )
        .unwrap();
        let strs = elan.to_strings();
        assert_eq!(strs.len(), 2);
        assert_eq!(strs[0], sample_eaf());
        assert_eq!(strs[1], sample_eaf_with_ref());
    }

    #[test]
    fn test_to_strings_round_trip() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string()],
            Some(vec!["test.eaf".to_string()]),
            false,
        )
        .unwrap();
        let strs = elan.to_strings();
        let elan2 = Elan::from_strs(strs, Some(vec!["test.eaf".to_string()]), false).unwrap();
        assert_eq!(elan.files()[0].tiers, elan2.files()[0].tiers);
    }

    #[test]
    fn test_write_files_single() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string()],
            Some(vec!["test.eaf".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("output");
        elan.write_files(out_dir.to_str().unwrap(), None).unwrap();
        let content = std::fs::read_to_string(out_dir.join("test.eaf")).unwrap();
        assert_eq!(content, sample_eaf());
    }

    #[test]
    fn test_write_files_directory() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string(), sample_eaf_with_ref().to_string()],
            Some(vec!["f1.eaf".to_string(), "f2.eaf".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("output");
        elan.write_files(out_dir.to_str().unwrap(), None).unwrap();
        let f1 = std::fs::read_to_string(out_dir.join("f1.eaf")).unwrap();
        let f2 = std::fs::read_to_string(out_dir.join("f2.eaf")).unwrap();
        assert_eq!(f1, sample_eaf());
        assert_eq!(f2, sample_eaf_with_ref());
    }

    #[test]
    fn test_write_files_custom_filenames() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string(), sample_eaf_with_ref().to_string()],
            Some(vec!["f1.eaf".to_string(), "f2.eaf".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("output");
        elan.write_files(
            out_dir.to_str().unwrap(),
            Some(vec!["alice.eaf".to_string(), "bob.eaf".to_string()]),
        )
        .unwrap();
        assert!(out_dir.join("alice.eaf").exists());
        assert!(out_dir.join("bob.eaf").exists());
    }

    #[test]
    fn test_write_files_validation_filename_mismatch() {
        let elan = Elan::from_strs(
            vec![sample_eaf().to_string(), sample_eaf_with_ref().to_string()],
            Some(vec!["f1.eaf".to_string(), "f2.eaf".to_string()]),
            false,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let result = elan.write_files(
            dir.path().to_str().unwrap(),
            Some(vec!["only_one.eaf".to_string()]),
        );
        assert!(matches!(result, Err(WriteError::Validation(_))));
    }
}
