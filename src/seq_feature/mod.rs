//! Configurable feature templates for sequence labeling models.
//!
//! Provides [`SeqFeatureTemplate`] for specifying what observations and labels
//! each model sees, plus extraction functions used by both averaged perceptron
//! and HMM models.

#[cfg(feature = "pyo3")]
mod py;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::persistence::ModelError;

#[cfg(feature = "pyo3")]
pub(crate) use py::register_module;

// ---------------------------------------------------------------------------
// Sentinel constants for boundary positions
// ---------------------------------------------------------------------------

/// Sentinel strings for positions before the start of a sequence.
pub const START: [&str; 4] = ["-START-", "-START2-", "-START3-", "-START4-"];

/// Sentinel strings for positions after the end of a sequence.
pub const END: [&str; 4] = ["-END-", "-END2-", "-END3-", "-END4-"];

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/// Get the first character of a string, or an empty string if empty.
#[inline]
pub fn first_char(s: &str) -> &str {
    s.chars().next().map(|c| &s[..c.len_utf8()]).unwrap_or("")
}

/// Get the final character of a string, or an empty string if empty.
#[inline]
pub fn final_char(s: &str) -> &str {
    s.chars()
        .next_back()
        .map(|c| &s[s.len() - c.len_utf8()..])
        .unwrap_or("")
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Transform applied to observation strings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeqTransform {
    /// Use the observation as-is.
    Identity,
    /// Use only the first character.
    FirstChar,
    /// Use only the final character.
    FinalChar,
}

/// Whether a feature template extracts from observations or labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum SeqFeatureKind {
    Obs,
    Label,
}

/// A single feature template for sequence labeling.
///
/// Specifies what to extract (observation or label), at which relative
/// positions, and with what transform.
#[cfg_attr(feature = "pyo3", pyclass(name = "SeqFeatureTemplate", from_py_object))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeqFeatureTemplate {
    pub(crate) kind: SeqFeatureKind,
    pub(crate) positions: Vec<i32>,
    pub(crate) transform: SeqTransform,
}

impl SeqFeatureTemplate {
    /// Create an observation feature template with a transform.
    pub fn obs(positions: &[i32], transform: SeqTransform) -> Self {
        Self {
            kind: SeqFeatureKind::Obs,
            positions: positions.to_vec(),
            transform,
        }
    }

    /// Create an observation feature template with identity transform.
    pub fn obs_identity(positions: &[i32]) -> Self {
        Self::obs(positions, SeqTransform::Identity)
    }

    /// Create a label feature template.
    pub fn label(positions: &[i32]) -> Self {
        Self {
            kind: SeqFeatureKind::Label,
            positions: positions.to_vec(),
            transform: SeqTransform::Identity,
        }
    }

    /// Whether this template extracts from labels (not observations).
    pub(crate) fn is_label(&self) -> bool {
        self.kind == SeqFeatureKind::Label
    }
}

/// Validated collection of feature templates with pre-computed metadata.
#[derive(Clone, Debug)]
pub struct SeqFeatureConfig {
    pub templates: Vec<SeqFeatureTemplate>,
    /// Whether all templates are observation-only (no label features).
    pub obs_only: bool,
}

impl SeqFeatureConfig {
    /// Build a config from a list of templates.
    pub fn new(templates: Vec<SeqFeatureTemplate>) -> Self {
        let obs_only = templates.iter().all(|t| !t.is_label());
        Self {
            templates,
            obs_only,
        }
    }
}

// ---------------------------------------------------------------------------
// Feature key generation helpers
// ---------------------------------------------------------------------------

/// Apply a transform to a string, returning a `Cow` to avoid allocation
/// when using Identity.
fn apply_transform<'a>(s: &'a str, transform: &SeqTransform) -> &'a str {
    match transform {
        SeqTransform::Identity => s,
        SeqTransform::FirstChar => first_char(s),
        SeqTransform::FinalChar => final_char(s),
    }
}

fn transform_suffix(transform: &SeqTransform) -> &'static str {
    match transform {
        SeqTransform::Identity => "",
        SeqTransform::FirstChar => ":first_char",
        SeqTransform::FinalChar => ":final_char",
    }
}

/// Resolve the observation at a relative position, using sentinels for
/// out-of-bounds positions.
fn resolve_obs<'a>(observations: &[&'a str], i: usize, offset: i32) -> &'a str {
    let pos = i as i64 + offset as i64;
    if pos < 0 {
        let sentinel_idx = (-1 - pos) as usize;
        START.get(sentinel_idx).copied().unwrap_or(START[3])
    } else if pos >= observations.len() as i64 {
        let sentinel_idx = (pos - observations.len() as i64) as usize;
        END.get(sentinel_idx).copied().unwrap_or(END[3])
    } else {
        observations[pos as usize]
    }
}

/// Resolve the label at a relative position, using sentinels for
/// out-of-bounds positions.
fn resolve_label<'a>(labels: &[&'a str], i: usize, offset: i32) -> &'a str {
    let pos = i as i64 + offset as i64;
    if pos < 0 {
        let sentinel_idx = (-1 - pos) as usize;
        START.get(sentinel_idx).copied().unwrap_or(START[3])
    } else if pos >= labels.len() as i64 {
        let sentinel_idx = (pos - labels.len() as i64) as usize;
        END.get(sentinel_idx).copied().unwrap_or(END[3])
    } else {
        labels[pos as usize]
    }
}

// ---------------------------------------------------------------------------
// AP feature extraction
// ---------------------------------------------------------------------------

/// A reusable buffer for building feature strings.
pub struct FeatureBuffer {
    features: Vec<String>,
}

impl Default for FeatureBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureBuffer {
    pub fn new() -> Self {
        Self {
            features: Vec::with_capacity(16),
        }
    }

    pub fn clear(&mut self) {
        self.features.clear();
    }

    pub fn push(&mut self, feature: String) {
        self.features.push(feature);
    }

    pub fn keys(&self) -> Vec<&str> {
        self.features.iter().map(|s| s.as_str()).collect()
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }
}

/// Extract AP features into `buf` for position `i`.
///
/// `labels` are the predicted labels so far (from the left-to-right pass).
/// Label features use negative positions to look back at previous labels.
pub fn extract_features(
    buf: &mut FeatureBuffer,
    config: &SeqFeatureConfig,
    observations: &[&str],
    i: usize,
    labels: &[&str],
) {
    buf.push("bias".to_string());

    for template in &config.templates {
        match template.kind {
            SeqFeatureKind::Obs => {
                let positions_str = template
                    .positions
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let tsuf = transform_suffix(&template.transform);

                if template.positions.len() == 1 {
                    let val = apply_transform(
                        resolve_obs(observations, i, template.positions[0]),
                        &template.transform,
                    );
                    buf.push(format!("obs:{}{} {}", positions_str, tsuf, val));
                } else {
                    // N-gram: concatenate transformed values with space separator.
                    let mut parts = String::new();
                    for (idx, &pos) in template.positions.iter().enumerate() {
                        if idx > 0 {
                            parts.push(' ');
                        }
                        parts.push_str(apply_transform(
                            resolve_obs(observations, i, pos),
                            &template.transform,
                        ));
                    }
                    buf.push(format!("obs:{}{} {}", positions_str, tsuf, parts));
                }
            }
            SeqFeatureKind::Label => {
                let positions_str = template
                    .positions
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",");

                if template.positions.len() == 1 {
                    let val = resolve_label(labels, i, template.positions[0]);
                    buf.push(format!("label:{} {}", positions_str, val));
                } else {
                    let mut parts = String::new();
                    for (idx, &pos) in template.positions.iter().enumerate() {
                        if idx > 0 {
                            parts.push(' ');
                        }
                        parts.push_str(resolve_label(labels, i, pos));
                    }
                    buf.push(format!("label:{} {}", positions_str, parts));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HMM observation extraction
// ---------------------------------------------------------------------------

/// Extract a single observation string for an HMM feature template at position `i`.
pub fn extract_observation(
    template: &SeqFeatureTemplate,
    observations: &[&str],
    i: usize,
) -> String {
    if template.positions.len() == 1 {
        apply_transform(
            resolve_obs(observations, i, template.positions[0]),
            &template.transform,
        )
        .to_string()
    } else {
        let mut result = String::new();
        for (idx, &pos) in template.positions.iter().enumerate() {
            if idx > 0 {
                result.push(' ');
            }
            result.push_str(apply_transform(
                resolve_obs(observations, i, pos),
                &template.transform,
            ));
        }
        result
    }
}

/// Like [`extract_observation`], but returns `Cow::Borrowed` when possible
/// (single-position templates) to avoid heap allocation.
pub fn extract_observation_cow<'a>(
    template: &SeqFeatureTemplate,
    observations: &[&'a str],
    i: usize,
) -> Cow<'a, str> {
    if template.positions.len() == 1 {
        Cow::Borrowed(apply_transform(
            resolve_obs(observations, i, template.positions[0]),
            &template.transform,
        ))
    } else {
        let mut result = String::new();
        for (idx, &pos) in template.positions.iter().enumerate() {
            if idx > 0 {
                result.push(' ');
            }
            result.push_str(apply_transform(
                resolve_obs(observations, i, pos),
                &template.transform,
            ));
        }
        Cow::Owned(result)
    }
}

// ---------------------------------------------------------------------------
// Default feature sets
// ---------------------------------------------------------------------------

/// Default features for the tagging averaged perceptron.
pub fn default_tagger_ap_features() -> Vec<SeqFeatureTemplate> {
    vec![
        SeqFeatureTemplate::obs(&[0], SeqTransform::FirstChar),
        SeqFeatureTemplate::obs(&[0], SeqTransform::FinalChar),
        SeqFeatureTemplate::obs(&[-1], SeqTransform::FirstChar),
        SeqFeatureTemplate::obs(&[-1], SeqTransform::FinalChar),
        SeqFeatureTemplate::label(&[-1]),
        SeqFeatureTemplate::obs(&[-2], SeqTransform::FirstChar),
        SeqFeatureTemplate::obs(&[-2], SeqTransform::FinalChar),
        SeqFeatureTemplate::label(&[-2]),
        SeqFeatureTemplate::obs(&[1], SeqTransform::FirstChar),
        SeqFeatureTemplate::obs(&[1], SeqTransform::FinalChar),
        SeqFeatureTemplate::obs(&[2], SeqTransform::FirstChar),
        SeqFeatureTemplate::obs(&[2], SeqTransform::FinalChar),
    ]
}

/// Default features for the wordseg averaged perceptron.
pub fn default_segmenter_ap_features() -> Vec<SeqFeatureTemplate> {
    vec![
        SeqFeatureTemplate::obs_identity(&[0]),
        SeqFeatureTemplate::obs_identity(&[-1]),
        SeqFeatureTemplate::obs_identity(&[-2]),
        SeqFeatureTemplate::obs_identity(&[1]),
        SeqFeatureTemplate::obs_identity(&[2]),
        SeqFeatureTemplate::obs_identity(&[-1, 0]),
        SeqFeatureTemplate::obs_identity(&[0, 1]),
        SeqFeatureTemplate::label(&[-1]),
        SeqFeatureTemplate::label(&[-2]),
    ]
}

/// Default features for the wordseg HMM.
pub fn default_segmenter_hmm_features() -> Vec<SeqFeatureTemplate> {
    vec![
        SeqFeatureTemplate::obs_identity(&[-1]),
        SeqFeatureTemplate::obs_identity(&[0]),
        SeqFeatureTemplate::obs_identity(&[1]),
        SeqFeatureTemplate::obs_identity(&[-1, 0]),
        SeqFeatureTemplate::obs_identity(&[0, 1]),
        SeqFeatureTemplate::obs_identity(&[-1, 1]),
    ]
}

/// Default features for the tagging HMM.
pub fn default_tagger_hmm_features() -> Vec<SeqFeatureTemplate> {
    vec![SeqFeatureTemplate::obs_identity(&[0])]
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a list of feature templates.
///
/// Returns an error string if validation fails.
pub fn validate_templates(
    templates: &[SeqFeatureTemplate],
    allow_labels: bool,
) -> Result<(), ModelError> {
    for template in templates {
        if template.positions.is_empty() {
            return Err(ModelError::ValidationError(
                "Feature template must have at least one position.".to_string(),
            ));
        }
        for &pos in &template.positions {
            if !(-4..=4).contains(&pos) {
                return Err(ModelError::ValidationError(format!(
                    "Position {} is out of range [-4, +4].",
                    pos
                )));
            }
        }
        if template.is_label() {
            if !allow_labels {
                return Err(ModelError::ValidationError(
                    "Label features (seq_label) are not supported for HMM models.".to_string(),
                ));
            }
            for &pos in &template.positions {
                if pos >= 0 {
                    return Err(ModelError::ValidationError(format!(
                        "seq_label positions must be negative (look back only), got {}.",
                        pos
                    )));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_char() {
        assert_eq!(first_char("hello"), "h");
        assert_eq!(first_char("世界"), "世");
        assert_eq!(first_char(""), "");
    }

    #[test]
    fn test_final_char() {
        assert_eq!(final_char("hello"), "o");
        assert_eq!(final_char("世界"), "界");
        assert_eq!(final_char(""), "");
    }

    #[test]
    fn test_resolve_obs_in_bounds() {
        let obs = vec!["a", "b", "c"];
        assert_eq!(resolve_obs(&obs, 1, 0), "b");
        assert_eq!(resolve_obs(&obs, 1, -1), "a");
        assert_eq!(resolve_obs(&obs, 1, 1), "c");
    }

    #[test]
    fn test_resolve_obs_boundary() {
        let obs = vec!["a", "b", "c"];
        assert_eq!(resolve_obs(&obs, 0, -1), START[0]);
        assert_eq!(resolve_obs(&obs, 0, -2), START[1]);
        assert_eq!(resolve_obs(&obs, 2, 1), END[0]);
        assert_eq!(resolve_obs(&obs, 2, 2), END[1]);
    }

    #[test]
    fn test_extract_features_obs_identity() {
        let config = SeqFeatureConfig::new(vec![SeqFeatureTemplate::obs_identity(&[0])]);
        let obs = vec!["a", "b", "c"];
        let labels: Vec<&str> = vec![];
        let mut buf = FeatureBuffer::new();

        extract_features(&mut buf, &config, &obs, 1, &labels);

        assert_eq!(buf.features().len(), 2); // bias + 1 template
        assert_eq!(buf.features()[0], "bias");
        assert_eq!(buf.features()[1], "obs:0 b");
    }

    #[test]
    fn test_extract_features_obs_bigram() {
        let config = SeqFeatureConfig::new(vec![SeqFeatureTemplate::obs_identity(&[-1, 0])]);
        let obs = vec!["a", "b", "c"];
        let labels: Vec<&str> = vec![];
        let mut buf = FeatureBuffer::new();

        extract_features(&mut buf, &config, &obs, 1, &labels);

        assert_eq!(buf.features()[1], "obs:-1,0 a b");
    }

    #[test]
    fn test_extract_features_obs_first_char() {
        let config =
            SeqFeatureConfig::new(vec![SeqFeatureTemplate::obs(&[0], SeqTransform::FirstChar)]);
        let obs = vec!["hello", "world"];
        let labels: Vec<&str> = vec![];
        let mut buf = FeatureBuffer::new();

        extract_features(&mut buf, &config, &obs, 0, &labels);

        assert_eq!(buf.features()[1], "obs:0:first_char h");
    }

    #[test]
    fn test_extract_features_label() {
        let config = SeqFeatureConfig::new(vec![SeqFeatureTemplate::label(&[-1])]);
        let obs = vec!["a", "b", "c"];
        let labels = vec!["X", "Y"];
        let mut buf = FeatureBuffer::new();

        extract_features(&mut buf, &config, &obs, 1, &labels);

        assert_eq!(buf.features()[1], "label:-1 X");
    }

    #[test]
    fn test_extract_features_label_boundary() {
        let config = SeqFeatureConfig::new(vec![SeqFeatureTemplate::label(&[-1])]);
        let obs = vec!["a", "b"];
        let labels: Vec<&str> = vec![];
        let mut buf = FeatureBuffer::new();

        extract_features(&mut buf, &config, &obs, 0, &labels);

        assert_eq!(buf.features()[1], "label:-1 -START-");
    }

    #[test]
    fn test_extract_observation_identity() {
        let template = SeqFeatureTemplate::obs_identity(&[0]);
        let obs = vec!["a", "b", "c"];
        assert_eq!(extract_observation(&template, &obs, 1), "b");
    }

    #[test]
    fn test_extract_observation_bigram() {
        let template = SeqFeatureTemplate::obs_identity(&[-1, 0]);
        let obs = vec!["a", "b", "c"];
        assert_eq!(extract_observation(&template, &obs, 1), "a b");
    }

    #[test]
    fn test_extract_observation_boundary() {
        let template = SeqFeatureTemplate::obs_identity(&[-1]);
        let obs = vec!["a", "b"];
        assert_eq!(extract_observation(&template, &obs, 0), START[0]);
    }

    #[test]
    fn test_config_obs_only() {
        let config = SeqFeatureConfig::new(vec![
            SeqFeatureTemplate::obs_identity(&[0]),
            SeqFeatureTemplate::obs_identity(&[-1, 0]),
        ]);
        assert!(config.obs_only);
    }

    #[test]
    fn test_validate_templates_ok() {
        let templates = default_tagger_ap_features();
        assert!(validate_templates(&templates, true).is_ok());
    }

    #[test]
    fn test_validate_templates_label_in_hmm() {
        let templates = vec![SeqFeatureTemplate::label(&[-1])];
        assert!(validate_templates(&templates, false).is_err());
    }

    #[test]
    fn test_validate_templates_label_at_zero() {
        let templates = vec![SeqFeatureTemplate::label(&[0])];
        assert!(validate_templates(&templates, true).is_err());
    }

    #[test]
    fn test_validate_templates_label_positive_position() {
        let templates = vec![SeqFeatureTemplate::label(&[1])];
        assert!(validate_templates(&templates, true).is_err());
    }

    #[test]
    fn test_validate_templates_position_out_of_range() {
        let templates = vec![SeqFeatureTemplate::obs_identity(&[5])];
        assert!(validate_templates(&templates, true).is_err());
    }

    #[test]
    fn test_validate_templates_empty_positions() {
        let templates = vec![SeqFeatureTemplate::obs_identity(&[])];
        assert!(validate_templates(&templates, true).is_err());
    }

    #[test]
    fn test_default_feature_sets() {
        // Just verify they don't panic and produce expected counts.
        assert_eq!(default_tagger_ap_features().len(), 12);
        assert_eq!(default_segmenter_ap_features().len(), 9);
        assert_eq!(default_segmenter_hmm_features().len(), 6);
        assert_eq!(default_tagger_hmm_features().len(), 1);
    }
}
