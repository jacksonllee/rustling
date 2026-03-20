//! Convert ELAN (.eaf) data to TextGrid (Praat) format.

use super::reader::ElanFile;
use crate::textgrid::serialize_textgrid_file;

/// Convert a single [`ElanFile`] to a TextGrid format string.
///
/// Each ELAN tier becomes an IntervalTier.
/// Annotations without time marks are skipped.
/// Times are converted from milliseconds to seconds.
pub(crate) fn elan_file_to_textgrid_str(file: &ElanFile) -> String {
    let mut tiers: Vec<crate::textgrid::TextGridTier> = Vec::new();
    let mut global_xmax: f64 = 0.0;

    for tier in &file.tiers {
        let mut intervals: Vec<crate::textgrid::Interval> = Vec::new();
        for ann in &tier.annotations {
            let (start, end) = match (ann.start_time, ann.end_time) {
                (Some(s), Some(e)) => (s, e),
                _ => continue,
            };
            let xmin = start as f64 / 1000.0;
            let xmax = end as f64 / 1000.0;
            if xmax > global_xmax {
                global_xmax = xmax;
            }
            intervals.push(crate::textgrid::Interval {
                xmin,
                xmax,
                text: ann.value.clone(),
            });
        }

        let tier_xmax = intervals.last().map(|i| i.xmax).unwrap_or(global_xmax);
        tiers.push(crate::textgrid::TextGridTier::IntervalTier {
            name: tier.id.clone(),
            xmin: 0.0,
            xmax: tier_xmax,
            intervals,
        });
    }

    let tg_file = crate::textgrid::TextGridFile {
        file_path: file.file_path.clone(),
        xmin: 0.0,
        xmax: global_xmax,
        tiers,
        raw_text: String::new(),
    };
    serialize_textgrid_file(&tg_file)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elan::reader::{Annotation, ElanFile, Tier};
    use crate::textgrid::{BaseTextGrid, TextGrid};

    fn make_ann(id: &str, start: i64, end: i64, value: &str) -> Annotation {
        Annotation {
            id: id.to_string(),
            start_time: Some(start),
            end_time: Some(end),
            value: value.to_string(),
            parent_id: None,
        }
    }

    fn make_tier(id: &str, annotations: Vec<Annotation>) -> Tier {
        Tier {
            id: id.to_string(),
            participant: String::new(),
            annotator: String::new(),
            linguistic_type_ref: "default-lt".to_string(),
            parent_id: None,
            child_ids: None,
            annotations,
        }
    }

    #[test]
    fn test_basic() {
        let file = ElanFile {
            file_path: "test.eaf".to_string(),
            tiers: vec![make_tier(
                "CHI",
                vec![make_ann("a1", 0, 1500, "hello world")],
            )],
            raw_xml: String::new(),
        };
        let tg_str = elan_file_to_textgrid_str(&file);

        assert!(tg_str.contains("\"IntervalTier\""));
        assert!(tg_str.contains("\"CHI\""));
        assert!(tg_str.contains("\"hello world\""));
        // 1500ms = 1.5s
        assert!(tg_str.contains("1.5"));
    }

    #[test]
    fn test_round_trip() {
        let file = ElanFile {
            file_path: "test.eaf".to_string(),
            tiers: vec![make_tier(
                "CHI",
                vec![
                    make_ann("a1", 0, 1500, "hello"),
                    make_ann("a2", 2000, 3500, "world"),
                ],
            )],
            raw_xml: String::new(),
        };
        let tg_str = elan_file_to_textgrid_str(&file);

        // Parse back through TextGrid.
        let tg = TextGrid::from_strs(vec![tg_str], Some(vec!["test.TextGrid".to_string()]), false)
            .unwrap();
        let tiers = tg.tiers_flat();
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].name(), "CHI");
        match &tiers[0] {
            crate::textgrid::TextGridTier::IntervalTier { intervals, .. } => {
                assert_eq!(intervals.len(), 2);
                assert_eq!(intervals[0].text, "hello");
                assert_eq!(intervals[0].xmin, 0.0);
                assert_eq!(intervals[0].xmax, 1.5);
                assert_eq!(intervals[1].text, "world");
            }
            _ => panic!("Expected IntervalTier"),
        }
    }

    #[test]
    fn test_skips_no_time() {
        let ann = Annotation {
            id: "a1".to_string(),
            start_time: None,
            end_time: None,
            value: "no times".to_string(),
            parent_id: None,
        };
        let file = ElanFile {
            file_path: "test.eaf".to_string(),
            tiers: vec![make_tier("CHI", vec![ann])],
            raw_xml: String::new(),
        };
        let tg_str = elan_file_to_textgrid_str(&file);

        assert!(tg_str.contains("intervals: size = 0"));
    }
}
