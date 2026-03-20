//! Convert TextGrid data to ELAN (.eaf) format.

use super::reader::{TextGridFile, TextGridTier};

/// Escape special XML characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Convert a single [`TextGridFile`] to an EAF XML string.
///
/// Only IntervalTiers are converted; TextTiers are skipped
/// (point annotations have no duration for ELAN alignable annotations).
/// Times are converted from seconds (f64) to milliseconds (i64).
pub(crate) fn textgrid_file_to_eaf_xml(file: &TextGridFile) -> String {
    let mut time_slots: Vec<(String, i64)> = Vec::new();
    let mut ts_counter = 1usize;

    // Collect interval tiers only.
    struct TierData {
        name: String,
        annotations: Vec<(String, String, String, String)>, // (ann_id, ts_ref1, ts_ref2, value)
    }
    let mut tier_data_list: Vec<TierData> = Vec::new();
    let mut ann_counter = 1usize;

    for tier in &file.tiers {
        if let TextGridTier::IntervalTier {
            name, intervals, ..
        } = tier
        {
            let mut annotations = Vec::new();
            for interval in intervals {
                if interval.text.is_empty() {
                    continue;
                }
                let ts_id1 = format!("ts{ts_counter}");
                ts_counter += 1;
                let ts_id2 = format!("ts{ts_counter}");
                ts_counter += 1;
                let ann_id = format!("a{ann_counter}");
                ann_counter += 1;

                let start_ms = (interval.xmin * 1000.0).round() as i64;
                let end_ms = (interval.xmax * 1000.0).round() as i64;

                time_slots.push((ts_id1.clone(), start_ms));
                time_slots.push((ts_id2.clone(), end_ms));
                annotations.push((ann_id, ts_id1, ts_id2, interval.text.clone()));
            }
            tier_data_list.push(TierData {
                name: name.clone(),
                annotations,
            });
        }
    }

    let mut xml = String::with_capacity(4096);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<ANNOTATION_DOCUMENT>\n");
    xml.push_str("    <HEADER MEDIA_FILE=\"\" TIME_UNITS=\"milliseconds\"/>\n");

    // TIME_ORDER
    xml.push_str("    <TIME_ORDER>\n");
    for (ts_id, ms) in &time_slots {
        xml.push_str(&format!(
            "        <TIME_SLOT TIME_SLOT_ID=\"{ts_id}\" TIME_VALUE=\"{ms}\"/>\n"
        ));
    }
    xml.push_str("    </TIME_ORDER>\n");

    // TIERs
    for td in &tier_data_list {
        xml.push_str(&format!(
            "    <TIER TIER_ID=\"{}\" PARTICIPANT=\"\" ANNOTATOR=\"\" LINGUISTIC_TYPE_REF=\"default-lt\">\n",
            xml_escape(&td.name)
        ));
        for (ann_id, ts1, ts2, value) in &td.annotations {
            xml.push_str("        <ANNOTATION>\n");
            xml.push_str(&format!(
                "            <ALIGNABLE_ANNOTATION ANNOTATION_ID=\"{ann_id}\" TIME_SLOT_REF1=\"{ts1}\" TIME_SLOT_REF2=\"{ts2}\">\n"
            ));
            xml.push_str(&format!(
                "                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>\n",
                xml_escape(value)
            ));
            xml.push_str("            </ALIGNABLE_ANNOTATION>\n");
            xml.push_str("        </ANNOTATION>\n");
        }
        xml.push_str("    </TIER>\n");
    }

    xml.push_str("</ANNOTATION_DOCUMENT>\n");
    xml
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elan::BaseElan;
    use crate::textgrid::reader::{Interval, TextGridFile};

    fn make_interval_tier(name: &str, intervals: Vec<Interval>) -> TextGridTier {
        let xmax = intervals.last().map(|i| i.xmax).unwrap_or(0.0);
        TextGridTier::IntervalTier {
            name: name.to_string(),
            xmin: 0.0,
            xmax,
            intervals,
        }
    }

    fn make_textgrid_file(tiers: Vec<TextGridTier>) -> TextGridFile {
        let xmax = tiers
            .iter()
            .map(|t| match t {
                TextGridTier::IntervalTier { xmax, .. } => *xmax,
                TextGridTier::TextTier { xmax, .. } => *xmax,
            })
            .fold(0.0f64, f64::max);
        TextGridFile {
            file_path: "test.TextGrid".to_string(),
            xmin: 0.0,
            xmax,
            tiers,
            raw_text: String::new(),
        }
    }

    #[test]
    fn test_single_tier() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "words",
            vec![
                Interval {
                    xmin: 0.0,
                    xmax: 1.5,
                    text: "hello".to_string(),
                },
                Interval {
                    xmin: 1.5,
                    xmax: 2.3,
                    text: "world".to_string(),
                },
            ],
        )]);
        let xml = textgrid_file_to_eaf_xml(&file);

        assert!(xml.contains("TIER_ID=\"words\""));
        assert!(xml.contains("TIME_VALUE=\"0\""));
        assert!(xml.contains("TIME_VALUE=\"1500\""));
        assert!(xml.contains("TIME_VALUE=\"2300\""));
        assert!(xml.contains("<ANNOTATION_VALUE>hello</ANNOTATION_VALUE>"));
        assert!(xml.contains("<ANNOTATION_VALUE>world</ANNOTATION_VALUE>"));
    }

    #[test]
    fn test_skips_texttier() {
        let file = make_textgrid_file(vec![TextGridTier::TextTier {
            name: "events".to_string(),
            xmin: 0.0,
            xmax: 1.0,
            points: vec![crate::textgrid::reader::Point {
                number: 0.5,
                mark: "click".to_string(),
            }],
        }]);
        let xml = textgrid_file_to_eaf_xml(&file);

        // No TIER should be generated for TextTier.
        assert!(!xml.contains("TIER_ID"));
    }

    #[test]
    fn test_skips_empty_text() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "words",
            vec![
                Interval {
                    xmin: 0.0,
                    xmax: 0.5,
                    text: String::new(),
                },
                Interval {
                    xmin: 0.5,
                    xmax: 1.0,
                    text: "hello".to_string(),
                },
            ],
        )]);
        let xml = textgrid_file_to_eaf_xml(&file);

        assert!(xml.contains("hello"));
        // Only one annotation, not two.
        assert_eq!(xml.matches("ALIGNABLE_ANNOTATION").count(), 2); // open + close
    }

    #[test]
    fn test_xml_escape() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "tier",
            vec![Interval {
                xmin: 0.0,
                xmax: 1.0,
                text: "a < b & c > d".to_string(),
            }],
        )]);
        let xml = textgrid_file_to_eaf_xml(&file);

        assert!(xml.contains("a &lt; b &amp; c &gt; d"));
    }

    #[test]
    fn test_round_trip_through_elan() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "CHI",
            vec![Interval {
                xmin: 0.0,
                xmax: 1.5,
                text: "hello world".to_string(),
            }],
        )]);
        let xml = textgrid_file_to_eaf_xml(&file);

        // Parse through ELAN.
        let elan =
            crate::elan::Elan::from_strs(vec![xml], Some(vec!["test.eaf".to_string()]), false)
                .unwrap();
        let tiers = elan.tiers_flat();
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].id, "CHI");
        assert_eq!(tiers[0].annotations.len(), 1);
        assert_eq!(tiers[0].annotations[0].value, "hello world");
        assert_eq!(tiers[0].annotations[0].start_time, Some(0));
        assert_eq!(tiers[0].annotations[0].end_time, Some(1500));
    }
}
