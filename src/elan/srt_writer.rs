//! Convert ELAN (.eaf) data to SRT (SubRip Subtitle) format.

use super::reader::ElanFile;
use crate::srt::format_srt_time;

/// A single subtitle entry collected from ELAN annotations.
struct SrtEntry {
    text: String,
    start_ms: i64,
    end_ms: i64,
}

/// Convert a single [`ElanFile`] to an SRT format string.
///
/// If `participants` is `Some`, only those tier IDs are treated as main tiers.
/// If `None`, auto-detect: parent tiers (no `parent_id`) with a 3-character ID.
/// Annotations without time marks are skipped.
/// When multiple tiers are selected, the subtitle text is prefixed
/// with the tier ID (e.g., `"CHI: more cookie ."`).
pub(crate) fn elan_file_to_srt_str(file: &ElanFile, participants: Option<&[String]>) -> String {
    // 1. Identify main tier IDs.
    let main_tier_ids: Vec<&str> = file
        .tiers
        .iter()
        .filter(|t| match participants {
            Some(codes) => codes.iter().any(|c| c == &t.id),
            None => t.parent_id.is_none() && t.id.len() == 3,
        })
        .map(|t| t.id.as_str())
        .collect();

    let multi = main_tier_ids.len() > 1;

    // 2. Collect entries from selected tiers.
    let mut entries: Vec<SrtEntry> = Vec::new();
    for tier in &file.tiers {
        if !main_tier_ids.contains(&tier.id.as_str()) {
            continue;
        }
        for ann in &tier.annotations {
            let (start, end) = match (ann.start_time, ann.end_time) {
                (Some(s), Some(e)) => (s, e),
                _ => continue,
            };
            let text = if multi {
                format!("{}: {}", tier.id, ann.value)
            } else {
                ann.value.clone()
            };
            entries.push(SrtEntry {
                text,
                start_ms: start,
                end_ms: end,
            });
        }
    }

    // 3. Sort by start time, then end time.
    entries.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    // 4. Generate SRT output.
    let mut output = String::with_capacity(4096);
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format!("{}\n", i + 1));
        output.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(entry.start_ms),
            format_srt_time(entry.end_ms),
        ));
        output.push_str(&entry.text);
        output.push('\n');
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elan::reader::{Annotation, ElanFile, Tier};

    fn make_alignable_ann(id: &str, start: i64, end: i64, value: &str) -> Annotation {
        Annotation {
            id: id.to_string(),
            start_time: Some(start),
            end_time: Some(end),
            value: value.to_string(),
            parent_id: None,
        }
    }

    fn make_main_tier(id: &str, participant: &str, annotations: Vec<Annotation>) -> Tier {
        Tier {
            id: id.to_string(),
            participant: participant.to_string(),
            annotator: String::new(),
            linguistic_type_ref: "default-lt".to_string(),
            parent_id: None,
            child_ids: None,
            annotations,
        }
    }

    fn make_elan_file(tiers: Vec<Tier>) -> ElanFile {
        ElanFile {
            file_path: "test.eaf".to_string(),
            tiers,
            raw_xml: String::new(),
        }
    }

    #[test]
    fn test_single_tier() {
        let file = make_elan_file(vec![make_main_tier(
            "CHI",
            "Target_Child",
            vec![
                make_alignable_ann("a1", 0, 1500, "hello world ."),
                make_alignable_ann("a2", 2000, 3500, "goodbye ."),
            ],
        )]);
        let srt = elan_file_to_srt_str(&file, None);

        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhello world ."));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:03,500\ngoodbye ."));
        // No prefix for single tier.
        assert!(!srt.contains("CHI:"));
    }

    #[test]
    fn test_multi_tier() {
        let file = make_elan_file(vec![
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a1", 0, 2000, "more cookie .")],
            ),
            make_main_tier(
                "MOT",
                "Mother",
                vec![make_alignable_ann("a2", 2500, 4000, "want more ?")],
            ),
        ]);
        let srt = elan_file_to_srt_str(&file, None);

        assert!(srt.contains("CHI: more cookie ."));
        assert!(srt.contains("MOT: want more ?"));
    }

    #[test]
    fn test_explicit_participants() {
        let file = make_elan_file(vec![
            make_main_tier(
                "Speaker1",
                "Alice",
                vec![make_alignable_ann("a1", 0, 1000, "hello")],
            ),
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a2", 1000, 2000, "hi")],
            ),
        ]);
        let participants = vec!["Speaker1".to_string()];
        let srt = elan_file_to_srt_str(&file, Some(&participants));

        assert!(srt.contains("hello"));
        assert!(!srt.contains("hi"));
        // Single participant after filtering, no prefix.
        assert!(!srt.contains("Speaker1:"));
    }

    #[test]
    fn test_skip_no_time_marks() {
        let ann = Annotation {
            id: "a1".to_string(),
            start_time: None,
            end_time: None,
            value: "no times".to_string(),
            parent_id: None,
        };
        let file = make_elan_file(vec![make_main_tier("CHI", "Target_Child", vec![ann])]);
        let srt = elan_file_to_srt_str(&file, None);

        assert!(srt.is_empty());
    }

    #[test]
    fn test_sorted_by_time() {
        let file = make_elan_file(vec![make_main_tier(
            "CHI",
            "Target_Child",
            vec![
                make_alignable_ann("a1", 3000, 4000, "second ."),
                make_alignable_ann("a2", 0, 1000, "first ."),
            ],
        )]);
        let srt = elan_file_to_srt_str(&file, None);

        let first_pos = srt.find("first .").unwrap();
        let second_pos = srt.find("second .").unwrap();
        assert!(first_pos < second_pos);
    }

    #[test]
    fn test_empty_file() {
        let file = make_elan_file(vec![]);
        let srt = elan_file_to_srt_str(&file, None);
        assert!(srt.is_empty());
    }

    #[test]
    fn test_auto_detect_skips_non_3char() {
        let file = make_elan_file(vec![
            make_main_tier(
                "Speaker1",
                "Alice",
                vec![make_alignable_ann("a1", 0, 1000, "hello")],
            ),
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a2", 1000, 2000, "hi")],
            ),
        ]);
        let srt = elan_file_to_srt_str(&file, None);

        // Only CHI (3 chars) should be picked up.
        assert!(srt.contains("hi"));
        assert!(!srt.contains("hello"));
    }
}
