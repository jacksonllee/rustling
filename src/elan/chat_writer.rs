//! Convert ELAN (.eaf) data to CHAT format.

use super::reader::ElanFile;
use std::collections::HashMap;

/// A single utterance collected from ELAN annotations for CHAT output.
struct ChatUtterance {
    participant: String,
    main_text: String,
    start_time: Option<i64>,
    end_time: Option<i64>,
    /// `(dep_tier_name_with_%, dep_text)` in encounter order.
    dep_tiers: Vec<(String, String)>,
}

/// Convert a single [`ElanFile`] to a CHAT format string.
///
/// If `participants` is `Some`, only those tier IDs are treated as main tiers.
/// If `None`, auto-detect: parent tiers (no `parent_id`) with a 3-character ID.
pub(crate) fn elan_file_to_chat_str(file: &ElanFile, participants: Option<&[String]>) -> String {
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

    let main_tier_set: std::collections::HashSet<&str> = main_tier_ids.iter().copied().collect();

    // 2. Build participant info: code -> display name.
    let participant_names: HashMap<&str, &str> = file
        .tiers
        .iter()
        .filter(|t| main_tier_set.contains(t.id.as_str()))
        .filter(|t| !t.participant.is_empty())
        .map(|t| (t.id.as_str(), t.participant.as_str()))
        .collect();

    // 3. Build child tier map: for each main tier, collect child tiers
    //    whose ID matches `{name}@{code}`.
    //    Key: (main_tier_id, annotation_id) -> Vec<(dep_tier_name_with_%, dep_text)>
    //
    //    First, build: child_tier_id -> (dep_name_with_%, &Tier)
    let mut child_tier_info: HashMap<&str, (&str, String)> = HashMap::new();
    for tier in &file.tiers {
        if let Some(ref parent_id) = tier.parent_id
            && main_tier_set.contains(parent_id.as_str())
        {
            // Check if tier.id matches `{name}@{code}`.
            if let Some(at_pos) = tier.id.rfind('@') {
                let name = &tier.id[..at_pos];
                let code = &tier.id[at_pos + 1..];
                if code == parent_id {
                    let dep_name = format!("%{name}");
                    child_tier_info.insert(tier.id.as_str(), (parent_id.as_str(), dep_name));
                }
            }
        }
    }

    // Build annotation-level child lookup:
    // parent_annotation_id -> Vec<(dep_tier_name_with_%, dep_text)>
    let mut child_annotations: HashMap<&str, Vec<(String, String)>> = HashMap::new();
    for tier in &file.tiers {
        if let Some((_, dep_name)) = child_tier_info.get(tier.id.as_str()) {
            for ann in &tier.annotations {
                if let Some(ref parent_ann_id) = ann.parent_id {
                    child_annotations
                        .entry(parent_ann_id.as_str())
                        .or_default()
                        .push((dep_name.clone(), ann.value.clone()));
                }
            }
        }
    }

    // 4. Collect all main-tier annotations as ChatUtterances.
    let mut utterances: Vec<ChatUtterance> = Vec::new();
    for tier in &file.tiers {
        if !main_tier_set.contains(tier.id.as_str()) {
            continue;
        }
        for ann in &tier.annotations {
            let dep_tiers = child_annotations
                .remove(ann.id.as_str())
                .unwrap_or_default();
            utterances.push(ChatUtterance {
                participant: tier.id.clone(),
                main_text: ann.value.clone(),
                start_time: ann.start_time,
                end_time: ann.end_time,
                dep_tiers,
            });
        }
    }

    // 5. Sort by (start_time, end_time). None sorts before Some.
    utterances.sort_by(|a, b| {
        a.start_time
            .cmp(&b.start_time)
            .then_with(|| a.end_time.cmp(&b.end_time))
    });

    // 6. Generate CHAT output.
    let mut output = String::with_capacity(4096);
    output.push_str("@UTF8\n");
    output.push_str("@Begin\n");

    // @Participants line.
    if !main_tier_ids.is_empty() {
        output.push_str("@Participants:\t");
        let parts: Vec<String> = main_tier_ids
            .iter()
            .map(|code| {
                let name = participant_names.get(code).copied().unwrap_or(code);
                format!("{code} {name}")
            })
            .collect();
        output.push_str(&parts.join(", "));
        output.push('\n');
    }

    // Utterances.
    for utt in &utterances {
        // Main tier line.
        output.push_str(&format!("*{}:\t{}", utt.participant, utt.main_text));
        if let (Some(start), Some(end)) = (utt.start_time, utt.end_time) {
            output.push_str(&format!(" \x15{start}_{end}\x15"));
        }
        output.push('\n');

        // Dependent tier lines.
        for (dep_name, dep_text) in &utt.dep_tiers {
            output.push_str(&format!("{dep_name}:\t{dep_text}\n"));
        }
    }

    output.push_str("@End\n");
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::BaseChat;
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

    fn make_ref_ann(id: &str, parent_id: &str, value: &str) -> Annotation {
        Annotation {
            id: id.to_string(),
            start_time: None,
            end_time: None,
            value: value.to_string(),
            parent_id: Some(parent_id.to_string()),
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

    fn make_dep_tier(id: &str, parent_ref: &str, annotations: Vec<Annotation>) -> Tier {
        Tier {
            id: id.to_string(),
            participant: String::new(),
            annotator: String::new(),
            linguistic_type_ref: "dependent-lt".to_string(),
            parent_id: Some(parent_ref.to_string()),
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
    fn test_basic_single_participant() {
        let file = make_elan_file(vec![make_main_tier(
            "CHI",
            "Target_Child",
            vec![make_alignable_ann("a1", 0, 1500, "hello world .")],
        )]);
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
        assert!(chat.contains("@Participants:\tCHI Target_Child"));
        assert!(chat.contains("*CHI:\thello world . \x150_1500\x15"));
    }

    #[test]
    fn test_multi_participant_with_dep_tiers() {
        let file = make_elan_file(vec![
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a1", 0, 2000, "more cookie .")],
            ),
            make_dep_tier(
                "mor@CHI",
                "CHI",
                vec![make_ref_ann("a2", "a1", "qn|more n|cookie .")],
            ),
            make_main_tier(
                "MOT",
                "Mother",
                vec![make_alignable_ann(
                    "a3",
                    2500,
                    5000,
                    "do you want more cookies ?",
                )],
            ),
            make_dep_tier(
                "mor@MOT",
                "MOT",
                vec![make_ref_ann(
                    "a4",
                    "a3",
                    "mod|do pro:per|you v|want qn|more n|cookie-PL ?",
                )],
            ),
        ]);
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("@Participants:\tCHI Target_Child, MOT Mother"));
        assert!(chat.contains("*CHI:\tmore cookie . \x150_2000\x15"));
        assert!(chat.contains("%mor:\tqn|more n|cookie ."));
        assert!(chat.contains("*MOT:\tdo you want more cookies ? \x152500_5000\x15"));
        assert!(chat.contains("%mor:\tmod|do pro:per|you v|want"));

        // CHI should come before MOT (sorted by start_time).
        let chi_pos = chat.find("*CHI:").unwrap();
        let mot_pos = chat.find("*MOT:").unwrap();
        assert!(chi_pos < mot_pos);
    }

    #[test]
    fn test_auto_detect_skips_non_3char_tiers() {
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
        // Auto-detect: only CHI (3 chars) should be picked up.
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("*CHI:\thi"));
        assert!(!chat.contains("Speaker1"));
        assert!(!chat.contains("hello"));
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
        // Explicitly request Speaker1.
        let participants = vec!["Speaker1".to_string()];
        let chat = elan_file_to_chat_str(&file, Some(&participants));

        assert!(chat.contains("*Speaker1:\thello"));
        assert!(chat.contains("@Participants:\tSpeaker1 Alice"));
        assert!(!chat.contains("*CHI:"));
    }

    #[test]
    fn test_no_time_marks() {
        let ann = Annotation {
            id: "a1".to_string(),
            start_time: None,
            end_time: None,
            value: "hello .".to_string(),
            parent_id: None,
        };
        let file = make_elan_file(vec![make_main_tier("CHI", "Target_Child", vec![ann])]);
        let chat = elan_file_to_chat_str(&file, None);

        // No bullet markers.
        assert!(chat.contains("*CHI:\thello ."));
        assert!(!chat.contains('\x15'));
    }

    #[test]
    fn test_child_tier_pattern_mismatch_skipped() {
        // A child tier "gloss" (not matching {name}@{code}) should be skipped.
        let file = make_elan_file(vec![
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a1", 0, 1000, "hello .")],
            ),
            Tier {
                id: "gloss".to_string(),
                participant: String::new(),
                annotator: String::new(),
                linguistic_type_ref: "dependent-lt".to_string(),
                parent_id: Some("CHI".to_string()),
                child_ids: None,
                annotations: vec![make_ref_ann("a2", "a1", "greeting")],
            },
        ]);
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("*CHI:\thello ."));
        // "gloss" doesn't match {name}@{code} pattern, so it's skipped.
        assert!(!chat.contains("gloss"));
        assert!(!chat.contains("greeting"));
    }

    #[test]
    fn test_empty_file() {
        let file = make_elan_file(vec![]);
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
        assert!(!chat.contains("@Participants"));
    }

    #[test]
    fn test_participant_name_falls_back_to_code() {
        // Empty participant name -> falls back to tier ID.
        let file = make_elan_file(vec![make_main_tier(
            "CHI",
            "",
            vec![make_alignable_ann("a1", 0, 500, "hi .")],
        )]);
        let chat = elan_file_to_chat_str(&file, None);

        assert!(chat.contains("@Participants:\tCHI CHI"));
    }

    #[test]
    fn test_round_trip_via_chat_parser() {
        let file = make_elan_file(vec![
            make_main_tier(
                "CHI",
                "Target_Child",
                vec![make_alignable_ann("a1", 0, 2000, "more cookie .")],
            ),
            make_dep_tier(
                "mor@CHI",
                "CHI",
                vec![make_ref_ann("a2", "a1", "qn|more n|cookie .")],
            ),
        ]);
        let chat_str = elan_file_to_chat_str(&file, None);

        // Parse the generated CHAT string.
        let (chat, _) =
            crate::chat::Chat::from_strs(vec![chat_str], None, false, Some("%mor"), Some("%gra"));
        let files = chat.files();
        assert_eq!(files.len(), 1);

        let utts: Vec<_> = files[0].real_utterances().collect();
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].participant.as_deref(), Some("CHI"));
        assert_eq!(utts[0].time_marks, Some((0, 2000)));

        // Check tiers.
        let tiers = utts[0].tiers.as_ref().unwrap();
        assert_eq!(tiers["CHI"], "more cookie . \x150_2000\x15");
        assert_eq!(tiers["%mor"], "qn|more n|cookie .");
    }
}
