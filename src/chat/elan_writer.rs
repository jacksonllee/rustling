//! Convert CHAT data to ELAN (.eaf) XML format.

use super::reader::ChatFile;
use std::collections::{HashMap, HashSet};

/// Escape text for use in XML content and attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Convert a single [`ChatFile`] to an EAF XML string.
pub(crate) fn chat_file_to_eaf_xml(file: &ChatFile) -> String {
    // Build participant code -> name lookup from headers.
    let participant_names: HashMap<&str, &str> = file
        .headers
        .participants
        .iter()
        .filter(|p| !p.name.is_empty())
        .map(|p| (p.code.as_str(), p.name.as_str()))
        .collect();

    // 1. Collect participants in encounter order.
    let mut participants: Vec<String> = Vec::new();
    let mut seen_participants: HashSet<String> = HashSet::new();

    // 2. Collect dependent tier names in encounter order.
    let mut dep_tier_names: Vec<String> = Vec::new();
    let mut seen_dep_tiers: HashSet<String> = HashSet::new();

    // 3. Build per-utterance data: (participant, main_text, time_marks, dep_tiers).
    //    We only consider real utterances (not changeable headers).
    struct UttData {
        participant: String,
        main_text: String,
        time_marks: Option<(i64, i64)>,
        dep_tiers: Vec<(String, String)>, // (tier_name_with_%, value)
    }

    let mut utt_data: Vec<UttData> = Vec::new();

    for utt in file.events.iter().filter(|u| u.changeable_header.is_none()) {
        let participant = match &utt.participant {
            Some(p) => p.clone(),
            None => continue,
        };

        if seen_participants.insert(participant.clone()) {
            participants.push(participant.clone());
        }

        let tiers = utt.tiers.as_ref();
        let main_text = tiers
            .and_then(|t| t.get(&participant))
            .cloned()
            .unwrap_or_default();

        let mut deps = Vec::new();
        if let Some(tiers) = tiers {
            for (key, value) in tiers {
                if key.starts_with('%') && !seen_dep_tiers.contains(key) {
                    seen_dep_tiers.insert(key.clone());
                    dep_tier_names.push(key.clone());
                }
                if key.starts_with('%') {
                    deps.push((key.clone(), value.clone()));
                }
            }
        }

        utt_data.push(UttData {
            participant,
            main_text,
            time_marks: utt.time_marks,
            dep_tiers: deps,
        });
    }

    // 4. Build time slots.
    let mut time_slots: Vec<(String, Option<i64>)> = Vec::new();
    let mut ts_counter: usize = 0;
    // Map: utterance index -> (ts_ref1, ts_ref2)
    let mut utt_ts_refs: Vec<(String, String)> = Vec::with_capacity(utt_data.len());

    for utt in &utt_data {
        ts_counter += 1;
        let ts1_id = format!("ts{ts_counter}");
        let ts1_val = utt.time_marks.map(|(start, _)| start);
        ts_counter += 1;
        let ts2_id = format!("ts{ts_counter}");
        let ts2_val = utt.time_marks.map(|(_, end)| end);
        time_slots.push((ts1_id.clone(), ts1_val));
        time_slots.push((ts2_id.clone(), ts2_val));
        utt_ts_refs.push((ts1_id, ts2_id));
    }

    // 5. Build annotations and tiers.
    let mut ann_counter: usize = 0;
    // Map: utterance index -> main annotation ID (for ref annotations)
    let mut utt_main_ann_ids: Vec<String> = vec![String::new(); utt_data.len()];

    let mut xml = String::with_capacity(4096);

    // XML declaration and root element.
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<ANNOTATION_DOCUMENT AUTHOR=\"\" FORMAT=\"3.0\" VERSION=\"3.0\"");
    xml.push_str(" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"");
    xml.push_str(" xsi:noNamespaceSchemaLocation=\"http://www.mpi.nl/tools/elan/EAFv3.0.xsd\">\n");

    // HEADER with optional MEDIA_DESCRIPTOR.
    xml.push_str("    <HEADER MEDIA_FILE=\"\" TIME_UNITS=\"milliseconds\"");
    if let Some(ref media) = file.headers.media_data {
        xml.push_str(">\n");
        let mime = match media.format.to_lowercase().as_str() {
            "audio" => "audio/x-wav",
            "video" => "video/mpeg",
            _ => "unknown",
        };
        xml.push_str(&format!(
            "        <MEDIA_DESCRIPTOR MEDIA_URL=\"file:///{}\" MIME_TYPE=\"{}\"/>\n",
            xml_escape(&media.filename),
            mime,
        ));
        xml.push_str("    </HEADER>\n");
    } else {
        xml.push_str("/>\n");
    }

    // TIME_ORDER.
    xml.push_str("    <TIME_ORDER>\n");
    for (id, value) in &time_slots {
        match value {
            Some(v) => {
                xml.push_str(&format!(
                    "        <TIME_SLOT TIME_SLOT_ID=\"{id}\" TIME_VALUE=\"{v}\"/>\n"
                ));
            }
            None => {
                xml.push_str(&format!("        <TIME_SLOT TIME_SLOT_ID=\"{id}\"/>\n"));
            }
        }
    }
    xml.push_str("    </TIME_ORDER>\n");

    // Main tiers (one per participant, alignable).
    for participant in &participants {
        let display_name = participant_names
            .get(participant.as_str())
            .copied()
            .unwrap_or(participant.as_str());
        xml.push_str(&format!(
            "    <TIER TIER_ID=\"{}\" LINGUISTIC_TYPE_REF=\"default-lt\" PARTICIPANT=\"{}\">\n",
            xml_escape(participant),
            xml_escape(display_name),
        ));

        for (i, utt) in utt_data.iter().enumerate() {
            if utt.participant != *participant {
                continue;
            }
            ann_counter += 1;
            let ann_id = format!("a{ann_counter}");
            utt_main_ann_ids[i] = ann_id.clone();
            let (ts1, ts2) = &utt_ts_refs[i];
            xml.push_str("        <ANNOTATION>\n");
            xml.push_str(&format!(
                "            <ALIGNABLE_ANNOTATION ANNOTATION_ID=\"{ann_id}\" \
                 TIME_SLOT_REF1=\"{ts1}\" TIME_SLOT_REF2=\"{ts2}\">\n"
            ));
            xml.push_str(&format!(
                "                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>\n",
                xml_escape(&utt.main_text),
            ));
            xml.push_str("            </ALIGNABLE_ANNOTATION>\n");
            xml.push_str("        </ANNOTATION>\n");
        }

        xml.push_str("    </TIER>\n");

        // Dependent tiers for this participant.
        for dep_tier in &dep_tier_names {
            let tier_name = dep_tier.strip_prefix('%').unwrap_or(dep_tier);
            let tier_id = format!("{tier_name}@{participant}");

            // Only emit the tier if at least one utterance for this participant has it.
            let has_any = utt_data.iter().any(|u| {
                u.participant == *participant && u.dep_tiers.iter().any(|(k, _)| k == dep_tier)
            });
            if !has_any {
                continue;
            }

            xml.push_str(&format!(
                "    <TIER TIER_ID=\"{}\" LINGUISTIC_TYPE_REF=\"dependent-lt\" \
                 PARENT_REF=\"{}\" PARTICIPANT=\"{}\">\n",
                xml_escape(&tier_id),
                xml_escape(participant),
                xml_escape(display_name),
            ));

            for (i, utt) in utt_data.iter().enumerate() {
                if utt.participant != *participant {
                    continue;
                }
                if let Some((_, value)) = utt.dep_tiers.iter().find(|(k, _)| k == dep_tier) {
                    ann_counter += 1;
                    let ann_id = format!("a{ann_counter}");
                    let parent_ann_id = &utt_main_ann_ids[i];
                    xml.push_str("        <ANNOTATION>\n");
                    xml.push_str(&format!(
                        "            <REF_ANNOTATION ANNOTATION_ID=\"{ann_id}\" \
                         ANNOTATION_REF=\"{parent_ann_id}\">\n"
                    ));
                    xml.push_str(&format!(
                        "                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>\n",
                        xml_escape(value),
                    ));
                    xml.push_str("            </REF_ANNOTATION>\n");
                    xml.push_str("        </ANNOTATION>\n");
                }
            }

            xml.push_str("    </TIER>\n");
        }
    }

    // LINGUISTIC_TYPE definitions.
    xml.push_str(
        "    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID=\"default-lt\" \
         TIME_ALIGNABLE=\"true\" GRAPHIC_REFERENCES=\"false\"/>\n",
    );
    xml.push_str(
        "    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID=\"dependent-lt\" \
         TIME_ALIGNABLE=\"false\" GRAPHIC_REFERENCES=\"false\" \
         CONSTRAINTS=\"Symbolic_Association\"/>\n",
    );

    // CONSTRAINT definition.
    xml.push_str(
        "    <CONSTRAINT DESCRIPTION=\"1-1 association between a parent annotation and a child \
         annotation\" STEREOTYPE=\"Symbolic_Association\"/>\n",
    );

    xml.push_str("</ANNOTATION_DOCUMENT>\n");
    xml
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::header::{Headers, Participant};
    use crate::chat::reader::ChatFile;
    use crate::chat::utterance::Utterance;
    use std::collections::HashMap;

    fn make_utterance(
        participant: &str,
        main_text: &str,
        time_marks: Option<(i64, i64)>,
        dep_tiers: Vec<(&str, &str)>,
    ) -> Utterance {
        let mut tiers = HashMap::new();
        tiers.insert(participant.to_string(), main_text.to_string());
        for (k, v) in dep_tiers {
            tiers.insert(k.to_string(), v.to_string());
        }
        Utterance {
            participant: Some(participant.to_string()),
            tokens: None,
            time_marks,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        }
    }

    fn make_participant(code: &str, name: &str) -> Participant {
        Participant {
            code: code.to_string(),
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn make_chat_file(events: Vec<Utterance>) -> ChatFile {
        ChatFile::new(
            "test.cha".to_string(),
            Headers::default(),
            events,
            vec![], // raw_lines
        )
    }

    #[test]
    fn test_single_participant_basic() {
        let utt = make_utterance("CHI", "hello world .", Some((0, 1500)), vec![]);
        let file = make_chat_file(vec![utt]);
        let xml = chat_file_to_eaf_xml(&file);

        assert!(xml.contains("TIER_ID=\"CHI\""));
        assert!(xml.contains("LINGUISTIC_TYPE_REF=\"default-lt\""));
        assert!(xml.contains("TIME_VALUE=\"0\""));
        assert!(xml.contains("TIME_VALUE=\"1500\""));
        assert!(xml.contains("<ANNOTATION_VALUE>hello world .</ANNOTATION_VALUE>"));
    }

    #[test]
    fn test_multi_participant_with_dep_tiers() {
        let chi_utt = make_utterance(
            "CHI",
            "more cookie .",
            Some((0, 2000)),
            vec![
                ("%mor", "qn|more n|cookie ."),
                ("%gra", "1|2|QUANT 2|0|INCROOT 3|2|PUNCT"),
            ],
        );
        let mot_utt = make_utterance(
            "MOT",
            "do you want more cookies ?",
            Some((2500, 5000)),
            vec![
                ("%mor", "mod|do pro:per|you v|want qn|more n|cookie-PL ?"),
                (
                    "%gra",
                    "1|3|AUX 2|3|SUBJ 3|0|ROOT 4|5|QUANT 5|3|OBJ 6|3|PUNCT",
                ),
            ],
        );
        let file = make_chat_file(vec![chi_utt, mot_utt]);
        let xml = chat_file_to_eaf_xml(&file);

        // Main tiers.
        assert!(xml.contains("TIER_ID=\"CHI\""));
        assert!(xml.contains("TIER_ID=\"MOT\""));

        // Dependent tiers.
        assert!(xml.contains("TIER_ID=\"mor@CHI\""));
        assert!(xml.contains("TIER_ID=\"gra@CHI\""));
        assert!(xml.contains("TIER_ID=\"mor@MOT\""));
        assert!(xml.contains("TIER_ID=\"gra@MOT\""));

        // Parent references.
        assert!(xml.contains("PARENT_REF=\"CHI\""));
        assert!(xml.contains("PARENT_REF=\"MOT\""));

        // Dep tier values.
        assert!(xml.contains("qn|more n|cookie ."));
        assert!(xml.contains("mod|do pro:per|you v|want"));
    }

    #[test]
    fn test_no_time_marks() {
        let utt = make_utterance("CHI", "hello .", None, vec![]);
        let file = make_chat_file(vec![utt]);
        let xml = chat_file_to_eaf_xml(&file);

        // Time slots without TIME_VALUE.
        assert!(xml.contains("<TIME_SLOT TIME_SLOT_ID=\"ts1\"/>"));
        assert!(xml.contains("<TIME_SLOT TIME_SLOT_ID=\"ts2\"/>"));
    }

    #[test]
    fn test_media_header() {
        use crate::chat::header::Media;

        let headers = Headers {
            media_data: Some(Media {
                filename: "example".to_string(),
                format: "video".to_string(),
                status: None,
            }),
            ..Default::default()
        };

        let utt = make_utterance("CHI", "hi .", Some((0, 500)), vec![]);
        let file = ChatFile::new("test.cha".to_string(), headers, vec![utt], vec![]);
        let xml = chat_file_to_eaf_xml(&file);

        assert!(xml.contains("MEDIA_DESCRIPTOR"));
        assert!(xml.contains("MEDIA_URL=\"file:///example\""));
        assert!(xml.contains("MIME_TYPE=\"video/mpeg\""));
    }

    #[test]
    fn test_xml_escaping() {
        let utt = make_utterance("CHI", "he said \"hi\" & <bye>", Some((0, 1000)), vec![]);
        let file = make_chat_file(vec![utt]);
        let xml = chat_file_to_eaf_xml(&file);

        assert!(xml.contains("he said &quot;hi&quot; &amp; &lt;bye&gt;"));
    }

    #[test]
    fn test_empty_file() {
        let file = make_chat_file(vec![]);
        let xml = chat_file_to_eaf_xml(&file);

        assert!(xml.contains("ANNOTATION_DOCUMENT"));
        assert!(xml.contains("TIME_ORDER"));
        // No tiers.
        assert!(!xml.contains("TIER_ID"));
    }

    #[test]
    fn test_round_trip_via_elan_parser() {
        let chi_utt = make_utterance(
            "CHI",
            "more cookie .",
            Some((0, 2000)),
            vec![("%mor", "qn|more n|cookie .")],
        );
        let mot_utt = make_utterance(
            "MOT",
            "want more ?",
            Some((2500, 4000)),
            vec![("%mor", "v|want qn|more ?")],
        );
        let file = make_chat_file(vec![chi_utt, mot_utt]);
        let xml = chat_file_to_eaf_xml(&file);

        // Parse the generated XML with the ELAN parser.
        let elan_file = crate::elan::parse_eaf_str(&xml, "test.eaf".to_string())
            .expect("Generated EAF XML should be parseable");

        // Main tiers: CHI, MOT.
        // Dependent tiers: mor@CHI, mor@MOT.
        assert_eq!(elan_file.tiers.len(), 4);

        let chi_tier = elan_file.tiers.iter().find(|t| t.id == "CHI").unwrap();
        assert_eq!(chi_tier.annotations.len(), 1);
        assert_eq!(chi_tier.annotations[0].value, "more cookie .");
        assert_eq!(chi_tier.annotations[0].start_time, Some(0));
        assert_eq!(chi_tier.annotations[0].end_time, Some(2000));
        assert!(chi_tier.parent_id.is_none());

        let mor_chi = elan_file.tiers.iter().find(|t| t.id == "mor@CHI").unwrap();
        assert_eq!(mor_chi.annotations.len(), 1);
        assert_eq!(mor_chi.annotations[0].value, "qn|more n|cookie .");
        assert_eq!(mor_chi.parent_id.as_deref(), Some("CHI"));
        // Ref annotation inherits parent's time.
        assert_eq!(mor_chi.annotations[0].start_time, Some(0));
        assert_eq!(mor_chi.annotations[0].end_time, Some(2000));

        let mot_tier = elan_file.tiers.iter().find(|t| t.id == "MOT").unwrap();
        assert_eq!(mot_tier.annotations.len(), 1);
        assert_eq!(mot_tier.annotations[0].value, "want more ?");
    }

    #[test]
    fn test_participant_names_from_headers() {
        let headers = Headers {
            participants: vec![
                make_participant("CHI", "Target_Child"),
                make_participant("MOT", "Mother"),
            ],
            ..Default::default()
        };
        let utt = make_utterance("CHI", "hi .", Some((0, 500)), vec![]);
        let file = ChatFile::new("test.cha".to_string(), headers, vec![utt], vec![]);
        let xml = chat_file_to_eaf_xml(&file);

        // TIER_ID uses the code, PARTICIPANT uses the name.
        assert!(xml.contains("TIER_ID=\"CHI\""));
        assert!(xml.contains("PARTICIPANT=\"Target_Child\""));
        assert!(!xml.contains("PARTICIPANT=\"CHI\""));
    }

    #[test]
    fn test_participant_falls_back_to_code() {
        // No headers -> PARTICIPANT falls back to the code.
        let utt = make_utterance("CHI", "hi .", Some((0, 500)), vec![]);
        let file = make_chat_file(vec![utt]);
        let xml = chat_file_to_eaf_xml(&file);

        assert!(xml.contains("PARTICIPANT=\"CHI\""));
    }
}
