//! Convert CHAT data to TextGrid (Praat) format.

use super::reader::ChatFile;
use crate::textgrid::serialize_textgrid_file;

/// Strip CHAT bullet markers (`\x15start_end\x15`) from text.
fn strip_bullet_markers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_bullet = false;
    for ch in text.chars() {
        if ch == '\x15' {
            in_bullet = !in_bullet;
        } else if !in_bullet {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

/// Convert a single [`ChatFile`] to a TextGrid format string.
///
/// If `participants` is `Some`, only those participant codes are included.
/// If `None`, all participants are included.
/// Utterances without time marks are skipped.
/// Each participant becomes an IntervalTier.
pub(crate) fn chat_file_to_textgrid_str(
    file: &ChatFile,
    participants: Option<&[String]>,
) -> String {
    // Collect utterances per participant.
    let mut participant_intervals: std::collections::HashMap<
        String,
        Vec<crate::textgrid::Interval>,
    > = std::collections::HashMap::new();
    let mut participant_order: Vec<String> = Vec::new();
    let mut global_xmax: f64 = 0.0;

    for utt in file.real_utterances() {
        let participant = match &utt.participant {
            Some(p) => p,
            None => continue,
        };

        if let Some(codes) = participants
            && !codes.iter().any(|c| c == participant)
        {
            continue;
        }

        let (start, end) = match utt.time_marks {
            Some((s, e)) => (s, e),
            None => continue,
        };

        let text = utt
            .tiers
            .as_ref()
            .and_then(|t| t.get(participant))
            .cloned()
            .unwrap_or_default();
        let text = strip_bullet_markers(&text);

        let xmin = start as f64 / 1000.0;
        let xmax = end as f64 / 1000.0;
        if xmax > global_xmax {
            global_xmax = xmax;
        }

        if !participant_order.contains(participant) {
            participant_order.push(participant.clone());
        }

        participant_intervals
            .entry(participant.clone())
            .or_default()
            .push(crate::textgrid::Interval { xmin, xmax, text });
    }

    // Build tiers.
    let tiers: Vec<crate::textgrid::TextGridTier> = participant_order
        .iter()
        .map(|p| {
            let intervals = participant_intervals.remove(p).unwrap_or_default();
            let tier_xmax = intervals.last().map(|i| i.xmax).unwrap_or(global_xmax);
            crate::textgrid::TextGridTier::IntervalTier {
                name: p.clone(),
                xmin: 0.0,
                xmax: tier_xmax,
                intervals,
            }
        })
        .collect();

    let tg_file = crate::textgrid::TextGridFile {
        file_path: file.file_path.to_string(),
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
    use crate::chat::header::Headers;
    use crate::chat::reader::ChatFile;
    use crate::chat::utterance::Utterance;
    use crate::textgrid::{BaseTextGrid, TextGrid};
    use std::collections::HashMap;

    fn make_utterance(
        participant: &str,
        main_text: &str,
        time_marks: Option<(i64, i64)>,
    ) -> Utterance {
        let mut tiers = HashMap::new();
        tiers.insert(participant.to_string(), main_text.to_string());
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

    fn make_chat_file(events: Vec<Utterance>) -> ChatFile {
        ChatFile::new("test.cha".to_string(), Headers::default(), events, vec![])
    }

    #[test]
    fn test_single_participant() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "hello world .", Some((0, 1500))),
            make_utterance("CHI", "goodbye .", Some((2000, 3500))),
        ]);
        let tg_str = chat_file_to_textgrid_str(&file, None);

        assert!(tg_str.contains("\"IntervalTier\""));
        assert!(tg_str.contains("\"CHI\""));
        assert!(tg_str.contains("\"hello world .\""));
        assert!(tg_str.contains("\"goodbye .\""));
    }

    #[test]
    fn test_round_trip() {
        let file = make_chat_file(vec![make_utterance(
            "CHI",
            "hello world . \x150_1500\x15",
            Some((0, 1500)),
        )]);
        let tg_str = chat_file_to_textgrid_str(&file, None);

        let tg = TextGrid::from_strs(vec![tg_str], Some(vec!["test.TextGrid".to_string()]), false)
            .unwrap();
        assert_eq!(tg.tiers_flat().len(), 1);
        assert_eq!(tg.tiers_flat()[0].name(), "CHI");
        match &tg.tiers_flat()[0] {
            crate::textgrid::TextGridTier::IntervalTier { intervals, .. } => {
                assert_eq!(intervals.len(), 1);
                assert_eq!(intervals[0].text, "hello world .");
                assert_eq!(intervals[0].xmin, 0.0);
                assert_eq!(intervals[0].xmax, 1.5);
            }
            _ => panic!("Expected IntervalTier"),
        }
    }

    #[test]
    fn test_skips_no_time_marks() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "hello .", Some((0, 1500))),
            make_utterance("CHI", "no times .", None),
        ]);
        let tg_str = chat_file_to_textgrid_str(&file, None);

        assert!(tg_str.contains("\"hello .\""));
        assert!(!tg_str.contains("no times"));
    }

    #[test]
    fn test_filter_participants() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "more cookie .", Some((0, 2000))),
            make_utterance("MOT", "want more ?", Some((2500, 4000))),
        ]);
        let participants = vec!["CHI".to_string()];
        let tg_str = chat_file_to_textgrid_str(&file, Some(&participants));

        assert!(tg_str.contains("\"CHI\""));
        assert!(!tg_str.contains("\"MOT\""));
    }
}
