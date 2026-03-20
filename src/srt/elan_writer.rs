//! Convert SRT data to ELAN (.eaf) XML format.

use super::reader::SrtFile;

/// Escape text for use in XML content and attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Convert a single [`SrtFile`] to an EAF XML string.
///
/// Creates a single alignable tier named `"SPK"` with one annotation per
/// subtitle block.
pub(crate) fn srt_file_to_eaf_xml(file: &SrtFile) -> String {
    let mut xml = String::with_capacity(4096);

    // XML declaration and root element.
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<ANNOTATION_DOCUMENT AUTHOR=\"\" FORMAT=\"3.0\" VERSION=\"3.0\"");
    xml.push_str(" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"");
    xml.push_str(" xsi:noNamespaceSchemaLocation=\"http://www.mpi.nl/tools/elan/EAFv3.0.xsd\">\n");

    // HEADER.
    xml.push_str("    <HEADER MEDIA_FILE=\"\" TIME_UNITS=\"milliseconds\"/>\n");

    // TIME_ORDER.
    xml.push_str("    <TIME_ORDER>\n");
    for (i, block) in file.blocks.iter().enumerate() {
        let ts1_id = format!("ts{}", i * 2 + 1);
        let ts2_id = format!("ts{}", i * 2 + 2);
        xml.push_str(&format!(
            "        <TIME_SLOT TIME_SLOT_ID=\"{ts1_id}\" TIME_VALUE=\"{}\"/>\n",
            block.start_ms,
        ));
        xml.push_str(&format!(
            "        <TIME_SLOT TIME_SLOT_ID=\"{ts2_id}\" TIME_VALUE=\"{}\"/>\n",
            block.end_ms,
        ));
    }
    xml.push_str("    </TIME_ORDER>\n");

    // Single main tier.
    xml.push_str(
        "    <TIER TIER_ID=\"SPK\" LINGUISTIC_TYPE_REF=\"default-lt\" PARTICIPANT=\"Speaker\">\n",
    );
    for (i, block) in file.blocks.iter().enumerate() {
        let ann_id = format!("a{}", i + 1);
        let ts1_id = format!("ts{}", i * 2 + 1);
        let ts2_id = format!("ts{}", i * 2 + 2);
        // Join multiline subtitle text with space for the annotation value.
        let text = block.text.replace('\n', " ");
        xml.push_str("        <ANNOTATION>\n");
        xml.push_str(&format!(
            "            <ALIGNABLE_ANNOTATION ANNOTATION_ID=\"{ann_id}\" \
             TIME_SLOT_REF1=\"{ts1_id}\" TIME_SLOT_REF2=\"{ts2_id}\">\n"
        ));
        xml.push_str(&format!(
            "                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>\n",
            xml_escape(&text),
        ));
        xml.push_str("            </ALIGNABLE_ANNOTATION>\n");
        xml.push_str("        </ANNOTATION>\n");
    }
    xml.push_str("    </TIER>\n");

    // LINGUISTIC_TYPE definition.
    xml.push_str(
        "    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID=\"default-lt\" \
         TIME_ALIGNABLE=\"true\" GRAPHIC_REFERENCES=\"false\"/>\n",
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
    use crate::srt::reader::{SrtBlock, SrtFile};

    fn make_srt_file(blocks: Vec<SrtBlock>) -> SrtFile {
        SrtFile {
            file_path: "test.srt".to_string(),
            blocks,
        }
    }

    #[test]
    fn test_basic_conversion() {
        let file = make_srt_file(vec![SrtBlock {
            index: 1,
            text: "Hello world.".to_string(),
            start_ms: 0,
            end_ms: 1500,
        }]);
        let xml = srt_file_to_eaf_xml(&file);

        assert!(xml.contains("ANNOTATION_DOCUMENT"));
        assert!(xml.contains("TIER_ID=\"SPK\""));
        assert!(xml.contains("PARTICIPANT=\"Speaker\""));
        assert!(xml.contains("TIME_VALUE=\"0\""));
        assert!(xml.contains("TIME_VALUE=\"1500\""));
        assert!(xml.contains("<ANNOTATION_VALUE>Hello world.</ANNOTATION_VALUE>"));
    }

    #[test]
    fn test_multiline_text() {
        let file = make_srt_file(vec![SrtBlock {
            index: 1,
            text: "Line one\nLine two".to_string(),
            start_ms: 0,
            end_ms: 2000,
        }]);
        let xml = srt_file_to_eaf_xml(&file);

        // Multiline joined with space.
        assert!(xml.contains("<ANNOTATION_VALUE>Line one Line two</ANNOTATION_VALUE>"));
    }

    #[test]
    fn test_xml_escaping() {
        let file = make_srt_file(vec![SrtBlock {
            index: 1,
            text: "he said \"hi\" & <bye>".to_string(),
            start_ms: 0,
            end_ms: 1000,
        }]);
        let xml = srt_file_to_eaf_xml(&file);

        assert!(xml.contains("he said &quot;hi&quot; &amp; &lt;bye&gt;"));
    }

    #[test]
    fn test_multiple_blocks() {
        let file = make_srt_file(vec![
            SrtBlock {
                index: 1,
                text: "First.".to_string(),
                start_ms: 0,
                end_ms: 1000,
            },
            SrtBlock {
                index: 2,
                text: "Second.".to_string(),
                start_ms: 1500,
                end_ms: 3000,
            },
        ]);
        let xml = srt_file_to_eaf_xml(&file);

        assert!(xml.contains("ANNOTATION_ID=\"a1\""));
        assert!(xml.contains("ANNOTATION_ID=\"a2\""));
        assert!(xml.contains("TIME_SLOT_ID=\"ts1\" TIME_VALUE=\"0\""));
        assert!(xml.contains("TIME_SLOT_ID=\"ts2\" TIME_VALUE=\"1000\""));
        assert!(xml.contains("TIME_SLOT_ID=\"ts3\" TIME_VALUE=\"1500\""));
        assert!(xml.contains("TIME_SLOT_ID=\"ts4\" TIME_VALUE=\"3000\""));
    }

    #[test]
    fn test_empty_file() {
        let file = make_srt_file(vec![]);
        let xml = srt_file_to_eaf_xml(&file);

        assert!(xml.contains("ANNOTATION_DOCUMENT"));
        assert!(xml.contains("TIER_ID=\"SPK\""));
        // No annotations.
        assert!(!xml.contains("ANNOTATION_ID"));
    }

    #[test]
    fn test_round_trip_via_elan_parser() {
        let file = make_srt_file(vec![
            SrtBlock {
                index: 1,
                text: "Hello world.".to_string(),
                start_ms: 0,
                end_ms: 1500,
            },
            SrtBlock {
                index: 2,
                text: "Goodbye.".to_string(),
                start_ms: 2000,
                end_ms: 3500,
            },
        ]);
        let xml = srt_file_to_eaf_xml(&file);

        // Parse with ELAN parser.
        let elan_file = crate::elan::parse_eaf_str(&xml, "test.eaf".to_string())
            .expect("Generated EAF XML should be parseable");

        assert_eq!(elan_file.tiers.len(), 1);
        let tier = &elan_file.tiers[0];
        assert_eq!(tier.id, "SPK");
        assert_eq!(tier.annotations.len(), 2);
        assert_eq!(tier.annotations[0].value, "Hello world.");
        assert_eq!(tier.annotations[0].start_time, Some(0));
        assert_eq!(tier.annotations[0].end_time, Some(1500));
        assert_eq!(tier.annotations[1].value, "Goodbye.");
        assert_eq!(tier.annotations[1].start_time, Some(2000));
        assert_eq!(tier.annotations[1].end_time, Some(3500));
    }
}
