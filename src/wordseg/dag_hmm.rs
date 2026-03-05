//! DAG + HMM hybrid word segmenter (jieba-style).
//!
//! Layer 1: Dictionary-based DAG with backward dynamic programming.
//! Layer 2: HMM fallback (BMES tagger) for out-of-vocabulary spans.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::PathBuf;

use pyo3::prelude::*;

use crate::persistence::ModelError;
use crate::seq_feature::SeqFeatureTemplate;
use crate::trie::CountTrie;
use crate::wordseg::{BaseHiddenMarkovModelSegmenter, HiddenMarkovModelSegmenter};

// ---------------------------------------------------------------------------
// Binary section helpers
// ---------------------------------------------------------------------------

/// Write a length-prefixed section: [8 bytes u64 LE length][data].
fn write_section<W: Write>(writer: &mut W, data: &[u8]) -> Result<(), ModelError> {
    let len = data.len() as u64;
    writer
        .write_all(&len.to_le_bytes())
        .map_err(|e| ModelError::Io(format!("Failed to write section length: {e}")))?;
    writer
        .write_all(data)
        .map_err(|e| ModelError::Io(format!("Failed to write section data: {e}")))
}

/// Read a length-prefixed section written by [`write_section`].
fn read_section<R: Read>(reader: &mut R) -> Result<Vec<u8>, ModelError> {
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .map_err(|e| ModelError::ParseError(format!("Failed to read section length: {e}")))?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    reader
        .read_exact(&mut data)
        .map_err(|e| ModelError::ParseError(format!("Failed to read section data: {e}")))?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// DagHmmSegmenter (pure Rust)
// ---------------------------------------------------------------------------

/// A jieba-style DAG + HMM hybrid word segmenter.
///
/// Layer 1: Dictionary-based DAG with backward dynamic programming.
/// Layer 2: HMM fallback (BMES tagger) for out-of-vocabulary spans.
///
/// For Python, use [`PyDagHmmSegmenter`].
#[derive(Clone)]
pub struct DagHmmSegmenter {
    /// Prefix dictionary stored as a character trie.
    /// Terminal nodes carry the word frequency; interior nodes are prefixes.
    dict: CountTrie<char>,
    /// Sum of all word frequencies (for log-probability normalization).
    total: f64,
    /// HMM segmenter for OOV character spans.
    hmm: HiddenMarkovModelSegmenter,
}

// ---------------------------------------------------------------------------
// Core algorithm (pure Rust, no PyO3)
// ---------------------------------------------------------------------------

impl DagHmmSegmenter {
    /// Create a new DagHmmSegmenter.
    pub fn new(
        n_iter: Option<usize>,
        tolerance: Option<f64>,
        gamma: Option<f64>,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Self {
        Self {
            dict: CountTrie::new(),
            total: 0.0,
            hmm: HiddenMarkovModelSegmenter::new(n_iter, tolerance, gamma, random_seed, features),
        }
    }

    /// Build the DAG for a sentence.
    ///
    /// Returns a vec where `dag[i]` contains all valid word-end positions
    /// (inclusive) reachable from position `i`.
    #[allow(clippy::needless_range_loop)]
    fn get_dag(&self, chars: &[char]) -> Vec<Vec<usize>> {
        let n = chars.len();
        let mut dag: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            // Walk the trie one character at a time from position i.
            // get_node(empty iterator) returns the root node.
            let mut node = self.dict.get_node(std::iter::empty::<char>());
            for j in i..n {
                node = node.and_then(|nd| nd.get_child(&chars[j]));
                match node {
                    None => break,
                    Some(nd) => {
                        if nd.terminal().copied().unwrap_or(0) > 0 {
                            dag[i].push(j);
                        }
                    }
                }
            }
            // Single char is always reachable as a fallback.
            if dag[i].is_empty() {
                dag[i].push(i);
            }
        }
        dag
    }

    /// Backward DP to find the highest-probability path through the DAG.
    ///
    /// Returns `route` where `route[i] = (log_prob, next_position)`.
    fn calc_route(&self, chars: &[char], dag: &[Vec<usize>]) -> Vec<(f64, usize)> {
        let n = chars.len();
        let log_total = self.total.ln();
        let mut route: Vec<(f64, usize)> = vec![(0.0, 0); n + 1];

        for i in (0..n).rev() {
            let mut best = (f64::NEG_INFINITY, 0usize);
            for &j in &dag[i] {
                let word_freq = self.dict.get_count(chars[i..=j].iter().copied());
                let log_prob = if word_freq > 0 {
                    (word_freq as f64).ln() - log_total
                } else {
                    // Unknown single char: use log(1) - log(total) = -log(total).
                    -log_total
                };
                let candidate = log_prob + route[j + 1].0;
                if candidate > best.0 {
                    best = (candidate, j + 1);
                }
            }
            route[i] = best;
        }
        route
    }

    /// Segment a single sentence using DAG + HMM.
    fn segment_one(&self, sent_str: &str) -> Vec<String> {
        let chars: Vec<char> = sent_str.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }

        // If no dictionary, fall back entirely to HMM.
        if self.dict.is_empty() {
            let result =
                BaseHiddenMarkovModelSegmenter::predict(&self.hmm, vec![sent_str.to_string()]);
            return result.into_iter().next().unwrap_or_default();
        }

        let dag = self.get_dag(&chars);
        let route = self.calc_route(&chars, &dag);

        let mut words: Vec<String> = Vec::new();
        let mut oov_buf: Vec<char> = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let next = route[i].1;
            let word_len = next - i;

            if word_len == 1 {
                // Single character: might be OOV. Buffer it.
                oov_buf.push(chars[i]);
            } else {
                // Multi-char word from dictionary.
                // Flush any buffered single-char spans first.
                if !oov_buf.is_empty() {
                    Self::flush_oov_buf(&mut oov_buf, &self.dict, &self.hmm, &mut words);
                }
                let word: String = chars[i..next].iter().collect();
                words.push(word);
            }
            i = next;
        }

        // Flush remaining buffer.
        if !oov_buf.is_empty() {
            Self::flush_oov_buf(&mut oov_buf, &self.dict, &self.hmm, &mut words);
        }

        words
    }

    /// Flush a buffer of single-char-route characters using the jieba three-way rule:
    ///
    /// 1. Single char -> output directly (HMM would always give "S", skip the call).
    /// 2. Span is IN the dictionary -> DAG chose to traverse it char-by-char because
    ///    individual chars were more probable; respect that by splitting char-by-char.
    /// 3. True OOV span (not in dict) -> delegate to HMM.
    fn flush_oov_buf(
        oov_buf: &mut Vec<char>,
        dict: &CountTrie<char>,
        hmm: &HiddenMarkovModelSegmenter,
        words: &mut Vec<String>,
    ) {
        if oov_buf.is_empty() {
            return;
        }
        let span: String = oov_buf.iter().collect();
        oov_buf.clear();

        if span.chars().count() == 1 {
            // Single char: output directly, no HMM needed.
            words.push(span);
        } else if dict.get_count(span.chars()) > 0 {
            // Span is in the dictionary but DAG preferred single-char paths.
            // Respect the DAG: split into individual characters.
            words.extend(span.chars().map(|c| c.to_string()));
        } else {
            // True OOV: delegate to HMM.
            let hmm_result = BaseHiddenMarkovModelSegmenter::predict(hmm, vec![span]);
            if let Some(hmm_words) = hmm_result.into_iter().next() {
                words.extend(hmm_words);
            }
        }
    }

    /// Segment unsegmented sentence strings.
    pub fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        sent_strs.iter().map(|s| self.segment_one(s)).collect()
    }

    /// Train the segmenter from supervised segmented sentences.
    pub fn fit_segmented(&mut self, sents: Vec<Vec<String>>) {
        // 1. Count word frequencies into the trie (prefix nodes created automatically).
        let mut dict = CountTrie::new();
        let mut total: u64 = 0;
        for sent in &sents {
            for word in sent {
                dict.increment(word.chars());
                total += 1;
            }
        }
        self.dict = dict;
        self.total = total as f64;

        // 2. Train HMM on the same data.
        BaseHiddenMarkovModelSegmenter::fit_segmented(&mut self.hmm, sents);
    }

    /// Train the HMM component with unsupervised EM on unsegmented sentences.
    pub fn fit_unsegmented(&mut self, sent_strs: Vec<String>) {
        BaseHiddenMarkovModelSegmenter::fit_unsegmented(&mut self.hmm, sent_strs);
    }

    /// Save the model and metadata to a binary file.
    ///
    /// Format: magic header "PSEG" + 3 length-prefixed sections:
    /// 1. Dict (JSON), 2. HMM (FlatBuffers), 3. Metadata (JSON).
    pub fn save(&self, path: &str, metadata: &HashMap<String, String>) -> Result<(), ModelError> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| ModelError::Io(format!("Failed to create file: {e}")))?;

        // Magic header.
        file.write_all(b"PSEG")
            .map_err(|e| ModelError::Io(format!("Failed to write magic: {e}")))?;

        // Section 1: dict (serde_json).
        // all_counts() returns only terminal (real word) entries — no prefix-only nodes.
        let freq_map: BTreeMap<String, u64> = self
            .dict
            .all_counts()
            .into_iter()
            .map(|(chars, count)| (chars.into_iter().collect::<String>(), count))
            .collect();
        let dict_bytes = serde_json::to_vec(&freq_map)
            .map_err(|e| ModelError::Io(format!("Failed to serialize dict: {e}")))?;
        write_section(&mut file, &dict_bytes)?;

        // Section 2: HMM (FlatBuffers binary, zstd-compressed).
        let mut hmm_bytes = Vec::new();
        BaseHiddenMarkovModelSegmenter::save_to_writer(&self.hmm, &mut hmm_bytes)
            .map_err(|e| ModelError::Io(format!("Failed to serialize HMM: {e:?}")))?;
        let hmm_compressed = zstd::bulk::compress(&hmm_bytes, 19)
            .map_err(|e| ModelError::Io(format!("Failed to compress HMM: {e}")))?;
        write_section(&mut file, &hmm_compressed)?;

        // Section 3: metadata (serde_json).
        let meta_bytes = serde_json::to_vec(metadata)
            .map_err(|e| ModelError::Io(format!("Failed to serialize metadata: {e}")))?;
        write_section(&mut file, &meta_bytes)?;

        Ok(())
    }

    /// Load a model and metadata from a binary file.
    ///
    /// Returns the metadata `HashMap` stored in the file.
    pub fn load(&mut self, path: &str) -> Result<HashMap<String, String>, ModelError> {
        let mut file = std::fs::File::open(path)
            .map_err(|_| ModelError::FileNotFound(format!("Can't locate model {path}")))?;

        // Check magic header.
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| ModelError::ParseError(format!("Failed to read magic: {e}")))?;
        if &magic != b"PSEG" {
            return Err(ModelError::ParseError(format!(
                "Not a valid PSEG model file (bad magic: {:?})",
                magic
            )));
        }

        // Section 1: dict.
        let dict_bytes = read_section(&mut file)?;
        let word_counts: HashMap<String, u64> = serde_json::from_slice(&dict_bytes)
            .map_err(|e| ModelError::ParseError(format!("Failed to parse dict: {e}")))?;
        let mut dict = CountTrie::new();
        let mut total: u64 = 0;
        for (word, count) in word_counts {
            dict.insert_count(word.chars(), count);
            total += count;
        }
        self.dict = dict;
        self.total = total as f64;

        // Section 2: HMM (FlatBuffers binary, zstd-compressed).
        let hmm_compressed = read_section(&mut file)?;
        let hmm_bytes = zstd::bulk::decompress(&hmm_compressed, 256 * 1024 * 1024)
            .map_err(|e| ModelError::ParseError(format!("Failed to decompress HMM: {e}")))?;
        BaseHiddenMarkovModelSegmenter::load_from_reader(&mut self.hmm, hmm_bytes.as_slice())
            .map_err(|e| ModelError::ParseError(format!("Failed to load HMM: {e:?}")))?;

        // Section 3: metadata.
        let meta_bytes = read_section(&mut file)?;
        let metadata: HashMap<String, String> = serde_json::from_slice(&meta_bytes)
            .map_err(|e| ModelError::ParseError(format!("Failed to parse metadata: {e}")))?;

        Ok(metadata)
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-exposed DAG + HMM hybrid segmenter.
///
/// Python users see this as `DAGHMMSegmenter`.
#[pyclass(name = "DAGHMMSegmenter", from_py_object)]
#[derive(Clone)]
pub struct PyDagHmmSegmenter {
    pub inner: DagHmmSegmenter,
}

#[pymethods]
impl PyDagHmmSegmenter {
    /// Create a new DAGHMMSegmenter.
    #[new]
    #[pyo3(signature = (*, n_iter=None, tolerance=None, gamma=None, random_seed=None, features=None))]
    fn new(
        n_iter: Option<usize>,
        tolerance: Option<f64>,
        gamma: Option<f64>,
        random_seed: Option<u64>,
        features: Option<Vec<SeqFeatureTemplate>>,
    ) -> Self {
        Self {
            inner: DagHmmSegmenter::new(n_iter, tolerance, gamma, random_seed, features),
        }
    }

    fn fit_segmented(&mut self, sents: Vec<Vec<String>>) {
        self.inner.fit_segmented(sents);
    }

    fn fit_unsegmented(&mut self, sent_strs: Vec<String>) {
        self.inner.fit_unsegmented(sent_strs);
    }

    fn predict(&self, sent_strs: Vec<String>) -> Vec<Vec<String>> {
        self.inner.predict(sent_strs)
    }

    fn save(&self, path: PathBuf, metadata: HashMap<String, String>) -> PyResult<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Path is not valid UTF-8"))?;
        Ok(self.inner.save(path_str, &metadata)?)
    }

    fn load(&mut self, path: PathBuf) -> PyResult<HashMap<String, String>> {
        let path_str = path
            .to_str()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Path is not valid UTF-8"))?;
        Ok(self.inner.load(path_str)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn training_data() -> Vec<Vec<String>> {
        vec![
            vec!["你好".into(), "世界".into()],
            vec!["我".into(), "喜歡".into(), "你".into()],
            vec!["你好".into(), "我".into(), "喜歡".into(), "世界".into()],
        ]
    }

    #[test]
    fn test_get_dag() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.dict.insert_count("你好".chars(), 2);
        seg.dict.insert_count("你".chars(), 5);
        seg.dict.insert_count("好".chars(), 3);
        seg.total = 10.0;

        let chars: Vec<char> = "你好".chars().collect();
        let dag = seg.get_dag(&chars);
        // Position 0: "你" (freq=5) and "你好" (freq=2).
        assert!(dag[0].contains(&0)); // "你" ends at 0
        assert!(dag[0].contains(&1)); // "你好" ends at 1
        // Position 1: "好" (freq=3).
        assert!(dag[1].contains(&1)); // "好" ends at 1
    }

    #[test]
    fn test_calc_route_prefers_longer_frequent_word() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.dict.insert_count("你好".chars(), 10);
        seg.dict.insert_count("你".chars(), 1);
        seg.dict.insert_count("好".chars(), 1);
        seg.total = 12.0;

        let chars: Vec<char> = "你好".chars().collect();
        let dag = seg.get_dag(&chars);
        let route = seg.calc_route(&chars, &dag);
        // Route should prefer "你好" (freq=10) over "你"+"好" (freq=1*1).
        assert_eq!(route[0].1, 2); // Jump from 0 to 2 (the whole word).
    }

    #[test]
    fn test_fit_and_predict_known_words() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.fit_segmented(training_data());
        let result = seg.predict(vec!["你好世界".into()]);
        assert_eq!(result[0], vec!["你好", "世界"]);
    }

    #[test]
    fn test_predict_empty_string() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.fit_segmented(training_data());
        let result = seg.predict(vec!["".into()]);
        assert_eq!(result[0], Vec::<String>::new());
    }

    #[test]
    fn test_predict_single_char_word() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.fit_segmented(training_data());
        // "我", "喜歡", "你" are all in the dict.
        // "我" and "你" are single-char words — output directly without HMM.
        // "喜歡" is a 2-char dict word — output from the DAG route.
        let result = seg.predict(vec!["我喜歡你".into()]);
        assert_eq!(result[0], vec!["我", "喜歡", "你"]);
    }

    #[test]
    fn test_predict_in_dict_span_split_char_by_char() {
        // If the entire buffered span IS in the dict, we split char-by-char
        // rather than calling HMM, because the DAG explicitly chose not to
        // take the multi-char path.
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        // Make "你" and "好" both more frequent than "你好" so DAG prefers single hops.
        seg.dict.insert_count("你好".chars(), 1);
        seg.dict.insert_count("你".chars(), 100);
        seg.dict.insert_count("好".chars(), 100);
        seg.total = 201.0;
        // DAG route will prefer "你"(100) + "好"(100) over "你好"(1).
        // flush_oov_buf sees "你好" in dict → split to ["你", "好"], not HMM.
        let result = seg.predict(vec!["你好".into()]);
        assert_eq!(result[0], vec!["你", "好"]);
    }

    #[test]
    fn test_predict_oov_delegated_to_hmm() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.fit_segmented(training_data());
        // "地球" is OOV: not in training data.
        let result = seg.predict(vec!["你好地球".into()]);
        // "你好" should come from the dictionary.
        assert_eq!(result[0][0], "你好");
        // The remaining chars ("地球") go to HMM.
        let rest: String = result[0][1..].join("");
        assert_eq!(rest, "地球"); // All chars preserved.
    }

    #[test]
    fn test_save_and_load_round_trip() {
        let mut seg = DagHmmSegmenter::new(None, None, None, None, None);
        seg.fit_segmented(training_data());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let path_str = path.to_str().unwrap();

        let metadata: HashMap<String, String> =
            [("foo".into(), "bar".into()), ("baz".into(), "qux".into())]
                .into_iter()
                .collect();

        seg.save(path_str, &metadata).unwrap();

        let mut loaded = DagHmmSegmenter::new(None, None, None, None, None);
        let loaded_metadata = loaded.load(path_str).unwrap();

        assert_eq!(metadata, loaded_metadata);

        let input = vec!["你好世界".into(), "我喜歡你".into()];
        assert_eq!(seg.predict(input.clone()), loaded.predict(input));
    }
}
