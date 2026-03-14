//! Index of Productive Syntax (IPSyn) scoring.
//!
//! Evaluates 56 syntactic structures (N1–N11, V1–V16, Q1–Q10, S1–S19)
//! across utterances, awarding 0–2 points per item (max 112).

use crate::chat::utterance::Utterance;

// ---------------------------------------------------------------------------
// Dependency graph (private, not exposed to Python)
// ---------------------------------------------------------------------------

struct Node {
    word: String,
    pos: String,
    mor: String,
}

/// A lightweight dependency graph built from an utterance's tokens.
///
/// Nodes are indexed 0..n where 0 = ROOT and 1..n correspond to tokens.
/// `dep_to_head[i]` gives the head of node `i`, `dep_to_rel[i]` its relation.
struct DependencyGraph {
    nodes: Vec<Node>,
    dep_to_head: Vec<usize>,
    dep_to_rel: Vec<String>,
    faulty: bool,
}

impl DependencyGraph {
    fn from_utterance(utterance: &Utterance) -> Self {
        let tokens = utterance.tokens.as_deref().unwrap_or(&[]);
        let n = tokens.len(); // number of actual tokens (positions 1..n)
        let total = n + 1; // +1 for ROOT at position 0

        let mut nodes = Vec::with_capacity(total);
        let mut dep_to_head = vec![0usize; total];
        let mut dep_to_rel = vec![String::new(); total];
        let mut faulty = false;

        // Position 0: ROOT node
        nodes.push(Node {
            word: "ROOT".to_string(),
            pos: "ROOT".to_string(),
            mor: "ROOT".to_string(),
        });

        for token in tokens {
            let word = token.word.clone();
            let pos = token.pos.clone().unwrap_or_default();
            let mor = token.mor.clone().unwrap_or_default();

            nodes.push(Node { word, pos, mor });

            if let Some(ref gra) = token.gra {
                let dep = gra.dep;
                if dep < total {
                    dep_to_head[dep] = gra.head;
                    dep_to_rel[dep] = gra.rel.clone();
                }
            } else {
                faulty = true;
            }
        }

        Self {
            nodes,
            dep_to_head,
            dep_to_rel,
            faulty,
        }
    }

    /// Number of nodes including ROOT.
    #[inline]
    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Scoring board
// ---------------------------------------------------------------------------

/// The 56 IPSyn items, ordered N1..N11, V1..V16, Q1..Q10, S1..S19.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Item {
    N1,
    N2,
    N3,
    N4,
    N5,
    N6,
    N7,
    N8,
    N9,
    N10,
    N11,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
    Q8,
    Q9,
    Q10,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
    S17,
    S18,
    S19,
}

const NUM_ITEMS: usize = 56;

struct ScoringBoard {
    scores: [u8; NUM_ITEMS],
    stopped: [bool; NUM_ITEMS],
}

impl ScoringBoard {
    fn new() -> Self {
        Self {
            scores: [0; NUM_ITEMS],
            stopped: [false; NUM_ITEMS],
        }
    }

    /// Add one point to `item`. If it reaches 2, stop further scoring.
    #[inline]
    fn add_point(&mut self, item: Item) {
        let i = item as usize;
        if !self.stopped[i] {
            self.scores[i] += 1;
            if self.scores[i] >= 2 {
                self.stopped[i] = true;
            }
        }
    }

    /// Credit a related item with one point if it hasn't already reached 2.
    #[inline]
    fn credit(&mut self, item: Item) {
        self.add_point(item);
    }

    #[inline]
    fn is_stopped(&self, item: Item) -> bool {
        self.stopped[item as usize]
    }

    /// Check if `item` has reached 2 points; if so, stop it. Returns true if stopped.
    #[inline]
    fn turn_off(&mut self, item: Item) -> bool {
        let i = item as usize;
        if self.scores[i] >= 2 {
            self.stopped[i] = true;
            true
        } else {
            false
        }
    }

    /// Force an item to score 2 and stop.
    #[inline]
    fn force_max(&mut self, item: Item) {
        let i = item as usize;
        self.scores[i] = 2;
        self.stopped[i] = true;
    }

    fn total(&self) -> usize {
        self.scores.iter().map(|&s| s as usize).sum()
    }
}

// ---------------------------------------------------------------------------
// Helper: extract lemma from mor field
// ---------------------------------------------------------------------------

fn get_lemma_from_mor(mor: &str) -> &str {
    let mor = mor.split('-').next().unwrap_or(mor);
    mor.split('&').next().unwrap_or(mor)
}

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

#[inline]
fn is_noun_pos(pos: &str) -> bool {
    pos == "n" || pos.starts_with("n:")
}

#[inline]
fn is_np_modifier_pos(pos: &str) -> bool {
    pos == "pro:poss:det" || pos == "adj" || pos == "qn"
}

#[inline]
fn is_punctuation_mor(mor: &str) -> bool {
    mor.is_empty() || mor == "beg" || mor == "end"
}

// ---------------------------------------------------------------------------
// Words that cause the whole utterance to be skipped
// ---------------------------------------------------------------------------

fn should_skip_utterance(graph: &DependencyGraph) -> bool {
    for i in 1..graph.n_nodes() {
        let w = graph.nodes[i].word.as_str();
        if w == "xxx" || w == "yyy" || w == "www" {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Noun phrase items (N1–N11)
// ---------------------------------------------------------------------------

fn score_n1(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N1) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if is_noun_pos(pos) {
            board.add_point(Item::N1);
        }
        if board.turn_off(Item::N1) {
            break;
        }
    }
}

fn score_n2(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N2) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos.starts_with("pro") && pos != "pro:poss:det" {
            board.add_point(Item::N2);
        }
        if board.turn_off(Item::N2) {
            break;
        }
    }
}

fn score_n3(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N3) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if is_np_modifier_pos(pos) {
            board.add_point(Item::N3);
        }
        if board.turn_off(Item::N3) {
            break;
        }
    }
}

fn score_n4(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::N4) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        if is_np_modifier_pos(pos1) && is_noun_pos(pos2) {
            board.add_point(Item::N4);
        }
        if board.turn_off(Item::N4) {
            break;
        }
    }
}

fn score_n5(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::N5) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        if pos1 == "det" && is_noun_pos(pos2) {
            board.add_point(Item::N5);
            board.credit(Item::N4);
        }
        if board.turn_off(Item::N5) {
            break;
        }
    }
}

fn score_n6(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() - 2 {
        if board.is_stopped(Item::N6) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        let pos3 = graph.nodes[i + 2].pos.as_str();
        if (pos1 == "v" || pos1 == "prep") && is_np_modifier_pos(pos2) && is_noun_pos(pos3) {
            board.add_point(Item::N6);
            board.credit(Item::N4);
        }
        if board.turn_off(Item::N6) {
            break;
        }
    }
}

fn score_n7(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N7) {
            break;
        }
        let mor = graph.nodes[i].mor.as_str();
        if mor.contains("-PL") {
            board.add_point(Item::N7);
        }
        if board.turn_off(Item::N7) {
            break;
        }
    }
}

fn score_n8(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() - 2 {
        if board.is_stopped(Item::N8) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        let pos3 = graph.nodes[i + 2].pos.as_str();
        if is_np_modifier_pos(pos1) && is_noun_pos(pos2) && pos3 == "v" {
            board.add_point(Item::N8);
            board.credit(Item::N4);
        }
        if board.turn_off(Item::N8) {
            break;
        }
    }
}

fn score_n9(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() - 2 {
        if board.is_stopped(Item::N9) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        let pos3 = graph.nodes[i + 2].pos.as_str();
        if is_np_modifier_pos(pos1) && (pos2 == "adj" || pos2 == "qn") && is_noun_pos(pos3) {
            board.add_point(Item::N9);
            board.credit(Item::N4);
        }
        if board.turn_off(Item::N9) {
            break;
        }
    }
}

fn score_n10(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N10) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos != "adv" {
            continue;
        }
        let head = graph.dep_to_head[i];
        if head < graph.n_nodes() {
            let head_pos = graph.nodes[head].pos.as_str();
            if head_pos == "adj" || head_pos == "n" {
                board.add_point(Item::N10);
                board.credit(Item::V8);
            }
        }
        if board.turn_off(Item::N10) {
            break;
        }
    }
}

fn score_n11(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::N11) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos == "n" || pos == "adj" || pos.starts_with("n:") {
            let mor = graph.nodes[i].mor.as_str();
            // Remove -PL, then check for remaining bound morphemes (marked by -)
            let stripped = mor.replace("-PL", "");
            if stripped.contains('-') {
                board.add_point(Item::N11);
            }
        }
        if board.turn_off(Item::N11) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Verb phrase items (V1–V16)
// ---------------------------------------------------------------------------

fn score_v1(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V1) {
            break;
        }
        if graph.nodes[i].pos == "v" {
            board.add_point(Item::V1);
        }
        if board.turn_off(Item::V1) {
            break;
        }
    }
}

fn score_v2(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V2) {
            break;
        }
        if graph.nodes[i].pos == "prep" {
            board.add_point(Item::V2);
        }
        if board.turn_off(Item::V2) {
            break;
        }
    }
}

fn score_v3(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::V3) {
            break;
        }
        if graph.dep_to_rel[dep] == "POBJ" {
            board.add_point(Item::V3);
            board.credit(Item::V2);
        }
        if board.turn_off(Item::V3) {
            break;
        }
    }
}

fn score_v4(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V4) {
            break;
        }
        if graph.nodes[i].pos != "cop" {
            continue;
        }
        let mut has_subject = false;
        let mut has_predicate = false;
        for dep in 1..graph.n_nodes() {
            if graph.dep_to_head[dep] != i {
                continue;
            }
            let rel = graph.dep_to_rel[dep].as_str();
            let dep_pos = graph.nodes[dep].pos.as_str();
            if rel == "SUBJ" && !dep_pos.ends_with("wh") {
                has_subject = true;
            } else if rel == "PRED" {
                has_predicate = true;
            }
        }
        if has_subject && has_predicate {
            board.add_point(Item::V4);
            board.credit(Item::V1);
        }
        if board.turn_off(Item::V4) {
            break;
        }
    }
}

fn score_v5(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    const PSEUDO_AUX: &[&str] = &[
        "hafta",
        "haf(ta)",
        "s'pose(da)",
        "s'poseda",
        "gonna",
        "gon(na)",
        "wanna",
        "wanta",
        "wan(t)(a)",
        "want(a)",
        "wan(na)",
        "gotta",
        "got(ta)",
        "better",
    ];
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::V5) {
            break;
        }
        let pos2 = graph.nodes[i + 1].pos.as_str();
        if pos2 != "v" {
            continue;
        }
        let word1 = graph.nodes[i].word.as_str();
        if PSEUDO_AUX.contains(&word1) {
            board.add_point(Item::V5);
        }
        if board.turn_off(Item::V5) {
            break;
        }
    }
}

fn score_v6(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V6) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        let mor = graph.nodes[i].mor.as_str();
        let lemma = get_lemma_from_mor(mor);
        if (pos == "aux" && !mor.starts_with("wi")) || (lemma == "do" && pos == "mod") {
            board.add_point(Item::V6);
            board.credit(Item::V5);
        }
        if board.turn_off(Item::V6) {
            break;
        }
    }
}

fn score_v7(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V7) {
            break;
        }
        if graph.nodes[i].mor.ends_with("PRESP") {
            board.add_point(Item::V7);
        }
        if board.turn_off(Item::V7) {
            break;
        }
    }
}

fn score_v8(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V8) {
            break;
        }
        if graph.nodes[i].pos == "adv" {
            board.add_point(Item::V8);
        }
        if board.turn_off(Item::V8) {
            break;
        }
    }
}

fn score_v9(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::V9) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        let word = graph.nodes[i].word.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        if pos.starts_with("mod") && pos2 == "v" && !word.is_empty() {
            board.add_point(Item::V9);
            board.credit(Item::V5);
        }
        if board.turn_off(Item::V9) {
            break;
        }
    }
}

fn score_v10(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V10) {
            break;
        }
        if graph.nodes[i].mor.contains("-3S") {
            board.add_point(Item::V10);
        }
        if board.turn_off(Item::V10) {
            break;
        }
    }
}

fn score_v11(graph: &DependencyGraph, board: &mut ScoringBoard) {
    const PAST_MODALS: &[&str] = &["could", "did", "might", "would", "wouldn't"];
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V11) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos != "mod" {
            continue;
        }
        let word = graph.nodes[i].word.as_str();
        if PAST_MODALS.contains(&word) {
            board.add_point(Item::V11);
            board.credit(Item::V9);
        }
        if board.turn_off(Item::V11) {
            break;
        }
    }
}

fn score_v12(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V12) {
            break;
        }
        let mor = graph.nodes[i].mor.as_str();
        if mor.contains("-PAST") && !mor.contains("-PASTP") {
            board.add_point(Item::V12);
        }
        if board.turn_off(Item::V12) {
            break;
        }
    }
}

fn score_v13(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V13) {
            break;
        }
        let mor = graph.nodes[i].mor.as_str();
        let pos = graph.nodes[i].pos.as_str();
        if mor.contains("&PAST") && (pos == "aux" || pos == "mod") {
            board.add_point(Item::V13);
            board.credit(Item::V6);
        }
        if board.turn_off(Item::V13) {
            break;
        }
    }
}

fn score_v14(graph: &DependencyGraph, board: &mut ScoringBoard) {
    // Medial adverb: not first or last word
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 2..graph.n_nodes() - 1 {
        if board.is_stopped(Item::V14) {
            break;
        }
        if graph.nodes[i].pos == "adv" {
            board.add_point(Item::V14);
            board.credit(Item::V8);
        }
        if board.turn_off(Item::V14) {
            break;
        }
    }
}

fn score_v15(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::V15) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        if pos1 != "cop" && pos1 != "aux" && pos1 != "mod" {
            continue;
        }
        let mor2 = graph.nodes[i + 1].mor.as_str();
        if is_punctuation_mor(mor2) {
            board.add_point(Item::V15);
            board.credit(Item::V4);
            board.credit(Item::V6);
            board.credit(Item::V9);
            board.credit(Item::V11);
            board.credit(Item::V13);
            board.credit(Item::V16);
        }
        if board.turn_off(Item::V15) {
            break;
        }
    }
}

fn score_v16(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::V16) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        let mor = graph.nodes[i].mor.as_str();
        if pos.starts_with("cop") && mor.contains("PAST") {
            board.add_point(Item::V16);
            board.credit(Item::V4);
        }
        if board.turn_off(Item::V16) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Question items (Q1–Q10)
// ---------------------------------------------------------------------------

fn score_q1(graph: &DependencyGraph, board: &mut ScoringBoard) {
    let n = graph.n_nodes();
    if n < 2 {
        return;
    }
    let final_word = graph.nodes[n - 1].word.as_str();
    if final_word != "?" {
        return;
    }
    let first_word = graph.nodes[1].word.to_lowercase();
    if matches!(
        first_word.as_str(),
        "what" | "why" | "how" | "which" | "where" | "when"
    ) {
        return;
    }
    board.add_point(Item::Q1);
    board.turn_off(Item::Q1);
}

fn score_q2(graph: &DependencyGraph, board: &mut ScoringBoard) {
    let n = graph.n_nodes();
    if n < 2 {
        return;
    }
    let final_word = graph.nodes[n - 1].word.as_str();
    if final_word != "?" {
        return;
    }
    let first_word = graph.nodes[1].word.to_lowercase();
    if !matches!(
        first_word.as_str(),
        "what" | "why" | "how" | "which" | "where" | "when"
    ) {
        return;
    }
    if n > 2 {
        board.add_point(Item::Q2);
    }
    board.turn_off(Item::Q2);
}

fn score_q3(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    const NEG_WORDS: &[&str] = &["no", "not", "can't", "don't"];
    for i in 1..graph.n_nodes() - 1 {
        if board.is_stopped(Item::Q3) {
            break;
        }
        let word1 = graph.nodes[i].word.as_str();
        let mor2 = graph.nodes[i + 1].mor.as_str();
        if NEG_WORDS.contains(&word1) && !is_punctuation_mor(mor2) {
            board.add_point(Item::Q3);
        }
        if board.turn_off(Item::Q3) {
            break;
        }
    }
}

fn score_q4(graph: &DependencyGraph, board: &mut ScoringBoard) {
    let n = graph.n_nodes();
    if n <= 2 {
        return;
    }
    let final_word = graph.nodes[n - 1].word.as_str();
    if final_word != "?" {
        return;
    }
    let first_word = graph.nodes[1].word.to_lowercase();
    if !matches!(
        first_word.as_str(),
        "what" | "why" | "how" | "which" | "where" | "when"
    ) {
        return;
    }
    // Check if the head of the first word is a verb
    let head = graph.dep_to_head[1];
    if head < n && graph.nodes[head].pos == "v" {
        board.add_point(Item::Q4);
    }
    if board.turn_off(Item::Q4) {
        board.force_max(Item::Q1);
        board.force_max(Item::Q2);
    }
}

fn score_q5(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::Q5) {
            break;
        }
        let head = graph.dep_to_head[dep];
        if dep >= head || head >= graph.n_nodes() {
            continue;
        }
        let rel = graph.dep_to_rel[dep].as_str();
        if rel != "SUBJ" {
            continue;
        }
        // Bug fix: old Python code had "V" (uppercase), should be "v" (lowercase)
        let head_pos = graph.nodes[head].pos.as_str();
        if head_pos != "v" {
            continue;
        }
        // Check for negation between subject and verb
        for j in (dep + 1)..head {
            if graph.nodes[j].pos == "neg" {
                board.add_point(Item::Q5);
                board.credit(Item::Q3);
                break;
            }
        }
        if board.turn_off(Item::Q5) {
            break;
        }
    }
}

fn score_q6(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::Q6) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos != "cop" && pos != "mod" && pos != "aux" {
            continue;
        }
        // Check for wh-word as a dependent that comes before this node
        for dep in 1..graph.n_nodes() {
            if graph.dep_to_head[dep] != i {
                continue;
            }
            if dep >= i {
                continue; // want inversion: wh-dep before head
            }
            if graph.nodes[dep].pos == "adv:wh" {
                board.add_point(Item::Q6);
            }
            if board.turn_off(Item::Q6) {
                break;
            }
        }
        if board.is_stopped(Item::Q6) {
            break;
        }
    }
}

fn score_q7(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::Q7) {
            break;
        }
        let pos = graph.nodes[i].pos.as_str();
        if pos != "mod" && pos != "cop" && pos != "aux" {
            continue;
        }
        for dep in 1..graph.n_nodes() {
            if graph.dep_to_head[dep] != i {
                continue;
            }
            if graph.nodes[dep].pos == "neg" {
                board.add_point(Item::Q7);
                board.credit(Item::Q5);
            }
            if board.turn_off(Item::Q7) {
                break;
            }
        }
        if board.is_stopped(Item::Q7) {
            break;
        }
    }
}

fn score_q8(graph: &DependencyGraph, board: &mut ScoringBoard) {
    let n = graph.n_nodes();
    if n <= 2 {
        return;
    }
    let final_word = graph.nodes[n - 1].word.as_str();
    if final_word != "?" {
        return;
    }
    for i in 1..n - 1 {
        if board.is_stopped(Item::Q8) {
            break;
        }
        let pos1 = graph.nodes[i].pos.as_str();
        // Check that the preceding word is not a wh-word
        let wh_test = if i > 1 {
            graph.nodes[i - 1].pos.as_str()
        } else {
            "dummy"
        };
        if (pos1 == "cop" || pos1 == "mod" || pos1 == "aux") && !wh_test.ends_with("wh") {
            // Check if the next word has a SUBJ relation
            if i + 1 < n {
                let next_head = graph.dep_to_head[i + 1];
                let next_rel = graph.dep_to_rel[i + 1].as_str();
                if next_rel == "SUBJ" && next_head < n {
                    board.add_point(Item::Q8);
                    if board.turn_off(Item::Q8) {
                        board.force_max(Item::Q1);
                        board.force_max(Item::Q2);
                        break;
                    }
                }
            }
        }
    }
}

fn score_q9(graph: &DependencyGraph, board: &mut ScoringBoard) {
    const WH_WORDS: &[&str] = &["why", "when", "which", "whose"];
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::Q9) {
            break;
        }
        let word = graph.nodes[i].word.as_str();
        if WH_WORDS.contains(&word) {
            board.add_point(Item::Q9);
        }
        if board.turn_off(Item::Q9) {
            break;
        }
    }
}

fn score_q10(graph: &DependencyGraph, board: &mut ScoringBoard) {
    let n = graph.n_nodes();
    if n <= 2 {
        return;
    }
    let final_word = graph.nodes[n - 1].word.as_str();
    if final_word != "?" {
        return;
    }
    // Part 1: "okay ?", "ok ?", "right ?"
    let second_final_word = graph.nodes[n - 2].word.as_str();
    if second_final_word == "okay" || second_final_word == "ok" || second_final_word == "right" {
        board.add_point(Item::Q10);
    }
    if board.turn_off(Item::Q10) {
        return;
    }
    // Part 2: POS pattern for tag questions
    let pos_seq: Vec<&str> = (1..n).map(|i| graph.nodes[i].pos.as_str()).collect();
    let test = pos_seq.join(" ");
    if test.contains("cop neg pro ?") || test.contains("cop pro ?") {
        board.add_point(Item::Q10);
    }
    board.turn_off(Item::Q10);
}

// ---------------------------------------------------------------------------
// Sentence items (S1–S19)
// ---------------------------------------------------------------------------

fn score_s1(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() > 2 {
        board.add_point(Item::S1);
    }
    board.turn_off(Item::S1);
}

fn score_s2(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S2) {
            break;
        }
        let head = graph.dep_to_head[dep];
        if dep >= head || head >= graph.n_nodes() {
            continue;
        }
        if graph.dep_to_rel[dep] != "SUBJ" {
            continue;
        }
        if graph.nodes[head].pos == "v" {
            board.add_point(Item::S2);
            board.credit(Item::S1);
        }
        if board.turn_off(Item::S2) {
            break;
        }
    }
}

fn score_s3(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 2 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S3) {
            break;
        }
        let head = graph.dep_to_head[dep];
        // Object comes after verb: dep > head
        if dep <= head || head >= graph.n_nodes() {
            continue;
        }
        if graph.dep_to_rel[dep] != "OBJ" {
            continue;
        }
        if graph.nodes[head].pos == "v" {
            board.add_point(Item::S3);
            board.credit(Item::S1);
        }
        if board.turn_off(Item::S3) {
            break;
        }
    }
}

fn score_s4(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::S4) {
            break;
        }
        if graph.nodes[i].pos != "v" {
            continue;
        }
        let mut has_subject = false;
        let mut has_object = false;
        for dep in 1..graph.n_nodes() {
            if graph.dep_to_head[dep] != i {
                continue;
            }
            let rel = graph.dep_to_rel[dep].as_str();
            if rel == "SUBJ" && dep < i {
                has_subject = true;
            }
            if rel == "OBJ" && dep > i {
                has_object = true;
            }
        }
        if has_subject && has_object {
            board.add_point(Item::S4);
            board.credit(Item::S2);
            board.credit(Item::S3);
        }
        if board.turn_off(Item::S4) {
            break;
        }
    }
}

fn score_s5(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::S5) {
            break;
        }
        if graph.nodes[i].pos == "conj" {
            board.add_point(Item::S5);
        }
        if board.turn_off(Item::S5) {
            break;
        }
    }
}

fn score_s6(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 4 {
        return;
    }
    // Collect heads that are verbs (from dependency edges)
    let mut verb_heads: Vec<usize> = Vec::new();
    for dep in 1..graph.n_nodes() {
        let head = graph.dep_to_head[dep];
        if head < graph.n_nodes() && graph.nodes[head].pos == "v" {
            verb_heads.push(head);
        }
    }
    // Deduplicate
    verb_heads.sort_unstable();
    verb_heads.dedup();
    // Two or more distinct verb heads
    if verb_heads.len() >= 2 {
        // Check that the two verb heads are not directly related
        let mut independent = false;
        for j in 0..verb_heads.len() {
            for k in (j + 1)..verb_heads.len() {
                let v1 = verb_heads[j];
                let v2 = verb_heads[k];
                // Two verbs are independent if neither is a dep of the other
                if graph.dep_to_head[v1] != v2 && graph.dep_to_head[v2] != v1 {
                    independent = true;
                    break;
                }
            }
            if independent {
                break;
            }
        }
        if independent {
            board.add_point(Item::S6);
        }
    }
    board.turn_off(Item::S6);
}

fn score_s7(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for i in 1..graph.n_nodes() - 2 {
        if board.is_stopped(Item::S7) {
            break;
        }
        let mor1 = graph.nodes[i].mor.as_str();
        let pos2 = graph.nodes[i + 1].pos.as_str();
        let mor3 = graph.nodes[i + 2].mor.as_str();
        if pos2 == "conj" && !is_punctuation_mor(mor1) && !is_punctuation_mor(mor3) {
            board.add_point(Item::S7);
            board.credit(Item::S5);
        }
        if board.turn_off(Item::S7) {
            break;
        }
    }
}

fn score_s8(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S8) {
            break;
        }
        if graph.nodes[dep].pos != "inf" {
            continue;
        }
        let inf_verb = graph.dep_to_head[dep];
        if inf_verb >= graph.n_nodes() {
            continue;
        }
        // Check that inf_verb's relation is not ROOT
        let inf_verb_head = graph.dep_to_head[inf_verb];
        if inf_verb_head < graph.n_nodes() {
            let inf_verb_rel = graph.dep_to_rel[inf_verb].as_str();
            if !inf_verb_rel.ends_with("ROOT") {
                board.add_point(Item::S8);
                board.credit(Item::S6);
                board.credit(Item::V5);
            }
        }
        if board.turn_off(Item::S8) {
            break;
        }
    }
}

fn score_s9(graph: &DependencyGraph, board: &mut ScoringBoard) {
    const TARGETS: &[&str] = &["let", "make", "help", "watch"];
    if graph.n_nodes() <= 2 {
        return;
    }
    // Check if the first word is one of the target words
    let first_word = graph.nodes[1].word.as_str();
    if !TARGETS.contains(&first_word) {
        return;
    }
    // Check if there's a verb depending on this word
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S9) {
            break;
        }
        if graph.dep_to_head[dep] == 1 && graph.nodes[dep].pos == "v" {
            board.add_point(Item::S9);
            break;
        }
    }
    board.turn_off(Item::S9);
}

fn score_s10(graph: &DependencyGraph, board: &mut ScoringBoard) {
    const EXCEPTIONS: &[&str] = &["and", "or", "then"];
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::S10) {
            break;
        }
        let word = graph.nodes[i].word.as_str();
        let pos = graph.nodes[i].pos.as_str();
        if pos == "conj" && !EXCEPTIONS.contains(&word) {
            board.add_point(Item::S10);
            board.credit(Item::S5);
        }
        if board.turn_off(Item::S10) {
            break;
        }
    }
}

fn score_s11(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    let mut subject_count = 0;
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S11) {
            break;
        }
        if graph.dep_to_rel[dep] == "SUBJ" && !graph.nodes[dep].word.is_empty() {
            subject_count += 1;
            if subject_count > 1 {
                board.add_point(Item::S11);
                board.credit(Item::S6);
            }
        }
        if board.turn_off(Item::S11) {
            break;
        }
    }
}

fn score_s12(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S12) {
            break;
        }
        if graph.nodes[dep].word != "and" {
            continue;
        }
        let head = graph.dep_to_head[dep];
        if head >= graph.n_nodes() {
            continue;
        }
        if graph.dep_to_rel[dep] == "CONJ" && graph.nodes[head].pos == "v" {
            board.add_point(Item::S12);
            board.credit(Item::S6);
            board.credit(Item::S5);
        }
        if board.turn_off(Item::S12) {
            break;
        }
    }
}

fn score_s13(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S13) {
            break;
        }
        let dep_pos = graph.nodes[dep].pos.as_str();
        if !dep_pos.ends_with("wh") {
            continue;
        }
        // Bug fix: old code called graph.nodes() which doesn't exist.
        // Check if next token is an infinitive marker.
        let inf = dep + 1 < graph.n_nodes() && graph.nodes[dep + 1].word == "INF";

        let head = graph.dep_to_head[dep];
        if head >= graph.n_nodes() {
            continue;
        }
        // We want the head of the wh-word to NOT have ROOT as its own relation
        let head_rel = graph.dep_to_rel[head].as_str();
        if head_rel != "ROOT" {
            board.add_point(Item::S13);
            board.credit(Item::S6);
            if inf {
                board.credit(Item::S8);
                board.credit(Item::S17);
            }
        }
        if board.turn_off(Item::S13) {
            break;
        }
    }
}

fn score_s14(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    // Find verb heads with multiple OBJ dependents
    // Collect (head) for each OBJ relation
    let mut obj_heads: Vec<usize> = Vec::new();
    for dep in 1..graph.n_nodes() {
        if graph.dep_to_rel[dep] == "OBJ" {
            obj_heads.push(graph.dep_to_head[dep]);
        }
    }
    // If any head appears more than once, it's bitransitive
    obj_heads.sort_unstable();
    let mut i = 0;
    while i < obj_heads.len() {
        let mut j = i + 1;
        while j < obj_heads.len() && obj_heads[j] == obj_heads[i] {
            j += 1;
        }
        if j - i > 1 {
            board.add_point(Item::S14);
            board.credit(Item::S3);
            break;
        }
        i = j;
    }
    board.turn_off(Item::S14);
}

fn score_s15(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    let verb_count: usize = (1..graph.n_nodes())
        .filter(|&i| graph.nodes[i].pos == "v")
        .count();
    if verb_count > 2 {
        board.add_point(Item::S15);
        board.credit(Item::S6);
    }
    board.turn_off(Item::S15);
}

fn score_s16(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S16) {
            break;
        }
        let head = graph.dep_to_head[dep];
        if dep <= head || head >= graph.n_nodes() {
            continue; // dep must be to the right of head
        }
        if graph.dep_to_rel[dep] != "CMOD" {
            continue;
        }
        // Check that "and" is not between head and dep
        let mut has_and = false;
        for j in (head + 1)..dep {
            if graph.nodes[j].word == "and" {
                has_and = true;
                break;
            }
        }
        if !has_and {
            board.add_point(Item::S16);
            board.credit(Item::S6);
        }
        if board.turn_off(Item::S16) {
            break;
        }
    }
}

fn score_s17(graph: &DependencyGraph, board: &mut ScoringBoard) {
    if graph.n_nodes() <= 3 {
        return;
    }
    for dep in 1..graph.n_nodes() {
        if board.is_stopped(Item::S17) {
            break;
        }
        let word = graph.nodes[dep].word.as_str();
        let pos = graph.nodes[dep].pos.as_str();
        if word != "to" || pos != "inf" {
            continue;
        }
        let inf_verb = graph.dep_to_head[dep]; // e.g., "go" in "he wants me to go"
        if inf_verb >= graph.n_nodes() {
            continue;
        }
        // Bug fix: old code did graph.edges()[inf_verb] which can KeyError.
        // Safe lookup: find what inf_verb depends on.
        let main_verb = graph.dep_to_head[inf_verb];
        if main_verb >= graph.n_nodes() {
            continue;
        }
        // Check if there's an OBJ of main_verb
        for test_dep in 1..graph.n_nodes() {
            if graph.dep_to_head[test_dep] == main_verb && graph.dep_to_rel[test_dep] == "OBJ" {
                board.add_point(Item::S17);
                board.credit(Item::S8);
                break;
            }
        }
        if board.turn_off(Item::S17) {
            break;
        }
    }
}

fn score_s18(graph: &DependencyGraph, board: &mut ScoringBoard) {
    for i in 1..graph.n_nodes() {
        if board.is_stopped(Item::S18) {
            break;
        }
        if graph.nodes[i].pos == "n:gerund" {
            board.add_point(Item::S18);
            board.credit(Item::V7);
        }
        if board.turn_off(Item::S18) {
            break;
        }
    }
}

fn score_s19(graph: &DependencyGraph, board: &mut ScoringBoard) {
    // Check if CONJ precedes two SUBJ positions
    let mut conj_position = graph.n_nodes(); // will be decremented if found
    let mut subj_positions: Vec<usize> = Vec::new();

    for dep in 1..graph.n_nodes() {
        if graph.nodes[dep].pos == "conj" && dep < conj_position {
            conj_position = dep;
        }
        if graph.dep_to_rel[dep] == "SUBJ" {
            subj_positions.push(dep);
        }
    }

    if subj_positions.len() >= 2
        && let Some(&min_subj) = subj_positions.iter().min()
        && conj_position < min_subj
    {
        board.add_point(Item::S19);
        board.credit(Item::S6);
    }
    board.turn_off(Item::S19);
}

// ---------------------------------------------------------------------------
// Main scoring function
// ---------------------------------------------------------------------------

type Scorer = fn(&DependencyGraph, &mut ScoringBoard);

/// All 56 scoring functions, invoked in order.
const SCORERS: &[(Item, Scorer)] = &[
    (Item::N1, score_n1),
    (Item::N2, score_n2),
    (Item::N3, score_n3),
    (Item::N4, score_n4),
    (Item::N5, score_n5),
    (Item::N6, score_n6),
    (Item::N7, score_n7),
    (Item::N8, score_n8),
    (Item::N9, score_n9),
    (Item::N10, score_n10),
    (Item::N11, score_n11),
    (Item::V1, score_v1),
    (Item::V2, score_v2),
    (Item::V3, score_v3),
    (Item::V4, score_v4),
    (Item::V5, score_v5),
    (Item::V6, score_v6),
    (Item::V7, score_v7),
    (Item::V8, score_v8),
    (Item::V9, score_v9),
    (Item::V10, score_v10),
    (Item::V11, score_v11),
    (Item::V12, score_v12),
    (Item::V13, score_v13),
    (Item::V14, score_v14),
    (Item::V15, score_v15),
    (Item::V16, score_v16),
    (Item::Q1, score_q1),
    (Item::Q2, score_q2),
    (Item::Q3, score_q3),
    (Item::Q4, score_q4),
    (Item::Q5, score_q5),
    (Item::Q6, score_q6),
    (Item::Q7, score_q7),
    (Item::Q8, score_q8),
    (Item::Q9, score_q9),
    (Item::Q10, score_q10),
    (Item::S1, score_s1),
    (Item::S2, score_s2),
    (Item::S3, score_s3),
    (Item::S4, score_s4),
    (Item::S5, score_s5),
    (Item::S6, score_s6),
    (Item::S7, score_s7),
    (Item::S8, score_s8),
    (Item::S9, score_s9),
    (Item::S10, score_s10),
    (Item::S11, score_s11),
    (Item::S12, score_s12),
    (Item::S13, score_s13),
    (Item::S14, score_s14),
    (Item::S15, score_s15),
    (Item::S16, score_s16),
    (Item::S17, score_s17),
    (Item::S18, score_s18),
    (Item::S19, score_s19),
];

/// Compute the IPSyn score for a single file's utterances.
///
/// Each utterance is converted to a dependency graph and tested against all
/// 56 items. The total score (0–112) is returned.
pub(crate) fn ipsyn_for_file(utterances: &[&Utterance]) -> usize {
    let mut board = ScoringBoard::new();

    for utterance in utterances {
        // Skip utterances with no tokens
        if utterance.tokens.as_ref().is_none_or(|t| t.is_empty()) {
            continue;
        }

        let graph = DependencyGraph::from_utterance(utterance);
        if graph.faulty {
            continue;
        }
        if should_skip_utterance(&graph) {
            continue;
        }

        for &(item, scorer) in SCORERS {
            if !board.is_stopped(item) {
                scorer(&graph, &mut board);
            }
        }
    }

    board.total()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::utterance::{Gra, Token};

    fn make_token(word: &str, pos: &str, mor: &str, dep: usize, head: usize, rel: &str) -> Token {
        Token {
            word: word.to_string(),
            pos: Some(pos.to_string()),
            mor: Some(mor.to_string()),
            gra: Some(Gra {
                dep,
                head,
                rel: rel.to_string(),
            }),
        }
    }

    fn make_utterance(tokens: Vec<Token>) -> Utterance {
        Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(tokens),
            time_marks: None,
            tiers: None,
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        }
    }

    #[test]
    fn test_empty_utterances() {
        let result = ipsyn_for_file(&[]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_n1_noun() {
        // "cookie ."
        let utt = make_utterance(vec![
            make_token("cookie", "n", "n|cookie", 1, 0, "ROOT"),
            make_token(".", "", "", 2, 1, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt]);
        assert!(result >= 1); // N1 should fire
    }

    #[test]
    fn test_n5_article_before_noun_credits_n4() {
        // "the cookie ."
        let utt = make_utterance(vec![
            make_token("the", "det", "det|the", 1, 2, "DET"),
            make_token("cookie", "n", "n|cookie", 2, 0, "ROOT"),
            make_token(".", "", "", 3, 2, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt, &utt]);
        // N1(2) + N3(0) + N4(>=1 from N5 credit) + N5(2) + S1(2) = at least 7
        assert!(result >= 7);
    }

    #[test]
    fn test_v4_copula_with_subject_and_predicate() {
        // "he is big ."
        let utt = make_utterance(vec![
            make_token("he", "pro", "pro|he", 1, 2, "SUBJ"),
            make_token("is", "cop", "cop|be&3S", 2, 0, "ROOT"),
            make_token("big", "adj", "adj|big", 3, 2, "PRED"),
            make_token(".", "", "", 4, 2, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt]);
        // Should fire V4 (copula+subj+pred), credits V1
        assert!(result >= 2);
    }

    #[test]
    fn test_s2_subject_verb() {
        // "he run ."
        let utt = make_utterance(vec![
            make_token("he", "pro", "pro|he", 1, 2, "SUBJ"),
            make_token("run", "v", "v|run", 2, 0, "ROOT"),
            make_token(".", "", "", 3, 2, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt]);
        // S2 (subj-verb), credits S1; also N2 (pronoun), V1 (verb)
        assert!(result >= 4);
    }

    #[test]
    fn test_skip_xxx_utterance() {
        let utt = make_utterance(vec![
            make_token("xxx", "n", "n|xxx", 1, 0, "ROOT"),
            make_token(".", "", "", 2, 1, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_q5_negation_between_subject_and_verb() {
        // "he not run ." — subject(1) -> verb(3), neg at position 2
        let utt = make_utterance(vec![
            make_token("he", "pro", "pro|he", 1, 3, "SUBJ"),
            make_token("not", "neg", "neg|not", 2, 3, "NEG"),
            make_token("run", "v", "v|run", 3, 0, "ROOT"),
            make_token(".", "", "", 4, 3, "PUNCT"),
        ]);
        let result = ipsyn_for_file(&[&utt]);
        // Q5 should fire (fixed from old Python bug where "V" != "v")
        // Q5 credits Q3; also N2, V1, S1, S2
        assert!(result >= 6);
    }

    #[test]
    fn test_faulty_graph_skipped() {
        // Token without gra
        let utt = Utterance {
            participant: Some("CHI".to_string()),
            tokens: Some(vec![Token {
                word: "cookie".to_string(),
                pos: Some("n".to_string()),
                mor: Some("n|cookie".to_string()),
                gra: None,
            }]),
            time_marks: None,
            tiers: None,
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        };
        let result = ipsyn_for_file(&[&utt]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_max_score_is_112() {
        // The max score per item is 2, and there are 56 items
        let board = ScoringBoard::new();
        assert_eq!(board.total(), 0);
        assert_eq!(NUM_ITEMS, 56);
    }
}
