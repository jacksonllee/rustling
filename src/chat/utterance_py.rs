use pyo3::prelude::*;

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::header::ChangeableHeader;
use super::utterance::{Gra, Token, Utterance, Utterances};

// ---------------------------------------------------------------------------
// Gra pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl Gra {
    #[new]
    fn new(dep: usize, head: usize, rel: String) -> Self {
        Self { dep, head, rel }
    }

    #[getter]
    fn dep(&self) -> usize {
        self.dep
    }

    #[getter]
    fn head(&self) -> usize {
        self.head
    }

    #[getter]
    fn rel(&self) -> String {
        self.rel.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Gra(dep={}, head={}, rel='{}')",
            self.dep, self.head, self.rel
        )
    }

    fn __eq__(&self, other: &Gra) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyToken
// ---------------------------------------------------------------------------

/// Python wrapper for [`Token`].
#[pyclass(name = "Token", from_py_object)]
#[derive(Clone)]
pub struct PyToken(pub Token);

#[pymethods]
impl PyToken {
    #[new]
    #[pyo3(signature = (word, pos=None, mor=None, gra=None))]
    fn new(word: String, pos: Option<String>, mor: Option<String>, gra: Option<Gra>) -> Self {
        Self(Token {
            word,
            pos,
            mor,
            gra,
        })
    }

    #[getter]
    fn word(&self) -> &str {
        &self.0.word
    }

    #[getter]
    fn pos(&self) -> Option<&str> {
        self.0.pos.as_deref()
    }

    #[getter]
    fn mor(&self) -> Option<&str> {
        self.0.mor.as_deref()
    }

    #[getter]
    fn gra(&self) -> Option<Gra> {
        self.0.gra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Token(word='{}', pos={}, mor={}, gra={})",
            self.0.word,
            match &self.0.pos {
                Some(p) => format!("'{p}'"),
                None => "None".to_string(),
            },
            match &self.0.mor {
                Some(m) => format!("'{m}'"),
                None => "None".to_string(),
            },
            match &self.0.gra {
                Some(g) => g.__repr__(),
                None => "None".to_string(),
            },
        )
    }

    fn __eq__(&self, other: &PyToken) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// PyUtterance
// ---------------------------------------------------------------------------

/// Python wrapper for [`Utterance`].
#[pyclass(name = "Utterance", from_py_object)]
#[derive(Clone)]
pub struct PyUtterance(pub Utterance);

#[pymethods]
impl PyUtterance {
    #[new]
    #[pyo3(signature = (*, participant=None, tokens=None, time_marks=None, tiers=None, changeable_header=None, mor_tier_name=Some("%mor".to_string()), gra_tier_name=Some("%gra".to_string())))]
    fn new(
        participant: Option<String>,
        tokens: Option<Vec<PyToken>>,
        time_marks: Option<(i64, i64)>,
        tiers: Option<HashMap<String, String>>,
        changeable_header: Option<ChangeableHeader>,
        mor_tier_name: Option<String>,
        gra_tier_name: Option<String>,
    ) -> Self {
        Self(Utterance {
            participant,
            tokens: tokens.map(|ts| ts.into_iter().map(|pt| pt.0).collect()),
            time_marks,
            tiers,
            changeable_header,
            mor_tier_name,
            gra_tier_name,
        })
    }

    #[getter]
    fn participant(&self) -> Option<&str> {
        self.0.participant.as_deref()
    }

    #[getter]
    fn tokens(&self) -> Option<Vec<PyToken>> {
        self.0
            .tokens
            .as_ref()
            .map(|ts| ts.iter().map(|t| PyToken(t.clone())).collect())
    }

    #[getter]
    fn time_marks(&self) -> Option<(i64, i64)> {
        self.0.time_marks
    }

    #[getter]
    fn tiers(&self) -> Option<HashMap<String, String>> {
        self.0.tiers.clone()
    }

    #[getter]
    fn changeable_header(&self) -> Option<ChangeableHeader> {
        self.0.changeable_header.clone()
    }

    #[getter]
    fn mor_tier_name(&self) -> Option<&str> {
        self.0.mor_tier_name.as_deref()
    }

    #[getter]
    fn gra_tier_name(&self) -> Option<&str> {
        self.0.gra_tier_name.as_deref()
    }

    /// Audibly faithful transcript of this utterance, or None for headers.
    #[getter]
    fn audible(&self) -> Option<String> {
        self.0.audible()
    }

    fn __repr__(&self) -> String {
        if let Some(ref ch) = self.0.changeable_header {
            return format!("Utterance(changeable_header={ch:?})");
        }
        format!(
            "Utterance(participant='{}', tokens=[...{} tokens], time_marks={:?})",
            self.0.participant.as_deref().unwrap_or(""),
            self.0.tokens.as_ref().map_or(0, |t| t.len()),
            self.0.time_marks,
        )
    }

    fn _repr_html_(&self) -> String {
        self.0.repr_html()
    }

    fn __eq__(&self, other: &PyUtterance) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash_into(&mut hasher);
        hasher.finish()
    }

    /// Return a plain text tabular representation of this utterance.
    pub fn to_str(&self) -> String {
        self.0.to_str()
    }
}

// ---------------------------------------------------------------------------
// PyUtterances
// ---------------------------------------------------------------------------

/// Python wrapper for [`Utterances`].
#[pyclass(name = "Utterances", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyUtterances(pub Utterances);

#[pymethods]
impl PyUtterances {
    fn __repr__(&self) -> String {
        self.0
            .utterances
            .iter()
            .map(|u| u.to_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn _repr_html_(&self) -> String {
        self.0
            .utterances
            .iter()
            .map(|u| u.repr_html())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn __len__(&self) -> usize {
        self.0.utterances.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PyUtterance> {
        let len = self.0.utterances.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        Ok(PyUtterance(self.0.utterances[idx as usize].clone()))
    }

    fn __eq__(&self, other: &PyUtterances) -> bool {
        self.0.utterances == other.0.utterances
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.utterances.len().hash(&mut hasher);
        for u in &self.0.utterances {
            u.hash_into(&mut hasher);
        }
        hasher.finish()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyUtterancesIter {
        PyUtterancesIter {
            inner: slf.0.utterances.clone(),
            index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PyUtterancesIter
// ---------------------------------------------------------------------------

/// Iterator for [`PyUtterances`].
#[pyclass]
struct PyUtterancesIter {
    inner: Vec<Utterance>,
    index: usize,
}

#[pymethods]
impl PyUtterancesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyUtterance> {
        if self.index < self.inner.len() {
            let item = self.inner[self.index].clone();
            self.index += 1;
            Some(PyUtterance(item))
        } else {
            None
        }
    }
}
