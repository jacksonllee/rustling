use pyo3::prelude::*;
use pyo3::types::PyDate;

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::header::{Age, Headers, Participant, parse_chat_date};

// ---------------------------------------------------------------------------
// Age pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl Age {
    #[getter]
    fn years(&self) -> u32 {
        self.years
    }

    #[getter]
    fn months(&self) -> Option<u32> {
        self.months
    }

    #[getter]
    fn days(&self) -> Option<u32> {
        self.days
    }

    fn __str__(&self) -> String {
        let mut s = format!("{}", self.years);
        if let Some(m) = self.months {
            s.push_str(&format!(";{m:02}"));
            if let Some(d) = self.days {
                s.push_str(&format!(".{d:02}"));
            }
        } else {
            s.push(';');
        }
        s
    }

    fn __repr__(&self) -> String {
        format!("Age('{}')", self.__str__())
    }

    fn __eq__(&self, other: &Age) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Return the age in total months as a float.
    fn in_months(&self) -> f64 {
        let mut total = self.years as f64 * 12.0;
        if let Some(m) = self.months {
            total += m as f64;
        }
        if let Some(d) = self.days {
            total += d as f64 / 30.0;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Participant pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl Participant {
    #[getter]
    fn code(&self) -> String {
        self.code.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn role(&self) -> String {
        self.role.clone()
    }

    #[getter]
    fn language(&self) -> Option<String> {
        self.language.clone()
    }

    #[getter]
    fn corpus(&self) -> Option<String> {
        self.corpus.clone()
    }

    #[getter]
    fn age(&self) -> Option<Age> {
        self.age.clone()
    }

    #[getter]
    fn sex(&self) -> Option<String> {
        self.sex.clone()
    }

    #[getter]
    fn group(&self) -> Option<String> {
        self.group.clone()
    }

    #[getter]
    fn ses(&self) -> Option<String> {
        self.ses.clone()
    }

    #[getter]
    fn education(&self) -> Option<String> {
        self.education.clone()
    }

    #[getter]
    fn custom(&self) -> Option<String> {
        self.custom.clone()
    }

    #[getter]
    fn birth(&self) -> Option<String> {
        self.birth.clone()
    }

    #[getter]
    fn birthplace(&self) -> Option<String> {
        self.birthplace.clone()
    }

    #[getter]
    fn l1(&self) -> Option<String> {
        self.l1.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Participant(code='{}', name='{}', role='{}')",
            self.code, self.name, self.role
        )
    }

    fn __eq__(&self, other: &Participant) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Headers pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl Headers {
    #[getter]
    fn pid(&self) -> Option<String> {
        self.pid.clone()
    }

    #[getter]
    fn languages(&self) -> Vec<String> {
        self.languages.clone()
    }

    #[getter]
    fn participants(&self) -> Vec<Participant> {
        self.participants.clone()
    }

    #[getter]
    fn options(&self) -> Option<String> {
        self.options.clone()
    }

    #[getter]
    fn location(&self) -> Option<String> {
        self.location.clone()
    }

    #[getter]
    fn number(&self) -> Option<String> {
        self.number.clone()
    }

    #[getter]
    fn recording_quality(&self) -> Option<String> {
        self.recording_quality.clone()
    }

    #[getter]
    fn room_layout(&self) -> Option<String> {
        self.room_layout.clone()
    }

    #[getter]
    fn tape_location(&self) -> Option<String> {
        self.tape_location.clone()
    }

    #[getter]
    fn time_duration(&self) -> Option<String> {
        self.time_duration.clone()
    }

    #[getter]
    fn time_start(&self) -> Option<String> {
        self.time_start.clone()
    }

    #[getter]
    fn transcriber(&self) -> Option<String> {
        self.transcriber.clone()
    }

    #[getter]
    fn transcription(&self) -> Option<String> {
        self.transcription.clone()
    }

    #[getter]
    fn types(&self) -> Option<String> {
        self.types.clone()
    }

    #[getter]
    fn videos(&self) -> Option<String> {
        self.videos.clone()
    }

    #[getter]
    fn warning(&self) -> Option<String> {
        self.warning.clone()
    }

    #[getter]
    fn situation(&self) -> Option<String> {
        self.situation.clone()
    }

    #[getter]
    fn comments(&self) -> Option<Vec<String>> {
        self.comments.clone()
    }

    #[getter]
    fn other(&self) -> HashMap<String, String> {
        self.other.clone()
    }

    #[getter]
    fn date(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.date {
            None => Ok(None),
            Some(s) => match parse_chat_date(s) {
                Some((year, month, day)) => {
                    let date = PyDate::new(py, year, month as u8, day as u8)?;
                    Ok(Some(date.into_any().unbind()))
                }
                None => Ok(None),
            },
        }
    }

    #[getter]
    fn media(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.media_data {
            None => Ok(None),
            Some(m) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("filename", &m.filename)?;
                dict.set_item("format", &m.format)?;
                dict.set_item("status", &m.status)?;
                Ok(Some(dict.into_any().unbind()))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Headers(languages={:?}, participants=[...{}], date={:?})",
            self.languages,
            self.participants.len(),
            self.date,
        )
    }

    fn __eq__(&self, other: &Headers) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_into(&mut hasher);
        hasher.finish()
    }
}
