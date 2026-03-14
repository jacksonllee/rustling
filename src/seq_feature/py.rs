use pyo3::prelude::*;

use super::{SeqFeatureTemplate, SeqTransform};

// ---------------------------------------------------------------------------
// PyO3 factory functions
// ---------------------------------------------------------------------------

/// Create an observation feature template.
#[pyfunction]
#[pyo3(signature = (*positions, transform=None))]
fn seq_obs(positions: Vec<i32>, transform: Option<&str>) -> PyResult<SeqFeatureTemplate> {
    if positions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seq_obs() requires at least one position.",
        ));
    }
    for &pos in &positions {
        if !(-4..=4).contains(&pos) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Position {} is out of range [-4, +4].",
                pos
            )));
        }
    }
    let transform = match transform {
        None => SeqTransform::Identity,
        Some("first_char") => SeqTransform::FirstChar,
        Some("final_char") => SeqTransform::FinalChar,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown transform '{}'. Use 'first_char' or 'final_char'.",
                other
            )));
        }
    };
    Ok(SeqFeatureTemplate::obs(&positions, transform))
}

/// Create a label feature template.
#[pyfunction]
#[pyo3(signature = (*positions))]
fn seq_label(positions: Vec<i32>) -> PyResult<SeqFeatureTemplate> {
    if positions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seq_label() requires at least one position.",
        ));
    }
    for &pos in &positions {
        if !(-4..=4).contains(&pos) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Position {} is out of range [-4, +4].",
                pos
            )));
        }
        if pos >= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "seq_label positions must be negative (look back only), got {}.",
                pos
            )));
        }
    }
    Ok(SeqFeatureTemplate::label(&positions))
}

/// Register the feature submodule with Python.
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let seq_feature_module = PyModule::new(parent_module.py(), "seq_feature")?;
    seq_feature_module.add_class::<SeqFeatureTemplate>()?;
    seq_feature_module.add_function(wrap_pyfunction!(seq_obs, &seq_feature_module)?)?;
    seq_feature_module.add_function(wrap_pyfunction!(seq_label, &seq_feature_module)?)?;
    parent_module.add_submodule(&seq_feature_module)?;
    Ok(())
}
