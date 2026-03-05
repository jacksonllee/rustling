"""Configurable feature templates for sequence labeling models."""

from rustling._lib_name import seq_feature as _seq_feature

SeqFeatureTemplate = _seq_feature.SeqFeatureTemplate
seq_obs = _seq_feature.seq_obs
seq_label = _seq_feature.seq_label

__all__ = ["SeqFeatureTemplate", "seq_obs", "seq_label"]
