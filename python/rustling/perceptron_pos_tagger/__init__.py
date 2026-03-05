"""POS tagging.

This module provides part-of-speech taggers that can be trained on
tagged sentences and used to predict POS tags for new text.
"""

from rustling._lib_name import perceptron_pos_tagger as _perceptron_pos_tagger

AveragedPerceptron = _perceptron_pos_tagger.AveragedPerceptron

__all__ = ["AveragedPerceptron"]
