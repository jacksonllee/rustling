"""POS taggers.

This module provides part-of-speech taggers that can be trained on
tagged sentences and used to predict POS tags for new text.
"""

from rustling._lib_name import taggers as _taggers

AveragedPerceptronTagger = _taggers.AveragedPerceptronTagger

__all__ = ["AveragedPerceptronTagger"]
