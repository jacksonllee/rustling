"""Configurable feature templates for sequence labeling models."""

class SeqFeatureTemplate:
    """A single feature template for sequence labeling.

    Do not instantiate directly. Use ``seq_obs()`` or ``seq_label()``
    factory functions instead.
    """

    ...

def seq_obs(*positions: int, transform: str | None = None) -> SeqFeatureTemplate:
    """Create an observation feature template.

    Args:
        *positions: One or more relative positions in ``[-4, +4]``.
            A single position extracts a unigram; multiple positions
            form an n-gram (e.g., ``seq_obs(-1, 0)`` is a bigram).
        transform: Optional transform applied to each observation.
            ``"first_char"`` or ``"final_char"``. Defaults to identity.

    Returns:
        A feature template for use in the ``features`` parameter of
        model constructors.

    Raises:
        ValueError: If no positions given, position out of range, or
            unknown transform.
    """
    ...

def seq_label(
    *positions: int,
) -> SeqFeatureTemplate:
    """Create a label feature template.

    Label features look back at previously predicted labels.
    Only supported for averaged perceptron models (not HMM).

    Args:
        *positions: One or more negative relative positions in ``[-4, -1]``.
            E.g., ``seq_label(-1)`` uses the label one step back.

    Returns:
        A feature template for use in the ``features`` parameter of
        model constructors.

    Raises:
        ValueError: If no positions given, position out of range ``[-4, +4]``,
            or position is non-negative.
    """
    ...

__all__ = ["SeqFeatureTemplate", "seq_obs", "seq_label"]
