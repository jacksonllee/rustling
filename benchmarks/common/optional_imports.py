"""Graceful optional import handling for benchmark dependencies.

This module provides utility functions to import optional benchmark dependencies
(NLTK, spaCy) with graceful degradation. If a library is not available, the
function returns a dict with available=False and an error message.
"""

from __future__ import annotations

from typing import Any


def try_import_nltk() -> dict[str, Any]:
    """Try to import NLTK components and download required data.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - available (bool): True if NLTK is available
        - pos_tag (callable): NLTK pos_tag function (if available)
        - pos_tag_sents (callable): NLTK pos_tag_sents function (if available)
        - nltk (module): NLTK module (if available)
        - error (str): Error message (if not available)
    """
    try:
        import nltk

        # Try to download required data (quietly, skip if already downloaded)
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("universal_tagset", quiet=True)
            nltk.download("brown", quiet=True)
            nltk.download("treebank", quiet=True)
        except Exception:
            # Data download failed, but maybe it's already downloaded
            pass

        from nltk import pos_tag, pos_tag_sents

        return {
            "available": True,
            "pos_tag": pos_tag,
            "pos_tag_sents": pos_tag_sents,
            "nltk": nltk,
        }
    except ImportError as e:
        return {
            "available": False,
            "error": f"NLTK not installed: {e}",
        }
    except Exception as e:
        return {
            "available": False,
            "error": f"NLTK error: {e}",
        }


def try_import_spacy(model: str = "en_core_web_sm") -> dict[str, Any]:
    """Try to import spaCy and load the specified model.

    Parameters
    ----------
    model : str, default="en_core_web_sm"
        spaCy model name to load

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - available (bool): True if spaCy is available
        - nlp (spacy.Language): Loaded spaCy model (if available)
        - model_name (str): Name of loaded model (if available)
        - error (str): Error message (if not available)
    """
    try:
        import spacy

        try:
            nlp = spacy.load(model)
            return {
                "available": True,
                "nlp": nlp,
                "model_name": model,
                "spacy": spacy,
            }
        except OSError as e:
            # Model not found
            return {
                "available": False,
                "error": f"spaCy model '{model}' not found: {e}\n"
                f"  Install with: python -m spacy download {model}",
            }
    except ImportError as e:
        return {
            "available": False,
            "error": f"spaCy not installed: {e}",
        }
    except Exception as e:
        return {
            "available": False,
            "error": f"spaCy error: {e}",
        }


def print_library_status(
    library_name: str,
    import_result: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Print the import status of a library.

    Parameters
    ----------
    library_name : str
        Name of the library (e.g., "NLTK", "spaCy")
    import_result : dict[str, Any]
        Result dictionary from try_import_* functions
    verbose : bool, default=True
        If True, print status message
    """
    if not verbose:
        return

    if import_result["available"]:
        extra_info = ""
        if "model_name" in import_result:
            extra_info = f" (model: {import_result['model_name']})"
        print(f"✓ {library_name} loaded successfully{extra_info}")
    else:
        error_msg = import_result.get("error", "Unknown error")
        print(f"✗ {library_name} not available: {error_msg}")
