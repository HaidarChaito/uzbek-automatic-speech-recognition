import math
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict

import jiwer

from scripts.uzbek_text_normalizer import normalize_text


class NormalizationLevel(Enum):
    """Predefined normalization levels for text comparison."""
    STRICT = "strict"  # Heavy normalization
    LIGHT = "light"    # Minimal normalization
    NONE = "none"      # No normalization at all

# Define normalization configurations
NORMALIZATION_CONFIGS: Dict[NormalizationLevel, Dict[str, bool]|None] = {
    NormalizationLevel.STRICT: {
        "should_normalize_numbers_to_words": True,
        "should_remove_punctuations": True,
        "should_normalize_capitalization": False,
        "should_lowercase_text": True,
        "should_remove_ellipsis": True,
    },
    NormalizationLevel.LIGHT: {
        "should_normalize_numbers_to_words": False,
        "should_remove_punctuations": False,
        "should_normalize_capitalization": True,
        "should_lowercase_text": False,
        "should_remove_ellipsis": False,
    },
    NormalizationLevel.NONE: None
}


def calculate(
    reference: str,
    hypothesis: str,
    normalization_level=NormalizationLevel.STRICT,
):
    """Calculate WER, CER, sequence_similarity metrics between reference and hypothesis texts.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text to compare
        normalization_level: Level of text normalization to apply. Options:
            - NormalizationLevel.STRICT (default): Aggressive normalization
            - NormalizationLevel.LIGHT: Minimal normalization
            - NormalizationLevel.NONE: No normalization
    """
    reference = _get_safe_string_if_nan(reference)
    hypothesis = _get_safe_string_if_nan(hypothesis)

    ref_normalized = reference
    hyp_normalized = hypothesis

    # Normalize both texts
    if (
        normalization_level is not None
        and normalization_level != NormalizationLevel.NONE
    ):
        config = NORMALIZATION_CONFIGS[normalization_level]
        ref_normalized = normalize_text(reference, **config)
        hyp_normalized = normalize_text(hypothesis, **config)

    # Word Error Rate (WER)
    wer = jiwer.wer(ref_normalized, hyp_normalized)

    # Character Error Rate (CER)
    cer = jiwer.cer(ref_normalized, hyp_normalized)

    # Sequence similarity (0-1 scale)
    sequence_similarity = SequenceMatcher(None, ref_normalized, hyp_normalized).ratio()

    return {
        "ref_normalized": ref_normalized,
        "hyp_normalized": hyp_normalized,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "sequence_similarity": round(sequence_similarity, 4),
        "ref_word_count": len(ref_normalized.split()),
        "hyp_word_count": len(hyp_normalized.split()),
    }


def _get_safe_string_if_nan(text) -> str:
    # Checks for NaN or empty string safely
    is_nan = isinstance(text, float) and math.isnan(text)
    if text is None or is_nan or str(text).strip() == "":
        return ""

    return str(text)
