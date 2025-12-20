import math
from difflib import SequenceMatcher

import jiwer

from scripts.uzbek_text_normalizer import normalize_text


def calculate(reference: str, hypothesis: str):
    """Calculate WER, CER, sequence_similarity metrics between reference and hypothesis texts."""
    reference = _get_safe_string_if_nan(reference)
    hypothesis = _get_safe_string_if_nan(hypothesis)

    # Normalize both texts
    ref_normalized = normalize_text(
        reference,
        should_normalize_numbers_to_words=True,
        should_remove_punctuations=True,
        should_normalize_capitalization=False,
        should_lowercase_text=True,
        should_remove_ellipsis=True,
    )
    hyp_normalized = normalize_text(
        hypothesis,
        should_normalize_numbers_to_words=True,
        should_remove_punctuations=True,
        should_normalize_capitalization=False,
        should_lowercase_text=True,
        should_remove_ellipsis=True,
    )

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
