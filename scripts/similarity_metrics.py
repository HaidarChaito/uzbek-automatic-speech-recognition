import math
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict

import jiwer

from scripts.uzbek_text_normalizer import normalize_text


class NormalizationLevel(Enum):
    """Predefined normalization levels for text comparison."""

    STRICT = "strict"  # Heavy normalization
    LIGHT = "light"  # Minimal normalization
    NONE = "none"  # No normalization at all


# Define normalization configurations
NORMALIZATION_CONFIGS: Dict[NormalizationLevel, Dict[str, bool] | None] = {
    NormalizationLevel.STRICT: {
        "should_normalize_numbers_to_words": True,
        "should_remove_punctuations": True,
        "should_lowercase_text": True,
        "should_remove_ellipsis": True,
    },
    NormalizationLevel.LIGHT: {
        "should_normalize_numbers_to_words": False,
        "should_remove_punctuations": False,
        "should_lowercase_text": False,
        "should_remove_ellipsis": False,
    },
    NormalizationLevel.NONE: None,
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


def calculate_batch(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Calculate corpus-level (macro-averaged) ASR metrics.

    Args:
        references: List of ground truth transcriptions
        hypotheses: List of predicted transcriptions

    Returns:
        Dict with WER, CER, and sequence similarity (both normalized and raw)
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )

    if len(references) == 0:
        raise ValueError("Empty input: references and hypotheses lists cannot be empty")

    config = NORMALIZATION_CONFIGS[NormalizationLevel.STRICT]
    refs_normalized = [normalize_text(ref, **config) for ref in references]
    hyps_normalized = [normalize_text(hyp, **config) for hyp in hypotheses]

    # Normalized corpus-level metrics
    wer = jiwer.wer(refs_normalized, hyps_normalized)
    cer = jiwer.cer(refs_normalized, hyps_normalized)
    sequence_similarity = compute_corpus_sequence_similarity(
        refs_normalized, hyps_normalized
    )

    # Raw [with punctuation, casing and numbers as digit] corpus-level metrics
    wer_raw = jiwer.wer(references, hypotheses)
    cer_raw = jiwer.cer(references, hypotheses)
    sequence_similarity_raw = compute_corpus_sequence_similarity(references, hypotheses)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "sequence_similarity": round(sequence_similarity, 4),
        "wer_raw": round(wer_raw, 4),
        "cer_raw": round(cer_raw, 4),
        "sequence_similarity_raw": round(sequence_similarity_raw, 4),
    }


def compute_corpus_sequence_similarity(refs: list[str], hyps: list[str]) -> float:
    """
    Compute corpus-level (macro) sequence similarity.

    Weights each sample's similarity by the average length of ref and hyp,
    so longer sequences contribute more to the final score.
    """
    total_matching_chars = 0
    total_chars = 0

    for ref, hyp in zip(refs, hyps):
        matcher = SequenceMatcher(None, ref, hyp)

        # Get matching blocks and sum their sizes
        matching_chars = sum(block.size for block in matcher.get_matching_blocks())

        # Total characters (average of ref and hyp length, which is what ratio() uses)
        total_chars += len(ref) + len(hyp)
        total_matching_chars += 2 * matching_chars

    if total_chars == 0:
        return 1.0

    return total_matching_chars / total_chars


def _get_safe_string_if_nan(text) -> str:
    # Checks for NaN or empty string safely
    is_nan = isinstance(text, float) and math.isnan(text)
    if text is None or is_nan or str(text).strip() == "":
        return ""

    return str(text)
