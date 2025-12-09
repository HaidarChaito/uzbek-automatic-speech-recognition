from difflib import SequenceMatcher

import jiwer

from scripts.uzbek_text_normalizer import normalize_text


def calculate(reference: str, hypothesis: str):
    """Calculate WER, CER, sequence_similarity metrics between reference and hypothesis texts."""
    # Normalize both texts
    ref_normalized = normalize_text(
        reference,
        should_remove_punctuations=True,
        should_normalize_capitalization=False,
        should_lowercase_text=True,
    )
    hyp_normalized = normalize_text(
        hypothesis,
        should_remove_punctuations=True,
        should_normalize_capitalization=False,
        should_lowercase_text=True,
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
