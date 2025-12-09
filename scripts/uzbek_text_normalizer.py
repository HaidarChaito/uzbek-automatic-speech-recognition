def normalize_text(
    text: str,
    should_remove_punctuations: bool = False,
    should_normalize_capitalization: bool = True,
    should_lowercase_text: bool = False,
    normalize_domains: bool = True,
    should_normalize_annotations: bool = True,
    lowercase_annotations: bool = True,
) -> str:
    """
    Normalize Uzbek transcribed text for ASR training.

    Applies a pipeline of normalization steps to clean and standardize
    transcribed audio text.

    Pipeline order:
    1. Remove newlines and carriage returns
    2. Clean bullet points and list markers (e.g. •, -, 1.)
    3. Normalize apostrophe variants to standard ASCII apostrophe (')
    4. Normalize annotation markers (optional)
    5. Remove special chars: quotes (", “, ”, «, ...), colons (:), ellipses (...) etc.
    6. Remove punctuations (optional): "!", ",", ".", ";", "?"
    7. Clean excessive whitespace
    8. Fix spacing around punctuation
    9. Normalize capitalization
    10. Lowercase entire text (optional)

    Args:
        text: Raw transcribed text to normalize
        should_remove_punctuations: If True, punctuation charecters will be removed (.,?!:)
        should_normalize_capitalization: If True, normalize capitalization (e.g. capitalization after punctuations)
        should_lowercase_text: If True, makes entire text to lowercase
        normalize_domains: If True, capitalize .uz domain names (e.g., "qalampir.uz" -> "Qalampir uz")
        should_normalize_annotations: If True, normalize annotation brackets: (text), *[text]* -> [text]
        lowercase_annotations: If True, lowercase text within annotations

    Returns:
        Normalized text ready for ASR training

    Examples:
        >>> normalize_text("men   o'qiyman\\n(musiqa)")
        "Men o'qiyman [musiqa]"

        >>> normalize_text("QALAMPIR.UZ  sayti", normalize_domains=True)
        "Qalampir uz sayti"
    """
    text = remove_new_lines(text)
    text = remove_list_markers(text)
    text = normalize_uzbek_apostrophes(text)

    if should_normalize_annotations:
        text = normalize_annotations(text, lowercase_annotation=lowercase_annotations)

    text = remove_special_chars(text)
    if should_remove_punctuations:
        text = remove_punctuations(text)

    text = remove_whitespaces(text)
    text = normalize_spacing_around_punc(text)

    if should_normalize_capitalization:
        text = normalize_capitalization(text, normalize_domains=normalize_domains)

    if should_lowercase_text:
        text = text.lower()

    return text


import re


def remove_whitespaces(text: str) -> str:
    text = text.strip()
    # Remove multiple spaces
    return re.sub(r"\s+", " ", text)


def remove_new_lines(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


def remove_list_markers(text: str) -> str:
    # Remove bullet points and list markers from the start
    bullet_pattern_start = r"^[•–—\-*>→⋅◦▪▫‣]\s*"
    text = re.sub(bullet_pattern_start, "", text)

    # Remove numbered list markers at the start (e.g., 1., 2), 1-, 1:)
    numbered_pattern_start = r"^\d+[\.\):]\s*"
    text = re.sub(numbered_pattern_start, "", text)

    # Remove bullet points in the middle of text
    mid_bullet_pattern = r"[•–—→⋅◦▪▫‣]"
    text = re.sub(mid_bullet_pattern, " ", text).rstrip()

    return text


def normalize_uzbek_apostrophes(text: str) -> str:
    """Normalize all apostrophe variants to straight apostrophe"""
    apostrophe_variants = [
        "‘",  # U+2018 (left single quotation mark) - common in Uzbek
        "’",  # U+2019 (right single quotation mark) - common in Uzbek
        "ʼ",  # U+02BC (modifier letter apostrophe)
        "ʻ",  # U+02BB (modifier letter turned comma)
        "ʽ",  # U+02BD (modifier letter reversed comma)
        "`",  # U+0060 (grave accent)
        "ˊ",  # U+02CA (modifier letter acute accent)
        "ˋ",  # U+02CB (modifier letter grave accent)
    ]

    for variant in apostrophe_variants:
        text = text.replace(variant, "'")

    return text


def normalize_annotations(text: str, lowercase_annotation=True) -> str:
    """Normalize single-word annotations [text], (musiqa), *text* to [text]."""

    # Single word pattern: no spaces allowed
    word = r"[A-Za-z0-9_-]+"

    text = re.sub(rf"\*\s*\[\s*({word})\s*]\s*\*", r"[\1]", text)  # *[word]* -> [word]
    text = re.sub(rf"\\\s*\[\s*({word})\s*]", r"[\1]", text)  # \[word] -> [word]

    # ONLY convert (musiqa) → [musiqa], not any word
    ANNOTATION_WORDS = {"musiqa"}  # add whatever you need
    allowed = "|".join(map(re.escape, ANNOTATION_WORDS))
    text = re.sub(rf"\(\s*({allowed})\s*\)", r"[\1]", text)

    text = re.sub(rf"\[\s*({word})\s*]", r"[\1]", text)  # [ word ] -> [word]

    if lowercase_annotation:
        text = re.sub(rf"\[\s*({word})\s*]", lambda m: f"[{m.group(1).lower()}]", text)

    return text


def remove_special_chars(text: str, remove_ellipsis: bool = False) -> str:
    """
    Remove special characters that don't affect ASR: quotes, colons, optionally ellipsis and others.
    Use remove_ellipsis True for read/book speech, False for conversational speech.
    """
    chars_to_remove = [
        '"',
        "“",
        "”",
        "„",
        "‟",
        "«",
        "»",
        ":",
        "#",
        "&",
        "*",
        "+",
        "/",
        "<",
        ">",
        "=",
        "@",
        "\\",
        "^",
        "_",
        "{",
        "|",
        "}",
        "~",
    ]

    for char in chars_to_remove:
        text = text.replace(char, "")

    if remove_ellipsis:
        text = text.replace("…", "")  # Single ellipsis character
        text = re.sub(r"\.{2,}", "", text)  # Remove 2 or more consecutive dots

    return text


def remove_punctuations(text: str):
    punctuations = ["!", ",", ".", ";", "?"]
    for punctuation in punctuations:
        text = text.replace(punctuation, "")

    return text


def normalize_spacing_around_punc(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # Remove space before punctuation
    return re.sub(
        r"([.,!?;:])([A-Za-z])", r"\1 \2", text
    )  # Add space after punctuation


def capitalize_first_character(text: str) -> str:
    """Ensures first character is capitalized"""
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def capitalize_after_punc(text: str) -> str:
    """Fix capitalization after sentence-ending punctuation [. ! ?]"""
    return re.sub(
        r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), text
    )


def capitalize_uz_domain(text: str) -> str:
    """Normalize Uzbek website/domain names. Capitalize first letter of domain names (Qalampir uz)"""
    return re.sub(
        r"\b([A-Za-z]+)\.uz\b",
        lambda m: m.group(1).capitalize() + " uz",
        text,
        flags=re.IGNORECASE,
    )


def normalize_capitalization(text: str, normalize_domains=True) -> str:
    """Capitalizes first character, after punctuations [. ! ?] and Uzbek domain"""
    text = capitalize_first_character(text)
    text = capitalize_after_punc(text)
    if normalize_domains:
        text = capitalize_uz_domain(text)
    return text
