import math
import re
from typing import AnyStr


def _get_word_from_char_position(text, current_char_at):
    # Find the start of the word
    start = current_char_at
    while start > 0 and text[start - 1].isalnum():
        start -= 1

    # Find the end of the word
    end = current_char_at
    while end < len(text) and text[end].isalnum():
        end += 1

    return text[start:end]


def _get_case_aware_replacement(
    text: str, to_be_replaced_char_at: int, replacement: str
):
    length = len(text) - 1
    assert to_be_replaced_char_at <= length

    char_to_be_replaced = text[to_be_replaced_char_at]

    word = _get_word_from_char_position(text, to_be_replaced_char_at)

    # Check if the word is all capitalized
    # АҚШ -> AQSH (not AQSh)
    if word.isupper():
        return replacement.upper()

    # Check if just the character to be replaced is capitalized
    # Шавкат -> Shavkat (not SHavkat)
    if char_to_be_replaced.isupper():
        return replacement[0].upper() + replacement[1:]

    return replacement


def _make_large_number_ordinary(
    pattern: re.Pattern[AnyStr], text: str, threshold: int = 300
):
    def replace_func(match):
        # Extract groups from the match
        full_match = match.group(0)
        number_str = match.group(1)
        suffix = match.group(2)

        try:
            # Handle ranges (e.g., "400-450") by splitting on the hyphen
            first_num = int(number_str.split("-")[0])

            if first_num > threshold:
                # Return the replacement format: number-suffix
                return f"{number_str}-{suffix}"

            # If below threshold, return the original matched text
            return full_match

        except (ValueError, IndexError):
            # Fallback if the group 1 isn't a valid integer
            return full_match

    # Apply the logic to all occurrences in the text
    return re.sub(pattern, replace_func, text)


def _to_latin_handle_dates(cyrillic_text):
    """
    Adds hyphen to years and dates in cyrillic text (represents ordinary numbers).
    Handles most of the cases correctly with rare exceptions. Check test cases.

    Example:
        Ўзбекистон 1991 йил 31 августда ўз мустақиллигини эълон қилди. -> Ўзбекистон 1991-йил 31-августда ўз мустақиллигини эълон қилди.

    :param cyrillic_text:
    :return cyrillic_text:
    """
    # Pre-compile patterns to optimize the performance

    # --- 1. Years (4-digit or ranges) ---
    # Covers: 1990 йил/йилларда/yilning/..., 1991-1995 йилларда
    # Misses (less likely): 1400 йил олдин
    four_digit_year_pattern = re.compile(
        r"\b(\d{4}(?:-\d{4})?)[\s-]*(йил[a-zA-Z]*)",
        flags=re.IGNORECASE,
    )
    cyrillic_text = re.sub(four_digit_year_pattern, r"\1-\2", cyrillic_text)

    # --- 2. Years (shortened 2-digit) ---
    # Covers: 80 йиллар бошида, 90 йилларда
    # Maybe misses cases that end with йиллар* and not ordinary (can't find such case though)
    cyrillic_two_digit_year_pattern = re.compile(
        r"\b(\d{2}(?:-\d{2})?)[\s-]*(йиллар[a-zA-Z]*)",
        flags=re.IGNORECASE,
    )
    cyrillic_text = re.sub(cyrillic_two_digit_year_pattern, r"\1-\2", cyrillic_text)

    # --- 3. Historical Context (милодий/асрнинг) ---
    # Covers: милодий 70 йилда, XIV асрнинг 50-60 йилларида
    up_to_three_digit_year_pattern = re.compile(
        r"\b(милоддан аввалги|милодий|асрнинг)\s+(\d{1,3}(?:-\d{1,3})?)[\s-]*(йил[a-zA-Z]*)",
        flags=re.IGNORECASE,
    )
    cyrillic_text = re.sub(up_to_three_digit_year_pattern, r"\1 \2-\3", cyrillic_text)

    # --- 4. Larger three-digit numbers (later handled with threshold: >300) ---
    # Covers: 722 йилда, 700-720 йилларда
    # Misses (less likely): Ўзбекистон тарихи 400 йилда or ordinary numbers less than threshold
    three_digit_year_pattern = re.compile(
        r"\b(\d{3}(?:-\d{3})?)\s*(йилда|йиллардаги|йилларда|йилнинг|йилги)\b",
        flags=re.IGNORECASE,
    )
    cyrillic_text = _make_large_number_ordinary(
        three_digit_year_pattern,
        cyrillic_text,
        threshold=300,
    )

    # --- Dates ---
    # Covers: 7 октябрь, 17-октябрда
    # omit 'ь' since adding any suffix will remove that: октябрь + да = октябрда
    dates_pattern = re.compile(
        r"\b(\d{1,2})[\s-]*((?:январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*)",
        flags=re.IGNORECASE,
    )
    cyrillic_text = re.sub(
        dates_pattern,
        r"\1-\2",
        cyrillic_text,
    )

    return cyrillic_text


def _to_latin_handle_name_initials(cyrillic_text):
    shortened_name_initials_pattern = re.compile(
        r"([А-ЯЁҚҒҲЎ])\.\s*(?:([А-ЯЁҚҒҲЎ])\.\s*)?[А-ЯЁҚҒҲЎ][а-яёқғҳў]+"  # Initials first (Я.Е. Қурбонов)
        r"|"  # OR
        r"[А-ЯЁҚҒҲЎ][а-яёқғҳў]+\s+([А-ЯЁҚҒҲЎ])\.\s*(?:([А-ЯЁҚҒҲЎ])\.)?"  # Surname first (Қурбонов Я.Е.)
    )

    matches = list(re.finditer(shortened_name_initials_pattern, cyrillic_text))

    y_letters = {"Я", "Ю", "Е", "Ё"}
    # Process matches in reverse order (right-to-left).
    # This prevents string indices from shifting when replacements
    # change the string length (e.g., 'Ч.' becoming 'Ch.').
    for match in reversed(matches):
        i = 1
        for matched_initial in match.groups():
            if matched_initial in y_letters:
                cyrillic_text = (
                    cyrillic_text[: match.start(i)]
                    + "Й"
                    + cyrillic_text[match.end(i) :]
                )
            # Let it process here otherwise _get_word_from_char_position("Чолқуши - Ч.Абдуллаев", 10) is returning 'Ч'
            # which cause the _get_case_aware_replacement to return 'CH' resulting at the end 'Cholqushi - CH.Abdullayev'
            elif matched_initial == "Ч":
                cyrillic_text = (
                    cyrillic_text[: match.start(i)]
                    + "Ch"
                    + cyrillic_text[match.end(i) :]
                )
            elif matched_initial == "Ш":
                cyrillic_text = (
                    cyrillic_text[: match.start(i)]
                    + "Sh"
                    + cyrillic_text[match.end(i) :]
                )
            i = i + 1

    return cyrillic_text


def _to_latin_handle_ғ_to_қ_assimilation(cyrillic_text):
    pattern = re.compile(
        r"\b(белбоққа|боққа|боғ-роққа|гулбоққа|буққа|доққа|тоққа|тиққа|сарёққа|ёруққа|маблаққа|уруққа|чўққа|зоққа)\b",
        flags=re.IGNORECASE,
    )

    def replace_keep_case(match):
        word = match.group(0)
        lower_word = word.lower()
        # "боққа" -> "боғга"
        replacement = re.sub("ққа$", "ғга", lower_word)

        # Handle Title Case (Boqqa -> Bog'ga)
        if word.istitle():
            return replacement.capitalize()
        # Handle Uppercase (BOQQA -> BOG'GA)
        if word.isupper():
            return replacement.upper()
        return replacement

    return re.sub(pattern, replace_keep_case, cyrillic_text)


def _to_latin_handle_в_and_илламоқ_combination(cyrillic_text):
    pattern = re.compile(
        r"\b(ҳувилла|ҳовилла|шувилла|шовилла|чувилла|вовилла|гувилла|ғувилла|ловилла|дувилла|зувилла|увилла)[а-яёқғҳў]*\b",
        flags=re.IGNORECASE,
    )

    def replace_keep_case(match):
        word = match.group(0)
        lower_word = word.lower()
        # "ловилла" -> "ловулла"
        replacement = re.sub("вилла", "вулла", lower_word)

        # Handle Uppercase (ЛОВИЛЛА -> ЛОВУЛЛА)
        if word.isupper():
            return replacement.upper()
        return replacement

    return re.sub(pattern, replace_keep_case, cyrillic_text)


def _to_latin_handle_even_words(cyrillic_text):
    pattern = re.compile(
        r"\b(чўлу биёбон|ору номус|кайфу сафо|шону шуҳрат|чангу ғубор|яхшию ёмонни|еру осмон|еру кўк|меҳнату машаққат|туну кун|кечаю кундуз|ёшу қари|дўсту душман|яккаю ягона)[а-яёқғҳў]*\b",
        flags=re.IGNORECASE,
    )

    def replace_keep_case(match):
        matched_str = match.group(0)
        words = matched_str.split()
        if len(words) != 2:
            print(
                f"Warning: failed to pre-process Cyrillic even word: {matched_str}. Expected number of space is 2 but got {len(words)}."
            )
            return matched_str

        # "еру осмон" -> "ер-у осмон"
        first_word = words[0]
        hyphenated_first_word = first_word[:-1] + "-" + first_word[-1]
        return hyphenated_first_word + " " + words[1]

    return re.sub(pattern, replace_keep_case, cyrillic_text)


def _to_latin_handle_even_words_not_hyphenated(cyrillic_text):
    # Hyphenated '...дан-...га' words, e.g. йилдан-йилга -> йилдан йилга
    pattern = re.compile(
        r"\b([а-яёқғҳў]+дан)-([а-яёқғҳў]+га)\b",
        flags=re.IGNORECASE,
    )
    cyrillic_text = re.sub(pattern, r"\1 \2", cyrillic_text)

    pattern = re.compile(
        r"\b(у ёқдан-бу ёққа|кўпдан-кўп|кундан-кун|тўғридан-тўғри|узундан-узоқ|узундан-узун|очиқдан-очиқ|қизигандан-қизиди|камдан-кам|текиндан-текин|йангидан-янги)\b",
        flags=re.IGNORECASE,
    )

    def replace_keep_case(match):
        matched_str = match.group(0)
        words = matched_str.split("-")
        if len(words) != 2:
            print(
                f"Warning: failed to pre-process Cyrillic even word: {matched_str}. Expected number of space is 2 but got {len(words)}."
            )
            return matched_str

        # кундан-кун -> кундан кун
        return f"{words[0]} {words[1]}"

    return re.sub(pattern, replace_keep_case, cyrillic_text)


def _to_latin_handle_foreign_words(cyrillic_text):
    # foreign word stem + 'и' -> foreign word stem + 'йи'
    # e.g. таржимаи ҳол -> tarjimayi hol: таржима + и => таржима + йи
    foreign_word_stems = [
        "нуқта",
        "таржима",
        "ойна",
        "бало",
        "адо",
        "аъзо",
        "расво",
        "дуо",
        "бало",
        "парво",
        "бухоро",
        "худо",
        "қазо",
        "авзо",
        "саҳро",
        "фидо",
    ]

    pattern = re.compile(
        rf"\b({"|".join(foreign_word_stems)})(и)\b",
        flags=re.IGNORECASE,
    )

    def replace_keep_case(match):
        stem = match.group(1)  # "Таржима"
        suffix = match.group(2)  # "и"

        # Determine if we should use 'й' or 'Й' based on the suffix's case
        replacement_char = "й" if suffix.islower() else "Й"

        return stem + replacement_char + suffix

    return re.sub(pattern, replace_keep_case, cyrillic_text)


def _to_latin_pre_processing(cyrillic_text):
    """
    Applies grammatical and orthographic normalization rules to Cyrillic text
    before converting it to Latin script.
    """
    # --- Rule 2: Handle dates as ordinary number ---
    cyrillic_text = _to_latin_handle_dates(cyrillic_text)

    # --- Rule 7: Handle shortened names with Я, Ю, Е, Ё, Ч, Ш initial letters ---
    # Covers: Я.Қурбонов, Ю. Давидова, А.Е.Қурбонов, Саъдиев Ё.
    # Misses: completely anonymized cases (Я.Е.) as it can be abbreviation too
    cyrillic_text = _to_latin_handle_name_initials(cyrillic_text)

    # --- Rule 11: Words ending with 'ғ' + -га = ққа ---
    # Example: In Cyrillic - боғ + га = боққа, in Latin it should be bog‘ga
    cyrillic_text = _to_latin_handle_ғ_to_қ_assimilation(cyrillic_text)

    # --- Rule 12: Words ending with 'в' + -илламоқ = -vullamoq (in Latin) ---
    # Example: In Cyrillic - шов + илламоқ = шовилламоқ, in Latin it should be shovullamoq
    cyrillic_text = _to_latin_handle_в_and_илламоқ_combination(cyrillic_text)

    # --- Rule 14: Even words are hyphenated in Latin ---
    # Example: In Cyrillic - 'еру осмон', in Latin it should be yer-u osmon
    cyrillic_text = _to_latin_handle_even_words(cyrillic_text)

    # --- Rule 15: Two words not hyphenated in Latin (first word + dan, second word + ga and some others) ---
    # Example: In Cyrillic - 'йилдан-йилга', 'бекордан-бекорга', in Latin it should be yildan yilga, bekordan-bekorga
    # Shuningdek, belgining ortiq darajasini bildiruvchi yangidan yangi, ochiqdan ochiq kabilar ajratib (chiziqchasiz) yoziladi.
    cyrillic_text = _to_latin_handle_even_words_not_hyphenated(cyrillic_text)

    # --- Rule 16: Izofali so'zlar unli bilan tugasa (таржимаи ҳол -> tarjimayi hol) ---
    # Izofa undosh bilan tugasa - 'дарди бедаво' -> 'dardi bedavo' kiril va lotinda o'zgarishsiz qoladi
    # Izofa unli bilan tugasa - 'адои тамом' -> 'adoyi tamom' - lotinda izofaga -yi qo'shimchasi qo'shiladi
    cyrillic_text = _to_latin_handle_foreign_words(cyrillic_text)

    # --- Rule 17: Fonetik yozuv -> morfologik yozuv (эрталабки -> ertalabgi). Ignored this rule (seems incorrect or no one is following) ---

    return cyrillic_text


def to_latin(cyrillic_text, normalize_apostrophes=False):
    """
    Transliterate Uzbek Cyrillic to Latin

    Handles common conversion nuances based on
    https://doi.org/10.5281/zenodo.7467371 (Yunus Jummayevich Davidov, 2022)

    Ignored rules: 13 and 17
    """
    # Check for NaN or empty string safely
    is_nan = isinstance(cyrillic_text, float) and math.isnan(cyrillic_text)
    if cyrillic_text is None or is_nan or str(cyrillic_text).strip() == "":
        return ""
    cyrillic_text = str(cyrillic_text)

    # TODO: Normalize apostrophe

    text = _to_latin_pre_processing(cyrillic_text)

    vowels = "аоиуўэёюяеАОИУЎЭЁЮЯЕ"

    # Base Mapping Dictionary
    # Note: 'ц', 'е', and 'ъ' are handled via logic
    mapping = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "ё": "yo",  # Rule 6
        "ж": "j",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "x",
        "ч": "ch",
        "ш": "sh",
        "щ": "sh",
        "ъ": "’",
        "ы": "i",
        "ь": "",  # Rule 5: ignore for the most part
        "э": "e",
        "ю": "yu",  # Rule 6
        "я": "ya",  # Rule 6
        "ў": "o‘",
        "ғ": "g‘",
        "қ": "q",
        "ҳ": "h",
    }

    result = []
    length = len(text)

    for i in range(length):
        char = text[i]
        lower_char = char.lower()

        # Get previous and next characters to determine context
        prev_char_lower = text[i - 1].lower() if i > 0 else None
        next_char_lower = text[i + 1].lower() if i + 1 < length else None

        is_word_start = (i == 0) or (not text[i - 1].isalnum())
        is_word_end = (i == length - 1) or (not text[i + 1].isalnum())
        is_after_vowel = prev_char_lower and prev_char_lower in vowels

        y_letters = ("я", "ю", "е", "ё")

        # --- Rule 1: Handle 'ў / ғ' + 'ъ' (not apostrophe) ---
        if lower_char == "ъ" and (prev_char_lower == "ў" or prev_char_lower == "ғ"):
            replacement = ""
            result.append(replacement)

        # --- Rule 4: Handle 'ъ' + 'я / ю / е / ё' (not apostrophe) ---
        elif lower_char == "ъ" and (next_char_lower in y_letters):
            # Capitalization
            replacement = _get_case_aware_replacement(text, i, "y")
            result.append(replacement)

        # --- Rule 3: Handle 'е' (ye vs e) ---
        # At start of word or after a vowel or 'ь' - 'ye', otherwise 'e'
        elif lower_char == "е":
            is_after_ь = prev_char_lower and prev_char_lower == "ь"

            replacement = (
                "ye" if (is_word_start or is_after_vowel or is_after_ь) else "e"
            )
            replacement = _get_case_aware_replacement(text, i, replacement)
            result.append(replacement)

        # --- Rule 5: Handle 'ь' + 'о' => 'y' (else ignore) ---
        elif lower_char == "ь" and (next_char_lower and next_char_lower == "о"):
            replacement = _get_case_aware_replacement(text, i, "y")
            result.append(replacement)

        # --- Rule 8: Handle 'ц' (ts vs s) ---
        # After a vowel - 'ts', 's' otherwise (after consonant, start and end of a word)
        elif lower_char == "ц":
            replacement = "ts" if is_after_vowel and not is_word_end else "s"
            replacement = _get_case_aware_replacement(text, i, replacement)
            result.append(replacement)

        # --- Handle Standard Mapping ---
        elif lower_char in mapping:
            replacement = mapping[lower_char]
            replacement = _get_case_aware_replacement(text, i, replacement)

            result.append(replacement)
        else:
            # Keep punctuation, numbers, and spaces as is
            result.append(char)

    latin_text = "".join(result)

    return latin_text
