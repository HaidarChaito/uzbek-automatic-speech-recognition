import unittest

from scripts.uzbek_text_normalizer import (
    remove_whitespaces,
    remove_list_markers,
    remove_new_lines,
    normalize_uzbek_apostrophes,
    normalize_annotations,
    normalize_spacing_around_punc,
    capitalize_first_character,
    capitalize_after_punc,
    capitalize_uz_domain,
    normalize_capitalization,
    remove_quotes_and_colons,
)


class TestRemoveNewLines(unittest.TestCase):
    def test_removes_newline(self):
        self.assertEqual(remove_new_lines("hello\nworld"), "hello world")

    def test_removes_carriage_return(self):
        self.assertEqual(remove_new_lines("hello\rworld"), "hello world")

    def test_removes_both_newline_and_carriage_return(self):
        self.assertEqual(remove_new_lines("hello\r\nworld"), "hello  world")

    def test_multiple_newlines(self):
        self.assertEqual(remove_new_lines("hello\n\n\nworld"), "hello   world")

    def test_no_newlines(self):
        self.assertEqual(remove_new_lines("hello world"), "hello world")


class TestListMarkers(unittest.TestCase):
    def test_remove_starting_bullet_point(self):
        self.assertEqual(remove_list_markers("• hello world"), "hello world")

    def test_remove_starting_bullet_point_without_space(self):
        self.assertEqual(remove_list_markers("•hello world"), "hello world")

    def test_remove_multiple_bullet_points(self):
        self.assertEqual(
            remove_list_markers("◦ hello\n  • world ▫"), "hello\n    world"
        )

    # Common in books
    def test_remove_multiple_middle_bullet_points(self):
        self.assertEqual(
            remove_list_markers("Non dema! — dedi. — nonni otini atama!"),
            "Non dema!   dedi.   nonni otini atama!",
        )
        self.assertEqual(
            remove_list_markers("Erk – manzilmas, erk – yo‘ldir"),
            "Erk   manzilmas, erk   yo‘ldir",
        )

    def test_remove_starting_numbered_list(self):
        self.assertEqual(remove_list_markers("1. hello world"), "hello world")

    def test_remove_starting_numbered_list_without_space(self):
        self.assertEqual(remove_list_markers("2) hello world"), "hello world")

    def test_handle_multiple_numbered_lists(self):
        self.assertEqual(
            remove_list_markers("1)2) hello 1. world"), "2) hello 1. world"
        )


class TestNormalizeUzbekApostrophes(unittest.TestCase):
    def test_left_single_quotation_mark(self):
        self.assertEqual(normalize_uzbek_apostrophes("o'zbek"), "o'zbek")

    def test_right_single_quotation_mark(self):
        self.assertEqual(normalize_uzbek_apostrophes("o'zbek"), "o'zbek")

    def test_modifier_letter_apostrophe(self):
        self.assertEqual(normalize_uzbek_apostrophes("oʼzbek"), "o'zbek")

    def test_modifier_letter_turned_comma(self):
        self.assertEqual(normalize_uzbek_apostrophes("oʻzbek"), "o'zbek")

    def test_modifier_letter_reversed_comma(self):
        self.assertEqual(normalize_uzbek_apostrophes("oʽzbek"), "o'zbek")

    def test_grave_accent(self):
        self.assertEqual(normalize_uzbek_apostrophes("o`zbek"), "o'zbek")

    def test_modifier_letter_acute_accent(self):
        self.assertEqual(normalize_uzbek_apostrophes("oˊzbek"), "o'zbek")

    def test_modifier_letter_grave_accent(self):
        self.assertEqual(normalize_uzbek_apostrophes("oˋzbek"), "o'zbek")

    def test_multiple_apostrophes(self):
        self.assertEqual(
            normalize_uzbek_apostrophes("o‘zbek maʼrifatini ta’minlashda"),
            "o'zbek ma'rifatini ta'minlashda",
        )

    def test_mixed_apostrophe_variants(self):
        self.assertEqual(normalize_uzbek_apostrophes("o'zbek tilʼi"), "o'zbek til'i")

    def test_no_apostrophes(self):
        self.assertEqual(normalize_uzbek_apostrophes("salom"), "salom")


class TestRemoveQuotesAndColons(unittest.TestCase):
    def test_double_quotation_mark1(self):
        self.assertEqual(
            remove_quotes_and_colons('Uning oldida "tirbandlik" paydo bo\'ldi'),
            "Uning oldida tirbandlik paydo bo'ldi",
        )

    def test_double_quotation_mark2(self):
        self.assertEqual(
            remove_quotes_and_colons("“Ahmoq!” o'yladim men. Kechikdi!"),
            "Ahmoq! o'yladim men. Kechikdi!",
        )
        self.assertEqual(
            remove_quotes_and_colons("U kelmadi. “Ahmoq! o'yladim men. Kechikdi!"),
            "U kelmadi. Ahmoq! o'yladim men. Kechikdi!",
        )

    def test_double_quotation_mark3(self):
        self.assertEqual(
            remove_quotes_and_colons("«Hovlini to'ldirib mevali daraxt ekdik, dedi."),
            "Hovlini to'ldirib mevali daraxt ekdik, dedi.",
        )
        self.assertEqual(
            remove_quotes_and_colons("Ular bunday sharoitda ma'nan «so'nib» qolishadi"),
            "Ular bunday sharoitda ma'nan so'nib qolishadi",
        )

    def test_no_apostrophes(self):
        self.assertEqual(
            remove_quotes_and_colons(
                "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi"
            ),
            "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi",
        )

    def test_remove_colons(self):
        self.assertEqual(
            remove_quotes_and_colons("Kitob haqida ma'lumot:"), "Kitob haqida ma'lumot"
        )
        self.assertEqual(
            remove_quotes_and_colons(
                "Birdan Stalin: «Lavrentiy, chora ko'r», deb qolardi.:"
            ),
            "Birdan Stalin Lavrentiy, chora ko'r, deb qolardi.",
        )


class TestNormalizeAnnotations(unittest.TestCase):
    def test_parentheses_to_brackets(self):
        self.assertEqual(
            normalize_annotations("hello (noise) world"), "hello [noise] world"
        )

    def test_asterisk_brackets_to_brackets(self):
        self.assertEqual(
            normalize_annotations("hello *[music]* world"), "hello [music] world"
        )

    def test_backslash_brackets_to_brackets(self):
        self.assertEqual(
            normalize_annotations("hello \\[laughter] world"), "hello [laughter] world"
        )

    def test_normalizes_spacing_in_brackets(self):
        self.assertEqual(
            normalize_annotations("hello [ noise ] world"), "hello [noise] world"
        )

    def test_normalizes_asterisk_and_spacing_in_brackets(self):
        self.assertEqual(
            normalize_annotations("hello *[ noise ]* world"), "hello [noise] world"
        )

    def test_normalizes_asterisk_and_backslash_brackets(self):
        self.assertEqual(
            normalize_annotations("hello *\\[noise]* world"), "hello *[noise]* world"
        )

    def test_lowercases_annotation_content(self):
        self.assertEqual(
            normalize_annotations("hello (NOISE) World"), "hello [noise] World"
        )
        self.assertEqual(
            normalize_annotations("Hello [ Noise ] world"), "Hello [noise] world"
        )

    def test_preserves_annotation_content_casing(self):
        self.assertEqual(
            normalize_annotations("hello (NOISE) World", lowercase_annotation=False),
            "hello [NOISE] World",
        )
        self.assertEqual(
            normalize_annotations("Hello [ Noise ] world", lowercase_annotation=False),
            "Hello [Noise] world",
        )

    def test_multiple_annotations(self):
        self.assertEqual(
            normalize_annotations("(music) hello *[noise]* world"),
            "[music] hello [noise] world",
        )

    def test_no_annotations(self):
        self.assertEqual(normalize_annotations("hello world"), "hello world")
        self.assertEqual(
            normalize_annotations("Tur, Mansur ketamiz. vaqt ketgani qoldi."),
            "Tur, Mansur ketamiz. vaqt ketgani qoldi.",
        )


class TestCleanWhitespaces(unittest.TestCase):
    def test_removes_leading_whitespace(self):
        self.assertEqual(remove_whitespaces("  hello"), "hello")

    def test_removes_trailing_whitespace(self):
        self.assertEqual(remove_whitespaces("hello  "), "hello")

    def test_removes_multiple_spaces(self):
        self.assertEqual(remove_whitespaces("hello    world"), "hello world")

    def test_handles_tabs_and_spaces(self):
        self.assertEqual(remove_whitespaces("hello\t\tworld"), "hello world")

    def test_empty_string(self):
        self.assertEqual(remove_whitespaces(""), "")

    def test_only_whitespace(self):
        self.assertEqual(remove_whitespaces("   \t  "), "")


class TestNormalizeSpacingAroundPunc(unittest.TestCase):
    def test_removes_space_before_period(self):
        self.assertEqual(normalize_spacing_around_punc("hello ."), "hello.")

    def test_removes_space_before_comma(self):
        self.assertEqual(normalize_spacing_around_punc("hello , world"), "hello, world")

    def test_removes_space_before_multiple_punctuation(self):
        self.assertEqual(
            normalize_spacing_around_punc("hello ! world ?"), "hello! world?"
        )

    def test_adds_space_after_period(self):
        self.assertEqual(normalize_spacing_around_punc("hello.world"), "hello. world")

    def test_adds_space_after_comma(self):
        self.assertEqual(normalize_spacing_around_punc("hello,world"), "hello, world")

    def test_handles_correct_spacing(self):
        self.assertEqual(
            normalize_spacing_around_punc("hello, world."), "hello, world."
        )

    def test_multiple_punctuation_fixes(self):
        self.assertEqual(
            normalize_spacing_around_punc("hello ,world .test"), "hello, world. test"
        )

    def test_no_punctuation(self):
        self.assertEqual(normalize_spacing_around_punc("hello world"), "hello world")


class TestCapitalizeAfterPunc(unittest.TestCase):
    def test_capitalizes_after_period(self):
        result = capitalize_after_punc(". hello world")
        self.assertEqual(result, ". Hello world")

    def test_capitalizes_after_exclamation(self):
        result = capitalize_after_punc(" ! hello world")
        self.assertEqual(result, " ! Hello world")

    def test_capitalizes_after_question_mark(self):
        result = capitalize_after_punc("? hello world")
        self.assertEqual(result, "? Hello world")

    def test_no_match_returns_original(self):
        self.assertEqual(capitalize_after_punc("hello. world"), "hello. World")

    def test_already_capitalized(self):
        result = capitalize_after_punc(". Hello world")
        self.assertEqual(result, ". Hello world")


class TestCapitalizeFirstCharacter(unittest.TestCase):
    def test_capitalizes_lowercase_first_letter(self):
        self.assertEqual(capitalize_first_character("hello"), "Hello")

    def test_keeps_uppercase_first_letter(self):
        self.assertEqual(capitalize_first_character("Hello"), "Hello")

    def test_handles_empty_string(self):
        self.assertEqual(capitalize_first_character(""), "")

    def test_handles_single_character(self):
        self.assertEqual(capitalize_first_character("a"), "A")

    def test_only_capitalizes_first_character(self):
        self.assertEqual(capitalize_first_character("hello WORLD"), "Hello WORLD")

    def test_handles_non_alphabetic_first_character(self):
        self.assertEqual(capitalize_first_character("123 hello"), "123 hello")


class TestCapitalizeUzDomain(unittest.TestCase):
    def test_capitalizes_uz_domain(self):
        result = capitalize_uz_domain("qalampir.uz")
        self.assertEqual(result, "Qalampir uz")

    def test_handles_uppercase_domain(self):
        result = capitalize_uz_domain("QALAMPIR.UZ")
        self.assertEqual(result, "Qalampir uz")

    def test_no_uz_domain_returns_original(self):
        self.assertEqual(capitalize_uz_domain("hello world"), "hello world")

    def test_mixed_case_domain(self):
        result = capitalize_uz_domain("QaLaMpIr.uz")
        self.assertEqual(result, "Qalampir uz")


class TestNormalizeCapitalization(unittest.TestCase):
    def test_capitalizes_first_character(self):
        result = normalize_capitalization("hello world", normalize_domains=False)
        self.assertEqual(result, "Hello world")

    def test_with_uz_domain_normalization(self):
        result = normalize_capitalization("qalampir.uz sayti", normalize_domains=True)
        self.assertEqual(result, "Qalampir uz sayti")

    def test_without_uz_domain_normalization(self):
        result = normalize_capitalization("qalampir.com sayti", normalize_domains=False)
        self.assertEqual(result, "Qalampir.com sayti")

    def test_empty_string(self):
        result = normalize_capitalization("", normalize_domains=False)
        self.assertEqual(result, "")

    def test_capitalization_after_comma_should_not_change(self):
        result = normalize_capitalization(
            "– Tur, Mansur ketamiz.", normalize_domains=True
        )
        self.assertEqual(result, "– Tur, Mansur ketamiz.")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""

    def test_full_normalization_pipeline(self):
        text = "— Muttasil o‘qib, kamolga intilmoq zarur!  men   o'qiyman\n\n\n(noise)  \nErk – manzilmas, erk – yo‘ldir"

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = remove_quotes_and_colons(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = normalize_capitalization(text)

        self.assertEqual(
            text,
            "Muttasil o'qib, kamolga intilmoq zarur! Men o'qiyman [noise] Erk manzilmas, erk yo'ldir",
        )

    def test_capitalization_pipeline(self):
        text = "– Tur, Mansur ketamiz! vaqt ketgani qoldi. odam bo'lmas ekan bu."

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = remove_quotes_and_colons(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = normalize_capitalization(text)

        self.assertEqual(
            text,
            "Tur, Mansur ketamiz! Vaqt ketgani qoldi. Odam bo'lmas ekan bu.",
        )

    def test_uzbek_text_with_various_apostrophes(self):
        text = "o'zbek tilʼi va oʻzbekiston"
        text = normalize_uzbek_apostrophes(text)

        self.assertEqual(text, "o'zbek til'i va o'zbekiston")

    def test_messy_transcription(self):
        text = "   salom  ,  dunyo   !   qalaysan   ?  "
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_first_character(text)

        self.assertEqual(text, "Salom, dunyo! qalaysan?")


if __name__ == "__main__":
    unittest.main()
