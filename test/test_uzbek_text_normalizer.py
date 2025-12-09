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
    remove_special_chars,
    remove_punctuations,
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


class TestRemoveSpecialChars(unittest.TestCase):
    def test_double_quotation_mark1(self):
        self.assertEqual(
            remove_special_chars('Uning oldida "tirbandlik" paydo bo\'ldi'),
            "Uning oldida tirbandlik paydo bo'ldi",
        )

    def test_double_quotation_mark2(self):
        self.assertEqual(
            remove_special_chars("“Ahmoq!” o'yladim men. Kechikdi!"),
            "Ahmoq! o'yladim men. Kechikdi!",
        )
        self.assertEqual(
            remove_special_chars("U kelmadi. “Ahmoq! o'yladim men. Kechikdi!"),
            "U kelmadi. Ahmoq! o'yladim men. Kechikdi!",
        )

    def test_double_quotation_mark3(self):
        self.assertEqual(
            remove_special_chars("«Hovlini to'ldirib mevali daraxt ekdik, dedi."),
            "Hovlini to'ldirib mevali daraxt ekdik, dedi.",
        )
        self.assertEqual(
            remove_special_chars("Ular bunday sharoitda ma'nan «so'nib» qolishadi"),
            "Ular bunday sharoitda ma'nan so'nib qolishadi",
        )

    def test_no_apostrophes(self):
        self.assertEqual(
            remove_special_chars(
                "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi"
            ),
            "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi",
        )

    def test_remove_colons(self):
        self.assertEqual(
            remove_special_chars("Kitob haqida ma'lumot:"), "Kitob haqida ma'lumot"
        )
        self.assertEqual(
            remove_special_chars(
                "Birdan Stalin: «Lavrentiy, chora ko'r», deb qolardi.:"
            ),
            "Birdan Stalin Lavrentiy, chora ko'r, deb qolardi.",
        )

    def test_remove_2dots(self):
        self.assertEqual(
            remove_special_chars(
                "Unda, sen chinovniklarni..! Chinovniklar to'nka!", remove_ellipsis=True
            ),
            "Unda, sen chinovniklarni! Chinovniklar to'nka!",
        )
        self.assertEqual(
            remove_special_chars("O'qimaysizmi?..", remove_ellipsis=True),
            "O'qimaysizmi?",
        )
        self.assertEqual(
            remove_special_chars(
                "Voy esim qursin.. Bug'doy nima bo'ldi...", remove_ellipsis=True
            ),
            "Voy esim qursin Bug'doy nima bo'ldi",
        )

    def test_remove_ellipses(self):
        self.assertEqual(
            remove_special_chars(
                "O'lsam, mozorimni kungayda qazing…", remove_ellipsis=True
            ),
            "O'lsam, mozorimni kungayda qazing",
        )
        self.assertEqual(
            remove_special_chars(
                "Yo'q, qoldi… u qoldi men bilan qoldi…", remove_ellipsis=True
            ),
            "Yo'q, qoldi u qoldi men bilan qoldi",
        )
        self.assertEqual(
            remove_special_chars(
                "Bundan tashqari... Biz oz emas-ko'p emas millionlab onalarni ularning orasida yosh kelinchaklar",
                remove_ellipsis=True,
            ),
            "Bundan tashqari Biz oz emas-ko'p emas millionlab onalarni ularning orasida yosh kelinchaklar",
        )
        self.assertEqual(
            remove_special_chars(
                "Ro'moli burchini ushlab, televizorga...", remove_ellipsis=True
            ),
            "Ro'moli burchini ushlab, televizorga",
        )
        self.assertEqual(
            remove_special_chars(
                "Kel... Deb xotirjamlik bilan gapirish kerak....", remove_ellipsis=True
            ),
            "Kel Deb xotirjamlik bilan gapirish kerak",
        )

    def test_not_remove_ellipses(self):
        self.assertEqual(
            remove_special_chars(
                "O'lsam, mozorimni kungayda qazing…", remove_ellipsis=False
            ),
            "O'lsam, mozorimni kungayda qazing…",
        )
        self.assertEqual(
            remove_special_chars(
                "Yo'q, qoldi… u qoldi men bilan qoldi…", remove_ellipsis=False
            ),
            "Yo'q, qoldi… u qoldi men bilan qoldi…",
        )
        self.assertEqual(
            remove_special_chars(
                "Kel.. Deb xotirjamlik bilan gapirish kerak....", remove_ellipsis=False
            ),
            "Kel.. Deb xotirjamlik bilan gapirish kerak....",
        )


class TestRemovePunctuations(unittest.TestCase):
    def test_question_mark(self):
        self.assertEqual(
            remove_punctuations('Uning oldida "tirbandlik" paydo bo\'ldimi?'),
            'Uning oldida "tirbandlik" paydo bo\'ldimi',
        )
        self.assertEqual(
            remove_punctuations("Nimalar deyapsan? Esing joyidami?"),
            "Nimalar deyapsan Esing joyidami",
        )
        self.assertEqual(
            remove_punctuations('"Musulmonman", deb namoz o\'qimaysanmi?!'),
            '"Musulmonman" deb namoz o\'qimaysanmi',
        )

    def test_exclamation_mark(self):
        self.assertEqual(
            remove_punctuations("“Ahmoq!” o'yladim men. Kechikdi!"),
            "“Ahmoq” o'yladim men Kechikdi",
        )
        self.assertEqual(
            remove_punctuations("To'xta! Ketma!!!"),
            "To'xta Ketma",
        )

    def test_comma(self):
        self.assertEqual(
            remove_punctuations("«Hovlini to'ldirib mevali daraxt ekdik, dedi."),
            "«Hovlini to'ldirib mevali daraxt ekdik dedi",
        )
        self.assertEqual(
            remove_punctuations("O'g'lim, ovqatingni yemaysanmi?"),
            "O'g'lim ovqatingni yemaysanmi",
        )

    def test_full_stop(self):
        self.assertEqual(
            remove_punctuations("Bir nechta. nuqtali gap. tekshiruv."),
            "Bir nechta nuqtali gap tekshiruv",
        )
        self.assertEqual(
            remove_punctuations("Uyga kirsam, menga baqrayib turar edi."),
            "Uyga kirsam menga baqrayib turar edi",
        )

    def test_semi_colon(self):
        self.assertEqual(
            remove_punctuations("Bir nechta. nuqtali gap; tekshiruv."),
            "Bir nechta nuqtali gap tekshiruv",
        )
        self.assertEqual(
            remove_punctuations("Uyga kirsam; menga baqrayib turar edi."),
            "Uyga kirsam menga baqrayib turar edi",
        )

    def test_no_punctuations(self):
        self.assertEqual(
            remove_punctuations(
                "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi"
            ),
            "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi",
        )
        self.assertEqual(
            remove_punctuations("O'g'lim ovqatingni yemaysanmi"),
            "O'g'lim ovqatingni yemaysanmi",
        )
        self.assertEqual(
            remove_punctuations(
                'Андижонда тандирлар "ҳавони ифлослантиряпти" деб топилди'
            ),
            'Андижонда тандирлар "ҳавони ифлослантиряпти" деб топилди',
        )

    def test_multiple_punctuations(self):
        self.assertEqual(
            remove_punctuations('"Ketma..." deya yalindi..'),
            '"Ketma" deya yalindi',
        )
        self.assertEqual(
            remove_punctuations("Uyga kirsam,,, menga baqrayib turar edi!!!"),
            "Uyga kirsam menga baqrayib turar edi",
        )
        self.assertEqual(
            remove_punctuations("Nimalar deyapsan? Esing joyidami???"),
            "Nimalar deyapsan Esing joyidami",
        )
        self.assertEqual(
            remove_punctuations("Nimalar deyapsan? Seni esing joyida emas?!?!"),
            "Nimalar deyapsan Seni esing joyida emas",
        )
        self.assertEqual(
            remove_punctuations('"Ketma.;" deya yalindi..'),
            '"Ketma" deya yalindi',
        )


class TestNormalizeAnnotations(unittest.TestCase):
    def test_parentheses_to_brackets(self):
        self.assertEqual(
            normalize_annotations("hello (musiqa) world"), "hello [musiqa] world"
        )

    def test_parentheses_to_no_brackets(self):
        self.assertEqual(
            normalize_annotations("hello (whatever) world"), "hello (whatever) world"
        )
        self.assertEqual(
            normalize_annotations("hello (NOISE) World"), "hello (NOISE) World"
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

    def test_normalizes_spacing_in_brackets_multiple_words(self):
        self.assertEqual(
            normalize_annotations("hello (noise detected) world"),
            "hello (noise detected) world",
        )
        self.assertEqual(
            normalize_annotations("hello [noise detected] world"),
            "hello [noise detected] world",
        )
        self.assertEqual(
            normalize_annotations("hello *[noise detected]* world"),
            "hello *[noise detected]* world",
        )
        self.assertEqual(
            normalize_annotations("hello *[noise_detected]* world"),
            "hello [noise_detected] world",
        )

    def test_normalizes_asterisk_and_backslash_brackets(self):
        self.assertEqual(
            normalize_annotations("hello *\\[noise]* world"), "hello *[noise]* world"
        )

    def test_lowercases_annotation_content(self):
        self.assertEqual(
            normalize_annotations("Hello [ Noise ] world"), "Hello [noise] world"
        )
        self.assertEqual(
            normalize_annotations("Hello [ NOISE ] world"), "Hello [noise] world"
        )

    def test_preserves_annotation_content_casing(self):
        self.assertEqual(
            normalize_annotations("hello (NOISE) World", lowercase_annotation=False),
            "hello (NOISE) World",
        )
        self.assertEqual(
            normalize_annotations("Hello [ Noise ] world", lowercase_annotation=False),
            "Hello [Noise] world",
        )

    def test_multiple_annotations(self):
        self.assertEqual(
            normalize_annotations("(music) hello *[noise]* world"),
            "(music) hello [noise] world",
        )
        self.assertEqual(
            normalize_annotations("[Music] hello *[noise]* world"),
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

    def test_case_insensitive_asr_normalization_pipeline(self):
        text = """
        Rasmiy  xabarda bu ishlar “havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari” deb atalgan. @Ya’ni andijonlik mulozimlarning fikricha, somsapazlar havoni > ifloslantirayotgan ekan. (random Noise) Shuning uchun ularning tandirini buzish kerak ekan.

        Shahrixon tumani hokimligi videoga qo‘shib e’lon qilgan fotojamlanmada tuman #hokimi Hikmatullo Dadaxonov ham aks etgan. ? u buzish ishlariga boshqa mas’ullar bilan birga, shaxsan (oddiy qavs ichida gap) o‘zi bosh-qosh bo‘lib turgan. Voqea katta shov-shuvni keltirib chiqargach, [ Annotatsiya ] video va rasmlar hokimlik kanalidan darhol o‘chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma’lum qildi.  
        """

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_annotations(text)
        text = remove_special_chars(text)
        text = remove_punctuations(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = text.lower()

        self.assertEqual(
            text,
            "rasmiy xabarda bu ishlar havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari deb atalgan ya'ni andijonlik mulozimlarning fikricha somsapazlar havoni ifloslantirayotgan ekan (random noise) shuning uchun ularning tandirini buzish kerak ekan shahrixon tumani hokimligi videoga qo'shib e'lon qilgan fotojamlanmada tuman hokimi hikmatullo dadaxonov ham aks etgan u buzish ishlariga boshqa mas'ullar bilan birga shaxsan (oddiy qavs ichida gap) o'zi bosh-qosh bo'lib turgan voqea katta shov-shuvni keltirib chiqargach [annotatsiya] video va rasmlar hokimlik kanalidan darhol o'chirildi viloyat hokimligi tezda bayonot bilan chiqib dadaxonovga hayfsan berilganini ma'lum qildi",
        )

    def test_case_sensitive_asr_normalization_pipeline(self):
        text = """
        Rasmiy xabarda bu ishlar : “havo ifloslanishining oldini {olish} maqsadidagi reyd tadbirlari” deb atalgan.ya’ni andijonlik mulozimlarning fikricha, somsapazlar havoni ifloslantirayotgan ekan.  shuning uchun ularning tandirini buzish kerak ekan.

        Shahrixon tumani hokimligi videoga qo‘shib e’lon qilgan fotojamlanmada tuman hokimi Hikmatullo Dadaxonov ham aks etgan.U buzish  ishlariga boshqa mas’ullar bilan < birga,shaxsan o‘zi bosh-qosh bo‘lib turgan. Voqea katta shov-shuvni keltirib chiqargach, video va rasmlar hokimlik kanalidan darhol o‘chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma’lum qildi.  
        """

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_annotations(text)
        text = remove_special_chars(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = normalize_capitalization(text)

        self.assertEqual(
            text,
            "Rasmiy xabarda bu ishlar havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari deb atalgan. Ya'ni andijonlik mulozimlarning fikricha, somsapazlar havoni ifloslantirayotgan ekan. Shuning uchun ularning tandirini buzish kerak ekan. Shahrixon tumani hokimligi videoga qo'shib e'lon qilgan fotojamlanmada tuman hokimi Hikmatullo Dadaxonov ham aks etgan. U buzish ishlariga boshqa mas'ullar bilan birga, shaxsan o'zi bosh-qosh bo'lib turgan. Voqea katta shov-shuvni keltirib chiqargach, video va rasmlar hokimlik kanalidan darhol o'chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma'lum qildi.",
        )

    def test_capitalization_pipeline(self):
        text = "– Tur, Mansur ketamiz! vaqt ketgani qoldi. odam bo'lmas ekan bu..."

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = remove_special_chars(text, remove_ellipsis=True)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = normalize_capitalization(text)

        self.assertEqual(
            text,
            "Tur, Mansur ketamiz! Vaqt ketgani qoldi. Odam bo'lmas ekan bu",
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
