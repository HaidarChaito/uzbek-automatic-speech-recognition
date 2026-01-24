import math
import unittest

from scripts.uzbek_number_normalizer import NumberToUzbekWord
from scripts.uzbek_text_normalizer import (
    remove_whitespaces,
    remove_list_markers,
    remove_new_lines,
    normalize_uzbek_apostrophes,
    normalize_annotations,
    normalize_spacing_around_punc,
    capitalize_after_punc,
    normalize_uz_domains,
    remove_special_chars,
    remove_punctuations,
    normalize_text,
    normalize_double_quotes,
    normalize_dashes,
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
        self.assertEqual(remove_new_lines("hello \tworld"), "hello \tworld")


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
    def test_remove_middle_bullet_points_after_punctuation_except_followed_by_dedi(
        self,
    ):
        self.assertEqual(
            remove_list_markers("Non dema! — dedi. — nonni otini atama!"),
            "Non dema! — dedi. nonni otini atama!",
        )
        self.assertEqual(
            remove_list_markers("— Olib keldim, o‘rtoq kapitan, kirsinmi? — Kirsin."),
            "Olib keldim, o‘rtoq kapitan, kirsinmi? Kirsin.",
        )
        self.assertEqual(
            remove_list_markers(
                "— Qoraquloq, — dedim sekin. — Ana, Qoraquloq! o‘zimning echkim."
            ),
            "Qoraquloq, — dedim sekin. Ana, Qoraquloq! o‘zimning echkim.",
        )
        self.assertEqual(
            remove_list_markers("Qanaqasiga ukam bo‘lsin? — dedim iljayib."),
            "Qanaqasiga ukam bo‘lsin? — dedim iljayib.",
        )
        self.assertEqual(
            remove_list_markers("Kim bilsin, — deb yelka qisdi yigit"),
            "Kim bilsin, — deb yelka qisdi yigit",
        )
        self.assertEqual(
            remove_list_markers(
                "Bu mutlaqo asossiz gaplar! - Shavkat Mirziyoyev Alisher Qodirov taklifiga munosabat bildirdi"
            ),
            "Bu mutlaqo asossiz gaplar! Shavkat Mirziyoyev Alisher Qodirov taklifiga munosabat bildirdi",
        )

    def test_remove_starting_numbered_list(self):
        self.assertEqual(remove_list_markers("1. hello world"), "hello world")

    def test_remove_starting_numbered_list_without_space(self):
        self.assertEqual(remove_list_markers("2) hello world"), "hello world")

    def test_handle_multiple_numbered_lists(self):
        self.assertEqual(
            remove_list_markers("1)2) hello 1. world"), "2) hello 1. world"
        )

    def test_keep_dashes(self):
        self.assertEqual(
            remove_list_markers("Erk – manzilmas, erk – yo‘ldir"),
            "Erk – manzilmas, erk – yo‘ldir",
        )
        self.assertEqual(
            remove_list_markers("Olimning adashgani — dunyoning buzilgani."),
            "Olimning adashgani — dunyoning buzilgani.",
        )
        self.assertEqual(
            remove_list_markers("asossiz gaplar! -Shavkat Mirziyoyev"),
            "asossiz gaplar! -Shavkat Mirziyoyev",
        )
        self.assertEqual(
            remove_list_markers("Asossiz gaplar- Shavkat Mirziyoyev"),
            "Asossiz gaplar- Shavkat Mirziyoyev",
        )
        self.assertEqual(
            remove_list_markers(
                "Yuziga kelgan ketma-ket zarbalar uni shoshiltirib qo'ydi."
            ),
            "Yuziga kelgan ketma-ket zarbalar uni shoshiltirib qo'ydi.",
        )
        self.assertEqual(
            remove_list_markers("Tashqarida - 5 gradus sovuq"),
            "Tashqarida - 5 gradus sovuq",
            "Should keep hyphen if both sides are not letters or punctuation.",
        )


class TestNormalizeDashes(unittest.TestCase):
    def test_dashes(self):
        self.assertEqual(
            normalize_dashes("Ism – insonning bir umrlik hamrohi."),
            "Ism – insonning bir umrlik hamrohi.",
        )
        self.assertEqual(
            normalize_dashes("Amr etaman: so‘ra! — dedi qirol shosha-pisha."),
            "Amr etaman: so‘ra! – dedi qirol shosha-pisha.",
        )
        self.assertEqual(
            normalize_dashes("— Voy onajon-a! — dedi ingrab."),
            "– Voy onajon-a! – dedi ingrab.",
        )
        self.assertEqual(
            normalize_dashes("Kuch ― birlikda."),
            "Kuch – birlikda.",
        )

    def keep_hyphens(self):
        self.assertEqual(
            normalize_dashes("O‘rtoqlar, o‘lan-laparlar aytmanglar!"),
            "O‘rtoqlar, o‘lan-laparlar aytmanglar!",
        )
        self.assertEqual(
            normalize_dashes("O‘rtoqlar, o‘lan—laparlar aytmanglar!"),
            "O‘rtoqlar, o‘lan—laparlar aytmanglar!",
        )
        self.assertEqual(
            normalize_dashes("Tashqarida -5 daraja sovuq."),
            "Tashqarida -5 daraja sovuq.",
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


class TestNormalizeDoubleQuotes(unittest.TestCase):
    def test_double_quotes(self):
        self.assertEqual(
            normalize_double_quotes("“Milan”ning xursandchiligi uzoqqa cho‘zilmadi."),
            '"Milan"ning xursandchiligi uzoqqa cho‘zilmadi.',
        )
        self.assertEqual(
            normalize_double_quotes(
                "Aytmoqchimanki, terma jamoalarimiz «planka»ni yuqori qo‘ydi."
            ),
            'Aytmoqchimanki, terma jamoalarimiz "planka"ni yuqori qo‘ydi.',
        )

    def test_keep_single_quotes(self):
        self.assertEqual(
            normalize_double_quotes(
                "Aytmoqchimanki, terma jamoalarimiz 'planka'ni yuqori qo‘ydi."
            ),
            "Aytmoqchimanki, terma jamoalarimiz 'planka'ni yuqori qo‘ydi.",
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


class TestNormalizeUzDomains(unittest.TestCase):
    def test_with_uz_domain_normalization(self):
        result = normalize_uz_domains("qalampir.uz sayti")
        self.assertEqual(result, "Qalampir uz sayti")

        result = normalize_uz_domains("Qalampir.uz sayti")
        self.assertEqual(result, "Qalampir uz sayti")

        result = normalize_uz_domains("Muzaffar Komilov kun.uz bilan suhbatda aynan")
        self.assertEqual(result, "Muzaffar Komilov Kun uz bilan suhbatda aynan")

    def test_without_uz_domain_normalization(self):
        result = normalize_uz_domains("qalampir.com sayti")
        self.assertEqual(result, "qalampir.com sayti")

        result = normalize_uz_domains("qalampir. uz sayti")
        self.assertEqual(result, "qalampir. uz sayti")

        result = normalize_uz_domains("qalampir  .uz sayti")
        self.assertEqual(result, "qalampir  .uz sayti")


class TestCapitalizeUzDomain(unittest.TestCase):
    def test_capitalizes_uz_domain(self):
        result = normalize_uz_domains("qalampir.uz")
        self.assertEqual(result, "Qalampir uz")

    def test_handles_uppercase_domain(self):
        result = normalize_uz_domains("QALAMPIR.UZ")
        self.assertEqual(result, "Qalampir uz")

    def test_no_uz_domain_returns_original(self):
        self.assertEqual(normalize_uz_domains("hello world"), "hello world")

    def test_mixed_case_domain(self):
        result = normalize_uz_domains("QaLaMpIr.uz")
        self.assertEqual(result, "Qalampir uz")


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

    def test_capitalization_after_comma_should_not_change(self):
        result = capitalize_after_punc("– Tur, Mansur ketamiz.")
        self.assertEqual(result, "– Tur, Mansur ketamiz.")

    def test_capitalizes_first_character(self):
        result = capitalize_after_punc("hello world")
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        result = capitalize_after_punc("")
        self.assertEqual(result, "")


class TestRemoveSpecialChars(unittest.TestCase):
    def test_remove_special_chars(self):
        self.assertEqual(
            remove_special_chars("Xonadondan qattiq-quruq gap tash­qariga chiqmasdi."),
            "Xonadondan qattiq-quruq gap tashqariga chiqmasdi.",
        )
        self.assertEqual(
            remove_special_chars(
                "Bizning shah­rimizda chet elni koʻrgan, qandaydir bir chet til­ni bilgan bitta odam bor"
            ),
            "Bizning shahrimizda chet elni koʻrgan, qandaydir bir chet tilni bilgan bitta odam bor",
        )

    def keep_some_special_chars(self):
        self.assertEqual(
            remove_special_chars("C++ni o'rganish davomida..."),
            "C++ni o'rganish davomida...",
        )
        self.assertEqual(
            remove_special_chars(
                "Foydalanuvchilar o‘z noroziliklarini #Uninstallsnapchat heshtegi ostida bildirmoqda."
            ),
            "Foydalanuvchilar o‘z noroziliklarini #Uninstallsnapchat heshtegi ostida bildirmoqda.",
        )
        self.assertEqual(
            remove_special_chars(
                "Cake&Bakega tashrif buyuring va hayotingizni yanada shirinroq qiling!"
            ),
            "Cake&Bakega tashrif buyuring va hayotingizni yanada shirinroq qiling!",
        )
        self.assertEqual(remove_special_chars("No specia char."), "No specia char.")


class TestRemovePunctuations(unittest.TestCase):
    def test_question_mark(self):
        self.assertEqual(
            remove_punctuations('Uning oldida "tirbandlik" paydo bo\'ldimi?'),
            "Uning oldida tirbandlik paydo bo'ldimi",
        )
        self.assertEqual(
            remove_punctuations("Nimalar deyapsan? Esing joyidami?"),
            "Nimalar deyapsan Esing joyidami",
        )
        self.assertEqual(
            remove_punctuations('"Musulmonman", deb namoz o\'qimaysanmi?!'),
            "Musulmonman deb namoz o'qimaysanmi",
        )

    def test_exclamation_mark(self):
        self.assertEqual(
            remove_punctuations("“Ahmoq!” o'yladim men. Kechikdi!"),
            "Ahmoq o'yladim men Kechikdi",
        )
        self.assertEqual(
            remove_punctuations("To'xta! Ketma!!!"),
            "To'xta Ketma",
        )

    def test_comma(self):
        self.assertEqual(
            remove_punctuations("«Hovlini to'ldirib mevali daraxt ekdik, dedi."),
            "Hovlini to'ldirib mevali daraxt ekdik dedi",
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
                "Андижонда тандирлар ҳавони ифлослантиряпти деб топилди"
            ),
            "Андижонда тандирлар ҳавони ифлослантиряпти деб топилди",
        )

    def test_multiple_punctuations(self):
        self.assertEqual(
            remove_punctuations('"Ketma..." deya yalindi..'),
            "Ketma... deya yalindi",
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
            "Ketma deya yalindi",
        )

    def test_double_quotation_mark1(self):
        self.assertEqual(
            remove_punctuations('Uning oldida "tirbandlik" paydo bo\'ldi'),
            "Uning oldida tirbandlik paydo bo'ldi",
        )
        self.assertEqual(
            remove_punctuations('Uning oldida "tirbandlik" paydo bo\'ldi'),
            "Uning oldida tirbandlik paydo bo'ldi",
        )

    def test_double_quotation_mark2(self):
        self.assertEqual(
            remove_punctuations("“Ahmoq!” o'yladim men. Kechikdi!"),
            "Ahmoq o'yladim men Kechikdi",
        )
        self.assertEqual(
            remove_punctuations("U kelmadi. “Ahmoq! o'yladim men. Kechikdi!"),
            "U kelmadi Ahmoq o'yladim men Kechikdi",
        )

    def test_double_quotation_mark3(self):
        self.assertEqual(
            remove_punctuations("«Hovlini to'ldirib mevali daraxt ekdik, dedi."),
            "Hovlini to'ldirib mevali daraxt ekdik dedi",
        )
        self.assertEqual(
            remove_punctuations("Ular bunday sharoitda ma'nan «so'nib» qolishadi"),
            "Ular bunday sharoitda ma'nan so'nib qolishadi",
        )

    def test_no_apostrophes(self):
        self.assertEqual(
            remove_punctuations(
                "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi"
            ),
            "Shu zahoti yana bir idish gumburlab portladi va Qurbonni ham jarohatladi",
        )

    def test_remove_colons(self):
        self.assertEqual(
            remove_punctuations("Kitob haqida ma'lumot:"), "Kitob haqida ma'lumot"
        )
        self.assertEqual(
            remove_punctuations(
                "Birdan Stalin: «Lavrentiy, chora ko'r», deb qolardi.:"
            ),
            "Birdan Stalin Lavrentiy chora ko'r deb qolardi",
        )

    def test_remove_dashes(self):
        self.assertEqual(
            remove_punctuations("Mening uyim ― mening qo'rg'onim."),
            "Mening uyim mening qo'rg'onim",
        )
        self.assertEqual(
            remove_punctuations("Ism – insonning bir umrlik hamrohi."),
            "Ism insonning bir umrlik hamrohi",
        )
        self.assertEqual(
            remove_punctuations("Ism - insonning bir umrlik hamrohi."),
            "Ism insonning bir umrlik hamrohi",
        )
        self.assertEqual(
            remove_punctuations(
                "— Qoraquloq, — dedim sekin. — Ana, Qoraquloq! o‘zimning echkim."
            ),
            "Qoraquloq dedim sekin Ana Qoraquloq o‘zimning echkim",
        )

    def test_keep_hyphen_if_not_followed_by_letter(self):
        self.assertEqual(
            remove_punctuations(
                "Yuziga kelgan ketma-ket zarbalar uni shoshiltirib qo'ydi."
            ),
            "Yuziga kelgan ketma-ket zarbalar uni shoshiltirib qo'ydi",
        )
        self.assertEqual(
            remove_punctuations("Tashqarida - 5 gradus sovuq"),
            "Tashqarida - 5 gradus sovuq",
        )

    def test_always_remove_2dots(self):
        self.assertEqual(
            remove_punctuations(
                "Unda, sen chinovniklarni..! Chinovniklar to'nka!",
                remove_ellipsis=False,
            ),
            "Unda sen chinovniklarni Chinovniklar to'nka",
        )
        self.assertEqual(
            remove_punctuations("O'qimaysizmi?..", remove_ellipsis=True),
            "O'qimaysizmi",
        )
        self.assertEqual(
            remove_punctuations("Voy esim qursin.. Bug'doy nima bo'ldi..."),
            "Voy esim qursin Bug'doy nima bo'ldi...",
        )

    def test_conversation_speech_keep_ellipses(self):
        self.assertEqual(
            remove_punctuations("O'lsam, mozorimni kungayda qazing…"),
            "O'lsam mozorimni kungayda qazing...",
        )
        self.assertEqual(
            remove_punctuations("Yo'q, qoldi… u qoldi men bilan qoldi…"),
            "Yo'q qoldi... u qoldi men bilan qoldi...",
        )
        self.assertEqual(
            remove_punctuations(
                "Bundan tashqari... Biz oz emas-ko'p emas millionlab onalarni ularning orasida yosh kelinchaklar"
            ),
            "Bundan tashqari... Biz oz emas-ko'p emas millionlab onalarni ularning orasida yosh kelinchaklar",
        )
        self.assertEqual(
            remove_punctuations("Ro'moli burchini ushlab, televizorga..."),
            "Ro'moli burchini ushlab televizorga...",
        )
        self.assertEqual(
            remove_punctuations("Kel... Deb xotirjamlik bilan gapirish kerak...."),
            "Kel... Deb xotirjamlik bilan gapirish kerak...",
        )

    def test_read_speech_remove_ellipsis(self):
        self.assertEqual(
            remove_punctuations(
                "O'lsam, mozorimni kungayda qazing…", remove_ellipsis=True
            ),
            "O'lsam mozorimni kungayda qazing",
        )
        self.assertEqual(
            remove_punctuations(
                "Yo'q, qoldi… u qoldi men bilan qoldi…", remove_ellipsis=True
            ),
            "Yo'q qoldi u qoldi men bilan qoldi",
        )
        self.assertEqual(
            remove_punctuations(
                "Kel.. Deb xotirjamlik bilan gapirish kerak....", remove_ellipsis=True
            ),
            "Kel Deb xotirjamlik bilan gapirish kerak",
        )


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""

    def test_case_insensitive_asr_normalization_pipeline(self):
        # Credits: Kun.uz
        text = """
        Rasmiy  xabarda bu ishlar “havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari” deb atalgan. 19-avgust @Ya’ni andijonlik mulozimlarning fikricha, somsapazlar havoni > ifloslantirayotgan ekan... (random Noise) Shuning uchun ularning tandirini buzish kerak ekan.

        Shahrixon tumani hokimligi videoga qo‘shib e’lon qilgan fotojamlanmada tuman #hokimi - Hikmatullo Dadaxonov ham aks etgan. ? u buzish ishlariga boshqa mas’ullar bilan birga, 3,05% ga shaxsan (oddiy qavs ichida gap) o‘zi bosh-qosh bo‘lib turgan. Voqea katta shov-shuvni keltirib chiqargach, [ Annotatsiya ] video va rasmlar hokimlik kanalidan darhol o‘chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma’lum qildi.  
        """
        number_to_uzbek_word = NumberToUzbekWord()

        # Step by step
        text = remove_new_lines(text)
        text = number_to_uzbek_word.normalize(text)
        text = remove_list_markers(text)
        text = normalize_dashes(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_double_quotes(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_uz_domains(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_after_punc(text)
        text = remove_special_chars(text)
        text = remove_punctuations(text, remove_ellipsis=True)
        text = text.lower()
        text = remove_whitespaces(text)

        self.assertEqual(
            text,
            "rasmiy xabarda bu ishlar havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari deb atalgan o'n to'qqizinchi avgust ya'ni andijonlik mulozimlarning fikricha somsapazlar havoni ifloslantirayotgan ekan (random noise) shuning uchun ularning tandirini buzish kerak ekan shahrixon tumani hokimligi videoga qo'shib e'lon qilgan fotojamlanmada tuman #hokimi hikmatullo dadaxonov ham aks etgan u buzish ishlariga boshqa mas'ullar bilan birga uch butun nol besh foizga shaxsan (oddiy qavs ichida gap) o'zi bosh-qosh bo'lib turgan voqea katta shov-shuvni keltirib chiqargach [annotatsiya] video va rasmlar hokimlik kanalidan darhol o'chirildi viloyat hokimligi tezda bayonot bilan chiqib dadaxonovga hayfsan berilganini ma'lum qildi",
        )

    def test_case_sensitive_asr_normalization_pipeline(self):
        text = """
        Rasmiy xabarda bu ishlar : “havo ifloslanishining oldini {olish} maqsadidagi reyd tadbirlari” deb atalgan.ya’ni andijonlik - mulozimlarning kun.uz fikricha, somsapazlar havoni ifloslantirayotgan ekan.  shuning uchun ularning tandirini buzish kerak ekan. 19-avgust

        Shahrixon tumani hokimligi videoga qo‘shib e’lon qilgan fotojamlanmada tuman hokimi Hikmatullo Dadaxonov ham aks etgan.U buzish  ishlariga boshqa mas’ullar bilan < birga,shaxsan o‘zi bosh-qosh bo‘lib turgan. Voqea katta shov-shuvni keltirib chiqargach, video va rasmlar hokimlik kanalidan darhol o‘chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma’lum qildi.  
        """
        number_to_uzbek_word = NumberToUzbekWord()

        # Step by step
        text = remove_new_lines(text)
        text = number_to_uzbek_word.normalize(text)
        text = remove_list_markers(text)
        text = normalize_dashes(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_double_quotes(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_uz_domains(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_after_punc(text)
        text = remove_special_chars(text)
        text = remove_whitespaces(text)

        self.assertEqual(
            text,
            "Rasmiy xabarda bu ishlar: \"havo ifloslanishining oldini olish maqsadidagi reyd tadbirlari\" deb atalgan. Ya'ni andijonlik – mulozimlarning Kun uz fikricha, somsapazlar havoni ifloslantirayotgan ekan. Shuning uchun ularning tandirini buzish kerak ekan. O'n to'qqizinchi avgust Shahrixon tumani hokimligi videoga qo'shib e'lon qilgan fotojamlanmada tuman hokimi Hikmatullo Dadaxonov ham aks etgan. U buzish ishlariga boshqa mas'ullar bilan birga, shaxsan o'zi bosh-qosh bo'lib turgan. Voqea katta shov-shuvni keltirib chiqargach, video va rasmlar hokimlik kanalidan darhol o'chirildi. Viloyat hokimligi tezda bayonot bilan chiqib, Dadaxonovga hayfsan berilganini ma'lum qildi.",
        )

    def test_case_extended_number_normalization_pipeline(self):
        text = """
        Yana bir jihati, maktab-internatning manzili ― Izboskan tumani, To‘rtko‘l shaharchasi, A.Navoiy ko‘chasi, 25-uyda, g‘oliblar esa xuddi shu ko‘chadagi 41A-uyda joylashgan.

        Gubkin nomidagi Rossiya davlat neft va gaz universitetining Toshkent shahridagi filiali konditsioner xarid qilish bo‘yicha ikkita tanlov o‘tkazdi. 1- xaridning boshlang‘ich qiymati 276 mln so‘m, 2-siniki esa 79 mln. so‘m etib belgilangan. Ikki xaridda ham Gree Air Prom, “Stroy biznes servis energiya”, Clivent va Gelios Line Technology MCHJlari ishtirok etgan.
  
        2025-yil yanvar-noyabr oylarida soʻm almashuv kursi 7,5 %ga mustahkamlandi. Markaziy bank tahliliga koʻra, bunga qator omillar taʼsir koʻrsatdi.

        Soʻm qadri sezilarli ortishida avvalo, migrantlar yuborgan pul oʻtkazmalari keskin o‘sgani muhim rol oʻynagan. Ya’ni yanvar-noyabr oylarida Oʻzbekistonga muhojirlardan $17,3 mlrd kelib tushgan (2024-yil mos davridagidan 25 % koʻp). Bu ichki valyuta bozoridagi taklifni ragʻbatlantirib, milliy valyuta qadri mustahkamlanishi jiddiy taʼsir etgan. 
        """
        number_to_uzbek_word = NumberToUzbekWord()

        # Step by step
        text = remove_new_lines(text)
        text = number_to_uzbek_word.normalize(text)
        text = remove_list_markers(text)
        text = normalize_dashes(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_double_quotes(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_uz_domains(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_after_punc(text)
        text = remove_special_chars(text)
        text = remove_whitespaces(text)

        self.assertEqual(
            text,
            "Yana bir jihati, maktab-internatning manzili – Izboskan tumani, To'rtko'l shaharchasi, A. Navoiy ko'chasi, yigirma beshinchi uyda, g'oliblar esa xuddi shu ko'chadagi qirq bir A-uyda joylashgan. Gubkin nomidagi Rossiya davlat neft va gaz universitetining Toshkent shahridagi filiali konditsioner xarid qilish bo'yicha ikkita tanlov o'tkazdi. Bir– xaridning boshlang'ich qiymati ikki yuz yetmish olti million so'm, ikkinchi siniki esa yetmish to'qqiz million so'm etib belgilangan. Ikki xaridda ham Gree Air Prom, \"Stroy biznes servis energiya\", Clivent va Gelios Line Technology MCHJlari ishtirok etgan. Ikki ming yigirma beshinchi yil yanvar-noyabr oylarida so'm almashuv kursi yetti butun besh foizga mustahkamlandi. Markaziy bank tahliliga ko'ra, bunga qator omillar ta'sir ko'rsatdi. So'm qadri sezilarli ortishida avvalo, migrantlar yuborgan pul o'tkazmalari keskin o'sgani muhim rol o'ynagan. Ya'ni yanvar-noyabr oylarida O'zbekistonga muhojirlardan o'n yetti butun uch dollar milliard kelib tushgan (ikki ming yigirma to'rtinchi yil mos davridagidan yigirma besh foiz ko'p). Bu ichki valyuta bozoridagi taklifni rag'batlantirib, milliy valyuta qadri mustahkamlanishi jiddiy ta'sir etgan.",
        )

    def test_capitalization_pipeline(self):
        text = "– Tur, Mansur ketamiz! vaqt ketgani qoldi. odam bo'lmas ekan bu..."

        # Step by step
        text = remove_new_lines(text)
        text = remove_list_markers(text)
        text = normalize_uzbek_apostrophes(text)
        text = normalize_annotations(text)
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_after_punc(text)
        text = remove_special_chars(text)
        text = remove_whitespaces(text)

        self.assertEqual(
            text,
            "Tur, Mansur ketamiz! Vaqt ketgani qoldi. Odam bo'lmas ekan bu...",
        )

    def test_uzbek_text_with_various_apostrophes(self):
        text = "o'zbek tilʼi va oʻzbekiston"
        text = normalize_uzbek_apostrophes(text)

        self.assertEqual(text, "o'zbek til'i va o'zbekiston")

    def test_messy_transcription(self):
        text = "   salom  ,  dunyo   !   qalaysan   ?  "
        text = remove_whitespaces(text)
        text = normalize_spacing_around_punc(text)
        text = capitalize_after_punc(text)

        self.assertEqual(text, "salom, dunyo! Qalaysan?")

    def test_nan_or_empty_string(self):
        text = " \t  "
        text = normalize_text(text)
        self.assertEqual(text, "")

        text = None
        text = normalize_text(text)
        self.assertEqual(text, "")

        text = math.nan
        text = normalize_text(text)
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
