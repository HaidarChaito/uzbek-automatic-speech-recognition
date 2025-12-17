import math
import unittest

from scripts.uzbek_number_normalizer import NumberToUzbekWord


class TestNumberToUzbekWord(unittest.TestCase):
    def setUp(self):
        """Set up test converter instance"""
        self.converter = NumberToUzbekWord()

    # ===== Basic Number Tests (0-9) =====
    def test_ones(self):
        self.assertEqual(self.converter._convert(0), "nol")
        self.assertEqual(self.converter._convert(1), "bir")
        self.assertEqual(self.converter._convert(5), "besh")
        self.assertEqual(self.converter._convert(7), "yetti")
        self.assertEqual(self.converter._convert(9), "to'qqiz")

    # ===== Tens Tests (10-99) =====
    def test_exact_tens(self):
        self.assertEqual(self.converter._convert(10), "o'n")
        self.assertEqual(self.converter._convert(20), "yigirma")
        self.assertEqual(self.converter._convert(50), "ellik")
        self.assertEqual(self.converter._convert(60), "oltmish")
        self.assertEqual(self.converter._convert(70), "yetmish")

    def test_two_digit_numbers(self):
        self.assertEqual(self.converter._convert(11), "o'n bir")
        self.assertEqual(self.converter._convert(25), "yigirma besh")
        self.assertEqual(self.converter._convert(47), "qirq yetti")
        self.assertEqual(self.converter._convert(99), "to'qson to'qqiz")

    # ===== Hundreds Tests (100-999) =====
    def test_exact_hundreds(self):
        self.assertEqual(self.converter._convert(100), "yuz")
        self.assertEqual(self.converter._convert(200), "ikki yuz")
        self.assertEqual(self.converter._convert(500), "besh yuz")
        self.assertEqual(self.converter._convert(900), "to'qqiz yuz")

    def test_three_digit_numbers(self):
        self.assertEqual(self.converter._convert(101), "bir yuz bir")
        self.assertEqual(self.converter._convert(234), "ikki yuz o'ttiz to'rt")
        self.assertEqual(self.converter._convert(567), "besh yuz oltmish yetti")
        self.assertEqual(self.converter._convert(999), "to'qqiz yuz to'qson to'qqiz")

    # ===== Thousands Tests =====
    def test_exact_thousands(self):
        self.assertEqual(self.converter._convert(1000), "ming")
        self.assertEqual(self.converter._convert(2000), "ikki ming")
        self.assertEqual(self.converter._convert(10_000), "o'n ming")
        self.assertEqual(self.converter._convert(100_000), "bir yuz ming")

    def test_complex_thousands(self):
        self.assertEqual(
            self.converter._convert(1234), "bir ming ikki yuz o'ttiz to'rt"
        )
        self.assertEqual(self.converter._convert(3456), "uch ming to'rt yuz ellik olti")
        self.assertEqual(
            self.converter._convert(5678), "besh ming olti yuz yetmish sakkiz"
        )
        self.assertEqual(self.converter._convert(77_100), "yetmish yetti ming bir yuz")
        self.assertEqual(
            self.converter._convert(99_999),
            "to'qson to'qqiz ming to'qqiz yuz to'qson to'qqiz",
        )

    # ===== Millions Tests =====
    def test_millions(self):
        self.assertEqual(
            self.converter._convert(1_000_000), "bir million"
        )  # be explicit here
        self.assertEqual(self.converter._convert(2_000_000), "ikki million")
        self.assertEqual(self.converter._convert(3_000_100), "uch million bir yuz")
        self.assertEqual(
            self.converter._convert(5_500_000), "besh million besh yuz ming"
        )
        self.assertEqual(
            self.converter._convert(7_060_050), "yetti million oltmish ming ellik"
        )
        self.assertEqual(
            self.converter._convert(1_234_567),
            "bir million ikki yuz o'ttiz to'rt ming besh yuz oltmish yetti",
        )

    # ===== Billions Tests =====
    def test_billions(self):
        self.assertEqual(
            self.converter._convert(1_000_000_000), "bir milliard"
        )  # be explicit here
        self.assertEqual(self.converter._convert(2_000_000_000), "ikki milliard")
        self.assertEqual(
            self.converter._convert(4_030_002_001),
            "to'rt milliard o'ttiz million ikki ming bir",
        )
        self.assertEqual(
            self.converter._convert(1_234_567_890),
            "bir milliard ikki yuz o'ttiz to'rt million besh yuz oltmish yetti ming sakkiz yuz to'qson",
        )

    # ===== Negative Numbers Tests =====
    def test_negative_numbers(self):
        self.assertEqual(self.converter._convert(-5), "minus besh")
        self.assertEqual(self.converter._convert(-15), "minus o'n besh")
        self.assertEqual(self.converter._convert(-99), "minus to'qson to'qqiz")
        self.assertEqual(self.converter._convert(-100), "minus yuz")
        self.assertEqual(
            self.converter._convert(-1234), "minus bir ming ikki yuz o'ttiz to'rt"
        )
        self.assertEqual(self.converter._convert(-10_100), "minus o'n ming bir yuz")

    # ===== Decimal Numbers Tests =====
    def test_decimal_numbers(self):
        self.assertEqual(self.converter._convert(3.5), "uch butun besh")
        self.assertEqual(self.converter._convert(10.25), "o'n butun yigirma besh")
        self.assertEqual(self.converter._convert(100.01), "yuz butun nol bir")
        self.assertEqual(
            self.converter._convert(123.001), "bir yuz yigirma uch butun nol nol bir"
        )
        self.assertEqual(self.converter._convert(123.000), "bir yuz yigirma uch")
        self.assertEqual(self.converter._convert(0.5), "nol butun besh")
        self.assertEqual(
            self.converter._convert(0.0065), "nol butun nol nol oltmish besh"
        )

    # ===== String Numbers Tests =====
    def test_correct_string_numbers(self):
        self.assertEqual(self.converter._convert("567 "), "besh yuz oltmish yetti")
        self.assertEqual(self.converter._convert(" 100_000"), "bir yuz ming")
        self.assertEqual(
            self.converter._convert("1_234_567\t"),
            "bir million ikki yuz o'ttiz to'rt ming besh yuz oltmish yetti",
        )

        self.assertEqual(self.converter._convert("\n1_000_000_000"), "bir milliard")
        self.assertEqual(
            self.converter._convert("4_030_002_001"),
            "to'rt milliard o'ttiz million ikki ming bir",
        )

        self.assertEqual(self.converter._convert("-100"), "minus yuz")
        self.assertEqual(
            self.converter._convert("-1_234"), "minus bir ming ikki yuz o'ttiz to'rt"
        )

        self.assertEqual(
            self.converter._convert("123.001"), "bir yuz yigirma uch butun nol nol bir"
        )
        self.assertEqual(self.converter._convert("123.000"), "bir yuz yigirma uch")
        self.assertEqual(self.converter._convert("   0.5"), "nol butun besh")
        self.assertEqual(
            self.converter._convert("0.0065"), "nol butun nol nol oltmish besh"
        )

    def test_incorrect_string_numbers(self):
        self.assertTrue(math.isnan(self.converter._convert("10 ta")))
        self.assertTrue(math.isnan(self.converter._convert("10ta")))
        self.assertTrue(math.isnan(self.converter._convert("10-")))
        self.assertTrue(math.isnan(self.converter._convert("- 10")))

    # ===== Preprocess Text Tests =====
    def test_correct_preprocess_text_abbreviations(self):
        self.assertEqual(
            self.converter._preprocess_text(
                "Oʻzbekistonda pul massasi 344,6 trln soʻmga yetdi."
            ),
            "Oʻzbekistonda pul massasi 344,6 trillion soʻmga yetdi.",
        )
        self.assertEqual(
            self.converter._preprocess_text(
                "11 oy ichida aholi banklarga 19,4 mlrd. dollar sotgan (oʻtgan yil mos davridagidan 4,8 mlrd dollar koʻp)"
            ),
            "11 oy ichida aholi banklarga 19,4 milliard dollar sotgan (oʻtgan yil mos davridagidan 4,8 milliard dollar koʻp)",
        )
        self.assertEqual(
            self.converter._preprocess_text(
                "2025-yil 1-dekabr holatiga ko‘ra 61,23 mln. dollarga yetdi."
            ),
            "2025-yil 1-dekabr holatiga ko‘ra 61,23 million dollarga yetdi.",
        )
        self.assertEqual(
            self.converter._preprocess_text(
                "2025-yil 1-dekabr holatiga ko‘ra olti mln. dollarga yetdi."
            ),
            "2025-yil 1-dekabr holatiga ko‘ra olti mln. dollarga yetdi.",
        )

    def test_correct_preprocess_text_spacing_old(self):
        self.assertEqual(
            self.converter._preprocess_text("Men 5ta kitob oldim"),
            "Men 5ta kitob oldim",
        )
        self.assertEqual(
            self.converter._preprocess_text("Men 5 ta kitob oldim"),
            "Men 5ta kitob oldim",
        )
        self.assertEqual(
            self.converter._preprocess_text("Men 5 \t ta kitob oldim"),
            "Men 5ta kitob oldim",
        )
        self.assertEqual(
            self.converter._preprocess_text("Menda 5kg olma bor"),
            "Menda 5 kg olma bor",
        )
        self.assertEqual(
            self.converter._preprocess_text("Menda 5  kg olma bor"),
            "Menda 5  kg olma bor",
        )
        self.assertEqual(
            self.converter._preprocess_text(
                "Yangi BMW X7 mashinasi taxminan 136,390 yevro turadi"
            ),
            "Yangi BMW X 7 mashinasi taxminan 136,390 yevro turadi",
        )

    def test_correct_preprocess_text_spacing_with_percentage(self):
        self.assertEqual(
            self.converter._preprocess_text("U 5%ga o'sdi"),
            "U 5%ga o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 5 % ga o'sdi"),
            "U 5%ga o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 5.5 % ga o'sdi"),
            "U 5.5%ga o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 5,5 % ga o'sdi"),
            "U 5,5%ga o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 50 500 % ga o'sdi"),
            "U 50 500%ga o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 50-55 % dan o'sdi"),
            "U 50-55%dan o'sdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 50.5-55.6 % ini tashkil qiladi"),
            "U 50.5-55.6%ini tashkil qiladi",
        )
        self.assertEqual(
            self.converter._preprocess_text("U 5  % ga o'sdi"),
            "U 5  % ga o'sdi",
        )

    def test_correct_preprocess_text_spacing_with_currencies(self):
        self.assertEqual(
            self.converter._preprocess_text("Tovar narxi $32,08ga oshdi"),
            "Tovar narxi $32,08ga oshdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("Tovar narxi $ 32.08 ga oshdi"),
            "Tovar narxi $ 32.08ga oshdi",
        )
        self.assertEqual(
            self.converter._preprocess_text("Tovar narxi 32,008€ ni tashkil qiladi"),
            "Tovar narxi 32,008€ni tashkil qiladi",
        )
        self.assertEqual(
            self.converter._preprocess_text("Tovar narxi 32,08 € ni tashkil qiladi"),
            "Tovar narxi 32,08 €ni tashkil qiladi",
        )
        self.assertEqual(
            self.converter._preprocess_text("Tovar narxi 30-35 € ni tashkil qiladi"),
            "Tovar narxi 30-35 €ni tashkil qiladi",
        )

    # ===== Normalize Function Tests =====
    def test_normalize_alphanumeric_numbers(self):
        self.assertEqual(
            self.converter.normalize("Men 5ta kitob oldim"), "Men beshta kitob oldim"
        )
        self.assertEqual(
            self.converter.normalize("Men 5 ta kitob oldim"), "Men beshta kitob oldim"
        )
        self.assertEqual(
            self.converter.normalize("Menda 5kg olma bor"), "Menda besh kg olma bor"
        )
        self.assertEqual(
            self.converter.normalize(
                "Yangi BMW X7 mashinasi taxminan 136,390 yevro turadi"
            ),
            "Yangi BMW X yetti mashinasi taxminan bir yuz o'ttiz olti ming uch yuz to'qson yevro turadi",
        )
        self.assertEqual(
            self.converter.normalize(
                "Mening kompyuterim protsessori - AMD Ryzen5 8640HS"
            ),
            "Mening kompyuterim protsessori - AMD Ryzen besh ming sakkiz yuz oltmish to'rtnol HS",
        )

    def test_normalize_simple_numbers(self):
        self.assertEqual(
            self.converter.normalize("Men 5 ta kitob oldim"), "Men beshta kitob oldim"
        )
        self.assertEqual(
            self.converter.normalize("Men 5ta kitob oldim"), "Men beshta kitob oldim"
        )
        self.assertEqual(
            self.converter.normalize("U 25 yoshda"), "U yigirma besh yoshda"
        )

    def test_normalize_small_range_values(self):
        self.assertEqual(
            self.converter.normalize("Men 2-3 kun ichida qaytaman"),
            "Men ikki - uch kun ichida qaytaman",
        )
        self.assertEqual(self.converter.normalize("10-15ta"), "o'n - o'n beshta")
        self.assertEqual(
            self.converter.normalize("U 10 - 15 ta kitob sotib oldi."),
            "U o'n - o'n beshta kitob sotib oldi.",
        )
        self.assertEqual(self.converter.normalize("-10-15ta"), "minus o'n - o'n beshta")
        self.assertEqual(self.converter.normalize("- 10-15ta"), "- o'n - o'n beshta")

    def test_normalize_big_range_values(self):
        self.assertEqual(
            self.converter.normalize("Menga 2,000,000-3,000,000 so'm pul kerak."),
            "Menga ikki million - uch million so'm pul kerak.",
        )
        self.assertEqual(
            self.converter.normalize("10 500 050-15 005 000ta"),
            "o'n million besh yuz ming ellik - o'n besh million besh mingta",
        )
        self.assertEqual(
            self.converter.normalize("10  500 050-15 005  000ta"),
            "o'n  besh yuz ming ellik - o'n besh ming besh  nolta",
        )
        self.assertEqual(
            self.converter.normalize("Uning xisobi -10 000 - 15 000 so'm edi."),
            "Uning xisobi minus o'n ming - o'n besh ming so'm edi.",
        )

    def test_normalize_decimal_range_values(self):
        self.assertEqual(
            self.converter.normalize("Menga 3.5-4.5 million so'm pul kerak."),
            "Menga uch butun besh - to'rt butun besh million so'm pul kerak.",
        )
        self.assertEqual(
            self.converter.normalize("Menga -3.5 - 4.5 million so'm pul kerak."),
            "Menga minus uch butun besh - to'rt butun besh million so'm pul kerak.",
        )

    def test_normalize_big_decimal_range_values(self):
        self.assertEqual(
            self.converter.normalize("Menga 2,000,000.50-3,000,000.70 so'm pul kerak."),
            "Menga ikki million butun besh - uch million butun yetti so'm pul kerak.",
        )
        self.assertEqual(
            self.converter.normalize("Uning xisobi -10 000.707 - 15 000.907 so'm edi."),
            "Uning xisobi minus o'n ming butun yetti yuz yetti - o'n besh ming butun to'qqiz yuz yetti so'm edi.",
        )

    def test_normalize_ordinal_numbers(self):
        self.assertEqual(self.converter.normalize("2-qism"), "ikkinchi qism")
        self.assertEqual(
            self.converter.normalize("2024-yilga nisbatan 25 foizga koʻp"),
            "ikki ming yigirma to'rtinchi yilga nisbatan yigirma besh foizga koʻp",
        )
        self.assertEqual(
            self.converter.normalize(
                "A.Navoiy ko‘chasi, 25-uy va shu ko‘chadagi 41A-uy"
            ),
            "A.Navoiy ko‘chasi, yigirma beshinchi uy va shu ko‘chadagi qirq bir A-uy",
        )

    def test_normalize_big_numbers(self):
        self.assertEqual(self.converter.normalize("3,000,000 so'm"), "uch million so'm")
        self.assertEqual(
            self.converter.normalize("1,234 ta"), "bir ming ikki yuz o'ttiz to'rtta"
        )
        self.assertEqual(
            self.converter.normalize("-1 234 ta"),
            "minus bir ming ikki yuz o'ttiz to'rtta",
        )
        self.assertEqual(
            self.converter.normalize("10  234 ta"), "o'n  ikki yuz o'ttiz to'rtta"
        )
        self.assertEqual(
            self.converter.normalize("-10  234 ta"),
            "minus o'n  ikki yuz o'ttiz to'rtta",
        )

    def test_normalize_big_decimal_numbers(self):
        self.assertEqual(
            self.converter.normalize("1,234.22 ta"),
            "bir ming ikki yuz o'ttiz to'rt butun yigirma ikkita",
        )
        self.assertEqual(
            self.converter.normalize("-1,234.22 ta"),
            "minus bir ming ikki yuz o'ttiz to'rt butun yigirma ikkita",
        )
        self.assertEqual(
            self.converter.normalize("3 000 000.04 so'm"),
            "uch million butun nol to'rt so'm",
        )
        self.assertEqual(
            self.converter.normalize(
                "1 yevro 50,98 so‘mga ko‘tarilib, 14 245,48 so‘m bo‘ldi"
            ),
            "bir yevro ellik butun to'qson sakkiz so‘mga ko‘tarilib, o'n to'rt ming ikki yuz qirq besh butun qirq sakkiz so‘m bo‘ldi",
        )

    def test_normalize_decimal_with_comma(self):
        self.assertEqual(self.converter.normalize("3,5 metr"), "uch butun besh metr")
        self.assertEqual(
            self.converter.normalize("-3,5 metr"), "minus uch butun besh metr"
        )
        self.assertEqual(
            self.converter.normalize("3,05 metr"), "uch butun nol besh metr"
        )

    def test_normalize_decimal_with_dot(self):
        self.assertEqual(
            self.converter.normalize("Narxi 10.5 dollar"), "Narxi o'n butun besh dollar"
        )
        self.assertEqual(
            self.converter.normalize("Narxi -10.5 dollar"),
            "Narxi minus o'n butun besh dollar",
        )
        self.assertEqual(
            self.converter.normalize("Narxi 10.05 dollar"),
            "Narxi o'n butun nol besh dollar",
        )

    def test_normalize_negative_in_text(self):
        self.assertEqual(
            self.converter.normalize("Harorat -15 daraja"),
            "Harorat minus o'n besh daraja",
        )
        self.assertEqual(
            self.converter.normalize("Harorat -15.7 daraja"),
            "Harorat minus o'n besh butun yetti daraja",
        )
        self.assertEqual(
            self.converter.normalize("Harorat --15.7 daraja"),
            "Harorat -minus o'n besh butun yetti daraja",
        )
        self.assertEqual(
            self.converter.normalize("Harorat - 15.7 daraja"),
            "Harorat - o'n besh butun yetti daraja",
        )

    def test_normalize_currency_prefix(self):
        self.assertEqual(
            self.converter.normalize("$ 3,000,000 miqdorida"),
            "uch million dollar miqdorida",
        )
        self.assertEqual(
            self.converter.normalize(
                "€1 50,98 so‘mga ko‘tarilib, 14 245,48 so‘m bo‘ldi"
            ),
            "bir yevro ellik butun to'qson sakkiz so‘mga ko‘tarilib, o'n to'rt ming ikki yuz qirq besh butun qirq sakkiz so‘m bo‘ldi",
        )
        self.assertEqual(
            self.converter.normalize(
                "₽  1 1,27 so‘mga oshib, 153,04 so‘mni tashkil etdi"
            ),
            "₽  bir bir butun yigirma yetti so‘mga oshib, bir yuz ellik uch butun nol to'rt so‘mni tashkil etdi",
        )

    def test_normalize_currency_suffix(self):
        self.assertEqual(
            self.converter.normalize("3,000,000 $ miqdorida"),
            "uch million dollar miqdorida",
        )
        self.assertEqual(
            self.converter.normalize(
                "1€ 50,98 so‘mga ko‘tarilib, 14 245,48 so‘m bo‘ldi"
            ),
            "bir yevro ellik butun to'qson sakkiz so‘mga ko‘tarilib, o'n to'rt ming ikki yuz qirq besh butun qirq sakkiz so‘m bo‘ldi",
        )
        self.assertEqual(
            self.converter.normalize(
                "1  ₽ 1,27 so‘mga oshib, 153,04 so‘mni tashkil etdi"
            ),
            "bir  bir butun yigirma yetti rubl so‘mga oshib, bir yuz ellik uch butun nol to'rt so‘mni tashkil etdi",
        )

    def test_normalize_percentage_suffix(self):
        self.assertEqual(
            self.converter.normalize("3,000,000% miqdorida"),
            "uch million foiz miqdorida",
        )
        self.assertEqual(
            self.converter.normalize(
                "soʻmning dollarga nisbatan kursi -3,4 % ga mustahkamlangandi"
            ),
            "soʻmning dollarga nisbatan kursi minus uch butun to'rt foizga mustahkamlangandi",
        )

    def test_normalize_multiple_numbers(self):
        result = self.converter.normalize("Men 3 ta kitob va 5 ta qalam oldim")
        self.assertEqual(result, "Men uchta kitob va beshta qalam oldim")

    def test_normalize_mixed_patterns(self):
        result = self.converter.normalize("1-bob: 2-3 kun, 1,000 so'm, 5.5 kg")
        self.assertEqual(
            result, "birinchi bob: ikki - uch kun, ming so'm, besh butun besh kg"
        )

    def test_normalize_mixed_patterns_extended(self):
        result = self.converter.normalize(
            """
Raqamlarga koʻra, oʻtgan oyda mamlakatga jami qiymati 68,2 mln dollarlik 4 597 dona elektromobil import qilingan. Taqqoslash uchun, sentyabrda 11 534 ta elektromobil olib kelingan (qariyb 60% ga kamaygan).

Respublikaga elektrda harakatlanuvchi avtomobillar importi 2025-yil iyuldan sezilarli oʻsa boshladi. Jumladan, iyunda chet eldan 4560 dona elektromobil olib kelingan boʻlsa, iyulda bu raqam 7,204 donani tashkil etgan.

Noyabr oyida import qilingan elektromobillar har bir donasining oʻrtacha qiymati keskin oshganini koʻrish mumkin — 14.8 ming dollar. Mazkur qiymat oktyabrda 10.3 ming dollardan toʻgʻri kelgandi. Bundan bir yil ilgari — 2024-yil noyabrda esa har bir dona elektormobil narxi 6 100 $ dan tushgan.

Yillik hisobda taqqoslansa, 2025-yil yanvar-noyabr oylarida Oʻzbekistonga umumiy qiymati 612,2 mln. dollarlik 51,856 dona elektromobil import qilingan. Yaʼni yillik import hajmi ikki baravardan yuqori oʻsgan.
            """
        )
        self.assertEqual(
            result,
            """
Raqamlarga koʻra, oʻtgan oyda mamlakatga jami qiymati oltmish sakkiz butun ikki million dollarlik to'rt ming besh yuz to'qson yetti dona elektromobil import qilingan. Taqqoslash uchun, sentyabrda o'n bir ming besh yuz o'ttiz to'rtta elektromobil olib kelingan (qariyb oltmish foizga kamaygan).

Respublikaga elektrda harakatlanuvchi avtomobillar importi ikki ming yigirma beshinchi yil iyuldan sezilarli oʻsa boshladi. Jumladan, iyunda chet eldan to'rt ming besh yuz oltmish dona elektromobil olib kelingan boʻlsa, iyulda bu raqam yetti ming ikki yuz to'rt donani tashkil etgan.

Noyabr oyida import qilingan elektromobillar har bir donasining oʻrtacha qiymati keskin oshganini koʻrish mumkin — o'n to'rt butun sakkiz ming dollar. Mazkur qiymat oktyabrda o'n butun uch ming dollardan toʻgʻri kelgandi. Bundan bir yil ilgari — ikki ming yigirma to'rtinchi yil noyabrda esa har bir dona elektormobil narxi olti ming bir yuz dollardan tushgan.

Yillik hisobda taqqoslansa, ikki ming yigirma beshinchi yil yanvar-noyabr oylarida Oʻzbekistonga umumiy qiymati olti yuz o'n ikki butun ikki million dollarlik ellik bir ming sakkiz yuz ellik olti dona elektromobil import qilingan. Yaʼni yillik import hajmi ikki baravardan yuqori oʻsgan.
            """,
        )

    # ===== Edge Cases =====
    def test_zero(self):
        self.assertEqual(self.converter._convert(0), "nol")
        self.assertEqual(self.converter.normalize("0 ta"), "nolta")

    def test_large_numbers(self):
        self.assertEqual(
            self.converter._convert(999_999_999),
            "to'qqiz yuz to'qson to'qqiz million to'qqiz yuz to'qson to'qqiz ming to'qqiz yuz to'qson to'qqiz",
        )

    def test_empty_string(self):
        self.assertEqual(self.converter.normalize(""), "")

    def test_text_without_numbers(self):
        text = "Bu oddiy matn"
        self.assertEqual(self.converter.normalize(text), text)

    def test_invalid_number_format(self):
        result = self.converter._convert("abc123xyz")
        self.assertTrue(math.isnan(result))


if __name__ == "__main__":
    unittest.main()
