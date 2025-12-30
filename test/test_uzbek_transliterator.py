import unittest

from scripts.uzbek_transliterator import to_latin


# Some texts are adopted from the book "FALASTIN: Agar o‘lishim lоzim bo‘lsa…"
# Credits to author: Nadеjda Kеvоrkоva and translator: Hamid Sodiq
class TestToLatin(unittest.TestCase):

    def test_basic_mapping(self):
        self.assertEqual(to_latin("салом"), "salom")
        self.assertEqual(to_latin("китоб"), "kitob")
        self.assertEqual(to_latin("мактаб"), "maktab")

    def test_special_uzbek_chars(self):
        self.assertEqual(to_latin("ўзбекистон"), "o‘zbekiston")
        self.assertEqual(to_latin("ғалла"), "g‘alla")
        self.assertEqual(to_latin("қалам"), "qalam")
        self.assertEqual(to_latin("ҳамма"), "hamma")

    def test_basic_mapping_in_text(self):
        self.assertEqual(
            to_latin(
                "Фаластин халқи Инжилда тавсифи келтирилган қандайдир мавҳум филистиминлар эмас."
            ),
            "Falastin xalqi Injilda tavsifi keltirilgan qandaydir mavhum filistiminlar emas.",
        )
        self.assertEqual(
            to_latin(
                "Уларнинг 24 000 таси яҳудий бўлиб, \nаксарияти ?  шаҳарликлар \tэди.…"
            ),
            "Ularning 24 000 tasi yahudiy bo‘lib, \naksariyati ?  shaharliklar \tedi.…",
        )

    # --- Rule 1: Handle 'ў / ғ' + 'ъ' (not apostrophe) ---
    def test_rule_ў_ғ_ъ(self):
        self.assertEqual(to_latin("мўъжиза"), "mo‘jiza")
        self.assertEqual(to_latin("мўътадил"), "mo‘tadil")
        self.assertEqual(to_latin("мўътабар"), "mo‘tabar")
        self.assertEqual(to_latin("мўғътабар"), "mo‘g‘tabar")
        self.assertEqual(
            to_latin("Лекин бу мактаб мўъжиза туфайли пайдо бўлган."),
            "Lekin bu maktab mo‘jiza tufayli paydo bo‘lgan.",
        )

    # --- Rule 2: Handle dates as ordinary number ---
    def test_rule_simple_years(self):
        self.assertEqual(to_latin("1999 йил"), "1999-yil")
        self.assertEqual(to_latin("1999  \t йил"), "1999-yil")
        self.assertEqual(to_latin("1999йил"), "1999-yil")
        self.assertEqual(to_latin("1999-Йил"), "1999-Yil")
        self.assertEqual(to_latin("1999--йил"), "1999-yil")
        self.assertEqual(to_latin("20000 йил"), "20000 yil")
        self.assertEqual(
            to_latin("Қуддусни 1917 йилда инглизлар босиб олди"),
            "Quddusni 1917-yilda inglizlar bosib oldi",
        )
        # TODO:
        # self.assertEqual(
        #     to_latin(
        #         "Ислом динининг улуғ китоби - Қуръони карим, 1400 йил аввал нозил бўлганига қарамай, ҳеч эскиргани йўқ."
        #     ),
        #     "Islom dinining ulug‘ kitobi - Qur'oni karim, 1400 yil avval nozil bo‘lganiga qaramay, hech eskirgani yo‘q.",
        # )

    def test_rule_simple_years_shortened(self):
        self.assertEqual(
            to_latin("Ўзбекистон 90 йилларга келиб, ..."),
            "O‘zbekiston 90-yillarga kelib, ...",
        )
        self.assertEqual(
            to_latin("Ўзбекистон 80 йилларга келиб, 90 йилларга келиб, ..."),
            "O‘zbekiston 80-yillarga kelib, 90-yillarga kelib, ...",
        )
        self.assertEqual(
            to_latin(
                "Ўтган асрнинг 70 йилларида улардан юқорида жойлашган тепаликдаги фаластинлик қўшнилари ўз боғларини кўчкиндиларга сотдилар."
            ),
            "O‘tgan asrning 70-yillarida ulardan yuqorida joylashgan tepalikdagi falastinlik qo‘shnilari o‘z bog‘larini ko‘chkindilarga sotdilar.",
        )
        self.assertEqual(
            to_latin("Мактаб 90  \t  йиллардан бошлаб давлат тасарруфида эмас."),
            "Maktab 90-yillardan boshlab davlat tasarrufida emas.",
        )

    def test_rule_range_years(self):
        self.assertEqual(
            to_latin("2021-2024 йиллар оралиғида"), "2021-2024-yillar oralig‘ida"
        )
        self.assertEqual(
            to_latin("2021-2024   \t Йиллар оралиғида"), "2021-2024-Yillar oralig‘ida"
        )
        self.assertEqual(to_latin("2021-2024йилларда"), "2021-2024-yillarda")
        self.assertEqual(
            to_latin("2021-2024 2025 йиллар оралиғида"),
            "2021-2024 2025-yillar oralig‘ida",
        )
        self.assertEqual(to_latin("2021-2024 йи"), "2021-2024 yi")

    def test_rule_simple_small_years(self):
        # --- With keywords: милодий, миллоддан аввалги, асрнинг ---
        self.assertEqual(to_latin("Милодий 70 йилда"), "Milodiy 70-yilda")
        self.assertEqual(to_latin("милодий 70  \tйилда"), "milodiy 70-yilda")
        self.assertEqual(
            to_latin("Милоддан аввалги 333 йилда Фаластинни"),
            "Miloddan avvalgi 333-yilda Falastinni",
        )
        self.assertEqual(
            to_latin("милоддан Аввалги  333йилда Фаластинни"),
            "miloddan Avvalgi 333-yilda Falastinni",
        )
        self.assertEqual(
            to_latin(
                "Милоддан аввалги 167 йилгача бу ерда эллинизм даври подшолари ҳукмронлик қилган."
            ),
            "Miloddan avvalgi 167-yilgacha bu yerda ellinizm davri podsholari hukmronlik qilgan.",
        )
        self.assertEqual(
            to_latin("XIV асрнинг 50 йилларида"), "XIV asrning 50-yillarida"
        )
        self.assertEqual(
            to_latin("XIV асрнинг 50--йилларида"), "XIV asrning 50-yillarida"
        )

        # --- Without keywords - match only yilda ---
        self.assertEqual(
            to_latin("Оссурияликлар Фаластинга 722 йилда ҳужум қилдилар."),
            "Ossuriyaliklar Falastinga 722-yilda hujum qildilar.",
        )
        self.assertEqual(
            to_latin("Абс 722 йилда 822 йилда ҳужум қилдилар."),
            "Abs 722-yilda 822-yilda hujum qildilar.",
        )
        self.assertEqual(
            to_latin("Оссурияликлар Фаластинга 722 \t  йилда ҳужум қилдилар."),
            "Ossuriyaliklar Falastinga 722-yilda hujum qildilar.",
        )
        # TODO:
        # self.assertEqual(
        #     to_latin("Some text 200 йилда Фаластинни"),
        #     "Some text 200-yilda Falastinni",
        # )

    def test_rule_simple_small_range_years(self):
        self.assertEqual(to_latin("Милодий 70-80 йилларда"), "Milodiy 70-80-yillarda")
        self.assertEqual(
            to_latin("614-629 йилларда Қуддусни форслар эгаллади."),
            "614-629-yillarda Quddusni forslar egalladi.",
        )

    def test_rule_years_mixed(self):
        self.assertEqual(
            to_latin(
                "1967 йилдан 2012 йилгача 45 йил давомида Исроил ҳукумати Ғарбий соҳилдаги фаластинликнинг 27 000 та уйини бузиб ташлаган."
            ),
            "1967-yildan 2012-yilgacha 45 yil davomida Isroil hukumati G‘arbiy sohildagi falastinlikning 27 000 ta uyini buzib tashlagan.",
        )
        self.assertEqual(
            to_latin(
                "1980 йиллардан бери у ҳар 7 йилда бир марта ноқонуний қурилиш учун судга чақирув қоғози олган"
            ),
            "1980-yillardan beri u har 7 yilda bir marta noqonuniy qurilish uchun sudga chaqiruv qog‘ozi olgan",
        )

    # --- Edge cases with years ---
    def test_rule_years_not_ordinary(self):
        # TODO:
        # self.assertEqual(
        #     to_latin("Ўзбекистон тарихи 400 йилда"), "O'zbekiston tarixi 400 yilda"
        # )

        self.assertEqual(
            to_latin(
                "... йилдан бошлаб 400 йил давомида Фаластин Усмонийлар империяси таркибида бўлди."
            ),
            "... yildan boshlab 400 yil davomida Falastin Usmoniylar imperiyasi tarkibida bo‘ldi.",
        )
        self.assertEqual(
            to_latin("16 йилдан бери кўрмаган акаси билан учрашмоқчи эди"),
            "16 yildan beri ko‘rmagan akasi bilan uchrashmoqchi edi",
        )
        self.assertEqual(
            to_latin("100 йилдан бери кўрмаган акаси билан учрашмоқчи эди"),
            "100 yildan beri ko‘rmagan akasi bilan uchrashmoqchi edi",
        )
        self.assertEqual(
            to_latin("Ғазода одамлар 10, 20 ва 30 йиллаб судсиз ўтиришади."),
            "G‘azoda odamlar 10, 20 va 30 yillab sudsiz o‘tirishadi.",
        )
        self.assertEqual(
            to_latin(
                "Агар бу станция бўлмаганида, сўнгги 7 йил ичида Ғазода умуман чироқ бўлмасди."
            ),
            "Agar bu stansiya bo‘lmaganida, so‘nggi 7 yil ichida G‘azoda umuman chiroq bo‘lmasdi.",
        )
        self.assertEqual(
            to_latin(
                "Уни ҳукм қилишга улгуришмади, чунки 12 йиллик қамоқ жазоси сўралаётган суд жараёни арафасида уни озод қилишди."
            ),
            "Uni hukm qilishga ulgurishmadi, chunki 12 yillik qamoq jazosi so‘ralayotgan sud jarayoni arafasida uni ozod qilishdi.",
        )
        self.assertEqual(
            to_latin("45 йилга чўзилди."),
            "45 yilga cho‘zildi.",
        )
        self.assertEqual(
            to_latin(
                "У ҳар 10 йилда бир марта ноқонуний қурилиш учун судга чақирув қоғози олган"
            ),
            "U har 10 yilda bir marta noqonuniy qurilish uchun sudga chaqiruv qog‘ozi olgan",
        )
        self.assertEqual(
            to_latin("40 йилда форслар босқин уюштиришди"),
            "40 yilda forslar bosqin uyushtirishdi",
        )
        self.assertEqual(
            to_latin("Тўрт йилга чўзилди."),
            "To‘rt yilga cho‘zildi.",
        )

    def test_dates(self):
        self.assertEqual(
            to_latin("7 ноябрь"),
            "7-noyabr",
        )
        self.assertEqual(
            to_latin("17 ноябрь"),
            "17-noyabr",
        )
        self.assertEqual(
            to_latin("17 \t сентябрь"),
            "17-sentyabr",
        )
        self.assertEqual(
            to_latin("17 сентабрь"),
            "17 sentabr",
        )
        self.assertEqual(
            to_latin("27-декабрь"),
            "27-dekabr",
        )
        self.assertEqual(
            to_latin("2023 йил 7 октябрда ҲАМАС чорасизлик босими остида ҳужумга ўтди"),
            "2023-yil 7-oktyabrda HAMAS chorasizlik bosimi ostida hujumga o‘tdi",
        )
        self.assertEqual(
            to_latin("Икки йил блокададан кейин, 2008 йил 23 январ куни"),
            "Ikki yil blokadadan keyin, 2008-yil 23-yanvar kuni",
        )
        self.assertEqual(
            to_latin(
                "Икки йил блокададан кейин, 2008 йил 23 январ ва 25 январ кунлари"
            ),
            "Ikki yil blokadadan keyin, 2008-yil 23-yanvar va 25-yanvar kunlari",
        )

    # --- Rule 3: Handle 'е' (ye vs e) ---
    def test_rule_ye(self):
        self.assertEqual(to_latin("ер"), "yer")
        self.assertEqual(to_latin("иерархия"), "iyerarxiya")
        self.assertEqual(to_latin("абс уер абс"), "abs uyer abs")
        self.assertEqual(to_latin("бер"), "ber")
        self.assertEqual(to_latin("Телефон"), "Telefon")
        self.assertEqual(to_latin("Ғаниев Алишер"), "G‘aniyev Alisher")
        self.assertEqual(to_latin("абс kупе абс"), "abs kupe abs")
        self.assertEqual(to_latin("ателье"), "atelye")
        self.assertEqual(to_latin("абс Рельеф абс"), "abs Relyef abs")
        self.assertEqual(to_latin("Премьера"), "Premyera")
        self.assertEqual(
            to_latin(
                "бомбалар остида таслим бўлмайди, ҳеч қаерга қочмайди, ҳақиқат ҳамда ғалаба улар томонида эканлигини қатъий билади."
            ),
            "bombalar ostida taslim bo‘lmaydi, hech qayerga qochmaydi, haqiqat hamda g‘alaba ular tomonida ekanligini qat’iy biladi.",
        )
        self.assertEqual(
            to_latin(
                "Унинг айтишича, УНИСEФ ўзига тегишли касалхоналар учун нималардир олиб келади, лекин бу етарли эмас."
            ),
            "Uning aytishicha, UNISEF o‘ziga tegishli kasalxonalar uchun nimalardir olib keladi, lekin bu yetarli emas.",
        )

    # --- Rule 4: Handle 'ъ' + 'я / ю / е / ё' (not apostrophe) ---
    def test_rule_я_ю_е_ё_ъ(self):
        self.assertEqual(to_latin("объект"), "obyekt")
        self.assertEqual(to_latin("съезд"), "syezd")
        self.assertEqual(
            to_latin(
                "Ҳеч ким тўхтамайди, саволларга жавоб бермайди, қадамини тезлатиб, фотообъективдан ўзини пана қилиб ўтиб кетади."
            ),
            "Hech kim to‘xtamaydi, savollarga javob bermaydi, qadamini tezlatib, fotoobyektivdan o‘zini pana qilib o‘tib ketadi.",
        )

    # --- Rule 5: Handle 'ь' ---
    def test_rule_ь(self):
        # Ignore
        self.assertEqual(to_latin("мебель"), "mebel")
        self.assertEqual(to_latin("компьютер"), "kompyuter")
        # Becomes y
        self.assertEqual(to_latin("павильон"), "pavilyon")
        self.assertEqual(to_latin("батальон"), "batalyon")

    # --- Rule 6: Handle 'я', 'ю', 'е', 'ё' ---
    def test_rule_я_ю_е_ё(self):
        self.assertEqual(to_latin("январь"), "yanvar")
        self.assertEqual(to_latin("юбилей"), "yubiley")
        self.assertEqual(to_latin("сентябрь"), "sentyabr")
        self.assertEqual(to_latin("молекуляр"), "molekulyar")
        self.assertEqual(to_latin("дунё"), "dunyo")

    # --- Rule 7: Handle shortened names with Я, Ю, Е, Ё, Ч, Ш initial letters ---
    def test_rule_shortened_name_initials_я_ю_е_ё_ч_ш(self):
        # Y-letters
        self.assertEqual(to_latin("Я.Қурбонов"), "Y.Qurbonov")
        self.assertEqual(to_latin("Ё. Саъдиев"), "Y. Sa’diyev")
        self.assertEqual(
            to_latin("Унинг исми Е. Саъдиев эди шакилли"),
            "Uning ismi Y. Sa’diyev edi shakilli",
        )
        self.assertEqual(
            to_latin("Унинг исми Юсуф - Ю. А. Саъдиев эди шакилли"),
            "Uning ismi Yusuf - Y. A. Sa’diyev edi shakilli",
        )
        self.assertEqual(
            to_latin("Унинг исми Юсуф - А.Ю. Саъдиева эди шакилли"),
            "Uning ismi Yusuf - A.Y. Sa’diyeva edi shakilli",
        )

        # Y-letters opposite
        self.assertEqual(
            to_latin("Унинг исми Юсуф - Саъдиев Ю. А. эди шакилли"),
            "Uning ismi Yusuf - Sa’diyev Y. A. edi shakilli",
        )
        self.assertEqual(
            to_latin("Унинг исми Юсуф - Саъдиева А.Ю. эди шакилли"),
            "Uning ismi Yusuf - Sa’diyeva A.Y. edi shakilli",
        )

        # Ч letter
        self.assertEqual(
            to_latin("Чолқуши - Ч.Абдуллаев some other text"),
            "Cholqushi - Ch.Abdullayev some other text",
        )
        self.assertEqual(
            to_latin("Унинг исми Юсуф - Ю. Ч. Саъдиев эди шакилли"),
            "Uning ismi Yusuf - Y. Ch. Sa’diyev edi shakilli",
        )

        # Ш letter
        self.assertEqual(
            to_latin("Шавкат Миромонович Мирзиёев - Ш.М.Мирзиёев"),
            "Shavkat Miromonovich Mirziyoyev - Sh.M.Mirziyoyev",
        )
        self.assertEqual(
            to_latin("Унинг исми Юсуф - Ю. Ч. Саъдиев эди шакилли"),
            "Uning ismi Yusuf - Y. Ch. Sa’diyev edi shakilli",
        )

        # Multiple
        self.assertEqual(
            to_latin(
                "Уларнинг исмлари Юсуф - Ю. Ч. Саъдиев ва Шерзод - Ш.Н. Абдуллаев эди шакилли"
            ),
            "Ularning ismlari Yusuf - Y. Ch. Sa’diyev va Sherzod - Sh.N. Abdullayev edi shakilli",
        )
        self.assertEqual(
            to_latin("Ш.М.Мирзиёев, Саъдиев А. Ю., Я.Қурбонов"),
            "Sh.M.Mirziyoyev, Sa’diyev A. Y., Y.Qurbonov",
        )

        # Non-capturing letters
        self.assertEqual(
            to_latin("дейди менга Наталя. Агар қамал давом этса"),
            "deydi menga Natalya. Agar qamal davom etsa",
        )
        self.assertEqual(
            to_latin("дейди Яҳё.Унинг чап қўли"), "deydi Yahyo.Uning chap qo‘li"
        )

    # --- Rule 8: Handle 'ц' (ts vs s) ---
    def test_rule_ts(self):
        self.assertEqual(to_latin("федерация"), "federatsiya")
        self.assertEqual(to_latin("абс милиция абс"), "abs militsiya abs")
        self.assertEqual(to_latin("цирк"), "sirk")
        self.assertEqual(to_latin("абс цирк абс"), "abs sirk abs")
        self.assertEqual(to_latin("концепт"), "konsept")
        self.assertEqual(to_latin("абс акция абс"), "abs aksiya abs")
        self.assertEqual(to_latin("шприц"), "shpris")
        self.assertEqual(to_latin("абс шприц абс"), "abs shpris abs")
        self.assertEqual(
            to_latin(
                "редакциядан келган қоғозларни ва аккредитация тўғрисидаги хатни қайта-қайта ўқирди."
            ),
            "redaksiyadan kelgan qog‘ozlarni va akkreditatsiya to‘g‘risidagi xatni qayta-qayta o‘qirdi.",
        )

    # --- Rule 9: Handle '-ev (-eva)', '-yev (-yeva)', '-ov (-ova)' surname endings ---
    def test_rule_surnames(self):
        self.assertEqual(to_latin("Абдулла Тошев"), "Abdulla Toshev")
        self.assertEqual(to_latin("Қиличева Азиза"), "Qilicheva Aziza")
        self.assertEqual(to_latin("Вафоев абс"), "Vafoyev abs")
        self.assertEqual(to_latin("абс Жумаева"), "abs Jumayeva")
        self.assertEqual(
            to_latin("абс. Абс Абдусамадов абс"), "abs. Abs Abdusamadov abs"
        )

    # --- Rule 10: Abbreviations ---
    def test_rule_abbreviations(self):
        self.assertEqual(to_latin("АҚШ"), "AQSH")
        self.assertEqual(to_latin("ТОШКEНТ"), "TOSHKENT")
        self.assertEqual(to_latin("абс. Абс ҚЎШЧИНОР абс"), "abs. Abs QO‘SHCHINOR abs")
        self.assertEqual(to_latin("абс. Абс ЮНЕСКО абс"), "abs. Abs YUNESKO abs")

    # --- Rule 11: Words ending with 'к', 'қ', 'ғ' + -га, -гач, -гунча, -гани, -гудек, -ган ---
    def test_rule_к_қ_ғ_word_ending_and_г_addition(self):
        # No difference in the Cyrillic and Latin rule
        self.assertEqual(to_latin("йўлаккача"), "yo‘lakkacha")
        self.assertEqual(to_latin("кечиккудек"), "kechikkudek")
        self.assertEqual(to_latin("қишлоққа"), "qishloqqa")
        self.assertEqual(to_latin("тўпиққача"), "to‘piqqacha")
        self.assertEqual(
            to_latin("Мени махсус хизматлар қийноққа солишди."),
            "Meni maxsus xizmatlar qiynoqqa solishdi.",
        )
        self.assertEqual(to_latin("педагогга"), "pedagogga")

        # In Cyrillic: 'ғ' + -га, -гач, -гунча, -гани, -гудек, -ган = ққ...
        # In Latin: bog‘ + ga = bog‘ga
        # Example: боғ + га = боққа but in Latin it's bog‘ga
        self.assertEqual(to_latin("белбоққа"), "belbog‘ga")
        self.assertEqual(to_latin("боққа"), "bog‘ga")
        self.assertEqual(to_latin("Боққа"), "Bog‘ga")
        self.assertEqual(to_latin("абс БОҚҚА абс"), "abs BOG‘GA abs")
        self.assertEqual(to_latin("боғ-роққа"), "bog‘-rog‘ga")
        self.assertEqual(to_latin("гулбоққа"), "gulbog‘ga")
        self.assertEqual(to_latin("буққа"), "bug‘ga")
        self.assertEqual(to_latin("доққа"), "dog‘ga")
        self.assertEqual(to_latin("тоққа"), "tog‘ga")
        self.assertEqual(to_latin("тиққа"), "tig‘ga")
        self.assertEqual(to_latin("сарёққа"), "saryog‘ga")
        self.assertEqual(to_latin("ёруққа"), "yorug‘ga")
        self.assertEqual(to_latin("маблаққа"), "mablag‘ga")
        self.assertEqual(to_latin("уруққа"), "urug‘ga")
        self.assertEqual(to_latin("чўққа"), "cho‘g‘ga")
        self.assertEqual(to_latin("зоққа"), "zog‘ga")
        self.assertEqual(
            to_latin("Белбоққа, боққа, боғ-роққа"), "Belbog‘ga, bog‘ga, bog‘-rog‘ga"
        )
        self.assertEqual(
            to_latin("Нонга етарли маблағга эга эмас."),
            "Nonga yetarli mablag‘ga ega emas.",
        )
        self.assertEqual(
            to_latin(
                "У ёшлигида карате билан шуғуллангани ва қора белбоққа эга бўлгани учунгина қамоқхонада соғлиғини сақлаб қолганини таъкидлайди"
            ),
            "U yoshligida karate bilan shug‘ullangani va qora belbog‘ga ega bo‘lgani uchungina qamoqxonada sog‘lig‘ini saqlab qolganini ta’kidlaydi",
        )
        self.assertEqual(
            to_latin("йоққа"),
            "yoqqa",
            "Special case: better don't convert here. It can be 'yog‘ + ga' or 'u yoqdan-bu yoqqa yurardi'",
        )

    # --- Rule 12: Words ending with 'в' + -илламоқ = vullamoq (in Latin) ---
    def test_rule_в_and_илламоқ_combination(self):
        # No difference in the Cyrillic and Latin rule
        self.assertEqual(to_latin("чирилламоқ"), "chirillamoq")
        self.assertEqual(to_latin("тақилла"), "taqilla")

        # Transform
        self.assertEqual(to_latin("шовилламоқ"), "shovullamoq")
        self.assertEqual(to_latin("Абс вовилла абс"), "Abs vovulla abs")
        self.assertEqual(
            to_latin("Эски шаҳар ҳувиллаб қолган."), "Eski shahar huvullab qolgan."
        )

        # No match
        self.assertEqual(
            to_latin("ота-оналарининг виллалари"), "ota-onalarining villalari"
        )

    def test_capitalization_logic(self):
        self.assertEqual(to_latin("Салом"), "Salom")
        self.assertEqual(to_latin("ЧАЙ"), "CHAY")
        self.assertEqual(to_latin("Цирк"), "Sirk")
        self.assertEqual(to_latin("АКЦИЯ"), "AKSIYA")
        self.assertEqual(to_latin("ТОсҲКеНТ"), "TOsHKeNT")

    def test_sh_ch_yo_yu_ya(self):
        self.assertEqual(to_latin("шаҳар"), "shahar")
        self.assertEqual(to_latin("чой"), "choy")
        self.assertEqual(to_latin("ёрдам"), "yordam")
        self.assertEqual(to_latin("юлдуз"), "yulduz")
        self.assertEqual(to_latin("ярим"), "yarim")

    def test_non_cyrillic_content(self):
        self.assertEqual(to_latin("123! @#"), "123! @#")
        self.assertEqual(to_latin("Hello Dunyo"), "Hello Dunyo")

    def test_soft_and_hard_signs(self):
        self.assertEqual(to_latin("маъно"), "ma’no")
        self.assertEqual(to_latin("лаънат"), "la’nat")
        # ь is usually ignored in Uzbek Latin
        self.assertEqual(to_latin("альбом"), "albom")

    def test_in_long_text(self):
        text = """
Мисрда инқилоб, ундан кейин эса аксилинқилоб юз берди, Сурия уруш
домида қолди, Фаластин 2014 йилда БМТ ва ЮНЕСКО томонидан тан олинди,
Фаластин партиялари атрофида катта сиёсий ўйинлар авжланди, Фаластин халқи
эса аввалгидай ўзининг аччиқ тақдири билан ёлғиз қолмоқда ва муқовамат
(қаршилик кўрсатишда) да давом этмоқда. Ҳа тўғри, баъзан Ғазони эсга олишади,
лекин ачиниш ва ғазабнок нутқлар тўкиб солишдан нарига ўтишмайди.

Фаластин халқи Инжилда тавсифи келтирилган қандайдир мавҳум филистиминлар эмас.
Ундай бўлса замонавий фаластинликлар - кимлар, қаердан пайдо бўлишган? Икки юз
йил олдин ҳеч қандай фаластин халқи йўқ эди, балки араб тилида сўзлашувчи Усмонли
империяси провинциясининг аҳолиси бор эди. Ушбу аҳолининг илдизи қадимий бўлиб,
уларнинг келиб чиқиши ҳақидаги мунозаралар ҳозиргача давом этмоқда, аммо улар
илк даврларда ўзларини замонавий тушунчадаги халқ (нация) деб ҳис қилишмаган.

Европалик яҳудийларнинг бу ерларга кўчириб келтирилиши арафасида Фаластин
вилоятининг аҳолиси 450 минг кишига яқин эди. 1880 йилда аҳолини рўйхатга олган
турклар уларнинг миллатини эмас, фақат конфессиясини қайд этган. Уларнинг 24 000 таси
яҳудий бўлиб, аксарияти шаҳарликлар эди. Яҳудийлар бошқа қўшнилари сингари араб
ва турк тилларида сўзлашар, аҳолининг қолган қисми мусулмон ва насронийлардан иборат
бўлган.

Ғазодаги энг йирик “Шифа” касалхонаси бош шифокори доктор Метат Аббос
шундай дейди: “Бомбардимонлардан кейин ҳалок бўлган болаларнинг жасадларида 33 та
заҳарли элемент, жумладан, уран борлиги аниқланди. Биз оқ фосфорнинг нафас олиш ва
иммун тизимига таъсирини ўрганаяпмиз”.
"""
        expected = """
Misrda inqilob, undan keyin esa aksilinqilob yuz berdi, Suriya urush
domida qoldi, Falastin 2014-yilda BMT va YUNESKO tomonidan tan olindi,
Falastin partiyalari atrofida katta siyosiy o‘yinlar avjlandi, Falastin xalqi
esa avvalgiday o‘zining achchiq taqdiri bilan yolg‘iz qolmoqda va muqovamat
(qarshilik ko‘rsatishda) da davom etmoqda. Ha to‘g‘ri, ba’zan G‘azoni esga olishadi,
lekin achinish va g‘azabnok nutqlar to‘kib solishdan nariga o‘tishmaydi.

Falastin xalqi Injilda tavsifi keltirilgan qandaydir mavhum filistiminlar emas.
Unday bo‘lsa zamonaviy falastinliklar - kimlar, qayerdan paydo bo‘lishgan? Ikki yuz
yil oldin hech qanday falastin xalqi yo‘q edi, balki arab tilida so‘zlashuvchi Usmonli
imperiyasi provinsiyasining aholisi bor edi. Ushbu aholining ildizi qadimiy bo‘lib,
ularning kelib chiqishi haqidagi munozaralar hozirgacha davom etmoqda, ammo ular
ilk davrlarda o‘zlarini zamonaviy tushunchadagi xalq (natsiya) deb his qilishmagan.

Yevropalik yahudiylarning bu yerlarga ko‘chirib keltirilishi arafasida Falastin
viloyatining aholisi 450 ming kishiga yaqin edi. 1880-yilda aholini ro‘yxatga olgan
turklar ularning millatini emas, faqat konfessiyasini qayd etgan. Ularning 24 000 tasi
yahudiy bo‘lib, aksariyati shaharliklar edi. Yahudiylar boshqa qo‘shnilari singari arab
va turk tillarida so‘zlashar, aholining qolgan qismi musulmon va nasroniylardan iborat
bo‘lgan.

G‘azodagi eng yirik “Shifa” kasalxonasi bosh shifokori doktor Metat Abbos
shunday deydi: “Bombardimonlardan keyin halok bo‘lgan bolalarning jasadlarida 33 ta
zaharli element, jumladan, uran borligi aniqlandi. Biz oq fosforning nafas olish va
immun tizimiga ta’sirini o‘rganayapmiz”.
"""
        self.assertEqual(to_latin(text), expected)


if __name__ == "__main__":
    unittest.main()
