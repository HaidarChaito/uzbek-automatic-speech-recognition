import math
import re


class NumberToUzbekWord:
    """
    Number to Uzbek Words Converter
    Converts digits to Uzbek word form for ASR transcription alignment

    Supports:
        - Basic integers: 42 → "qirq ikki"
        - Negative numbers: -15 → "minus o'n besh"
        - Decimals: 3.14 → "uch butun o'n to'rt"
        - Large numbers with separators: 1,234,567 → "bir million ikki yuz..."
        - European format: 14 245,48 → "o'n to'rt ming..."
        - Ranges: 5-10 → "besh - o'n"
        - Ordinals: 2-qism → "ikkinchi qism"
        - Currencies: $5.5, 100€, ₽50 → "besh butun besh dollar", "yuz yevro", "ellik rubl"
        - Percentages: 60% → "oltmish foiz"
        - Abbreviations: 17,3 mlrd → "17,3 milliard"
        - Quantifiers: 5ta → "beshta"
        - Case suffixes: $32,08 dan, 60% ga → oltmish foizga

    Limitations:
        - Doesn't appropriately support time, roman numerals, phone numbers, and fractions
    """

    def __init__(self):
        # Basic numbers 0-9
        self.ones = {
            0: "nol",
            1: "bir",
            2: "ikki",
            3: "uch",
            4: "to'rt",
            5: "besh",
            6: "olti",
            7: "yetti",
            8: "sakkiz",
            9: "to'qqiz",
        }

        # Tens
        self.tens = {
            10: "o'n",
            20: "yigirma",
            30: "o'ttiz",
            40: "qirq",
            50: "ellik",
            60: "oltmish",
            70: "yetmish",
            80: "sakson",
            90: "to'qson",
        }

        # Scale words
        self.scales = {
            100: "yuz",
            1_000: "ming",
            1_000_000: "million",
            1_000_000_000: "milliard",
            1_000_000_000_000: "trillion",
        }

        # Currency words
        self.currencies = {
            "$": "dollar",
            "€": "yevro",
            "£": "funt sterling",
            "₽": "rubl",
        }

    def normalize(self, text):
        """
        Find and convert all number patterns in Uzbek text

        Args:
            text: Input text with numbers

        Returns:
            Text with all numbers converted to words
        """
        currency_symbols_regex = "[" + "".join(self.currencies.keys()) + "]"

        def convert_match(match):
            original = match.group(0)

            additional_suffix = ""
            number_part = original

            # 1. Extract currency prefix if present
            prefix_match = re.match(rf"^({currency_symbols_regex}) ?", original)
            if prefix_match:
                currency_symbol = prefix_match.group(1)
                additional_suffix = " " + self.currencies[currency_symbol]
                number_part = original[prefix_match.end() :]

            # 2. Extract percentage or currency suffix if present
            suffix_match = re.search(rf"(%|{currency_symbols_regex})$", number_part)
            if suffix_match:
                matched_symbol = suffix_match.group(1)
                if matched_symbol == "%":
                    additional_suffix = " foiz"
                elif matched_symbol in self.currencies:
                    additional_suffix = " " + self.currencies[matched_symbol]
                else:
                    print(f"Error: Unhandled suffix type detected: {matched_symbol}")
                number_part = number_part[: suffix_match.start()].strip()

            # 3. Range values with formatted numbers: 2,000,000-3,000,000 or 2-3 or 1.14-3.14
            range_match = re.match(
                r"^(-?\d{1,3}(?:[, ]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)"
                + r"-"
                + r"(\d{1,3}(?:[, ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)$",  # can be even 2,000.89-2,500.95
                number_part,
            )
            if range_match:
                num1_str = range_match.group(1).replace(",", "").replace(" ", "")
                num2_str = range_match.group(2).replace(",", "").replace(" ", "")

                # Convert to float or int depending on whether it contains a decimal point
                num1 = self._convert(
                    float(num1_str) if "." in num1_str else int(num1_str)
                )
                num2 = self._convert(
                    float(num2_str) if "." in num2_str else int(num2_str)
                )

                # Don't add space around hyphen if one word number
                is_first_number_one_word = len(num1.split(" ")) == 1
                if is_first_number_one_word:
                    return f"{num1}-{num2}{additional_suffix}"

                return f"{num1} - {num2}{additional_suffix}"

            # 4. Ordinal numbers: 2-qism
            ordinal_number_match = re.match(r"(-?\d+)-([a-zA-Z]+)", number_part)
            if ordinal_number_match:
                number_str = ordinal_number_match.group(1)  # 2
                suffix = ordinal_number_match.group(2)  # qism

                number_in_words = self._convert(int(number_str))
                if number_in_words.lower().endswith("i"):
                    suffix = "nchi " + suffix
                else:
                    suffix = "inchi " + suffix

                return f"{number_in_words}{suffix}{additional_suffix}"

            # 5. Big numbers with commas or spaces: 3,000,000.14 or 3 000 000.14 or 14 245,48
            if re.match(r"-?\d{1,3}(?:[, ]\d{3})+", number_part):
                # Check if this is European-style decimal (space as thousand separator, comma as decimal)
                # Pattern: digits + spaces (for thousands) + comma + 1-2 digits
                european_decimal_match = re.match(
                    r"^(-?\d{1,3}(?: \d{3})+),(\d{1,2})$", number_part
                )

                if european_decimal_match:
                    # European style: "14 245,48" → "14245.48"
                    cleaned = number_part.replace(" ", "").replace(",", ".")
                else:
                    # Standard style with comma/space as thousand separators: "3,000,000" or "3 000 000"
                    cleaned = number_part.replace(",", "").replace(" ", "")

                # Convert to float or int depending on whether it contains a decimal point
                number_in_words = self._convert(
                    float(cleaned) if "." in cleaned else int(cleaned)
                )
                return f"{number_in_words}{additional_suffix}"

            # 6. Decimal numbers: 3.5 or 3,5
            if re.match(r"-?\d+[.,]\d+", number_part):
                decimal_str = number_part.replace(",", ".")
                return f"{self._convert(decimal_str)}{additional_suffix}"

            # 7. Just numbers (including negative): -15, 6
            if number_part.lstrip("-").isdigit():
                return f"{self._convert(int(number_part))}{additional_suffix}"

            return original

        # Pattern that matches all number types (order matters - more specific patterns first)
        pattern = (
            rf"(?:{currency_symbols_regex} ?)?"  # optional currency prefix: $, €, £, ₽
            + r"(?:(?<!\w)-\d+|\d+)"  # required main number (can be negative but exclude hyphen words: e.g. covid-19 => match 19, not -19): 10, -156
            + r"(?:(?:,\d{3})+| (?:\d{3} )+\d{3})?"  # optional thousand separators (all commas OR all spaces): ,000,000 |  000 000
            + r"(?:[, ]\d{3})*"  # optional thousand separators (comma or space):
            + r"(?:[.,]\d+)?"  # optional decimal part: .1415 | ,1415
            + r"(?:-\d+(?:[, ]\d{3})*(?:[.,]\d+)?)?"  # optional range with second number (can be formatted or decimal nuber): -10,000 | -3.14 [full e.g. 8,000-10,000 | 1.14-3.14]
            + r"(?:-[a-zA-Z]+)?"  # optional ordinal suffix with word: -qism
            + rf"(?: ?%| ?{currency_symbols_regex})?"  # optional percentage or currency suffix (with whitespace): % | $
        )

        # Preprocess to separate alphanumeric combinations
        text = self._preprocess_text(text)

        result = re.sub(pattern, convert_match, text)

        # Ensure assimilation rule: bir + ta = bitta [not birta]
        result = re.sub(r"\bbirta\b", "bitta", result)

        return result

    def _preprocess_text(self, text):
        """
        Handle alphanumeric combinations and abbreviations:
            1. Expand abbreviated number words ONLY after numbers: "17,3 mlrd" → "17,3 milliard"
            2. Separate numbers from other letters: "Intel Core i10" → "Intel Core i 10"
            3. Merge numbers with "ta" quantifier: "5 ta" → "5ta"
            4. Merge complex numbers with percentage and Uzbek cases: "1,234.56% ga" → "1,234.56%ga"
            5. Merge complex numbers with currency and Uzbek cases: "$32,08 ga" → "$32,08ga"
        """
        # Expand abbreviated large number words
        abbreviations = {
            "mln": "million",
            "mlrd": "milliard",
            "trln": "trillion",
        }

        for abbr, full in abbreviations.items():
            # Match: number (with optional decimals/commas) + optional space + abbreviation + optional dot
            # Examples: "17,3 mlrd", "344.6mlrd", "5 mln."
            pattern = r"(\d+(?:[.,]\d+)?)\s*" + re.escape(abbr) + r"\.?"
            text = re.sub(pattern, rf"\1 {full}", text)

        # Separate ALL alphanumeric combinations
        def separate_match(match):
            full = match.group(0)

            # Letter(s) + number: X7 → X 7, i10 → i 10
            if re.match(r"^[A-Za-z]+\d+", full):
                split_point = re.search(r"\d", full).start()
                return full[:split_point] + " " + full[split_point:]

            # Number + letter(s): 41A → 41 A, 5kg → 5 kg
            if re.match(r"^\d+[A-Za-z]+", full):
                split_point = re.search(r"[A-Za-z]", full).start()
                return full[:split_point] + " " + full[split_point:]

            return full

        pattern = r"\b[A-Za-z]*\d+[A-Za-z]+\b|\b[A-Za-z]+\d+\b"
        text = re.sub(pattern, separate_match, text)

        # ==== Merge Back Special Cases ====

        # Merge "number + space + ta" → "number+ta"
        # "5 ta" → "5ta", "10 ta" → "10ta"
        text = re.sub(r"\b(\d+)\s+(ta)\b", r"\1\2", text)

        # Define complex number pattern (same as the main regex structure)
        currency_symbols_regex = "[" + "".join(self.currencies.keys()) + "]"
        complex_number = (
            r"-?\d+"  # main number
            + r"(?:(?:,\d{3})+| (?:\d{3} )+\d{3})?"  # thousand separators
            + r"(?:[, ]\d{3})*"  # additional separators
            + r"(?:[.,]\d+)?"  # decimal part
            + r"(?:-\d+(?:[, ]\d{3})*(?:[.,]\d+)?)?"  # range
        )

        uzbek_cases_regex = r"ning|ni|ga|ka|qa|da|dan|ini"

        # Merge "complex_number% + space + cases" → "complex_number%cases"
        # "1,234.56% ga" → "1,234.56%ga", "60% dan" → "60%dan"
        text = re.sub(
            rf"\b({complex_number}) ?% ?({uzbek_cases_regex})\b", r"\1%\2", text
        )

        # Merge "currency + number + space + cases" → "currency+number+cases"
        # "$1,234.56 ga" → "$1,234.56ga", "1,234.56$ ni" → "1,234.56$ni"
        text = re.sub(
            rf"({currency_symbols_regex} ?{complex_number}|{complex_number} ?{currency_symbols_regex}) ?({uzbek_cases_regex})\b",
            r"\1\2",
            text,
        )

        return text

    def _convert_up_to_two_digits(self, n):
        """Convert numbers 0-99 to words from groups"""
        if n < 10:
            return self.ones[n]

        tens_digit = (n // 10) * 10
        ones_digit = n % 10

        if ones_digit == 0:
            return self.tens[tens_digit]
        else:
            return f"{self.tens[tens_digit]} {self.ones[ones_digit]}"

    def _convert_up_to_three_digits(self, n):
        """Convert numbers 0-999 to words from groups"""
        if n == 0:
            return ""

        if n < 100:
            return self._convert_up_to_two_digits(n)

        hundreds = n // 100
        remainder = n % 100

        # Handle hundreds
        if hundreds == 1:
            result = "bir yuz"
        else:
            result = f"{self.ones[hundreds]} yuz"

        # Add remainder if exists
        if remainder > 0:
            result += f" {self._convert_up_to_two_digits(remainder)}"

        return result

    def _convert_integer(self, n):
        """Convert any integer to Uzbek words"""
        if n < 0:
            return f"minus {self._convert_integer(-n)}"

        if n in self.ones:
            return self.ones[n]

        # be explicit starting with "million" => "bir million"
        if n < 1_000_000 and n in self.scales:
            return self.scales[n]

        # Break number into groups of three digits
        parts = []
        scale_index = 0
        scales_list = [1, 1_000, 1_000_000, 1_000_000_000]

        while n > 0:
            group = n % 1_000
            if group > 0:
                group_words = self._convert_up_to_three_digits(group)

                # Add scale word if needed
                if scale_index > 0:
                    scale_word = self.scales[scales_list[scale_index]]
                    parts.append(f"{group_words} {scale_word}")
                else:
                    parts.append(group_words)

            n = n // 1_000  # floor division
            scale_index += 1

        return " ".join(reversed(parts))

    def _convert_decimal(self, number_str):
        """Convert decimal numbers like 3.14 or 10.001 to words"""
        parts = number_str.split(".")
        integer_part = int(parts[0])

        result = self._convert_integer(integer_part)

        if len(parts) > 1:
            decimal_part_str = parts[1]

            # Count leading zeros
            leading_zeros_count = len(decimal_part_str) - len(
                decimal_part_str.lstrip("0")
            )
            # If decimal part is all zeros, ignore it
            if decimal_part_str.replace("0", "") == "":
                return result

            # Convert the actual numeric value to int (e.g., "001" -> 1)
            decimal_part_int = int(decimal_part_str)

            result += " butun "

            # Add "nol" for each leading zero
            if leading_zeros_count > 0:
                result += "nol " * leading_zeros_count

            result += self._convert_integer(decimal_part_int)

        return result

    def _convert(self, number):
        """
        Main conversion method
        Args:
            number: Can be int, float, or string representation
        Returns:
            Uzbek word representation
        """
        # Convert to string to handle various input types
        number_str = str(number).strip()

        try:
            # Handle decimal numbers
            if "." in number_str:
                return self._convert_decimal(number_str)

            # Handle integers
            n = int(number_str)
            return self._convert_integer(n)
        except ValueError as err:
            print(f"Error: Invalid number format: {number_str}\n{err}")
            return math.nan


# TODO: handle time, roman numbers, phone numbers
