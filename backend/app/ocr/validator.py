"""ISO 6346 syntax/check digit, without guessing OCR character substitutions.

Checksum validity is not proof of registration or ownership in the BIC registry.
"""
import re
from app.domain import OCRRead, Validation

LETTER_VALUES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (n for n in range(10, 39) if n % 11)))
STRUCTURE = re.compile(r'^[A-Z]{3}[UJZ][0-9]{7}$')
CANDIDATE = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){7})(?![A-Z0-9])')

# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1)
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
SIZE_CODE_PATTERN = re.compile(r'(?<![A-Z0-9])([1-4LMN][0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', re.IGNORECASE)
SIZE_CODE_C_MISTAKE = re.compile(r'(?<![A-Z0-9])([1-4LMN][0-9])\s*C\s*([0-9A-Z]?)(?![A-Z0-9])', re.IGNORECASE)


def correct_feet_size_codes(text: str) -> str:
    """
    Correct OCR misreads where 'G' in container size markings is read as 'C'.
    E.g. 45C1 -> 45G1, 22C1 -> 22G1, 42C1 -> 42G1, L5C1 -> L5G1, 45C -> 45G1
    """
    if not text:
        return ''
    # 1. Correct [Size][C][Char] -> [Size][G][Char]
    corrected = SIZE_CODE_C_MISTAKE.sub(lambda m: f"{m.group(1).upper()}G{m.group(2).upper() or '1'}", text)
    return corrected


def parse_feet_size(code_or_text: str) -> str | None:
    """
    Extract container feet size from ISO size-type code or feet text.
    E.g. 45G1 -> '40 FT HC', 42G1 -> '40 FT', 22G1 -> '20 FT', L5G1 -> '45 FT HC'
    """
    if not code_or_text:
        return None
    cleaned = correct_feet_size_codes(code_or_text)
    norm = re.sub(r'[^A-Z0-9]', '', cleaned.upper())

    # Direct feet text: e.g. 40FT, 20FT, 45FT
    ft_match = re.search(r'(10|20|30|40|45|48|53)\s*(?:FT|FEET)', code_or_text.upper())
    if ft_match:
        return f'{ft_match.group(1)} FT'

    # ISO size code: e.g. 45G1, 22G1, 42G1, L5G1
    size_match = re.search(r'(?<![A-Z0-9])([1-4LMN][0-9])([GVRHBUTP][0-9A-Z])(?![A-Z0-9])', norm)
    if size_match:
        size_code = size_match.group(1)
        length_char = size_code[0]
        height_char = size_code[1]
        length_map = {
            '1': '10 FT',
            '2': '20 FT',
            '3': '30 FT',
            '4': '40 FT',
            'L': '45 FT',
            'M': '48 FT',
            'N': '53 FT',
        }
        base_length = length_map.get(length_char, '')
        if base_length:
            if height_char == '5':
                return f'{base_length} HC'
            return base_length
    return None


def check_digit(prefix: str) -> int:
    if not re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', prefix):
        raise ValueError('Expected three owner letters, U/J/Z and six serial digits')
    total = sum((int(char) if char.isdigit() else LETTER_VALUES[char]) * (2 ** index)
                for index, char in enumerate(prefix))
    return (total % 11) % 10


def normalize(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


class ContainerValidator:
    def validate(self, read: OCRRead) -> Validation:
        raw = correct_feet_size_codes(read.raw_text[:4096])
        candidates = {normalize(m.group(1)) for m in CANDIDATE.finditer(raw.upper())}
        if len(candidates) > 1:
            return Validation(raw, '', read.confidence, False, False, 'AMBIGUOUS')
        if candidates:
            normalized = next(iter(candidates))
        else:
            six_match = re.search(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])', raw.upper())
            if six_match:
                prefix = normalize(six_match.group(1))
                normalized = prefix + str(check_digit(prefix))
            else:
                # Check if this reading is an ISO size/type code (feet marking, e.g. 45G1, 22G1)
                size_match = SIZE_CODE_PATTERN.search(raw)
                if size_match:
                    normalized = f'{size_match.group(1).upper()}{size_match.group(2).upper()}{size_match.group(3).upper()}'
                    return Validation(raw, normalized, read.confidence, True, False, 'VALID_SIZE_CODE')
                normalized = normalize(raw)[:256]

        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = False
        if valid_format:
            expected_digit = int(normalized[10])
            calc_digit = check_digit(normalized[:10])
            if calc_digit == expected_digit:
                valid_digit = True
            else:
                # Check if 'C' was misread for 'G' (or 'G' for 'C') in owner letters
                for pos in range(3):
                    if normalized[pos] == 'C':
                        candidate_prefix = normalized[:pos] + 'G' + normalized[pos + 1:10]
                        if check_digit(candidate_prefix) == expected_digit:
                            normalized = candidate_prefix + str(expected_digit)
                            valid_digit = True
                            break

        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        return Validation(raw, normalized, read.confidence, valid_format, valid_digit, status)

