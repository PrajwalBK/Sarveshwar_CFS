"""ISO 6346 syntax/check digit, without guessing OCR character substitutions.

Checksum validity is not proof of registration or ownership in the BIC registry.
"""
import re
from app.domain import OCRRead, Validation

LETTER_VALUES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (n for n in range(10, 39) if n % 11)))
STRUCTURE = re.compile(r'^[A-Z]{3}[UJZ][0-9]{7}$')
CANDIDATE = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){7})(?![A-Z0-9])')


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
        raw = read.raw_text[:4096]
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
                normalized = normalize(raw)[:256]
        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = valid_format and check_digit(normalized[:10]) == int(normalized[10])
        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        return Validation(raw, normalized, read.confidence, valid_format, valid_digit, status)
