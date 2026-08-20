"""
Plate/OCR text validation to prevent prompt and instruction text from being stored as plates.

The OCR model (e.g. Qwen-VL) can sometimes echo back the prompt or instruction phrases
instead of actual image text. Such output must be rejected before storage.
"""

# Maximum length for a single plate/trailer ID field (reasonable upper bound)
MAX_PLATE_LENGTH = 50

# Minimum sensible length for a plate / trailer id. Anything shorter is noise
# (single character OCR fragments, stray punctuation, etc.).
MIN_PLATE_LENGTH = 2

# Strings that look like "OCR found nothing" rather than a real plate.
# Compared case-insensitively against the FULL trimmed text — so a real plate
# that happens to start with "NO" (unlikely but possible) is not rejected.
# Add new entries here if you see another placeholder in the wild.
PLACEHOLDER_VALUES = frozenset({
    "",
    "-",
    "--",
    "?",
    "??",
    "???",
    ".",
    "..",
    "...",
    "NONE",
    "NULL",
    "NIL",
    "N/A",
    "NA",
    "UNKNOWN",
    "NO TEXT",
    "NO PLATE",
    "NO_PLATE",
    "NO TRAILER",
    "NOT FOUND",
    "NOTFOUND",
    "EMPTY",
    "BLANK",
    "?????",
    "XXXX",
    "XXXXX",
    "XXXXXX",
    # Blacklist prompt formatting examples to reject hallucinations
    "A96904",
    "538148",
    "HMKD 808154",
    "HMKD808154",
    "3000R7560",
    "3000R",
    "53124",
    "INYU 500434",
    "INYU500434",
    "TSFZ 562124",
    "TSFZ562124",
    "711538",
    "500434",
    "372",
})

# Phrases that indicate the text is from the OCR prompt/instruction, not from the image.
# If any of these appear in the OCR result, treat it as invalid (prompt leak).
PROMPT_LEAK_PHRASES = (
    "CRITICAL",
    "Look for VERTICAL",
    "VERTICAL text",
    "written from top",
    "Pay EXTREME attention",
    "Extract ALL text",
    "Extract ALL trailer",
    "Scan the entire image",
    "Scan entire image",
    "left side center right",
    "trailer identification",
    "trailer identification numbers",
    "BOTH horizontal AND",
    "company names (like",
    "List each unique",
    "List unique element",
    "Do not skip",
    "not skip even",
    "Output ONLY text",
    "Output that actually",
    "example strings",
    "this instruction",
    "Read each character",
    "low-contrast",
    "systematically",
    "from this image",
    "FORMAT EXAMPLES ONLY",
    "Never output those",
    "text found",  # Literal "text found" often appears in echoed instructions
    "separated by spaces",
    "appears faint",
    "Do not guess",
    "respond with exactly",
    "no explanations",
    "If NO trailer",
    "visible in the image",
)


def is_prompt_leak(text: str) -> bool:
    """
    Return True if the text looks like OCR prompt/instruction leak (should not be stored as plate).

    Checks for:
    - Distinctive phrases from the OCR prompt
    - Text starting with "text found"
    - Suspiciously long text (likely full prompt)
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False

    # Long text is almost certainly prompt or paragraph, not a plate
    if len(t) > 150:
        return True

    lower = t.lower()
    # Literal "text found" at start or as a distinct phrase
    if lower.startswith("text found") or " text found " in lower or lower.startswith("text found."):
        return True

    # Check for prompt phrases (case-insensitive for most)
    for phrase in PROMPT_LEAK_PHRASES:
        if phrase.lower() in lower:
            return True

    return False


def is_placeholder_value(text: str) -> bool:
    """Return True if ``text`` is one of the known 'no detection' placeholders
    (UNKNOWN / NONE / N/A / dashes / question marks / etc.).

    Matches the entire trimmed text case-insensitively against
    ``PLACEHOLDER_VALUES`` — substring matches are intentionally NOT done so
    a real plate that happens to contain "NA" or "NONE" isn't rejected.
    """
    if not text or not isinstance(text, str):
        return True
    t_clean = text.strip().upper()
    if t_clean in PLACEHOLDER_VALUES:
        return True
    
    # Also extract digits and check if they match any of the blacklisted prompt example digit sequences
    digits = "".join(c for c in t_clean if c.isdigit())
    if digits in {"96904", "538148", "808154", "500434", "562124", "711538"}:
        return True
        
    return False


def is_valid_plate_for_storage(text: str) -> bool:
    """
    Return True if the text is acceptable to store as a plate/trailer ID.

    Rejects:
    - Empty or whitespace
    - Too short (< MIN_PLATE_LENGTH) — single-char noise from OCR
    - Known placeholder values (UNKNOWN / NONE / N/A / etc. — see is_placeholder_value)
    - Prompt/instruction leak (is_prompt_leak)
    - Too long for a single plate field
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if len(t) < MIN_PLATE_LENGTH:
        return False
    if is_placeholder_value(t):
        return False
    if is_prompt_leak(t):
        return False
    if len(t) > MAX_PLATE_LENGTH:
        return False
    return True


def sanitize_plate_for_storage(text: str) -> str:
    """
    Return a plate string safe for storage, or empty string if invalid.

    Use this when you need a value to store: returns stripped text if valid,
    otherwise empty string so callers can skip storing or treat as no plate.

    Reuses ``is_valid_plate_for_storage`` so all rejection rules
    (empty / too short / placeholders like UNKNOWN-NONE-N/A / prompt leak /
    too long) stay in one place.
    """
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    if not is_valid_plate_for_storage(t):
        return ""
    return t
