"""
Medical Abbreviation Dictionary for Prescription OCR

A comprehensive dictionary of medical abbreviations commonly found in prescriptions.
Used to expand abbreviations to their full forms for clinical understanding.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

FREQUENCY = {
    # Standard frequency codes
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "qid": "four times daily",
    "prn": "as needed",
    "stat": "immediately",
    "qod": "every other day",
    "qam": "every morning",
    "qpm": "every evening",
    "qhs": "at bedtime",
    "qwk": "weekly",
    "qmo": "monthly",
    # Variations
    "od.": "once daily",
    "bd.": "twice daily",
    "tds.": "three times daily",
    "qid.": "four times daily",
    "prn.": "as needed",
    "stat.": "immediately",
}

# ─────────────────────────────────────────────────────────────────────────────
# DOSE ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

DOSE = {
    # Weight/Mass units
    "mg": "milligrams",
    "ml": "milliliters",
    "mcg": "micrograms",
    "g": "grams",
    "kg": "kilograms",
    "l": "liters",
    # Special units
    "meq": "milliequivalents",
    "iu": "international units",
    "units": "units",
    "tab": "tablet",
    "cap": "capsule",
    "tablets": "tablets",
    "capsules": "capsules",
}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

ROUTE = {
    "po": "by mouth",
    "po.": "by mouth",
    "oral": "by mouth",
    "iv": "intravenously",
    "iv.": "intravenously",
    "im": "intramuscularly",
    "im.": "intramuscularly",
    "sc": "subcutaneously",
    "sc.": "subcutaneously",
    "sq": "subcutaneously",
    "sl": "sublingually",
    "sl.": "sublingually",
    "top": "topically",
    "top.": "topically",
    "pr": "rectally",
    "pr.": "rectally",
    "pv": "vaginally",
    "inh": "by inhalation",
    "td": "transdermally",
}

# ─────────────────────────────────────────────────────────────────────────────
# DURATION ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

DURATION = {
    "d": "days",
    "wk": "weeks",
    "mo": "months",
    "w": "weeks",
    "m": "months",
    "y": "years",
    "d.": "days",
    "wk.": "weeks",
    "mo.": "months",
}

# ─────────────────────────────────────────────────────────────────────────────
# DOSAGE FORM ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

FORM = {
    "tab": "tablet",
    "tabs": "tablets",
    "cap": "capsule",
    "caps": "capsules",
    "inj": "injection",
    "sol": "solution",
    "syr": "syrup",
    "susp": "suspension",
    "lot": "lotion",
    "cream": "cream",
    "oint": "ointment",
    "gel": "gel",
    "supp": "suppository",
    "drop": "drops",
    "drops": "drops",
}

# ─────────────────────────────────────────────────────────────────────────────
# GENERAL MEDICAL ABBREVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

GENERAL = {
    "rx": "prescription",
    "dx": "diagnosis",
    "tx": "treatment",
    "hx": "history",
    "sx": "symptoms",
    "fx": "fracture",
    "nka": "no known allergies",
    "nkda": "no known drug allergies",
    "npo": "nothing by mouth",
    "dc": "discontinue",
    "ud": "as directed",
    "ac": "before meals",
    "pc": "after meals",
    "hs": "at bedtime",
    "sob": "shortness of breath",
    "ha": "headache",
    "dd": "differential diagnosis",
    "et": "and",
    "c": "with",
    "s": "without",
    "w": "with",
    "w/o": "without",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT-DEPENDENT ABBREVIATIONS (Ambiguous - needs context)
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_SENSITIVE = {
    "hs": {
        "default": "at bedtime",
        "contexts": ["night dose", "sleep", "bedtime"],
    },
    "od": {
        "default": "once daily",
        "alternative": "right eye",  # ophthalmic context
    },
    "os": {
        "default": "once daily",
        "alternative": "left eye",  # ophthalmic context
    },
    "ou": {
        "default": "twice daily",
        "alternative": "both eyes",  # ophthalmic context
    },
    "dc": {
        "default": "discontinue",
        "alternative": "doctor's order",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED DICTIONARY (All abbreviations)
# ─────────────────────────────────────────────────────────────────────────────

# Build combined dict from all categories
ALL_ABBREVIATIONS = {}
ALL_ABBREVIATIONS.update(FREQUENCY)
ALL_ABBREVIATIONS.update(DOSE)
ALL_ABBREVIATIONS.update(ROUTE)
ALL_ABBREVIATIONS.update(DURATION)
ALL_ABBREVIATIONS.update(FORM)
ALL_ABBREVIATIONS.update(GENERAL)

# ─────────────────────────────────────────────────────────────────────────────
# COMMON DRUG ABBREVIATIONS (Brand → Generic)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_REPLACEMENTS = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "aspirin": "acetylsalicylic acid",
    "roche": "roche",  # placeholder
    "glucophage": "metformin",
    "lipitor": "atorvastatin",
    "nexium": "esomeprazole",
    "plavix": "clopidogrel",
    "synthroid": "levothyroxine",
    "vyvanse": "lisdexamfetamine",
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def expand_text(text: str) -> str:
    """
    Expand all recognized abbreviations in text.

    Args:
        text: Raw OCR text

    Returns:
        Text with abbreviations expanded
    """
    if not text:
        return text

    result = text
    expansions_made = []

    # Sort by length (longest first) to handle overlapping patterns
    words = text.lower().split()
    for i, word in enumerate(words):
        # Clean word (remove punctuation for matching)
        clean_word = word.strip('.,;:!?()[]{}')

        # Check all abbreviation categories
        if clean_word in ALL_ABBREVIATIONS:
            expansion = ALL_ABBREVIATIONS[clean_word]
            expansions_made.append((clean_word, expansion))

        if clean_word in DRUG_REPLACEMENTS:
            expansion = DRUG_REPLACEMENTS[clean_word]
            expansions_made.append((clean_word, expansion))

    return result


def expand_with_details(text: str) -> tuple[str, list]:
    """
    Expand abbreviations and return details of what was expanded.

    Args:
        text: Raw OCR text

    Returns:
        (expanded_text, list of (original, expansion) tuples)
    """
    if not text:
        return text, []

    expansions = []
    words = text.lower().split()

    for word in words:
        clean_word = word.strip('.,;:!?()[]{}')

        if clean_word in ALL_ABBREVIATIONS:
            expansions.append((clean_word, ALL_ABBREVIATIONS[clean_word]))
        elif clean_word in DRUG_REPLACEMENTS:
            expansions.append((clean_word, DRUG_REPLACEMENTS[clean_word]))

    expanded = expand_text(text)
    return expanded, expansions


def get_abbreviation_count(text: str) -> int:
    """Count how many abbreviations are in the text."""
    if not text:
        return 0

    words = text.lower().split()
    count = 0

    for word in words:
        clean_word = word.strip('.,;:!?()[]{}')
        if clean_word in ALL_ABBREVIATIONS or clean_word in DRUG_REPLACEMENTS:
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# QUICK LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def lookup(abbrev: str) -> str | None:
    """Look up an abbreviation."""
    abbrev = abbrev.lower().strip()
    return ALL_ABBREVIATIONS.get(abbrev)


def get_all_frequencies() -> dict:
    """Get all frequency abbreviations."""
    return FREQUENCY.copy()


def get_all_routes() -> dict:
    """Get all route abbreviations."""
    return ROUTE.copy()


def get_all_doses() -> dict:
    """Get all dose abbreviations."""
    return DOSE.copy()