"""
Intent detection for chat messages.

Parses user messages to determine which Moondream capability to invoke:
- "describe" / "caption" / "what's in this" → caption
- "find" / "detect" / "locate" / "where are the" → detect
- "point to" / "point at" / "where is the" → point
- Everything else → query (visual question answering)

Also extracts the subject for detect/point operations (e.g. "find all
cars" → subject is "cars").

This is intentionally simple keyword matching — no ML needed here.
The patterns cover natural phrasing well enough, and the fallback
to query (which is Moondream's most flexible mode) means misclassified
messages still produce useful results.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Capability(Enum):
    QUERY = "query"
    CAPTION = "caption"
    DETECT = "detect"
    POINT = "point"


@dataclass
class Intent:
    """Parsed intent from a user message."""

    capability: Capability
    subject: str | None = None  # for detect/point: what to look for
    question: str | None = None  # for query: the full question
    caption_length: str = "normal"  # for caption: short/normal/long


# Patterns are checked in order. First match wins.
# Each tuple: (compiled regex, capability, group index for subject extraction)
# Group index None means no subject extraction needed.

_CAPTION_PATTERNS = [
    r"^describe\b",
    r"^caption\b",
    r"^generate\s+(?:a\s+)?(?:short|long|normal)?\s*caption\b",
    r"\bwhat(?:'s| is) (?:in |going on in )?this (?:image|picture|photo)\b",
    r"\bwhat do you see\b",
    r"^summarize this\b",
    r"^tell me about this (?:image|picture|photo)\b",
]

_DETECT_PATTERNS = [
    # "find all cars", "detect people", "locate the dogs"
    (r"^(?:find|detect|locate|spot)\s+(?:all\s+|the\s+|every\s+)?(.+)", 1),
    (r"^where are (?:the |all (?:the )?)?(.+?)(?:\?|$)", 1),
    (r"^show me (?:all|every) (?:the\s+)?(.+?)(?:\?|$)", 1),
    (r"^how many (.+?)(?:\s+are\s+there)?(?:\?|$)", 1),
    (r"^count (?:the |all (?:the )?)?(.+?)(?:\?|$)", 1),
]

_POINT_PATTERNS = [
    (r"^point (?:to|at) (?:the\s+)?(.+?)(?:\?|$)", 1),
    (r"^where(?:'s| is) (?:the\s+)?(.+?)(?:\?|$)", 1),
    (r"^show me (?:the\s+|where\s+(?:the\s+)?)?(.+?)(?:\s+is)?(?:\?|$)", 1),
]

# Caption length keywords
_LENGTH_KEYWORDS = {
    "short": "short",
    "brief": "short",
    "quick": "short",
    "detailed": "long",
    "long": "long",
    "verbose": "long",
    "thorough": "long",
}


def parse_intent(message: str) -> Intent:
    """
    Parse a user message into an Intent.

    Checks patterns in order: caption → detect → point → query (fallback).
    This ordering matters — detect patterns like "find X" should be
    checked before falling through to generic query.

    Args:
        message: Raw user message text.

    Returns:
        Intent with the detected capability, subject, and/or question.
    """
    text = message.strip()
    text_lower = text.lower()

    # --- Caption ---
    for pattern in _CAPTION_PATTERNS:
        if re.search(pattern, text_lower):
            length = "normal"
            for keyword, length_val in _LENGTH_KEYWORDS.items():
                if keyword in text_lower:
                    length = length_val
                    break
            return Intent(capability=Capability.CAPTION, caption_length=length)

    # --- Detect ---
    for pattern, group_idx in _DETECT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            subject = match.group(group_idx).strip().rstrip("?.,!")
            return Intent(capability=Capability.DETECT, subject=subject)

    # --- Point ---
    for pattern, group_idx in _POINT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            subject = match.group(group_idx).strip().rstrip("?.,!")
            return Intent(capability=Capability.POINT, subject=subject)

    # --- Fallback: Query ---
    return Intent(capability=Capability.QUERY, question=text)
