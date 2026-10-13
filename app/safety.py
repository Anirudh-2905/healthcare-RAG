from __future__ import annotations

import re


INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"act as (the )?(system|developer)",
    r"exfiltrat(e|ion)",
    r"print .*api.?key",
    r"send .*secret",
]

MEDICAL_ADVICE_PATTERNS = [
    r"diagnose me",
    r"what (medicine|drug|dose) should i (take|use)",
    r"should i stop taking",
    r"prescribe",
    r"is this an emergency",
]


def contains_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def requests_personal_medical_advice(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in MEDICAL_ADVICE_PATTERNS)


def safe_excerpt(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"

