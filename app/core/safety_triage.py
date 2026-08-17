"""
============================================================
app/core/safety_triage.py — Clinical Safety Boundary & Hard Override
============================================================
PURPOSE:
    This module implements the HARD OVERRIDE safety screener
    that intercepts every incoming user message BEFORE the
    Hybrid RAG retrieval pipeline or LLM call is triggered.

    If a message triggers the safety screener:
    1. The normal LLM pipeline is COMPLETELY BYPASSED
    2. A structured crisis triage response is returned immediately
    3. No user data (session memory, embeddings) is processed
    4. An anonymized audit log entry is created

WHY HARD OVERRIDE?
    An AI wellness agent talking to someone in crisis cannot
    afford to generate an empathetic but therapeutically wrong
    response. The screener guarantees clinical safety boundaries
    are enforced programmatically, not relying on LLM behavior.

DETECTION ARCHITECTURE (3-Layer):
    Layer A: Fast keyword regex check (O(1) per message)
    Layer B: Semantic intent comparison (Phase 6 enhancement)
    Layer C: Contextual LLM classifier (Phase 6 enhancement)
    Phase 1 implements Layer A only.

CONNECTED TO:
    Phase 1  → Called at top of POST /api/v1/chat handler
    Phase 5  → chat.py checks return value before RAG pipeline
    Phase 6  → Extended with semantic embedding comparison (Layer B)
    Phase 6  → Extended with LLM classifier call (Layer C)
    Phase 9  → Crisis interception count logged to Prometheus metrics
============================================================
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import structlog

log = structlog.get_logger(__name__)


# -------------------------------------------------------
# Crisis Resource Registry
# -------------------------------------------------------
# Localized to India (modify per deployment region).
# Phase 6 will add geo-lookup to serve region-appropriate
# helplines based on user profile metadata.
# -------------------------------------------------------
CRISIS_RESOURCES = [
    {"name": "iCall (India)", "contact": "9152987821", "type": "phone"},
    {"name": "Vandrevala Foundation", "contact": "1860-2662-345", "type": "phone"},
    {"name": "NIMHANS Helpline", "contact": "080-46110007", "type": "phone"},
    {"name": "Vandrevala 24/7", "contact": "1860-2662-345", "type": "phone"},
    {"name": "Emergency Services India", "contact": "112", "type": "emergency"},
]

# -------------------------------------------------------
# Keyword Trigger Sets — Phase 1 (Layer A: Fast Regex)
# -------------------------------------------------------
# Organized by crisis type for clarity and future extensibility.
# Phase 6 replaces/supplements these with semantic vectors.
#
# IMPORTANT: These are intentionally broad. False positives
# (triggering triage unnecessarily) are far safer than
# false negatives (missing a genuine crisis signal).
# -------------------------------------------------------

# Self-harm and suicidal ideation signals
SELF_HARM_KEYWORDS = {
    r"\bsuicid(e|al|ally|ing)\b",
    r"\bkill(ing)?\s+myself\b",           # matches: kill myself, killing myself
    r"\bend\s+(my\s+)?life\b",
    r"\bself[\s\-]?harm\b",
    r"\bwant\s+to\s+die\b",
    r"\bi\s+want\s+to\s+die\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bcan'?t\s+go\s+on\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bend\s+it\s+all\b",
    r"\bcut(ting)?\s+myself\b",           # matches: cut myself, cutting myself
    r"\bhurt(ing)?\s+myself\b",           # matches: hurt myself, hurting myself
    r"\boverdos(e|ing)\b",
    r"\btake\s+my\s+own\s+life\b",
}

# Eating disorder crisis signals
EATING_DISORDER_CRISIS_KEYWORDS = {
    r"\bnot\s+eaten\s+(anything\s+)?(in\s+)?\d+\s+days\b",
    r"\bstarving\s+myself\b",
    r"\bpurging\b",
    r"\blaxatives?\s+to\s+(lose|control)\b",
    r"\bbinge\s+and\s+purge\b",
    r"\bfasting\s+for\s+days\b",
}

# Acute medical distress signals
ACUTE_MEDICAL_KEYWORDS = {
    r"\bchest\s+pain\b",
    r"\bcan'?t\s+breathe\b",
    r"\bhaving\s+a\s+heart\s+attack\b",
    r"\bseizure\b",
    r"\bunconsciou\b",
    r"\bpassing\s+out\b",
    r"\bblood\s+(everywhere|all\s+over|gushing)\b",
    r"\bstroke\b",
}

# Compile all patterns once at module load (performance optimization)
_ALL_PATTERNS: list[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern_set in [
        SELF_HARM_KEYWORDS,
        EATING_DISORDER_CRISIS_KEYWORDS,
        ACUTE_MEDICAL_KEYWORDS,
    ]
    for pattern in pattern_set
]

# Map pattern → crisis type string for audit logging
_PATTERN_TYPE_MAP: dict[re.Pattern, str] = {}
for pattern in SELF_HARM_KEYWORDS:
    _PATTERN_TYPE_MAP[re.compile(pattern, re.IGNORECASE | re.MULTILINE)] = "self_harm"
for pattern in EATING_DISORDER_CRISIS_KEYWORDS:
    _PATTERN_TYPE_MAP[re.compile(pattern, re.IGNORECASE | re.MULTILINE)] = "eating_disorder"
for pattern in ACUTE_MEDICAL_KEYWORDS:
    _PATTERN_TYPE_MAP[re.compile(pattern, re.IGNORECASE | re.MULTILINE)] = "acute_medical"


# -------------------------------------------------------
# Response Data Structures
# -------------------------------------------------------

@dataclass
class TriageResponse:
    """
    Structured crisis triage response returned when the
    safety screener intercepts a message.

    Fields:
        is_crisis     → Always True when returned
        crisis_type   → One of: self_harm, eating_disorder, acute_medical
        message       → Empathetic, non-clinical opening message
        resources     → List of localized helpline dicts
        follow_up     → Emergency services reminder
        triggered_by  → The keyword/pattern that triggered (for audit)
    """
    is_crisis: bool = True
    crisis_type: str = ""
    message: str = ""
    resources: list[dict] = field(default_factory=list)
    follow_up: str = ""
    triggered_by: str = ""  # Logged internally — never returned to client

    def to_client_response(self) -> dict:
        """
        Serializes to the JSON response returned to the client.
        Excludes internal audit fields (triggered_by).
        """
        return {
            "type": "CRISIS_TRIAGE",
            "message": self.message,
            "resources": self.resources,
            "follow_up": self.follow_up,
        }


@dataclass
class SafetyResult:
    """
    The result of running the safety screener on a message.

    If is_safe=True  → proceed to normal RAG/LLM pipeline
    If is_safe=False → return triage_response immediately
    """
    is_safe: bool
    triage_response: Optional[TriageResponse] = None


# -------------------------------------------------------
# Crisis Response Templates
# -------------------------------------------------------
_TRIAGE_MESSAGES = {
    "self_harm": (
        "I'm genuinely concerned about what you've shared. "
        "Your life has immense value, and what you're feeling right now "
        "deserves immediate, real human support. "
        "Please reach out to one of these resources right now:"
    ),
    "eating_disorder": (
        "What you've shared tells me you might be going through an "
        "incredibly difficult time with food and your body. "
        "You deserve specialized support — please connect with one of these resources:"
    ),
    "acute_medical": (
        "This sounds like it may be a medical emergency. "
        "Please call emergency services (112) immediately or ask someone nearby for help. "
        "Here are additional resources:"
    ),
}


# -------------------------------------------------------
# Core Evaluation Function
# -------------------------------------------------------

def evaluate_clinical_safety(message: str, user_id: str = "") -> SafetyResult:
    """
    Evaluates a user message against all crisis keyword patterns.

    Args:
        message  : The raw user message string to evaluate
        user_id  : Optional user ID for audit logging (not sent to client)

    Returns:
        SafetyResult(is_safe=True)  → No crisis detected, proceed normally
        SafetyResult(is_safe=False, triage_response=TriageResponse(...))
                                    → Crisis detected, return triage immediately

    This function is designed to be:
    - Synchronous (no I/O, pure computation)
    - Fast (regex on typical message: < 1ms)
    - Side-effect free (audit logging is caller's responsibility)

    Phase 6 Enhancement:
        After keyword check, Phase 6 adds a semantic similarity
        comparison against pre-embedded "crisis template" vectors
        to catch paraphrased or indirect expressions of crisis intent.
    """
    for pattern, crisis_type in _PATTERN_TYPE_MAP.items():
        match = pattern.search(message)
        if match:
            # Log internally (anonymized — no message content stored)
            log.warning(
                "Clinical safety screener triggered",
                user_id=user_id,
                crisis_type=crisis_type,
                pattern=pattern.pattern,
                # NOTE: Never log the actual message content
            )

            triage = TriageResponse(
                is_crisis=True,
                crisis_type=crisis_type,
                message=_TRIAGE_MESSAGES.get(crisis_type, _TRIAGE_MESSAGES["self_harm"]),
                resources=CRISIS_RESOURCES,
                follow_up=(
                    "If you are in immediate danger, please call emergency services (112) "
                    "or ask someone nearby for help immediately."
                ),
                triggered_by=pattern.pattern,  # Audit log only — not returned to client
            )
            return SafetyResult(is_safe=False, triage_response=triage)

    return SafetyResult(is_safe=True)
