"""
============================================================
app/services/mood_tracker.py — In-Session Valence Scoring (Phase 3)
============================================================
PURPOSE:
    Provides lightweight, zero-latency per-message sentiment scoring
    using a curated keyword lexicon. Scores are accumulated in the
    session metadata HASH in Redis and used to detect significant
    mood drops within a session.

DESIGN RATIONALE:
    An LLM-based sentiment call would add ~200ms latency and token cost
    to every message. This keyword approach adds ~0ms and zero API cost.
    It mirrors the architecture of the Phase 1 clinical safety screener:
    fast regex/keyword matching as a first layer, with semantic analysis
    reserved for a later phase (Phase 6) where it matters most.

SCORING MODEL:
    Each user message gets a valence score in [-1.0, +1.0].

    Raw score = (positive_hits - negative_hits) / total_content_words
    Amplifiers: negative words preceded by crisis amplifiers
                (e.g. "can't cope" → weight × 1.5)
    Clamp to [-1.0, +1.0]

MOOD DROP DETECTION:
    A MOOD_DROP_ALERT is flagged if:
        first_message_valence - current_average_valence > MOOD_DROP_THRESHOLD (0.4)

    This is conservative — a threshold of 0.4 on a [-1.0, +1.0] scale
    means the user needs to go from neutral (0) to noticeably negative (-0.4)
    before the alert fires. False positives are a much smaller concern here
    than in clinical safety (which hard-overrides the LLM). Mood drop alerts
    are informational only — they enrich the synthesis payload in Phase 5.

CONNECTED TO:
    Phase 3 → app/services/session_lifecycle.py (update_session_meta calls score_message)
    Phase 5 → synthesis payload includes mood_drop_flag from session metadata
    Phase 6 → Phase 6 semantic screener may supersede this for crisis detection
============================================================
"""
import re
import structlog
from typing import Optional

log = structlog.get_logger(__name__)

# -------------------------------------------------------
# Mood Drop Threshold
# -------------------------------------------------------
# If average session valence drops more than this amount from
# the first-message baseline, we flag a mood drop alert.
MOOD_DROP_THRESHOLD: float = 0.4

# -------------------------------------------------------
# Lexicons
# -------------------------------------------------------
# All lowercase. Word-boundary matching is applied on the
# scored text (lowercased, punctuation stripped).
# -------------------------------------------------------

# Words that push valence positive (+1 each)
POSITIVE_LEXICON: frozenset[str] = frozenset({
    "better", "good", "great", "happy", "calm", "hopeful", "grateful",
    "improving", "progress", "excited", "motivated", "energetic", "relieved",
    "peaceful", "confident", "positive", "optimistic", "rested", "recovered",
    "okay", "fine", "alright", "manageable", "coping", "proud", "loved",
    "supported", "connected", "focused", "productive", "balanced", "refreshed",
    "cheerful", "content", "satisfied", "thankful", "joyful", "well",
})

# Words that push valence negative (-1 each)
NEGATIVE_LEXICON: frozenset[str] = frozenset({
    "anxious", "anxiety", "stressed", "stress", "depressed", "depression",
    "hopeless", "worthless", "exhausted", "tired", "overwhelmed", "scared",
    "afraid", "worried", "worry", "panic", "panicking", "lonely", "alone",
    "numb", "empty", "lost", "confused", "miserable", "suffering", "pain",
    "hurt", "broken", "stuck", "helpless", "useless", "failing", "failed",
    "terrible", "horrible", "awful", "dread", "dreading", "crying", "cried",
    "sad", "upset", "irritable", "angry", "frustrated", "trapped", "burdened",
    "defeated", "ashamed", "guilty", "shame", "rejected", "worthless",
    "sleepless", "insomnia", "nightmare", "nightmare", "isolated",
})

# Words that amplify the weight of adjacent negative words (×1.5)
CRISIS_AMPLIFIERS: frozenset[str] = frozenset({
    "can't", "cannot", "never", "nothing", "always", "completely", "totally",
    "absolutely", "utterly", "everything", "anymore", "no one", "nobody",
    "pointless", "forever", "keep", "constantly", "every", "all",
})

# Precompile word tokenizer for efficiency
_WORD_PATTERN = re.compile(r"\b[a-z']+\b")


def score_message(text: str) -> float:
    """
    Scores a single message's emotional valence.

    Returns a float in [-1.0, +1.0]:
        +1.0 = maximally positive / stable
         0.0 = neutral
        -1.0 = maximally negative / distressed

    Algorithm:
        1. Lowercase + tokenize into words
        2. For each word, check positive/negative lexicon membership
        3. If a negative word is preceded by a crisis amplifier, weight it ×1.5
        4. Normalize by total word count (prevents long messages dominating)
        5. Clamp to [-1.0, +1.0]

    Args:
        text: Raw user message text (any case, any punctuation)

    Returns:
        Valence float in [-1.0, +1.0]
    """
    if not text or not text.strip():
        return 0.0

    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return 0.0

    total_words = len(words)
    raw_score = 0.0
    prev_word = ""

    for word in words:
        if word in POSITIVE_LEXICON:
            raw_score += 1.0
        elif word in NEGATIVE_LEXICON:
            # Apply amplifier if the previous word was a crisis amplifier
            multiplier = 1.5 if prev_word in CRISIS_AMPLIFIERS else 1.0
            raw_score -= multiplier
        prev_word = word

    # Normalize by word count so long messages don't dominate
    normalized = raw_score / total_words

    # Clamp to [-1.0, +1.0]
    return max(-1.0, min(1.0, normalized))


def detect_mood_drop(
    mood_sum: float,
    mood_count: int,
    first_score: Optional[float],
) -> bool:
    """
    Determines whether a significant mood drop has occurred
    within the current session.

    A mood drop is flagged if the current session average valence
    is more than MOOD_DROP_THRESHOLD below the first-message score.
    This catches cases where a user starts the session neutral but
    deteriorates over the course of the conversation.

    Args:
        mood_sum    : Running sum of valence scores for this session
        mood_count  : Number of user messages scored so far
        first_score : Valence of the very first user message this session

    Returns:
        True if a significant mood drop is detected, False otherwise.
    """
    if mood_count < 2 or first_score is None:
        # Need at least 2 data points to detect a trend
        return False

    current_average = mood_sum / mood_count
    return (first_score - current_average) > MOOD_DROP_THRESHOLD
