"""
============================================================
tests/test_safety_triage.py — Clinical Safety Screener Unit Tests
============================================================
PURPOSE:
    Verifies that the safety screener correctly:
    1. Detects all three crisis categories
    2. Does NOT flag safe wellness messages
    3. Returns properly structured TriageResponse objects
    4. Is case-insensitive (handles "SUICIDE", "suicide", "SuIcIdE")
============================================================
"""
import pytest
from app.core.safety_triage import evaluate_clinical_safety


# ── SELF-HARM DETECTION ──────────────────────────────────────────

class TestSelfHarmDetection:
    def test_detects_suicide_keyword(self):
        result = evaluate_clinical_safety("I've been thinking about suicide a lot lately")
        assert result.is_safe is False
        assert result.triage_response.crisis_type == "self_harm"

    def test_detects_want_to_die(self):
        result = evaluate_clinical_safety("I just want to die, nothing matters")
        assert result.is_safe is False

    def test_detects_end_it_all(self):
        result = evaluate_clinical_safety("I feel like I want to end it all")
        assert result.is_safe is False

    def test_detects_self_harm(self):
        result = evaluate_clinical_safety("I've been cutting myself again")
        assert result.is_safe is False

    def test_detects_kill_myself(self):
        result = evaluate_clinical_safety("Sometimes I think about killing myself")
        assert result.is_safe is False

    def test_case_insensitive(self):
        result = evaluate_clinical_safety("SUICIDE is something I think about")
        assert result.is_safe is False

    def test_triage_has_resources(self):
        result = evaluate_clinical_safety("I want to kill myself")
        assert result.triage_response is not None
        assert len(result.triage_response.resources) > 0

    def test_client_response_excludes_internal_fields(self):
        result = evaluate_clinical_safety("I want to die")
        response = result.triage_response.to_client_response()
        assert "triggered_by" not in response  # Internal audit field — never sent to client
        assert "type" in response
        assert "message" in response
        assert "resources" in response


# ── EATING DISORDER DETECTION ─────────────────────────────────────

class TestEatingDisorderDetection:
    def test_detects_starving(self):
        result = evaluate_clinical_safety("I've been starving myself for the past week")
        assert result.is_safe is False
        assert result.triage_response.crisis_type == "eating_disorder"

    def test_detects_purging(self):
        result = evaluate_clinical_safety("I've been purging after every meal")
        assert result.is_safe is False


# ── ACUTE MEDICAL DETECTION ───────────────────────────────────────

class TestAcuteMedicalDetection:
    def test_detects_chest_pain(self):
        result = evaluate_clinical_safety("I'm having really bad chest pain right now")
        assert result.is_safe is False
        assert result.triage_response.crisis_type == "acute_medical"

    def test_detects_cant_breathe(self):
        result = evaluate_clinical_safety("I can't breathe properly and feel faint")
        assert result.is_safe is False


# ── SAFE MESSAGES (NO FALSE POSITIVES) ───────────────────────────

class TestSafeMessages:
    def test_normal_anxiety_message(self):
        result = evaluate_clinical_safety("I've been feeling anxious before my exams")
        assert result.is_safe is True

    def test_sleep_complaint(self):
        result = evaluate_clinical_safety("I haven't been sleeping well this week")
        assert result.is_safe is True

    def test_mood_check_in(self):
        result = evaluate_clinical_safety("My mood has been around 5 out of 10 today")
        assert result.is_safe is True

    def test_coping_mention(self):
        result = evaluate_clinical_safety("The box breathing technique really helped me today")
        assert result.is_safe is True
        assert result.triage_response is None
