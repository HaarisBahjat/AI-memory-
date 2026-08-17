"""
============================================================
tests/test_time_decay.py — Mathematical Verification of Decay Formula
============================================================
PURPOSE:
    Verifies the mathematical correctness of the exponential
    time-decay scoring function:

        S_adjusted = S_raw × e^(-λ × Δt)

    These tests are CRITICAL for Phase 9 benchmark validity.
    If the decay math is wrong, retrieved memories will be
    incorrectly scored, ruining retrieval quality metrics.
============================================================
"""
import pytest
import math
from datetime import datetime, timezone, timedelta
from app.services.retrieval_engine import (
    apply_time_decay,
    apply_time_decay_batch,
    filter_by_decay,
)


class TestTimeDeratingMath:
    """
    Mathematical correctness tests for S_adjusted = S_raw × e^(-λ × Δt)
    """

    def test_fresh_memory_no_decay(self):
        """A memory created NOW should have zero decay (e^0 = 1.0)."""
        now = datetime.now(timezone.utc)
        adjusted = apply_time_decay(raw_score=0.85, created_at=now)
        assert adjusted == pytest.approx(0.85, abs=1e-3)

    def test_30_day_old_memory_decay(self):
        """
        A 30-day old memory with S_raw=0.82 should decay:
        S_adj = 0.82 × e^(-0.005 × 30) = 0.82 × e^(-0.15) ≈ 0.82 × 0.8607 ≈ 0.706
        This is above the 0.65 threshold → KEPT
        """
        created_at = datetime.now(timezone.utc) - timedelta(days=30)
        adjusted = apply_time_decay(raw_score=0.82, created_at=created_at)
        expected = 0.82 * math.exp(-0.005 * 30)
        assert adjusted == pytest.approx(expected, rel=1e-4)
        assert adjusted > 0.65  # Should still pass threshold

    def test_60_day_old_memory_below_threshold(self):
        """
        A 60-day old memory with S_raw=0.70 should decay below 0.65:
        S_adj = 0.70 × e^(-0.005 × 60) = 0.70 × e^(-0.30) ≈ 0.70 × 0.7408 ≈ 0.519
        Below 0.65 threshold → DISCARDED
        """
        created_at = datetime.now(timezone.utc) - timedelta(days=60)
        adjusted = apply_time_decay(raw_score=0.70, created_at=created_at)
        expected = 0.70 * math.exp(-0.005 * 60)
        assert adjusted == pytest.approx(expected, rel=1e-4)
        assert adjusted < 0.65  # Must fail threshold

    def test_high_score_very_old_memory(self):
        """
        Even a very high S_raw score decays over long periods.
        S_adj = 1.0 × e^(-0.005 × 365) = e^(-1.825) ≈ 0.161
        A 1-year old memory (λ=0.005) has only 16% of its original score.
        """
        created_at = datetime.now(timezone.utc) - timedelta(days=365)
        adjusted = apply_time_decay(raw_score=1.0, created_at=created_at)
        expected = math.exp(-0.005 * 365)
        assert adjusted == pytest.approx(expected, rel=1e-4)
        assert adjusted < 0.65  # 0.161 << 0.65 threshold → discarded

    def test_custom_lambda_faster_decay(self):
        """
        Higher lambda = faster decay.
        With λ=0.05 (10x faster), a 30-day memory decays to:
        S_adj = 0.80 × e^(-0.05 × 30) = 0.80 × e^(-1.5) ≈ 0.80 × 0.2231 ≈ 0.178
        """
        created_at = datetime.now(timezone.utc) - timedelta(days=30)
        adjusted = apply_time_decay(
            raw_score=0.80,
            created_at=created_at,
            lambda_decay=0.05  # Custom lambda
        )
        expected = 0.80 * math.exp(-0.05 * 30)
        assert adjusted == pytest.approx(expected, rel=1e-4)

    def test_score_never_exceeds_raw(self):
        """Decay can only reduce a score, never increase it."""
        created_at = datetime.now(timezone.utc) - timedelta(days=1)
        raw = 0.75
        adjusted = apply_time_decay(raw_score=raw, created_at=created_at)
        assert adjusted <= raw

    def test_score_always_non_negative(self):
        """Scores should never go negative (e^x is always positive)."""
        created_at = datetime.now(timezone.utc) - timedelta(days=10000)
        adjusted = apply_time_decay(raw_score=0.9, created_at=created_at)
        assert adjusted >= 0.0


class TestFilterByDecay:
    """Tests for the full filter_by_decay pipeline (threshold + ranking)."""

    def _make_memory(self, text, score, days_old, category="coping_mechanism"):
        return {
            "id": "test-id",
            "text": text,
            "category": category,
            "similarity_score": score,
            "created_at": datetime.now(timezone.utc) - timedelta(days=days_old),
            "reinforcement_count": 1,
        }

    def test_filters_below_threshold(self):
        """Memories with adjusted score < 0.65 should be discarded."""
        memories = [
            self._make_memory("Old memory", score=0.70, days_old=200),  # Will decay below 0.65
        ]
        result = filter_by_decay(memories, threshold=0.65)
        assert len(result) == 0

    def test_keeps_fresh_memories(self):
        """Fresh memories with good scores should be kept."""
        memories = [
            self._make_memory("Fresh insight", score=0.80, days_old=2),
        ]
        result = filter_by_decay(memories, threshold=0.65)
        assert len(result) == 1

    def test_max_memories_limit(self):
        """Should never return more than max_memories (default 3)."""
        memories = [
            self._make_memory(f"Memory {i}", score=0.80, days_old=i)
            for i in range(10)
        ]
        result = filter_by_decay(memories, max_memories=3)
        assert len(result) <= 3

    def test_sorted_by_adjusted_score(self):
        """Results should be sorted highest adjusted score first."""
        memories = [
            self._make_memory("Lower score", score=0.70, days_old=1),
            self._make_memory("Higher score", score=0.90, days_old=1),
        ]
        result = filter_by_decay(memories)
        if len(result) >= 2:
            assert result[0]["adjusted_score"] >= result[1]["adjusted_score"]


class TestBatchDecay:
    """Tests for vectorized batch time-decay scoring (Phase 9 benchmarking)."""

    def test_batch_matches_individual_scores(self):
        """Batch results should match individual apply_time_decay() calls."""
        times = [
            datetime.now(timezone.utc) - timedelta(days=d)
            for d in [0, 10, 30, 60, 90]
        ]
        scores = [0.80, 0.75, 0.70, 0.85, 0.90]

        batch_results = apply_time_decay_batch(scores, times)
        individual_results = [
            apply_time_decay(s, t) for s, t in zip(scores, times)
        ]

        for batch, individual in zip(batch_results, individual_results):
            assert batch == pytest.approx(individual, rel=1e-5)
