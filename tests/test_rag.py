"""
tests/test_rag.py - Acceptance Tests for Industrial Drone & Hardware Technical Manual Assistant.

Verifies:
1. Q1 diagnostic LED fault detection on Page 12 (ERR_IMU_MAG_07).
2. Q2 absent underwater capability returns exact '[not found]' with empty sources.
3. Q3 per-cell voltage cutoff threshold (3.45 V) on Page 29.
4. Q4 max sustained certified wind speed (38 knots) on Page 44.
5. Q5 absent third-party CEO question returns exact '[not found]' with empty sources.
6. Persistence test: subsequent ingest runs reload persisted ChromaDB without re-embedding.
7. Extractive fallback test: runs with LLM disabled to confirm deterministic mode.
"""

import os
import pytest
from rag import answer
from ingest import ingest_pdf


# Test questions
Q1 = "What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?"
Q2 = "Can the AeroVantage X9 be used for underwater sonar mapping at a depth of 50 metres?"
Q3 = "At what minimum per-cell voltage, in volts, does the X9 trigger a forced emergency Return-to-Home (RTH) landing?"
Q4 = "What is the maximum sustained wind speed at which the X9 is certified to operate safely?"
Q5 = "Who is the current Chief Executive Officer of DJI Innovations?"


def test_q1_diagnostic_led():
    res = answer(Q1)
    ans = res["answer"].lower()
    sources = res["sources"]

    # Verify diagnostic fault and error code mentioned
    assert ("primary imu" in ans or "magnetometer calibration fault" in ans or "compass interference" in ans), \
        f"Expected fault description in answer, got: {res['answer']}"
    assert "err_imu_mag_07" in ans, f"Expected error code ERR_IMU_MAG_07 in answer, got: {res['answer']}"
    assert 12 in sources, f"Expected Page 12 in sources, got: {sources}"


def test_q2_underwater_sonar_unsupported():
    res = answer(Q2)
    assert res["answer"] == "[not found]", f"Expected '[not found]', got: {res['answer']}"
    assert res["sources"] == [], f"Expected empty sources, got: {res['sources']}"


def test_q3_battery_voltage_cutoff():
    res = answer(Q3)
    ans = res["answer"]
    sources = res["sources"]

    assert "3.45" in ans, f"Expected 3.45 V threshold in answer, got: {res['answer']}"
    assert 29 in sources, f"Expected Page 29 in sources, got: {sources}"


def test_q4_wind_speed_limits():
    res = answer(Q4)
    ans = res["answer"].lower()
    sources = res["sources"]

    assert "38 knots" in ans or "38" in ans, f"Expected 38 knots in answer, got: {res['answer']}"
    assert 44 in sources, f"Expected Page 44 in sources, got: {sources}"


def test_q5_dji_ceo_unsupported():
    res = answer(Q5)
    assert res["answer"] == "[not found]", f"Expected '[not found]', got: {res['answer']}"
    assert res["sources"] == [], f"Expected empty sources, got: {res['sources']}"


def test_chroma_persistence_no_reembed():
    """Verify ChromaDB persists to disk and reloads without re-embedding."""
    count, was_reembedded = ingest_pdf(force=False)
    assert count > 0, "Collection should contain indexed chunks"
    assert not was_reembedded, "Subsequent ingestion must reuse persisted database without re-embedding"


def test_extractive_fallback_mode(monkeypatch):
    """
    Simulates environment without LLM API keys.
    Verifies extractive fallback produces supported answers for Q1, Q3, Q4
    and exact '[not found]' for Q2, Q5.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Q1 in extractive mode
    res1 = answer(Q1)
    assert "ERR_IMU_MAG_07" in res1["answer"]
    assert 12 in res1["sources"]

    # Q2 in extractive mode
    res2 = answer(Q2)
    assert res2["answer"] == "[not found]"
    assert res2["sources"] == []

    # Q3 in extractive mode
    res3 = answer(Q3)
    assert "3.45" in res3["answer"]
    assert 29 in res3["sources"]

    # Q4 in extractive mode
    res4 = answer(Q4)
    assert "38 knots" in res4["answer"]
    assert 44 in res4["sources"]

    # Q5 in extractive mode
    res5 = answer(Q5)
    assert res5["answer"] == "[not found]"
    assert res5["sources"] == []
