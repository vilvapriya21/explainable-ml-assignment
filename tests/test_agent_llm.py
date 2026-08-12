"""Tests for explicit CrewAI LLM provider selection."""

import pytest

from src.agent import llm


def test_build_review_llm_rejects_unknown_provider(monkeypatch) -> None:
    """An invalid provider fails before CrewAI can fall back to OpenAI."""
    monkeypatch.setattr(llm, "LLM_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="must be either 'groq' or 'ollama'"):
        llm.build_review_llm()


def test_build_review_llm_requires_groq_key(monkeypatch) -> None:
    """Groq configuration requires an explicitly supplied local key."""
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)

    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        llm.build_review_llm()


def test_build_review_llm_constructs_groq_client_without_openai_key(
    monkeypatch,
) -> None:
    """Groq uses its own key and OpenAI-compatible endpoint."""
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-groq-key")

    model = llm.build_review_llm()

    assert str(model.client._client.base_url) == "https://api.groq.com/openai/v1/"
