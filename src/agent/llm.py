"""LLM configuration for the CrewAI model-review workflow."""

from typing import Any

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def build_review_llm() -> Any:
    """Build the LLM selected by the local environment configuration.

    Returns:
        A LangChain chat model suitable for CrewAI agents.

    Raises:
        ValueError: If the configured provider is unsupported or lacks required
            configuration.
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required when LLM_PROVIDER is set to 'groq'. "
                "Add it to the local .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
    raise ValueError(
        "LLM_PROVIDER must be either 'groq' or 'ollama'; "
        f"received {LLM_PROVIDER!r}."
    )
