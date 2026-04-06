"""Vendor-agnostic LLM and Embeddings factory.

Supported providers (value of *_MODEL_PROVIDER / EMBEDDING_PROVIDER env var):
  - "openai_compatible"  Active now — covers OpenAI, OpenRouter, vLLM, Ollama,
                         Groq, DeepSeek API, LM Studio, and any base_url override.
  - "anthropic"          Planned — requires ``langchain-anthropic``.
  - "google"             Planned — requires ``langchain-google-genai``.
  - "ollama"             Planned — requires ``langchain-ollama``.

To add a new provider: install the matching ``langchain-<provider>`` package,
add an ``elif`` branch in ``build_chat_model`` / ``build_embeddings``, and
update the ``LLMProvider`` literal.
"""

import logging
from typing import Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai_compatible", "anthropic", "google", "ollama"]

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://audit-assistant.app",
    "X-Title": "AI RAG Audit Assistant",
}


def build_chat_model(
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: str,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Instantiate a LangChain chat model for the given provider.

    Args:
        provider: One of the ``LLMProvider`` literals.
        model: Model name / ID (provider-specific format).
        base_url: Optional base URL override (used for openai_compatible only).
        api_key: API key for the provider.
        temperature: Sampling temperature.

    Returns:
        A ``BaseChatModel`` instance.

    Raises:
        NotImplementedError: If ``provider`` is not yet implemented.
        ValueError: If ``provider`` is not a recognised value.
    """
    if provider == "openai_compatible":
        return _build_openai_compatible_chat(model, base_url, api_key, temperature)

    if provider == "anthropic":
        raise NotImplementedError(
            "Provider 'anthropic' is not yet active. "
            "Add 'langchain-anthropic>=0.3.0' to pyproject.toml and implement the branch."
        )

    if provider == "google":
        raise NotImplementedError(
            "Provider 'google' is not yet active. "
            "Add 'langchain-google-genai>=2.0.0' to pyproject.toml and implement the branch."
        )

    if provider == "ollama":
        raise NotImplementedError(
            "Provider 'ollama' is not yet active. "
            "Add 'langchain-ollama>=0.2.0' to pyproject.toml and implement the branch."
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Valid values: openai_compatible, anthropic, google, ollama."
    )


def build_embeddings(
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: str,
    dimensions: int,
) -> Embeddings:
    """Instantiate a LangChain embeddings client for the given provider.

    Args:
        provider: One of the ``LLMProvider`` literals.
        model: Embedding model name / ID.
        base_url: Optional base URL override.
        api_key: API key for the provider.
        dimensions: Output vector dimensions (MRL models support variable dims).

    Returns:
        A ``langchain_core.embeddings.Embeddings`` instance.

    Raises:
        NotImplementedError: If ``provider`` is not yet implemented.
        ValueError: If ``provider`` is not a recognised value.
    """
    if provider == "openai_compatible":
        return _build_openai_compatible_embeddings(model, base_url, api_key, dimensions)

    if provider in ("anthropic", "google", "ollama"):
        raise NotImplementedError(
            f"Embeddings provider '{provider}' is not yet active. "
            f"Use 'openai_compatible' with the appropriate base_url for now."
        )

    raise ValueError(
        f"Unknown embeddings provider '{provider}'. "
        f"Valid values: openai_compatible, anthropic, google, ollama."
    )


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_openai_compatible_chat(
    model: str,
    base_url: Optional[str],
    api_key: str,
    temperature: float,
) -> ChatOpenAI:
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "openai_api_key": api_key,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url
        if "openrouter.ai" in base_url:
            kwargs["default_headers"] = _OPENROUTER_HEADERS

    logger.debug(
        "Building openai_compatible chat model (model=%s, base_url=%s)",
        model,
        base_url or "default",
    )
    return ChatOpenAI(**kwargs)


def _build_openai_compatible_embeddings(
    model: str,
    base_url: Optional[str],
    api_key: str,
    dimensions: int,
) -> OpenAIEmbeddings:
    kwargs: dict = {
        "model": model,
        "dimensions": dimensions,
        "openai_api_key": api_key,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url

    logger.debug(
        "Building openai_compatible embeddings (model=%s, dims=%d, base_url=%s)",
        model,
        dimensions,
        base_url or "default",
    )
    return OpenAIEmbeddings(**kwargs)
