"""LLM client wrapper for generation."""

import logging
from typing import Literal, Optional

from langchain_openai import ChatOpenAI

from src.config import settings
from src.generation.provider_factory import build_chat_model

logger = logging.getLogger(__name__)

ModelTier = Literal["small", "mid", "frontier"]

_llm_client: Optional[ChatOpenAI] = None


def get_llm_client(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Get or create the LLM client (singleton for default params).

    Args:
        model: Override LLM model name.
        temperature: Override temperature.

    Returns:
        ChatOpenAI instance.
    """
    global _llm_client

    _model = model or settings.llm_model
    _temperature = temperature if temperature is not None else settings.llm_temperature

    # Return cached client if params match defaults
    if (
        _llm_client is not None
        and _model == settings.llm_model
        and _temperature == settings.llm_temperature
    ):
        return _llm_client

    client = ChatOpenAI(
        model=_model,
        temperature=_temperature,
        openai_api_key=settings.openai_api_key,
    )

    # Cache only default clients
    if _model == settings.llm_model and _temperature == settings.llm_temperature:
        _llm_client = client

    logger.info("LLM client created (model=%s, temperature=%.2f)", _model, _temperature)
    return client


def resolve_tier_config(tier: ModelTier) -> dict[str, str]:
    """Resolve provider, model name, base_url, and api_key for a given tier.

    API key fallback chain: tier-specific key → openrouter_api_key → openai_api_key.

    Returns:
        Dict with keys: provider, model, base_url, api_key.
    """
    _fallback_key = settings.openrouter_api_key or settings.openai_api_key

    if tier == "small":
        provider = settings.small_model_provider
        model = settings.small_model_name or settings.llm_model
        base_url = settings.small_model_base_url or None
        api_key = settings.small_model_api_key or _fallback_key
    elif tier == "mid":
        provider = settings.mid_model_provider
        model = settings.mid_model_name or settings.llm_model
        base_url = settings.mid_model_base_url or None
        api_key = settings.mid_model_api_key or _fallback_key
    elif tier == "frontier":
        provider = settings.frontier_model_provider
        model = settings.frontier_model_name or settings.llm_model
        base_url = settings.frontier_model_base_url or None
        api_key = settings.frontier_model_api_key or _fallback_key
    else:
        provider = "openai_compatible"
        model = settings.llm_model
        base_url = None
        api_key = _fallback_key

    return {"provider": provider, "model": model, "base_url": base_url, "api_key": api_key}


def get_llm_client_by_tier(
    tier: ModelTier,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Get an LLM client configured for the specified model tier.

    Args:
        tier: One of 'small', 'mid', 'frontier'.
        temperature: Override temperature (default from settings).

    Returns:
        BaseChatModel instance for the tier (currently always ChatOpenAI).
    """
    cfg = resolve_tier_config(tier)
    _temperature = temperature if temperature is not None else settings.llm_temperature

    client = build_chat_model(
        provider=cfg["provider"],
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=_temperature,
    )
    logger.info(
        "Tier LLM client created (tier=%s, provider=%s, model=%s, base_url=%s)",
        tier,
        cfg["provider"],
        cfg["model"],
        cfg["base_url"] or "default",
    )
    return client


async def invoke_llm_by_tier(
    messages: list[dict[str, str]],
    tier: ModelTier,
    temperature: Optional[float] = None,
) -> str:
    """Invoke an LLM at the specified tier.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        tier: One of 'small', 'mid', 'frontier'.
        temperature: Override temperature.

    Returns:
        The LLM response content string.
    """
    client = get_llm_client_by_tier(tier=tier, temperature=temperature)
    response = await client.ainvoke(messages)
    return response.content


async def invoke_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Invoke the LLM with a list of messages and return the response content.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        model: Override LLM model.
        temperature: Override temperature.

    Returns:
        The LLM response content string.
    """
    client = get_llm_client(model=model, temperature=temperature)
    response = await client.ainvoke(messages)
    return response.content
