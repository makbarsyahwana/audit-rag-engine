"""LLM client wrapper for generation (OpenAI/Anthropic)."""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

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
