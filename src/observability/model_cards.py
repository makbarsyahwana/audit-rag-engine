"""Model cards: track LLM and embedding model versions in use."""

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


class ModelCard(BaseModel):
    """Metadata card for an AI model in use."""

    model_id: str
    provider: str                      # openai, anthropic, local, etc.
    model_name: str                    # e.g. "gpt-4o-mini"
    version: str = ""
    purpose: str                       # generation, embedding, reranking, extraction
    context_window: int = 0
    max_output_tokens: int = 0
    temperature: float = 0.0
    cost_per_1k_input: float = 0.0     # USD per 1k input tokens
    cost_per_1k_output: float = 0.0    # USD per 1k output tokens
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    ethical_notes: str = ""
    registered_at: float = Field(default_factory=time.time)


class ModelRegistry:
    """Registry of all AI models used by the RAG engine."""

    def __init__(self) -> None:
        self._cards: dict[str, ModelCard] = {}

    def register(self, card: ModelCard) -> None:
        """Register a model card."""
        self._cards[card.model_id] = card
        logger.info("Model card registered: %s (%s)", card.model_id, card.model_name)

    def get(self, model_id: str) -> Optional[ModelCard]:
        """Get a model card by ID."""
        return self._cards.get(model_id)

    def list_all(self) -> list[ModelCard]:
        """List all registered model cards."""
        return list(self._cards.values())

    def to_dict(self) -> list[dict]:
        """Export all model cards as dicts."""
        return [c.model_dump() for c in self._cards.values()]


# Singleton
model_registry = ModelRegistry()


def register_default_models() -> None:
    """Register model cards for the default models configured in settings."""
    model_registry.register(ModelCard(
        model_id="llm-primary",
        provider="openai",
        model_name=settings.llm_model,
        purpose="generation",
        context_window=128000,
        max_output_tokens=16384,
        temperature=settings.llm_temperature,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        capabilities=[
            "audit Q&A with citations",
            "workpaper narrative drafting",
            "finding drafting",
            "entity/relationship extraction",
            "faithfulness judging",
        ],
        limitations=[
            "Cannot access real-time data",
            "May hallucinate if context is insufficient",
            "Audit conclusions require human review",
        ],
        ethical_notes=(
            "All generated content is grounded in retrieved evidence. "
            "Confidence scoring and abstention mechanisms reduce hallucination risk. "
            "Human-in-the-loop is required for formal audit deliverables."
        ),
    ))

    model_registry.register(ModelCard(
        model_id="embedding-primary",
        provider="openai",
        model_name=settings.embedding_model,
        purpose="embedding",
        context_window=8191,
        cost_per_1k_input=0.00002,
        capabilities=[
            "document chunk embedding",
            "query embedding for vector search",
        ],
        limitations=[
            "Fixed dimension output",
            "English-optimized",
        ],
    ))
