"""Answer faithfulness evaluation using LLM-as-judge."""

import logging
import re

from pydantic import BaseModel, Field

from src.evaluation.golden_set import GoldenSet
from src.generation.llm import invoke_llm

logger = logging.getLogger(__name__)


FAITHFULNESS_JUDGE_PROMPT = """You are an evaluation judge. Your task is to assess whether \
an AI-generated answer is faithful to the provided context (retrieved chunks).

Faithfulness means the answer ONLY contains claims that are directly supported by the context. \
Any claim not grounded in the context is considered unfaithful (hallucination).

For each answer, evaluate:
1. **Supported claims**: Statements directly backed by the context.
2. **Unsupported claims**: Statements not found in or contradicted by the context.
3. **Faithfulness score**: A score from 0.0 (completely unfaithful) to 1.0 (fully faithful).

Respond in this exact format:
SUPPORTED_CLAIMS: <count>
UNSUPPORTED_CLAIMS: <count>
FAITHFULNESS_SCORE: <0.0-1.0>
REASONING: <brief explanation>
"""


class FaithfulnessResult(BaseModel):
    """Result of faithfulness evaluation for a single answer."""

    question_id: str
    question: str
    answer: str
    context_snippet: str = ""
    supported_claims: int = 0
    unsupported_claims: int = 0
    faithfulness_score: float = 0.0
    reasoning: str = ""


class FaithfulnessReport(BaseModel):
    """Aggregate faithfulness evaluation report."""

    golden_set_name: str
    domain: str
    total_evaluated: int = 0
    mean_faithfulness: float = 0.0
    fully_faithful_count: int = 0
    hallucination_count: int = 0
    results: list[FaithfulnessResult] = Field(default_factory=list)
    passed: bool = False
    threshold: float = 0.7


async def evaluate_faithfulness(
    golden_set: GoldenSet,
    answers: list[dict],
    contexts: list[list[str]],
    threshold: float = 0.7,
) -> FaithfulnessReport:
    """Evaluate answer faithfulness using LLM-as-judge.

    Args:
        golden_set: Golden set used for evaluation.
        answers: List of generated answers (one per golden question).
        contexts: List of context chunk lists (one list per question).
        threshold: Minimum mean faithfulness score to pass.

    Returns:
        FaithfulnessReport with per-answer and aggregate scores.
    """
    results: list[FaithfulnessResult] = []

    for i, question in enumerate(golden_set.questions):
        if i >= len(answers):
            break

        answer = answers[i].get("answer", "") if isinstance(answers[i], dict) else str(answers[i])
        context_chunks = contexts[i] if i < len(contexts) else []
        context_text = "\n\n---\n\n".join(context_chunks[:5])

        # Ask LLM judge to evaluate faithfulness
        messages = [
            {"role": "system", "content": FAITHFULNESS_JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question.question}\n\n"
                    f"Answer: {answer}\n\n"
                    "Evaluate the faithfulness of this answer."
                ),
            },
        ]

        try:
            judge_response = await invoke_llm(messages, temperature=0.0)
            parsed = _parse_judge_response(judge_response)
        except Exception as e:
            logger.warning("Faithfulness judge failed for %s: %s", question.id, e)
            parsed = {
                "supported": 0, "unsupported": 0,
                "score": 0.0, "reasoning": f"Judge error: {e}",
            }

        results.append(FaithfulnessResult(
            question_id=question.id,
            question=question.question,
            answer=answer,
            context_snippet=context_text[:500],
            supported_claims=parsed["supported"],
            unsupported_claims=parsed["unsupported"],
            faithfulness_score=parsed["score"],
            reasoning=parsed["reasoning"],
        ))

    # Aggregate
    n = len(results) or 1
    mean_faith = sum(r.faithfulness_score for r in results) / n
    fully_faithful = sum(1 for r in results if r.faithfulness_score >= 0.95)
    hallucinations = sum(1 for r in results if r.unsupported_claims > 0)

    report = FaithfulnessReport(
        golden_set_name=golden_set.name,
        domain=golden_set.domain,
        total_evaluated=len(results),
        mean_faithfulness=round(mean_faith, 4),
        fully_faithful_count=fully_faithful,
        hallucination_count=hallucinations,
        results=results,
        passed=mean_faith >= threshold,
        threshold=threshold,
    )

    logger.info(
        "Faithfulness eval [%s]: mean=%.3f, faithful=%d/%d, passed=%s",
        golden_set.domain, mean_faith, fully_faithful, len(results), report.passed,
    )

    return report


def _parse_judge_response(response: str) -> dict:
    """Parse the LLM judge response for faithfulness metrics."""
    supported = 0
    unsupported = 0
    score = 0.0
    reasoning = ""

    supported_match = re.search(r"SUPPORTED_CLAIMS:\s*(\d+)", response)
    if supported_match:
        supported = int(supported_match.group(1))

    unsupported_match = re.search(r"UNSUPPORTED_CLAIMS:\s*(\d+)", response)
    if unsupported_match:
        unsupported = int(unsupported_match.group(1))

    score_match = re.search(r"FAITHFULNESS_SCORE:\s*([\d.]+)", response)
    if score_match:
        score = min(1.0, max(0.0, float(score_match.group(1))))

    reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return {
        "supported": supported,
        "unsupported": unsupported,
        "score": score,
        "reasoning": reasoning,
    }
