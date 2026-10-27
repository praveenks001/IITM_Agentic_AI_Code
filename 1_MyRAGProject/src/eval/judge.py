"""src/eval/judge.py — LLM-as-judge (task-success grader).

Lineage: this is the W5 judge, carried forward into the capstone and lightly
hardened for W11. In W5 it was the headline eval tool; from W11 onwards RAGAS
owns the four RAG-specific metrics (faithfulness, answer relevance, context
precision, context recall) and this judge is repurposed as the *task-success*
grader for KPI #1 — "did the answer actually meet the user's goal end-to-end".

Two changes vs the W5 reference:
  1. Lazy client — we never instantiate AsyncOpenAI at import time, so the
     module imports cleanly in pytest with no API key set (W7/W8 footgun).
  2. A thin sync wrapper `judge_one_sync` so the eval suite and KPI step can
     call the judge without managing an event loop.

The 3-dimension rubric (accuracy / groundedness / format, 1-4 each) is the
W5 rubric, unchanged. Adapt the dimension definitions if your capstone domain
differs; the structure stays put.
"""
from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Lazy client (W11 hardening) ─────────────────────────────────────────────
def _async_openai(api_key: str | None = None):
    """Create the AsyncOpenAI client on first use, not at import.

    Importing this module must never require an API key — unit tests import it
    with no environment set. The `openai` import is also deferred so a machine
    without the package can still import the rubric constants.
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key)


# ─── Output schema ───────────────────────────────────────────────────────────
class JudgeScore(BaseModel):
    """One judge verdict on one candidate answer."""

    accuracy:     int = Field(..., ge=1, le=4)
    groundedness: int = Field(..., ge=1, le=4)
    format:       int = Field(..., ge=1, le=4)
    reasoning:    str = Field(..., min_length=1)

    @property
    def mean(self) -> float:
        """Average across the three dimensions — useful for sorting/convergence."""
        return (self.accuracy + self.groundedness + self.format) / 3.0

    @property
    def task_success(self) -> bool:
        """KPI #1 rule: a 'successful' answer is accuracy>=3 AND groundedness>=3.

        Format alone never makes or breaks task success — a well-formatted but
        wrong answer is still a failure. Tune this bar in your ADR (D5).
        """
        return self.accuracy >= 3 and self.groundedness >= 3


# ─── Rubric prompt — Enterprise Knowledge Assistant capstone ─────────────────
RUBRIC_PROMPT: str = """\
You are a senior evaluator for an Enterprise Knowledge Assistant — an LLM-backed
service that answers employees' questions about company policies, processes,
and documents.

You rate each candidate answer on three dimensions, each on a 1-4 scale.

ACCURACY — Is the answer factually correct given the ideal answer?
  1 — Wrong. States something that contradicts the ideal answer.
  2 — Partially correct. Some facts right, some wrong or missing critical detail.
  3 — Correct. All stated facts align with the ideal; minor omissions acceptable.
  4 — Correct and complete. Captures every fact in the ideal answer.

GROUNDEDNESS — Does every claim trace to the ideal/source, or does the answer
honestly admit uncertainty when relevant?
  1 — Hallucinates. Invents facts not in the source.
  2 — Mostly grounded but adds 1-2 unsupported claims.
  3 — Grounded. All claims trace to the source. May lack citations.
  4 — Grounded and explicit. Claims trace clearly; cites where possible; admits
      uncertainty if the ideal answer would.

FORMAT — Is the structure, length, and presentation right for the question?
  1 — Wrong shape. Too long/short, missing required structure, garbled.
  2 — Functional but rough. Right info, wrong shape.
  3 — Appropriate. Right length, clear structure, reasonable presentation.
  4 — Polished. Length and structure match what a senior writer would produce.

ALWAYS call the rate_answer tool. Never reply in plain text. The reasoning field
should be 2-4 sentences naming what's strong and what's weak.
"""


# ─── Tool schema for structured output ───────────────────────────────────────
RATE_ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "rate_answer",
        "description": (
            "Rate a candidate answer on three dimensions, 1-4 each, "
            "and explain your reasoning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "accuracy":     {"type": "integer", "minimum": 1, "maximum": 4},
                "groundedness": {"type": "integer", "minimum": 1, "maximum": 4},
                "format":       {"type": "integer", "minimum": 1, "maximum": 4},
                "reasoning":    {"type": "string"},
            },
            "required": ["accuracy", "groundedness", "format", "reasoning"],
        },
    },
}


# ─── Judge call ──────────────────────────────────────────────────────────────
async def judge_one(
    question: str,
    candidate_answer: str,
    ideal_answer: str,
    *,
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
) -> JudgeScore:
    """Run the judge and close the HTTP client before the event loop exits."""

    async with _async_openai(api_key) as client:

        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"IDEAL ANSWER:\n{ideal_answer}\n\n"
            f"CANDIDATE ANSWER:\n{candidate_answer}\n\n"
            "Rate the candidate using the rate_answer tool."
        )

        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=[RATE_ANSWER_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "rate_answer"},
            },
        )

        tool_calls = resp.choices[0].message.tool_calls or []

        if not tool_calls:
            raise RuntimeError(
                "Judge did not call the rate_answer tool."
            )

        args = json.loads(tool_calls[0].function.arguments)

        return JudgeScore.model_validate(args)


def judge_one_sync(
    question: str,
    candidate_answer: str,
    ideal_answer: str,
    *,
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
) -> JudgeScore:
    """Synchronous convenience wrapper around `judge_one`.

    The eval suite (test_kpi.py) is synchronous pytest; this lets it grade
    task success without managing an event loop.
    """
    return asyncio.run(
        judge_one(
            question, candidate_answer, ideal_answer,
            judge_model=judge_model, api_key=api_key,
        )
    )
