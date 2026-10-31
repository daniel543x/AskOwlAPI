"""Tool-call contracts for the ask pipeline.

Every LLM interaction in the pipeline goes through one of these native tool
contracts; free-form generation is never trusted as an interface.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class SubmitSearchQuery(BaseModel):
    """Contract for the query-rewriting step."""

    query: str = Field(
        description=(
            "The optimized web search query: entities and nouns only, no "
            "conversational filler, same language as the user's question."
        )
    )


class AskBody(BaseModel):
    """The unified v3 ask body (protocol v3, PR 21).

    One body schema for every ask route; search adds its retrieval
    extras, persona requires persona_id. Routes ignore fields that do
    not apply to them.
    """

    query: str
    stream: bool = True
    conversation_id: uuid.UUID | None = None
    thinking: bool | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None
    source_types: list[str] = []
    background: bool = False
    attachment_ids: list[uuid.UUID] = []
    depth: Literal["quick", "research", "deep"] | None = None
    persona_id: uuid.UUID | None = None
    allow_model_fallback: bool = Field(
        default=False,
        description=(
            "Consent to continue on the standard resolution chain when the "
            "persona's own model is unusable (webapp API-NOTES #5). Without "
            "it, a persona turn ends with a model_confirm event (SSE) or a "
            "409 (JSON/background) instead of silently swapping the model."
        ),
    )


class CitedAnswer(BaseModel):
    """Contract for the answer-synthesis step."""

    answer: str = Field(
        description=(
            "The final answer in Markdown. Base it strictly on the provided "
            "context chunks and cite sources inline as [Source](URL) right "
            "after each claim. If the context does not contain the answer, "
            "say so explicitly instead of guessing."
        )
    )
