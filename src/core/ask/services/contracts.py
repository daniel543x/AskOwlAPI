"""Tool-call contracts for the ask pipeline.

Every LLM interaction in the pipeline goes through one of these native tool
contracts; free-form generation is never trusted as an interface.
"""

from pydantic import BaseModel, Field


class SubmitSearchQuery(BaseModel):
    """Contract for the query-rewriting step."""

    query: str = Field(
        description=(
            "The optimized standalone web search query: resolves all "
            "pronouns and omissions from the conversation, entities and "
            "nouns only, no conversational filler, same language as the "
            "user's question. Add a year qualifier when recency matters."
        )
    )
    standalone: bool = Field(
        default=True,
        description=(
            "False if the raw query needed no conversation context and "
            "was returned essentially unchanged."
        ),
    )


class CitedAnswer(BaseModel):
    """Contract for the answer-synthesis step."""

    answer: str = Field(
        description=(
            "The final answer in Markdown. Base it strictly on the provided "
            "context chunks and cite them with bracketed numbers referring "
            "to the numbered context chunks, e.g. [1] or [3], placed right "
            "after each claim; only cite numbers that exist among the "
            "context chunks. If the context does not contain the answer, "
            "say so explicitly instead of guessing."
        )
    )
