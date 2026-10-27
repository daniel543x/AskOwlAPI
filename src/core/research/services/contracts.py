"""Tool-call contracts for the research pipeline and the agent loop.

Same rule as the ask pipeline: the LLM is never trusted with free-form
output. Planning, report synthesis, and the agent's final outcome must all
arrive as native tool calls.
"""

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Contract for the research planner step."""

    subquestions: list[str] = Field(
        description=(
            "3 to 6 focused web-search sub-questions that together cover the "
            "user's question. Each must be a standalone search query "
            "(entities and nouns, same language as the question), not a "
            "restatement of the whole topic."
        )
    )


class ResearchReport(BaseModel):
    """Contract for the research synthesis step."""

    report: str = Field(
        description=(
            "The final research report in Markdown: one section per "
            "sub-question, findings grounded strictly in the provided "
            "context chunks, and inline citations as [Source](URL) after "
            "each claim. State explicitly when the context does not cover "
            "something instead of guessing."
        )
    )


class SubmitOutcome(BaseModel):
    """Contract for the agent loop's final answer."""

    answer: str = Field(
        description=(
            "The agent's final answer in Markdown, grounded in the context "
            "gathered through the tools, with inline citations as "
            "[Source](URL). Call this tool exactly once when the goal is "
            "answered; say what could not be established instead of guessing."
        )
    )


class MissingInfo(BaseModel):
    """Reflection step (WU-7): what is still unknown, and the next search."""

    missing: list[str] = Field(
        description=(
            "Concrete facts or angles still missing from the research "
            "report. Empty when the gathered context already suffices."
        )
    )
    follow_up_queries: list[str] = Field(
        description=(
            "Up to 3 standalone web-search queries for the missing items. "
            "Empty when nothing is missing."
        ),
        max_length=3,
    )


class ContextBrief(BaseModel):
    """Compression step (WU-8): merge new context into a running brief."""

    brief: str = Field(
        description=(
            "Updated running summary: previously established facts merged "
            "with the new context, every numeric value kept exact (never "
            "rounded or generalized), source URLs kept inline."
        )
    )
