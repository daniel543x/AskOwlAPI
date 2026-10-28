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


class ReportSection(BaseModel):
    """One planned section of the note-assembled report (PR 18)."""

    idx: int = Field(description="The note's idx from the TOC.")
    include: bool = Field(default=True, description="Include this section.")
    order: int = Field(description="Position of this section in the report.")
    lead_in: str = Field(
        default="",
        description=(
            "One sentence connecting this section to the query (optional)."
        ),
    )


class SubmitReport(BaseModel):
    """Editorial contract: assemble the report from the notes' TOC only.

    The model sees the TOC (never the full note contents) and plans the
    report; the backend streams the sections deterministically.
    """

    sections: list[ReportSection] = Field(
        description="Sections to include, each referencing a note idx."
    )
    intro: str = Field(
        default="",
        description="A short intro paragraph for the report (optional).",
    )
