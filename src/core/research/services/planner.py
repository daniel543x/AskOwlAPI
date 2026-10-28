"""Research planner: decompose the question via a native tool call."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ....tools.dates import current_date_line
from .contracts import ResearchPlan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a Research Planner for a search engine.
Decompose the user's question into focused web-search sub-questions:
1. Coverage: together the sub-questions must fully cover the original
   question; avoid near-duplicates.
2. Searchability: each sub-question is a standalone search query (entities
   and nouns, no filler), in the user's language.
3. Brevity: at most the requested number of sub-questions.
You MUST respond by calling the ResearchPlan tool. Do not answer the
question and do not write any prose."""


async def plan_subquestions(
    model: BaseChatModel,
    query: str,
    max_subquestions: int,
    today: str | None = None,
) -> tuple[list[str], bool]:
    """Plan sub-questions via a native tool call.

    Returns (subquestions, used_tool_call). On a tool-protocol error the
    raw query becomes the single sub-question (the deterministic fallback
    path; free-form model text is never parsed or trusted).
    """
    bound = model.bind_tools([ResearchPlan])
    request = HumanMessage(
        content=f"Question: {query}\nAt most {max_subquestions} sub-questions."
    )
    response = await bound.ainvoke(
        [
            SystemMessage(
                content=f"{current_date_line(today)}\n{PLANNER_SYSTEM}"
            ),
            request,
        ]
    )

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "ResearchPlan":
            raw = tool_call.get("args", {}).get("subquestions", [])
            subquestions = [str(item).strip() for item in raw if str(item).strip()]
            if subquestions:
                return subquestions[:max_subquestions], True

    logger.warning(
        "Planning did not use the ResearchPlan tool; falling back to the "
        "raw query as the only sub-question (tool-protocol error path)."
    )
    return [query], False


REFLECT_SYSTEM = """You are reviewing a research run's material so far.
Given the query and a digest of what was gathered, decide whether any
gaps remain. Call `RevisePlan` exactly once: add ONLY standalone search
sub-questions for material the digest does not cover. If the material is
sufficient, add an empty list. You MUST respond with the RevisePlan tool
call; prose is a protocol violation."""


async def revise_research_plan(
    model: BaseChatModel,
    query: str,
    digest: str,
    today: str | None = None,
) -> tuple[list[str], str]:
    """One reflection round (PR 19): returns (added, rationale).

    A missing or prose response means "no additions" — reflection is
    optional and must never fail the run.
    """
    from ..services.contracts import RevisePlan

    bound = model.bind_tools([RevisePlan])
    try:
        response = await bound.ainvoke(
            [
                SystemMessage(
                    content=f"{current_date_line(today)}\n{REFLECT_SYSTEM}"
                ),
                HumanMessage(
                    content=f"Query: {query}\n\nGathered so far:\n{digest}"
                ),
            ]
        )
        for tool_call in response.tool_calls or []:
            if tool_call.get("name") == "RevisePlan":
                plan = RevisePlan.model_validate(tool_call.get("args", {}))
                seen: dict[str, None] = {}
                for q in plan.add:
                    cleaned_q = q.strip()
                    if cleaned_q:
                        seen.setdefault(cleaned_q, None)
                return list(seen), plan.rationale
    except Exception:
        logger.debug("RevisePlan reflection failed; treating as no-op.")
    return [], ""
