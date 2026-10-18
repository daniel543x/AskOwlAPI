"""Offline tests for pipeline pre-scores and their degradation paths."""

import pytest

from src.core.tools.pre_scores import (
    choose_query,
    filter_chunks,
    filter_search_results,
    grounding_confidence,
)
from src.modules.decision.base import (
    DecisionError,
    IDecisionMaker,
)
from src.modules.search_engine.providers.base import SearchResult
from src.tools.settings import settings


class FakeDecisionMaker(IDecisionMaker):
    def __init__(
        self,
        noul_yes=0.9,
        scores=None,
        choice="optimized",
        fail=False,
    ):
        self.noul_yes = noul_yes
        self.scores = scores or {}
        self.choice_value = choice
        self.fail = fail
        self.calls = []

    async def ask(self, state, questions):
        self.calls.append(("ask", state, questions))
        if self.fail:
            raise DecisionError("backend down")
        qid, question = next(iter(questions.items()))
        if question["type"] == "noul":
            return {qid: {"type": "noul", "noul": self.noul_yes}}
        if question["type"] == "choice":
            winner = self.choice_value
            loser = "raw" if winner == "optimized" else "optimized"
            return {
                qid: {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": {winner: 0.7, loser: 0.3},
                }
            }
        return self._score_answer(3.0)

    @staticmethod
    def _score_answer(score):
        probs = {str(i): 0.2 for i in range(5)}
        return {"q": {"type": "score", "score": score,
                      "probabilities": probs, "legend": {}}}

    async def ask_batch(self, states, questions):
        self.calls.append(("batch", len(states)))
        if self.fail:
            raise DecisionError("backend down")
        question = next(iter(questions.values()))
        out = []
        for index in range(len(states)):
            if question["type"] == "noul":
                yes = self.noul_yes if index in self.scores.get("keep", [0]) else 0.1
                out.append({"q": {"type": "noul", "noul": yes}})
            else:
                score = self.scores.get("chunks", [3.0] * len(states))[index]
                out.append(self._score_answer(score))
        return out


def results(*urls):
    return [SearchResult(title=f"T {u}", snippet=f"S {u}", url=u) for u in urls]


# <----- search-result filter ----->


@pytest.mark.asyncio
async def test_search_filter_drops_irrelevant_results(monkeypatch):
    monkeypatch.setattr(settings, "PRE_SCORE_RELEVANCE_THRESHOLD", 0.5)
    maker = FakeDecisionMaker(scores={"keep": [0]})
    kept = await filter_search_results(
        maker, "rust", results("http://a", "http://b")
    )

    assert [r.url for r in kept] == ["http://a"]


@pytest.mark.asyncio
async def test_search_filter_never_empties_the_result_set():
    maker = FakeDecisionMaker(noul_yes=0.01)  # everything "irrelevant"
    kept = await filter_search_results(
        maker, "rust", results("http://a", "http://b")
    )

    assert kept  # degrades to keeping everything rather than zero results


@pytest.mark.asyncio
async def test_search_filter_failure_keeps_everything():
    maker = FakeDecisionMaker(fail=True)
    kept = await filter_search_results(
        maker, "rust", results("http://a", "http://b")
    )

    assert len(kept) == 2


@pytest.mark.asyncio
async def test_search_filter_without_maker_is_a_noop():
    kept = await filter_search_results(None, "rust", results("http://a"))

    assert len(kept) == 1


# <----- chunk filter ----->


@pytest.mark.asyncio
async def test_chunk_filter_drops_below_minimum(monkeypatch):
    monkeypatch.setattr(settings, "PRE_SCORE_CHUNK_MIN_SCORE", 3.0)
    maker = FakeDecisionMaker(scores={"chunks": [4.2, 1.1, 3.0]})
    chunks = [{"content": f"c{i}"} for i in range(3)]

    kept = await filter_chunks(maker, "q", chunks)

    assert kept == [{"content": "c0"}, {"content": "c2"}]


@pytest.mark.asyncio
async def test_chunk_filter_failure_keeps_all():
    maker = FakeDecisionMaker(fail=True)
    chunks = [{"content": "c"}]

    assert await filter_chunks(maker, "q", chunks) == chunks


# <----- query choice ----->


@pytest.mark.asyncio
async def test_choose_query_picks_the_decisions_candidate():
    maker = FakeDecisionMaker(choice="raw")
    chosen = await choose_query(maker, "raw query", "optimized query")

    assert chosen == "raw query"


@pytest.mark.asyncio
async def test_choose_query_defaults_to_optimized_on_failure():
    maker = FakeDecisionMaker(fail=True)

    assert await choose_query(maker, "raw", "optimized") == "optimized"


@pytest.mark.asyncio
async def test_choose_query_skips_identical_candidates():
    maker = FakeDecisionMaker()

    assert await choose_query(maker, "same ", "same") == "same"
    assert maker.calls == []


# <----- grounding ----->


@pytest.mark.asyncio
async def test_grounding_returns_yes_probability():
    maker = FakeDecisionMaker(noul_yes=0.82)

    assert await grounding_confidence(maker, "q", "ctx", "answer") == pytest.approx(
        0.82
    )


@pytest.mark.asyncio
async def test_grounding_failure_returns_none():
    maker = FakeDecisionMaker(fail=True)

    assert await grounding_confidence(maker, "q", "ctx", "answer") is None


@pytest.mark.asyncio
async def test_grounding_without_maker_returns_none():
    assert await grounding_confidence(None, "q", "ctx", "answer") is None


# <----- toolbox wiring (toggles gate the filters) ----->


@pytest.mark.asyncio
async def test_toolbox_search_filter_respects_the_toggle(monkeypatch):
    from src.core.tools.toolbox import AskToolbox
    from tests.core_ask.fakes import FakeRanker, FakeScraper, FakeSearchProvider

    maker = FakeDecisionMaker(noul_yes=0.01)
    monkeypatch.setattr(settings, "PRE_SCORE_SEARCH_FILTER", False)
    toolbox = AskToolbox(
        FakeSearchProvider(), FakeScraper(), FakeRanker(), decision_maker=maker
    )
    await toolbox.web_search("rust")

    assert len(toolbox.ranked_sources) == 2  # filter off: nothing dropped

    monkeypatch.setattr(settings, "PRE_SCORE_SEARCH_FILTER", True)
    toolbox2 = AskToolbox(
        FakeSearchProvider(), FakeScraper(), FakeRanker(), decision_maker=maker
    )
    await toolbox2.web_search("rust")

    # Everything scored irrelevant -> keeps the set (never empty).
    assert toolbox2.ranked_sources


def test_pre_score_tool_only_with_decision_maker():
    from src.core.tools.toolbox import AskToolbox, build_tools
    from tests.core_ask.fakes import FakeRanker, FakeScraper, FakeSearchProvider

    bare = build_tools(
        AskToolbox(FakeSearchProvider(), FakeScraper(), FakeRanker())
    )
    equipped = build_tools(
        AskToolbox(
            FakeSearchProvider(), FakeScraper(), FakeRanker(),
            decision_maker=FakeDecisionMaker(),
        )
    )

    assert [t.name for t in bare] == ["web_search", "fetch_pages", "get_context"]
    assert [t.name for t in equipped] == [
        "pre_score", "web_search", "fetch_pages", "get_context",
    ]


@pytest.mark.asyncio
async def test_pre_score_tool_degrades_without_backend():
    from src.core.tools.toolbox import AskToolbox, _pre_score
    from tests.core_ask.fakes import FakeRanker, FakeScraper, FakeSearchProvider

    toolbox = AskToolbox(
        FakeSearchProvider(), FakeScraper(), FakeRanker(),
        decision_maker=FakeDecisionMaker(fail=True),
    )
    import json

    result = json.loads(await _pre_score(toolbox, "q?", "text"))
    assert "error" in result
