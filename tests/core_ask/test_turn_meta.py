"""Turn metadata on stored user messages (webapp API-NOTES #11).

The UI renders a "Search · qwen3.5 (Local)" caption under each question.
Live SSE turns carry the mode and model resolution, but the persisted
turns did not, so the caption vanished after a page reload. The user
message of every persisted turn now stores a small `meta` object filled
at turn creation from the ask request's mode and its resolved chat model.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from src.core.ask.services.search import build_turn_meta, search_json_adapter
from src.modules.history.model import Conversation, Message, MessageRead
from src.modules.history.repository import HistoryRepository
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    make_model,
    make_scraper,
    tool_call,
)


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self, *_args, **_kwargs):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class RecordingSession:
    """Minimal AsyncSession stand-in that records transaction boundaries."""

    def __init__(self, get_result=None):
        self.commits = 0
        self.added: list[Any] = []
        self._get_result = get_result

    async def get(self, _model, key):
        return self._get_result

    async def refresh(self, instance, attribute_names=None):
        return None

    def add(self, instance):
        self.added.append(instance)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def execute(self, _statement, params=None):
        return FakeResult([])

    def exec(self, _statement):
        return FakeResult([])


def model_with_metadata() -> Any:
    return SimpleNamespace(
        metadata={
            "catalog_entry_id": "3f9d1cce-1111-4a5a-9d7a-000000000001",
            "catalog_label": "qwen3.5",
            "catalog_provider": "Local Ollama",
        },
        model_name="qwen3.5",
    )


def builtin_model() -> Any:
    return SimpleNamespace(metadata={}, model_name="qwen3:8b")


# <----- build_turn_meta: the caption payload ----->


def test_turn_meta_carries_mode_and_catalog_model():
    meta = build_turn_meta("search", model_with_metadata())

    assert meta == {
        "mode": "search",
        "mode_label": "Search",
        "model": "3f9d1cce-1111-4a5a-9d7a-000000000001",
        "model_label": "qwen3.5",
        "model_provider": "Local Ollama",
    }


def test_turn_meta_labels_every_ask_mode():
    assert build_turn_meta("search", None)["mode_label"] == "Search"
    assert build_turn_meta("research", None)["mode_label"] == "Research"
    assert build_turn_meta("deep_research", None)["mode_label"] == "Deep research"
    assert build_turn_meta("persona", None)["mode_label"] == "Persona"


def test_turn_meta_falls_back_for_builtin_models():
    meta = build_turn_meta("search", builtin_model())

    assert meta["model"] is None
    assert meta["model_label"] == "qwen3:8b"


# <----- Persistence: meta lands on the user message only ----->


@pytest.mark.asyncio
async def test_add_turn_stores_meta_on_the_user_message_only():
    session: Any = RecordingSession()
    conversation = Conversation(id=uuid4(), user_id=uuid4(), title="t")

    user_message, assistant_message = await HistoryRepository(session).add_turn(
        conversation,
        "what is rust?",
        "Rust is a systems programming language.",
        meta={"mode": "search", "model_label": "qwen3.5"},
    )

    assert user_message.meta == {"mode": "search", "model_label": "qwen3.5"}
    assert assistant_message.meta is None


@pytest.mark.asyncio
async def test_add_turn_without_meta_keeps_null():
    session: Any = RecordingSession()
    conversation = Conversation(id=uuid4(), user_id=uuid4(), title="t")

    user_message, _assistant = await HistoryRepository(session).add_turn(
        conversation, "q", "a"
    )

    assert user_message.meta is None


@pytest.mark.asyncio
async def test_search_turn_persists_the_request_meta():
    session: Any = RecordingSession(
        get_result=Conversation(id=uuid4(), user_id=uuid4(), title="t")
    )
    model = make_model(
        optimized_query="rust programming language",
        answer="Rust is safe. [1]",
    )
    model.responses = [
        tool_call("SubmitSearchQuery", {"query": "rust programming language"}),
        tool_call("CitedAnswer", {"answer": "Rust is safe. [1]"}),
    ]

    result = await search_json_adapter(
        "what is rust?",
        FakeSearchProvider(),
        model,
        make_scraper(),
        FakeRanker(),
        session=session,
        user_id=uuid4(),
        turn_meta=build_turn_meta("search", model_with_metadata()),
    )
    assert result["answer"]

    messages = [item for item in session.added if isinstance(item, Message)]
    by_role = {m.role: m for m in messages}
    user_meta = by_role["user"].meta
    assert user_meta is not None
    assert user_meta["mode"] == "search"
    assert user_meta["model_label"] == "qwen3.5"
    assert by_role["assistant"].meta is None


def test_message_read_exposes_meta():
    read = MessageRead(
        id=uuid4(),
        role="user",
        content="q",
        created_at=datetime.now(UTC),
        meta={"mode": "search"},
    )

    assert read.meta == {"mode": "search"}
