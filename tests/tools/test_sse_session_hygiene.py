"""DB transaction hygiene inside streaming responses (webapp API-NOTES #10).

Every SSE route shares the request session with its auth dependency. The
auth SELECT used to open a transaction that nothing closed until the
dependency teardown after the stream ended, so each open stream parked one
pool connection idle-in-transaction; reconnect loops exhausted the pool
and took the whole API down. These tests pin the release points: the auth
read, the conversation/history reads, and every store read must end their
transaction as soon as their work is done.
"""

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.exc import SQLAlchemyError

from src.core.ask.services.search import _resolve_conversation
from src.modules.chunk_store.repository import ChunkRepository
from src.modules.history.model import Conversation
from src.modules.identity.auth.security import get_current_user
from src.modules.identity.auth.service import create_access_token
from src.modules.notes.repository import NoteRepository


class FakeResult:
    """Mimics the result surface the repositories actually use."""

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

    def __init__(self, get_result=None, execute_rows=(), exec_rows=()):
        self.commits = 0
        self.rollbacks = 0
        self.added = []
        self._get_result = get_result
        self._execute_rows = list(execute_rows)
        self._exec_rows = list(exec_rows)

    async def get(self, _model, key):
        return self._get_result

    async def refresh(self, instance, attribute_names=None):
        return None

    def add(self, instance):
        self.added.append(instance)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1

    async def execute(self, _statement, params=None):
        return FakeResult(self._execute_rows)

    async def exec(self, _statement):
        return FakeResult(self._exec_rows)


# <----- Auth ----->


@pytest.mark.asyncio
async def test_get_current_user_releases_the_auth_transaction():
    user = SimpleNamespace(id=uuid4(), is_active=True)
    session: Any = RecordingSession(get_result=user)
    request: Any = SimpleNamespace(state=SimpleNamespace())
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=create_access_token({"sub": str(user.id)})
    )

    resolved = await get_current_user(request, credentials, None, session)

    assert resolved is user
    assert session.commits >= 1


# <----- Conversation reads ----->


@pytest.mark.asyncio
async def test_resolve_conversation_releases_the_read_transaction():
    conversation = Conversation(id=uuid4(), user_id=uuid4(), title="t")
    session: Any = RecordingSession(get_result=conversation)

    resolved, history, created = await _resolve_conversation(
        session, conversation.user_id, conversation.id, "question"
    )

    assert resolved is conversation
    assert history == []
    assert created is False
    assert session.commits >= 1
    assert conversation.id == conversation.id


@pytest.mark.asyncio
async def test_resolve_conversation_releases_the_transaction_when_missing():
    session: Any = RecordingSession(get_result=None)

    with pytest.raises(KeyError):
        await _resolve_conversation(session, uuid4(), uuid4(), "question")

    assert session.commits >= 1


@pytest.mark.asyncio
async def test_resolve_conversation_releases_the_transaction_after_create():
    session: Any = RecordingSession(get_result=None)

    _conversation, _history, created = await _resolve_conversation(
        session, uuid4(), None, "question"
    )

    assert created is True
    assert session.commits >= 2


# <----- Chunk store reads ----->


@pytest.mark.asyncio
async def test_chunk_repository_reads_release_the_transaction():
    session: Any = RecordingSession(
        execute_rows=[SimpleNamespace(content_hash="h1")]
    )

    hashes = await ChunkRepository(session).existing_hashes(["h1"])

    assert hashes == {"h1"}
    assert session.commits == 1
    assert session.rollbacks == 0


@pytest.mark.asyncio
async def test_chunk_repository_vector_read_releases_the_transaction():
    row = SimpleNamespace(
        id=1,
        content="chunk text",
        metadata={},
        source_url="http://example.com",
        distance=0.1,
    )
    session: Any = RecordingSession(execute_rows=[row])

    candidates = await ChunkRepository(session).vector_candidates([1.0, 0.0], k=1)

    assert candidates
    assert session.commits == 1
    assert session.rollbacks == 0


@pytest.mark.asyncio
async def test_chunk_repository_failures_still_roll_back():
    class ExplodingSession(RecordingSession):
        async def execute(self, statement, params=None):
            raise SQLAlchemyError("boom")

    session: Any = ExplodingSession()

    with pytest.raises(SQLAlchemyError):
        await ChunkRepository(session).existing_hashes(["h1"])

    assert session.rollbacks == 1
    assert session.commits == 0


# <----- Note reads ----->


@pytest.mark.asyncio
async def test_note_reads_release_the_transaction():
    note = SimpleNamespace(id=uuid4())
    session: Any = RecordingSession(exec_rows=[note])

    found = await NoteRepository(session).get_by_ids([note.id])

    assert found == [note]
    assert session.commits == 1
    assert session.rollbacks == 0
