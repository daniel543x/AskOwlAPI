"""Per-user job event fan-out for the Inbox SSE stream (PR 13).

In-process pub-sub (single-node, like the JobQueue itself): the queue
publishes on every job transition, each connected Inbox listener gets a
`job_update` payload on its own asyncio.Queue. A Redis-backed bus
arrives with the multi-node topology (decision 19).
"""

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class JobEventBus:
    def __init__(self) -> None:
        self._listeners: dict[uuid.UUID, set[asyncio.Queue]] = {}

    def subscribe(self, user_id: uuid.UUID) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._listeners.setdefault(user_id, set()).add(queue)
        return queue

    def unsubscribe(self, user_id: uuid.UUID, queue: asyncio.Queue) -> None:
        listeners = self._listeners.get(user_id)
        if listeners is not None:
            listeners.discard(queue)
            if not listeners:
                self._listeners.pop(user_id, None)

    def publish(self, user_id: uuid.UUID, payload: dict) -> None:
        for queue in self._listeners.get(user_id, ()):
            queue.put_nowait(payload)

    def listener_count(self, user_id: uuid.UUID) -> int:
        return len(self._listeners.get(user_id, ()))
