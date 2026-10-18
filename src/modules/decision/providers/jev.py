"""Jev-compatible HTTP decision client (`laya[serve]` sidecar or Jev cloud).

Wire protocol (POST /v1/systemone, /v1/systemone/batch): a state plus
typed questions (choice / score / noul); answers carry probabilities that
we temperature-scale before use. Egress goes through the shared client
factory, so PROXY_URL applies.
"""

import logging
from typing import Any

import httpx

from ....tools.http import make_client
from ..base import DecisionError, IDecisionMaker

logger = logging.getLogger(__name__)


class JevCompatibleClient(IDecisionMaker):
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        calibration: dict[str, float] | None = None,
        model: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.calibration = dict(calibration or {})
        self.model = model
        # Injectable transport for tests.
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        return payload

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with make_client(
                timeout=self.timeout,
                headers=self._headers(),
                transport=self._transport,
            ) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
        except httpx.RequestError as e:
            raise DecisionError(f"Decision backend unreachable: {e}") from e

        if response.status_code != 200:
            raise DecisionError(
                f"Decision backend returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise DecisionError("Decision backend returned invalid JSON.") from e

    async def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        body = await self._post("/v1/systemone", self._payload(state, questions))
        answers = body.get("answers")
        if not isinstance(answers, dict) or not answers:
            raise DecisionError("Decision backend response missing answers.")
        return answers

    async def ask_batch(
        self, states: list[str], questions: dict[str, dict]
    ) -> list[dict[str, dict]]:
        if not states:
            return []
        body = await self._post(
            "/v1/systemone/batch",
            {"states": states, **self._payload(states[0], questions)},
        )
        # Batch answers come as a list of {question_id: answer} objects.
        results = body.get("results")
        if results is None:
            answers = body.get("answers")
            results = [answers] * len(states) if isinstance(answers, dict) else None
        if not isinstance(results, list) or len(results) != len(states):
            raise DecisionError("Decision backend batch response malformed.")
        return results
