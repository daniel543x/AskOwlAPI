"""Decision-model interface: typed answers with calibrated probabilities.

One Jev-compatible HTTP client serves both backends (PLAN decision #2,
2026-10-18): a self-hosted `laya[serve]` sidecar or the Jev cloud. Raw
checkpoints are over-confident — probabilities pass through temperature
scaling (per question type and option count) before anyone gates on them.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DecisionError(Exception):
    """The decision backend failed or answered malformed data."""


@dataclass
class ChoiceDecision:
    question_id: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class ScoreDecision:
    question_id: str
    score: float
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


@dataclass
class NoulDecision:
    question_id: str
    yes: float
    confidence: float


def temperature_scale(
    probabilities: dict[str, float], temperature: float
) -> dict[str, float]:
    """Softmax(logits / T) expressed on probabilities.

    T > 1 flattens an over-confident distribution (the shipped Laya
    checkpoints need this), T < 1 sharpens, T = 1 is identity.
    """
    if temperature <= 0 or abs(temperature - 1.0) < 1e-9:
        return dict(probabilities)
    scaled = {
        key: max(value, 0.0) ** (1.0 / temperature)
        for key, value in probabilities.items()
    }
    total = sum(scaled.values())
    if total <= 0:
        return dict(probabilities)
    return {key: value / total for key, value in scaled.items()}


class IDecisionMaker(ABC):
    """Typed decisions over a state (text) plus questions.

    Implementations only need `ask` / `ask_batch`; the typed helpers below
    build on them and apply calibration.
    """

    # Per-(question-type, option-count) temperature overrides. Class-level
    # empty mapping (never mutated); implementations assign their own dict.
    calibration: dict[str, float] = {}

    def _temperature_for(self, question_type: str, option_count: int) -> float:
        return self.calibration.get(
            f"{question_type}:{option_count}", self._default_temperature()
        )

    def _default_temperature(self) -> float:
        return self.calibration.get("default", 1.0)

    @abstractmethod
    async def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """One state, many typed questions; raw answer objects per id."""

    @abstractmethod
    async def ask_batch(
        self, states: list[str], questions: dict[str, dict]
    ) -> list[dict[str, dict]]:
        """Same questions over many states (one decision call ideally)."""

    # <----- typed helpers ----->

    async def noul(self, instructions: str, state: str) -> NoulDecision:
        answers = await self.ask(
            state, {"q": {"type": "noul", "instructions": instructions}}
        )
        answer = answers["q"]
        yes = float(answer["noul"])
        scaled = temperature_scale(
            {"yes": yes, "no": 1.0 - yes}, self._temperature_for("noul", 2)
        )
        return NoulDecision(
            question_id="q",
            yes=scaled["yes"],
            confidence=max(scaled.values()),
        )

    async def score(
        self, instructions: str, criteria: list[str], state: str
    ) -> ScoreDecision:
        answers = await self.ask(
            state,
            {
                "q": {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        )
        answer = answers["q"]
        probabilities = {
            str(key): float(value) for key, value in answer["probabilities"].items()
        }
        scaled = temperature_scale(
            probabilities, self._temperature_for("score", len(probabilities))
        )
        return ScoreDecision(
            question_id="q",
            score=float(answer["score"]),
            probabilities=scaled,
            legend={str(k): str(v) for k, v in (answer.get("legend") or {}).items()},
            confidence=max(scaled.values()),
        )

    async def choice(
        self,
        instructions: str,
        criteria: dict[str, str] | list[str],
        state: str,
    ) -> ChoiceDecision:
        if isinstance(criteria, list):
            criteria = {item: item for item in criteria}
        answers = await self.ask(
            state,
            {
                "q": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        )
        answer = answers["q"]
        probabilities = {
            str(key): float(value) for key, value in answer["probabilities"].items()
        }
        scaled = temperature_scale(
            probabilities, self._temperature_for("choice", len(probabilities))
        )
        choice = max(scaled, key=scaled.get)
        return ChoiceDecision(
            question_id="q",
            choice=choice,
            probabilities=scaled,
            confidence=scaled[choice],
        )

    async def batch_noul(
        self, instructions: str, states: list[str]
    ) -> list[NoulDecision]:
        results = await self.ask_batch(
            states, {"q": {"type": "noul", "instructions": instructions}}
        )
        decisions = []
        for answers in results:
            yes = float(answers["q"]["noul"])
            scaled = temperature_scale(
                {"yes": yes, "no": 1.0 - yes},
                self._temperature_for("noul", 2),
            )
            decisions.append(
                NoulDecision(
                    question_id="q", yes=scaled["yes"], confidence=max(scaled.values())
                )
            )
        return decisions

    async def batch_score(
        self, instructions: str, criteria: list[str], states: list[str]
    ) -> list[ScoreDecision]:
        results = await self.ask_batch(
            states,
            {
                "q": {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        )
        decisions = []
        for answers in results:
            answer = answers["q"]
            probabilities = {
                str(key): float(value) for key, value in answer["probabilities"].items()
            }
            scaled = temperature_scale(
                probabilities, self._temperature_for("score", len(probabilities))
            )
            decisions.append(
                ScoreDecision(
                    question_id="q",
                    score=float(answer["score"]),
                    probabilities=scaled,
                    legend={
                        str(k): str(v) for k, v in (answer.get("legend") or {}).items()
                    },
                    confidence=max(scaled.values()),
                )
            )
        return decisions
