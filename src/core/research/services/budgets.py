"""Budgets bounding the agent loop and the research pipeline.

Three dimensions, all best-effort guards against runaway LLM/tool usage:
- iterations: tool-call rounds in the agent loop,
- time: wall-clock deadline for the whole run,
- tokens: estimated tokens (characters // 4) pushed through LLM calls.

Exhausting a budget never raises: callers degrade to whatever context was
already gathered.
"""

import time
from dataclasses import dataclass, field

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-independent token estimate."""
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


@dataclass
class Budget:
    """Shared mutable budget for one pipeline/loop run."""

    max_iterations: int
    time_budget_seconds: float
    token_budget: int
    _deadline: float = field(init=False)
    _tokens_spent: int = field(default=0, init=False)
    _iterations_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._deadline = time.monotonic() + self.time_budget_seconds

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    @property
    def time_exhausted(self) -> bool:
        return time.monotonic() >= self._deadline

    @property
    def tokens_exhausted(self) -> bool:
        return self._tokens_spent >= self.token_budget

    @property
    def iterations_left(self) -> int:
        return max(0, self.max_iterations - self._iterations_used)

    @property
    def exhausted(self) -> bool:
        return self.time_exhausted or self.tokens_exhausted

    def spend_tokens(self, *texts: str) -> None:
        for text in texts:
            self._tokens_spent += estimate_tokens(text)

    def spend_iteration(self) -> None:
        self._iterations_used += 1

    def stop_reason(self) -> str | None:
        """Human-readable reason if a budget is exhausted, else None."""
        if self.time_exhausted:
            return "time budget exhausted"
        if self.tokens_exhausted:
            return "token budget exhausted"
        if self.iterations_left == 0:
            return "iteration budget exhausted"
        return None
