"""Deterministic numbered citations, shared by every answer mode.

The model keeps citing markdown links; this module assigns the numbers
after the fact: each distinct known source gets a dense `[1..N]` by
first appearance (never-cited sources trail after the cited prefix), a
rewriter turns known links into `[n]` (streaming-safe via a bounded
carry buffer), and a validator reports the state of the sequence as a
verdict — never an exception. URLs are compared through
`normalize_url` so tracking params and `www.` cannot hide a match.
"""

import re
from dataclasses import dataclass, field

from ...tools.urls import normalize_url

# One pending link is never longer than a sane URL + label; beyond the
# bound it is flushed verbatim instead of buffering forever.
MAX_PENDING_LINK = 2048
MAX_PENDING_TAIL = 4096

MARKDOWN_LINK = re.compile(r"\[(?P<label>[^\]]*)\]\((?P<url>[^)\s]+)\)")
PROSE_MARKER = re.compile(r"\[(\d+)\]")
SOURCES_HEADING = "\n## Sources"


def build_registry(
    known_urls: list[str], cited_order: list[str]
) -> dict[str, int]:
    """Dense 1-based mapping for `known_urls`.

    Cited URLs (in `cited_order`, the first-appearance order) number
    first; the remaining known URLs follow in list order. Keys are the
    original `known_urls` strings; comparison is normalized.
    """
    original_by_norm: dict[str, str] = {}
    for url in known_urls:
        original_by_norm.setdefault(normalize_url(url), url)
    registry: dict[str, int] = {}
    for url in cited_order:
        original = original_by_norm.get(normalize_url(url))
        if original is not None and original not in registry:
            registry[original] = len(registry) + 1
    for url in known_urls:
        if url not in registry:
            registry[url] = len(registry) + 1
    return registry


def rewrite_citations(
    text: str, registry: dict[str, int]
) -> tuple[str, dict[str, int]]:
    """Rewrite known markdown links to `[n]`; leave everything else."""
    number_by_norm = {normalize_url(url): n for url, n in registry.items()}

    def replace(match: re.Match[str]) -> str:
        number = number_by_norm.get(normalize_url(match.group("url")))
        if number is None:
            return match.group(0)
        return f"[{number}]"

    rewritten = MARKDOWN_LINK.sub(replace, text)
    stats = {"rewritten": 0, "left_as_is": 0}
    for match in MARKDOWN_LINK.finditer(text):
        if number_by_norm.get(normalize_url(match.group("url"))) is not None:
            stats["rewritten"] += 1
        else:
            stats["left_as_is"] += 1
    return rewritten, stats


@dataclass
class CitationReport:
    """Verdict of one validation pass (never raised)."""

    cited_numbers: list[int] = field(default_factory=list)
    missing_numbers: list[int] = field(default_factory=list)
    unresolved_markers: list[int] = field(default_factory=list)
    uncited_sources: list[int] = field(default_factory=list)


def body_before_sources(text: str) -> str:
    """Everything before the `## Sources` heading (whole text if absent)."""
    index = text.find(SOURCES_HEADING)
    return text if index < 0 else text[:index]


def validate_citations(
    body: str, registry: dict[str, int]
) -> CitationReport:
    """Check the prose `[n]` markers against the registry."""
    total = len(registry)
    numbers = {int(m) for m in PROSE_MARKER.findall(body)}
    cited = sorted(n for n in numbers if 1 <= n <= total)
    highest = max(cited, default=0)
    return CitationReport(
        cited_numbers=cited,
        missing_numbers=[n for n in range(1, highest + 1) if n not in cited],
        unresolved_markers=sorted(n for n in numbers if n < 1 or n > total),
        uncited_sources=[n for n in range(1, total + 1) if n not in cited],
    )


class CitationRewriter:
    """Streaming rewriter: numbers citations in first-appearance order.

    `feed` emits completed text only; a tail that could still grow into
    a markdown link is held back (bounded, then flushed verbatim).
    `flush` ends the answer; the mapping — cited in appearance order,
    never-cited known URLs appended — is exposed by `registry` and
    `citations_payload`.
    """

    def __init__(self, known_urls: list[str]):
        # First original wins per normalized form; uncited entries are
        # appended to the mapping when it is finalized.
        self._original_by_norm: dict[str, str] = {}
        for url in known_urls:
            self._original_by_norm.setdefault(normalize_url(url), url)
        self._numbers: dict[str, int] = {}  # normalized -> n (cited only)
        self._buffer = ""
        self._finalized = False

    def _number_for(self, url: str) -> int | None:
        normalized = normalize_url(url)
        if normalized not in self._original_by_norm:
            return None
        if normalized not in self._numbers:
            self._numbers[normalized] = len(self._numbers) + 1
        return self._numbers[normalized]

    def feed(self, delta: str) -> str:
        self._buffer += delta
        out: list[str] = []
        while True:
            start = self._buffer.find("[")
            if start < 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start:]

            label_end = self._buffer.find("]")
            if label_end < 0:
                if len(self._buffer) > MAX_PENDING_TAIL:
                    out.append(self._buffer)  # stale, not a link
                    self._buffer = ""
                    break
                break  # wait for the label to close
            following = self._buffer[label_end + 1 : label_end + 2]
            if following == "":
                break  # "(" may still arrive in the next delta
            if following != "(":
                out.append(self._buffer[: label_end + 1])
                self._buffer = self._buffer[label_end + 1 :]
                continue
            url_end = self._buffer.find(")", label_end + 2)
            if url_end < 0:
                if len(self._buffer) > MAX_PENDING_LINK:
                    out.append(self._buffer)  # stale, not a link
                    self._buffer = ""
                    break
                break  # wait for the URL to close
            link = self._buffer[: url_end + 1]
            url = MARKDOWN_LINK.match(link).group("url")  # type: ignore[union-attr]
            number = self._number_for(url)
            out.append(f"[{number}]" if number is not None else link)
            self._buffer = self._buffer[url_end + 1 :]
        return "".join(out)

    def flush(self) -> str:
        tail, self._buffer = self._buffer, ""
        self._finalized = True
        return tail

    def registry(self) -> dict[str, int]:
        """Cited URLs in appearance order, then never-cited known URLs."""
        if not self._finalized:
            raise RuntimeError("Flush the rewriter before reading the registry.")
        final: dict[str, int] = {}
        for normalized, number in self._numbers.items():
            final[self._original_by_norm[normalized]] = number
        for normalized, original in self._original_by_norm.items():
            if normalized not in self._numbers:
                final[original] = len(final) + 1
        return final

    def citations_payload(self) -> list[dict]:
        return [
            {"n": number, "url": url}
            for url, number in self.registry().items()
        ]
