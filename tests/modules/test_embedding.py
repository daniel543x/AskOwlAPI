import asyncio

import httpx
import pytest

from src.modules.embedding.factory import EmbeddingFactory
from src.modules.embedding.providers.ollama import OllamaEmbedder
from src.modules.embedding.providers.openai_compatible import OpenAICompatibleEmbedder
from src.modules.llm_provider.selection import ResolvedModel


def _install_mock(monkeypatch, handler):
    def fake_build_client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OllamaEmbedder, "_build_client", fake_build_client)


async def test_embed_texts_batches_and_preserves_order(monkeypatch):
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        import json

        body = json.loads(payload)
        requests_seen.append([t[:4] for t in body["input"]])
        # Echo embeddings identifiable by text prefix.
        embeddings = [[float(len(t)), 1.0] for t in body["input"]]
        return httpx.Response(200, json={"embeddings": embeddings})

    _install_mock(monkeypatch, handler)
    embedder = OllamaEmbedder(
        base_url="http://ollama:11434/", model_name="bge-m3", dims=2, batch_size=10
    )

    texts = [f"text-{i}" for i in range(25)]
    result = await embedder.embed_texts(texts)

    assert len(requests_seen) == 3  # 10 + 10 + 5
    assert [len(batch) for batch in requests_seen] == [10, 10, 5]
    assert len(result) == 25
    assert result[0] == [6.0, 1.0]  # "text-0".__len__() == 6
    assert result[24] == [7.0, 1.0]  # "text-24" -> 7 chars


async def test_embed_sends_model_and_input(monkeypatch):
    import json

    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.read()))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    _install_mock(monkeypatch, handler)
    embedder = OllamaEmbedder(
        base_url="http://ollama:11434", model_name="bge-m3", dims=2
    )

    await embedder.embed_texts(["hello"])

    assert seen_payloads == [{"model": "bge-m3", "input": ["hello"]}]


async def test_embed_mismatched_count_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1]]})  # too few

    _install_mock(monkeypatch, handler)
    embedder = OllamaEmbedder(
        base_url="http://ollama:11434", model_name="bge-m3", dims=2
    )

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        await embedder.embed_texts(["a", "b"])


async def test_embed_empty_input_no_requests(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    _install_mock(monkeypatch, handler)
    embedder = OllamaEmbedder(
        base_url="http://ollama:11434", model_name="bge-m3", dims=2
    )

    assert await embedder.embed_texts([]) == []


def test_dims_property():
    embedder = OllamaEmbedder(
        base_url="http://ollama:11434", model_name="bge-m3", dims=1024
    )
    assert embedder.dims == 1024


def test_factory_builds_ollama_embedder_from_resolved():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="bge-m3",
        provider_type="ollama",
        base_url="http://ollama:11434",
        vector_dims=1024,
        default_params={"batch_size": 8, "timeout": 12.0},
    )

    embedder = factory.create_embedder(resolved)

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "bge-m3"
    assert embedder.dims == 1024
    assert embedder.batch_size == 8
    assert embedder.timeout == 12.0


def test_factory_rejects_unknown_provider_type():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="x",
        provider_type="mystery",
        base_url="http://x",
        vector_dims=768,
    )

    with pytest.raises(ValueError, match="No embedder implementation"):
        factory.create_embedder(resolved)


def test_factory_requires_vector_dims():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="bge-m3",
        provider_type="ollama",
        base_url="http://ollama:11434",
    )

    with pytest.raises(ValueError, match="vector_dims"):
        factory.create_embedder(resolved)


def test_factory_requires_base_url_for_ollama():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="bge-m3",
        provider_type="ollama",
        vector_dims=1024,
    )

    with pytest.raises(ValueError, match="base_url"):
        factory.create_embedder(resolved)


# <----- OpenAI-compatible (OpenRouter, OpenAI, vLLM, ...) ----->


def _install_openai_mock(monkeypatch, handler):
    def fake_build_client(self):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    monkeypatch.setattr(OpenAICompatibleEmbedder, "_build_client", fake_build_client)


def _openai_response(embeddings):
    data = [
        {"object": "embedding", "index": i, "embedding": vector}
        for i, vector in enumerate(embeddings)
    ]
    return httpx.Response(200, json={"object": "list", "data": data})


async def test_openai_embed_batches_and_preserves_order(monkeypatch):
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read())
        requests_seen.append(payload["input"])
        embeddings = [[float(len(t)), 1.0] for t in payload["input"]]
        return _openai_response(embeddings)

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1",
        model_name="liquid/lfm-2.5-embedding-350m:free",
        dims=2,
        api_key="sk-test",
        batch_size=10,
    )

    texts = [f"text-{i}" for i in range(25)]
    result = await embedder.embed_texts(texts)

    assert [len(batch) for batch in requests_seen] == [10, 10, 5]
    assert len(result) == 25
    assert result[0] == [6.0, 1.0]  # "text-0".__len__() == 6
    assert result[24] == [7.0, 1.0]  # "text-24" -> 7 chars


async def test_openai_embed_normalizes_shuffled_indexes(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        # Server answers with indexes out of order (allowed by the spec).
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"index": 1, "embedding": [0.2, 0.2]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                ],
            },
        )

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1", model_name="m", dims=2, api_key="k"
    )

    result = await embedder.embed_texts(["first", "second"])

    assert result == [[0.1, 0.1], [0.2, 0.2]]


async def test_openai_embed_sends_model_input_and_auth_header(monkeypatch):
    import json

    seen_payloads = []
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.read()))
        seen_headers.append(dict(request.headers))
        return _openai_response([[0.1, 0.2]])

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1", model_name="m", dims=2, api_key="k"
    )

    await embedder.embed_texts(["hello"])

    assert seen_payloads == [{"model": "m", "input": ["hello"]}]
    assert seen_headers[0]["authorization"] == "Bearer k"


async def test_openai_embed_mismatched_count_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _openai_response([[0.1]])  # too few

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1", model_name="m", dims=2, api_key="k"
    )

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        await embedder.embed_texts(["a", "b"])


async def test_openai_embed_wrong_dims_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _openai_response([[0.1, 0.2, 0.3]])  # 3 dims, declared 2

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1", model_name="m", dims=2, api_key="k"
    )

    with pytest.raises(RuntimeError, match="3-dimensional"):
        await embedder.embed_texts(["a"])


async def test_openai_embed_empty_input_no_requests(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1", model_name="m", dims=2, api_key="k"
    )

    assert await embedder.embed_texts([]) == []


def test_factory_builds_openai_compatible_embedder_from_resolved():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="liquid/lfm-2.5-embedding-350m:free",
        provider_type="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        vector_dims=1024,
        default_params={
            "batch_size": 8,
            "timeout": 12.0,
            "max_concurrency": 3,
            "max_retries": 5,
        },
    )

    embedder = factory.create_embedder(resolved)

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_name == "liquid/lfm-2.5-embedding-350m:free"
    assert embedder.dims == 1024
    assert embedder.batch_size == 8
    assert embedder.timeout == 12.0
    assert embedder.max_concurrency == 3
    assert embedder.max_retries == 5


def test_factory_requires_api_key_for_openai():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="m",
        provider_type="openai",
        base_url="https://openrouter.ai/api/v1",
        vector_dims=1024,
    )

    with pytest.raises(ValueError, match="API key"):
        factory.create_embedder(resolved)


def test_factory_requires_base_url_for_openai():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="m",
        provider_type="openai",
        api_key="sk-test",
        vector_dims=1024,
    )

    with pytest.raises(ValueError, match="base_url"):
        factory.create_embedder(resolved)


def test_factory_requires_vector_dims_for_openai():
    factory = EmbeddingFactory()
    resolved = ResolvedModel(
        kind="embedding",
        model_name="m",
        provider_type="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
    )

    with pytest.raises(ValueError, match="vector_dims"):
        factory.create_embedder(resolved)


async def test_openai_embed_retries_429_then_succeeds(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            # Retry-After: 0 keeps the test fast while exercising the header.
            return httpx.Response(
                429, headers={"Retry-After": "0"}, json={"error": "rate limited"}
            )
        return _openai_response([[0.1, 0.2]])

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1",
        model_name="m",
        dims=2,
        api_key="k",
        retry_base_delay=0.0,
    )

    result = await embedder.embed_texts(["hello"])

    assert result == [[0.1, 0.2]]
    assert len(calls) == 3


async def test_openai_embed_raises_after_exhausted_retries(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={"error": "rate limited"})

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1",
        model_name="m",
        dims=2,
        api_key="k",
        max_retries=2,
        retry_base_delay=0.0,
    )

    with pytest.raises(httpx.HTTPStatusError, match="429"):
        await embedder.embed_texts(["hello"])

    assert len(calls) == 3  # initial attempt + 2 retries


async def test_openai_embed_respects_max_concurrency(monkeypatch):
    import json

    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        # Overlap requests so the semaphore is actually contended.
        await asyncio.sleep(0.01)
        active -= 1
        payload = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"index": i, "embedding": [0.1, 0.2]}
                    for i in range(len(payload["input"]))
                ],
            },
        )

    def fake_build_client(self):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    monkeypatch.setattr(
        OpenAICompatibleEmbedder, "_build_client", fake_build_client
    )
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1",
        model_name="m",
        dims=2,
        api_key="k",
        batch_size=1,
        max_concurrency=2,
    )

    result = await embedder.embed_texts([f"t{i}" for i in range(10)])

    assert len(result) == 10
    assert result == [[0.1, 0.2]] * 10
    assert peak > 1  # requests did overlap...
    assert peak <= 2  # ...but never beyond the configured concurrency


async def test_openai_embed_retries_transient_503(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _openai_response([[0.3, 0.4]])

    _install_openai_mock(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://openrouter.ai/api/v1",
        model_name="m",
        dims=2,
        api_key="k",
        retry_base_delay=0.0,
    )

    result = await embedder.embed_texts(["hello"])

    assert result == [[0.3, 0.4]]
    assert len(calls) == 2
