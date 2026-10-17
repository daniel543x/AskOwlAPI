import httpx
import pytest

from src.modules.embedding.factory import EmbeddingFactory
from src.modules.embedding.providers.ollama import OllamaEmbedder
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
