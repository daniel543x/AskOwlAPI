from src.core.ask.services.tools import AskToolbox
from tests.core_ask.fakes import FakeRanker, FakeSearchProvider
from tests.core_ask.test_eval import FakeChunkRepo, FakeEmbedder


class ExplodingEmbedder:
    @property
    def dims(self) -> int:
        return 2

    async def embed_texts(self, texts):
        raise RuntimeError("embedding service down")


async def test_fetch_pages_stores_embeddings_when_available():
    from tests.core_ask.fakes import make_scraper

    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=FakeEmbedder(
            {"http://example.com/a": [1.0, 0.0], "http://example.com/b": [0.0, 1.0]}
        ),
        chunk_repo=FakeChunkRepo(),
    )

    await toolbox.web_search("rust")
    await toolbox.fetch_pages()

    assert toolbox.chunk_repo.stored, "chunks should be embedded and stored"


async def test_fetch_pages_degrades_when_embedding_fails():
    from tests.core_ask.fakes import make_scraper

    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=ExplodingEmbedder(),
        chunk_repo=FakeChunkRepo(),
    )

    await toolbox.web_search("rust")
    preview = await toolbox.fetch_pages()

    # Pipeline continues with the in-memory path.
    assert toolbox.chunks
    assert "first paragraph" in preview


async def test_get_context_uses_vector_retrieval_when_available():
    from tests.core_ask.fakes import make_scraper

    chunk_repo = FakeChunkRepo()
    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=FakeEmbedder(
            {"http://example.com/a": [1.0, 0.0], "http://example.com/b": [0.0, 1.0]}
        ),
        chunk_repo=chunk_repo,
    )

    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    context = await toolbox.get_context("http://example.com/a topic")

    assert toolbox.used_vector_retrieval is True
    assert "first paragraph" in context


async def test_get_context_falls_back_without_embedder():
    from tests.core_ask.fakes import make_scraper

    toolbox = AskToolbox(FakeSearchProvider(), make_scraper(), FakeRanker())

    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    context = await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is False
    assert "first paragraph" in context


async def test_get_context_falls_back_when_vector_retrieval_fails():
    from tests.core_ask.fakes import make_scraper

    class ExplodingRepo:
        async def store_chunks(self, chunks, embeddings):
            return 0

        async def vector_candidates(self, query_embedding, k):
            raise RuntimeError("pgvector unavailable")

    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=FakeEmbedder({"http://example.com/a": [1.0, 0.0]}),
        chunk_repo=ExplodingRepo(),
    )

    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    context = await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is False
    assert "first paragraph" in context
