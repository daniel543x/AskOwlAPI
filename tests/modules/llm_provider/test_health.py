import httpx

from src.modules.llm_provider.health import probe_provider


async def test_probe_ollama_uses_tags_endpoint():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"models": []})

    status = await probe_provider(
        "ollama", "http://ollama:11434", 3.0, transport=httpx.MockTransport(handler)
    )

    assert status == "up"
    assert seen == ["http://ollama:11434/api/tags"]


async def test_probe_trailing_slash_normalized():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    await probe_provider(
        "ollama", "http://ollama:11434/", 3.0, transport=httpx.MockTransport(handler)
    )
    assert seen == ["http://ollama:11434/api/tags"]


async def test_probe_generic_provider_pings_base_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(404)  # any HTTP answer counts as reachable

    status = await probe_provider(
        "openai", "https://api.example.com", 3.0, transport=httpx.MockTransport(handler)
    )

    assert status == "up"
    assert seen == ["https://api.example.com"]


async def test_probe_connection_error_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    status = await probe_provider(
        "ollama", "http://nowhere:11434", 0.1, transport=httpx.MockTransport(handler)
    )
    assert status == "down"


async def test_probe_server_error_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    status = await probe_provider(
        "ollama", "http://sick:11434", 0.1, transport=httpx.MockTransport(handler)
    )
    assert status == "down"


async def test_no_base_url_returns_local():
    assert await probe_provider("flashrank", None, 3.0) == "local"
