from src.core.tools.chunking import chunk_markdown_data


def test_splits_markdown_by_headers_and_keeps_url():
    data = [
        {
            "url": "http://example.com/a",
            "content": (
                "# Title\nparagraph under title\n\n"
                "## Subsection\nparagraph under subsection"
            ),
        }
    ]

    chunks = chunk_markdown_data(data)

    assert len(chunks) == 2
    assert all(c["metadata"]["url"] == "http://example.com/a" for c in chunks)
    assert chunks[0]["metadata"].get("Header 1") == "Title"
    assert chunks[1]["metadata"].get("Header 2") == "Subsection"
    assert "paragraph under title" in chunks[0]["content"]
    assert "paragraph under subsection" in chunks[1]["content"]


def test_long_content_is_resplit_to_max_chunk_size():
    paragraph = "word " * 400  # ~2000 chars, above the 1000-char chunk size
    data = [{"url": "http://example.com/b", "content": paragraph}]

    chunks = chunk_markdown_data(data)

    assert len(chunks) > 1
    assert all(len(c["content"]) <= 1200 for c in chunks)
    assert all(c["metadata"]["url"] == "http://example.com/b" for c in chunks)


def test_multiple_pages_are_chunked_independently():
    data = [
        {"url": "http://example.com/1", "content": "# One\nfirst page body"},
        {"url": "http://example.com/2", "content": "# Two\nsecond page body"},
    ]

    chunks = chunk_markdown_data(data)

    urls = {c["metadata"]["url"] for c in chunks}
    assert urls == {"http://example.com/1", "http://example.com/2"}


def test_empty_inputs_produce_no_chunks():
    assert chunk_markdown_data([]) == []
    assert chunk_markdown_data([{"url": "http://example.com/x", "content": ""}]) == []
