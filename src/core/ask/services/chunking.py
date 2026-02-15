from typing import Dict, List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

"""
Processes a list of scraped Markdown contents into smaller chunks suitable.

This function iterates through the provided data, splitting the Markdown content
first by headers to preserve semantic structure, and then by character count
if chunks exceed the specified size. It ensures that metadata, specifically the
source URL and header hierarchy, is preserved in each resulting chunk.

Args:
    data (List[Dict[str, str]]): A list of dictionaries where each dictionary
        contains a 'url' key and a 'content' key with Markdown text.

Returns:
    List[Dict[str, Any]]: A list of chunked dictionaries, each containing the
        'content' text, the 'url', and any extracted header metadata.
"""


def chunk_markdown_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = []

    for item in data:
        url = item.get("url", "")
        content = item.get("content", "")

        md_docs = markdown_splitter.split_text(content)

        docs = text_splitter.split_documents(md_docs)

        for doc in docs:
            chunk_metadata = doc.metadata
            chunk_metadata["url"] = url

            final_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": chunk_metadata,
                }
            )

    return final_chunks
