import json

from pydantic import BaseModel, ValidationError

from ....modules.llm_provider.providers.base import LLMProviderBase
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ..services.chunking import chunk_markdown_data


def make_query_for_search_engine(query: str):
    return f"""
    # Role
    You are a Web Search Query Optimizer API. You convert user questions into search keywords.

    # Task
    Analyze the User Query and generate a search_query.

    # Guidelines
    1. Noise Removal: Remove conversational filler.
    2. Keyword Optimization: Focus on entities and nouns.
    3. Language Preservation: Maintain original language.

    # Constraints (CRITICAL)
    - Output MUST be a valid JSON object.
    - Do NOT include markdown code blocks (no ```json).
    - Do NOT include explanations, introductions, or notes.
    - Start the response directly with {{ and end with }}.

    # Output Format
    {{"search_query": "string"}}

    # Input
    User Query: "{query}"

    # Response
    {{"search_query": "
    """


def make_answer_for_search(query: str, data):
    return f"""
    # Role
    You are an expert AI assistant for a search engine.

    # Task
    Answer the user's question based strictly on the provided context chunks.

    # Guidelines
    1. Analyze the user's question and the provided context.
    2. Formulate a precise and accurate answer based ONLY on the information found in the context.
    3. Do not use outside knowledge or hallucinate facts. If the context does not contain the answer, state that clearly.

    # Constraints (CRITICAL)
    - You MUST cite your sources. Whenever you use information from a specific chunk, append the source URL immediately after the claim or at the end of the paragraph using Markdown link format: [Source](URL).
    - If multiple sources support the same point, cite all of them.
    - Structure the answer clearly using bullet points if necessary.

    User query: # Role
    You are an expert AI assistant for a search engine.

    # Task
    Answer the user's question based strictly on the provided context chunks.

    # Guidelines
    1. Analyze the user's question and the provided context.
    2. Formulate a precise and accurate answer based ONLY on the information found in the context.
    3. Do not use outside knowledge or hallucinate facts. If the context does not contain the answer, state that clearly.

    # Constraints (CRITICAL)
    - You MUST cite your sources. Whenever you use information from a specific chunk, append the source URL immediately after the claim or at the end of the paragraph using Markdown link format: [Source](URL).
    - If multiple sources support the same point, cite all of them.
    - Structure the answer clearly using bullet points if necessary.

    User Query:"{query}"

    Context Chunks:{data}
    """


class SearchQuerySchema(BaseModel):
    search_query: str


async def sse_search_generator(
    query: str,
    searching: ISearchProvider,
    model: LLMProviderBase,
    scraper: IScraper,
    ranker: IRanker,
):
    try:
        # <----- LLM building query ----->
        # Step: 1
        # search_query = await model.generate(make_query_for_search_engine(query))
        # search_query = json.loads(search_query)
        yield f"event: status\ndata: {json.dumps({'step': 'Building query for search engine...'})}\n\n"

        raw_search_query = await model.generate(make_query_for_search_engine(query))
        print("\nraw_search_query:", raw_search_query, "\n")

        clean_json = raw_search_query.strip().strip("`").removeprefix("json\n")
        print("\nclean_json:", clean_json, "\n")

        try:
            parsed_query = SearchQuerySchema.model_validate_json(clean_json)
            print("\nparsed_query:", parsed_query, "\n")
            actual_search_query = parsed_query.search_query
            print("\nactual_search_query:", actual_search_query, "\n")
        except ValidationError:
            actual_search_query = query

        yield f"event: query_generated\ndata: {json.dumps({'query': actual_search_query})}\n\n"

        # <----- Searching ----->
        # Step: 2
        # sources = await searching.search(query=search_query.get("search_query"))
        yield f"event: status\ndata: {json.dumps({'step': 'Searching URLs...'})}\n\n"
        sources = await searching.search(query=actual_search_query)
        print("\nSource:", sources, "\n")

        if not sources:
            yield f"event: error\ndata: {json.dumps({'error': 'No search results found.'})}\n\n"
            return

        # <----- Ranking searching results ----->
        # Step: 3
        # sources = ranker.rank_web_search(query, sources)
        yield f"event: status\ndata: {json.dumps({'step': 'Ranking URLs...'})}\n\n"
        ranked_sources = ranker.rank_web_search(query, sources)
        print("\nranked_sources:", ranked_sources, "\n")

        # <----- Scraping ----->
        # Step: 4
        # content = await scraper.scrape(sources)
        yield f"event: status\ndata: {json.dumps({'step': 'Downloading data...'})}\n\n"
        content = await scraper.scrape(ranked_sources)
        # print("\ncontent:", content, "\n")

        # <----- Chunking content data ----->
        # Step: 5
        # if content:
        #    content = chunk_markdown_data(content)
        best_content = None
        if content:
            yield f"event: status\ndata: {json.dumps({'step': 'Chunking data...'})}\n\n"
            chunked_content = chunk_markdown_data(content)
        else:
            yield f"event: warning\ndata: {json.dumps({'warning': 'Empty source. No response can be give.'})}\n\n"
            return

        # <----- Ranking chunked data ----->
        # Step: 6
        # content = ranker.rank_chunks(query, content, 20)
        yield f"event: status\ndata: {json.dumps({'step': 'Choose the best data...'})}\n\n"
        best_content = ranker.rank_chunks(query, chunked_content, 20)
        print("\nbest_content:", best_content, "\n")

        # <----- LLM building answer  ----->
        # Step: 7
        answer = await model.generate(make_answer_for_search(query, content))
        yield f"event: message\ndata: {json.dumps({'text': answer})}\n\n\n"

        # !!! ------- To Do:  -------- !!!
        # - Add streaming to llm call
        # - Replace answer function
        #
        # async for chunk in model.generate(make_answer_for_search(query, content), stream=True):
        #     yield f"event: message\ndata: {json.dumps({'text': chunk})}\n\n"

        # Step: 8 For end
        yield f"event: done\ndata: {json.dumps({'status': 'Completed'})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'Critical error: {str(e)}'})}\n\n"
