import json

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ..services.chunking import chunk_markdown_data


# <----- Pydantic Schemas ----->
class SearchQuerySchema(BaseModel):
    search_query: str


query_parser = JsonOutputParser(pydantic_object=SearchQuerySchema)


# <----- Prompts ----->
query_optimization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Web Search Query Optimizer API.
Your task is to convert the user's question into optimized search keywords.

Guidelines:
1. Noise Removal: Remove conversational filler.
2. Keyword Optimization: Focus on entities and nouns.
3. Language Preservation: Maintain original language.

{format_instructions}
""",
        ),
        ("human", "User Query: {query}"),
    ]
)

answer_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI assistant for a search engine.
Answer the user's question based strictly on the provided context chunks.

Guidelines:
1. Analyze the user's question and the provided context.
2. Formulate a precise answer based ONLY on the context.
3. You MUST cite sources using Markdown link format: [Source](URL) immediately after the claim.
4. If context doesn't contain the answer, state that clearly.

Context Chunks:
{data}""",
        ),
        ("human", "User Query: {query}"),
    ]
)


# <----- Meat ----->
async def sse_search_generator(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
):
    try:
        # <----- LLM building query ----->
        # Step: 1
        # search_query = await model.generate(make_query_for_search_engine(query))
        # search_query = json.loads(search_query)
        yield f"event: status\ndata: {json.dumps({'step': 'Building query for search engine...'})}\n\n"

        query_chain = query_optimization_prompt | model | query_parser

        try:
            parsed_output = await query_chain.ainvoke(
                {
                    "query": query,
                    "format_instructions": query_parser.get_format_instructions(),
                }
            )
            actual_search_query = parsed_output["search_query"]
        except ValidationError as e:
            print(f"Error: {e}")
            actual_search_query = query  # Fallback

        yield f"event: query_generated\ndata: {json.dumps({'query': actual_search_query})}\n\n"

        # <----- Searching ----->
        # Step: 2
        # sources = await searching.search(query=search_query.get("search_query"))
        yield f"event: status\ndata: {json.dumps({'step': 'Searching URLs...'})}\n\n"
        sources = await searching.search(query=actual_search_query)

        if not sources:
            yield f"event: error\ndata: {json.dumps({'error': 'No search results found.'})}\n\n"
            return

        # <----- Ranking searching results ----->
        # Step: 3
        # sources = ranker.rank_web_search(query, sources)
        yield f"event: status\ndata: {json.dumps({'step': 'Ranking URLs...'})}\n\n"
        ranked_sources = ranker.rank_web_search(query, sources)

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
        if not content:
            yield f"event: warning\ndata: {json.dumps({'warning': 'Empty source. No response can be give.'})}\n\n"
            return

        yield f"event: status\ndata: {json.dumps({'step': 'Chunking data...'})}\n\n"
        chunked_content = chunk_markdown_data(content)

        # <----- Ranking chunked data ----->
        #
        # content = ranker.rank_chunks(query, content, 20)
        yield f"event: status\ndata: {json.dumps({'step': 'Choose the best data...'})}\n\n"
        best_content = ranker.rank_chunks(query, chunked_content, 20)

        context_str = ""
        if best_content:
            if isinstance(best_content, list) and isinstance(best_content[0], dict):
                context_str = "\n\n---\n\n".join(
                    item.get("content", str(item)) for item in best_content
                )
            else:
                context_str = "\n\n---\n\n".join(str(item) for item in best_content)

        # <----- LLM building answer  ----->
        # Step: 7
        yield f"event: status\ndata: {json.dumps({'step': 'Generating answer...'})}\n\n"
        answer_chain = answer_generation_prompt | model
        async for chunk in answer_chain.astream({"query": query, "data": context_str}):
            if chunk.content:
                yield f"event: message\ndata: {json.dumps({'text': chunk.content})}\n\n"

        # Step: 8 For end
        yield f"event: done\ndata: {json.dumps({'status': 'Completed'})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'Critical error: {str(e)}'})}\n\n"
