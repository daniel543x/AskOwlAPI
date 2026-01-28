from jinja2 import BaseLoader, Environment

env = Environment(loader=BaseLoader())

# Mode 1: Search (RAG)
RAG_ANSWER_TEMPLATE = env.from_string("""
Based on the following context retrieved from the web, answer the user's question.
If the context doesn't contain the answer, say that you couldn't find the information in the provided sources.

Context:
---
{{ context }}
---

User's Question: {{ query }}

Answer:
""")

# Mode 2: Research - Step 1: Generating sub-question
RESEARCH_SUBQUERIES_TEMPLATE = env.from_string("""
You are a research assistant. Your task is to break down the user's main query into 3-5 specific sub-queries.
These sub-queries will be used to gather information from a search engine.

Main Query: {{ query }}

Provide the sub-queries as a JSON list of strings.
Example: ["query 1", "query 2", "query 3"]

Sub-queries:
""")

# Mode 2: Research - Step 2: Make full raport
RESEARCH_SYNTHESIS_TEMPLATE = env.from_string("""
You are a research analyst. Your task is to synthesize a comprehensive report based on the answers to several sub-queries.
Each sub-query answer is a separate piece of information. Combine them into a coherent, well-structured final report that addresses the original main query.

Main Query: {{ main_query }}

Sub-query Answers:
---
{% for answer in answers %}
Answer to sub-query {{ loop.index }}:
{{ answer }}
---
{% endfor %}

Final Report:
""")
