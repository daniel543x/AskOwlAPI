import json
from typing import Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from .base import LLMProviderBase


class OllamaProvider(LLMProviderBase):
    def __init__(self, model_name: str, extra_params: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        params = {}
        if extra_params:
            try:
                params = json.loads(extra_params)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in extra_params for Ollama provider.")

        # host.containers.internal:11434
        # host.docker.internal:11434
        # ollama:11434
        base_url = params.get("base_url", "http://localhost:11434")

        self.llm = Ollama(base_url=base_url, model=self.model_name)
        self.embeddings = OllamaEmbeddings(base_url=base_url, model=self.model_name)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += prompt

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm.invoke, full_prompt)
