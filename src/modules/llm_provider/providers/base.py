# llm_provider/base.py
from abc import ABC, abstractmethod
from typing import List, Optional


class LLMProviderBase(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def get_embeddings(self, text: str) -> List[float]:
        pass
