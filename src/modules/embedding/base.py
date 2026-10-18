from abc import ABC, abstractmethod


class IEmbedder(ABC):
    """Turns texts into vectors. Implementations must be fully async."""

    @property
    @abstractmethod
    def dims(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order."""
