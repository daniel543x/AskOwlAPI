import logging
import uuid

from fastapi import Depends, HTTPException, Query, status

from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from ..llm_provider.model import ModelKind
from ..llm_provider.repository import ModelRepository, get_repository
from ..llm_provider.selection import ResolvedModel, SelectionError
from .base import IEmbedder
from .providers.ollama import OllamaEmbedder
from .providers.openai_compatible import OpenAICompatibleEmbedder

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Builds embedders from registry-resolved configuration."""

    REGISTRY: dict[str, type] = {
        "ollama": OllamaEmbedder,
        "openai": OpenAICompatibleEmbedder,
    }

    def create_embedder(self, resolved: ResolvedModel) -> IEmbedder:
        embedder_cls = self.REGISTRY.get(resolved.provider_type)
        if embedder_cls is None:
            raise ValueError(
                f"No embedder implementation for provider type "
                f"'{resolved.provider_type}' (known: {sorted(self.REGISTRY)})."
            )

        if resolved.provider_type == "ollama":
            if not resolved.base_url:
                raise ValueError(
                    "Ollama embedding entries need a provider base_url."
                )
            if not resolved.vector_dims:
                raise ValueError(
                    f"Embedding entry '{resolved.model_name}' has no vector_dims "
                    "set; entries must declare it."
                )
            return OllamaEmbedder(
                base_url=resolved.base_url,
                model_name=resolved.model_name,
                dims=resolved.vector_dims,
                batch_size=int(resolved.default_params.get("batch_size", 16)),
                timeout=float(resolved.default_params.get("timeout", 60.0)),
                api_key=resolved.api_key,
            )

        if resolved.provider_type == "openai":
            if not resolved.base_url:
                raise ValueError(
                    "OpenAI-compatible embedding entries need a provider "
                    "base_url."
                )
            if not resolved.api_key:
                raise ValueError(
                    "OpenAI-compatible embedding entries need a provider API key."
                )
            if not resolved.vector_dims:
                raise ValueError(
                    f"Embedding entry '{resolved.model_name}' has no vector_dims "
                    "set; entries must declare it."
                )
            return OpenAICompatibleEmbedder(
                base_url=resolved.base_url,
                model_name=resolved.model_name,
                dims=resolved.vector_dims,
                api_key=resolved.api_key,
                batch_size=int(resolved.default_params.get("batch_size", 16)),
                timeout=float(resolved.default_params.get("timeout", 60.0)),
                max_concurrency=int(
                    resolved.default_params.get("max_concurrency", 2)
                ),
                max_retries=int(resolved.default_params.get("max_retries", 4)),
            )

        raise ValueError(
            f"Provider type '{resolved.provider_type}' is registered but has no "
            "embedder construction path yet."
        )


# Dependency
def get_embedding_factory() -> EmbeddingFactory:
    return EmbeddingFactory()


# Dependency: resolves the embedding model for the requesting user.
async def get_embedder(
    repo: ModelRepository = Depends(get_repository),
    factory: EmbeddingFactory = Depends(get_embedding_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None, alias="embedder_id"),
) -> IEmbedder:
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.EMBEDDING.value,
            user_id=user.id,
            user_roles=[role.name for role in user.roles],
            requested_model_id=request_model_id,
        )
        return factory.create_embedder(resolved)
    except SelectionError as e:
        code = (
            status.HTTP_403_FORBIDDEN
            if e.tier == "permission"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        raise HTTPException(status_code=code, detail=e.reason) from e


# Dependency: optional embedder for hybrid retrieval. Degrades to None when
# no embedding model is configured instead of failing the request.
async def get_optional_embedder(
    repo: ModelRepository = Depends(get_repository),
    factory: EmbeddingFactory = Depends(get_embedding_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None, alias="embedder_id"),
) -> IEmbedder | None:
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.EMBEDDING.value,
            user_id=user.id,
            user_roles=[role.name for role in user.roles],
            requested_model_id=request_model_id,
        )
        return factory.create_embedder(resolved)
    except (SelectionError, ValueError) as e:
        logger.info(
            "Hybrid retrieval disabled for this request (no usable "
            "embedder): %s",
            e,
        )
        return None
