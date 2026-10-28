import logging
import uuid
from datetime import UTC, datetime

from fastapi import Depends
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ...tools.cryptography import CryptoService, get_crypto_service
from ...tools.db import get_session
from ...tools.settings import settings
from .health import probe_provider
from .model import (
    DefaultSettings,
    ModelCatalogEntry,
    ModelEntryCreate,
    ModelEntryUpdate,
    ModelKind,
    ModelProvider,
    ProviderCreate,
    ProviderUpdate,
    UserModelPreference,
)
from .selection import (
    TIER_BUILTIN,
    CatalogView,
    ModelInfo,
    ResolvedModel,
    SelectionError,
    has_role_access,
    resolve_model_id,
)

logger = logging.getLogger(__name__)

DEFAULT_KEY_PREFIX = "default_model_id:"
FLASHRANK_PROVIDER_NAME = "flashrank-local"
OLLAMA_PROVIDER_NAME = "Local Ollama"


class ModelRepository:
    """Database access for the model registry (providers, entries, defaults)."""

    def __init__(self, session: AsyncSession, crypto: CryptoService | None = None):
        self.session = session
        self.crypto = crypto or get_crypto_service()

    # <----- Resolution ----->

    async def resolve_model(
        self,
        kind: str,
        user_id: uuid.UUID,
        user_roles: list[str],
        requested_model_id: uuid.UUID | None = None,
        profile_model_id: uuid.UUID | None = None,
    ) -> ResolvedModel:
        view = await self._catalog_view(user_id)

        try:
            model_id, tier = resolve_model_id(
                kind,
                requested_model_id,
                view,
                user_roles,
                profile_model_id=profile_model_id,
            )
        except SelectionError:
            builtin = self._builtin_model(kind)
            if builtin is None:
                raise
            return builtin

        return await self._resolved_from_entry(model_id, tier)

    async def _catalog_view(self, user_id: uuid.UUID) -> CatalogView:
        entries = (await self.session.exec(select(ModelCatalogEntry))).all()
        preferences = (
            await self.session.exec(
                select(UserModelPreference).where(
                    UserModelPreference.user_id == user_id
                )
            )
        ).all()
        defaults_rows = (
            await self.session.exec(
                select(DefaultSettings).where(
                    DefaultSettings.key.startswith(DEFAULT_KEY_PREFIX)  # type: ignore[attr-defined]
                )
            )
        ).all()

        view = CatalogView(
            models={
                entry.id: ModelInfo(
                    id=entry.id,
                    kind=entry.kind,
                    required_role=entry.required_role,
                    is_active=entry.is_active,
                )
                for entry in entries
            },
            user_preferences={
                pref.kind: pref.selected_model_id for pref in preferences
            },
            defaults={
                row.key.removeprefix(DEFAULT_KEY_PREFIX): uuid.UUID(row.value)
                for row in defaults_rows
            },
        )
        return view

    async def _resolved_from_entry(
        self, model_id: uuid.UUID, tier: str
    ) -> ResolvedModel:
        entry = await self.session.get(ModelCatalogEntry, model_id)
        if entry is None:
            raise SelectionError(f"Model {model_id} disappeared from the catalog.")
        provider = await self.session.get(ModelProvider, entry.provider_id)

        api_key = None
        if provider and provider.api_key_encrypted:
            try:
                api_key = self.crypto.decrypt(provider.api_key_encrypted)
            except Exception:
                logger.exception(
                    "Failed to decrypt API key of provider '%s'.", provider.name
                )

        return ResolvedModel(
            kind=entry.kind,
            model_name=entry.model_name,
            provider_type=provider.provider_type if provider else "unknown",
            base_url=provider.base_url if provider else None,
            api_key=api_key,
            default_params=dict(entry.default_params or {}),
            capabilities=list(entry.capabilities or []),
            effort_levels=list(entry.effort_levels or []),
            vector_dims=entry.vector_dims,
            entry_id=entry.id,
            provider_id=provider.id if provider else None,
            max_concurrency=self._provider_concurrency(provider),
            source=tier,
        )

    @staticmethod
    def _provider_concurrency(provider: ModelProvider | None) -> int:
        if provider is None:
            return 1
        if provider.max_concurrency:
            return provider.max_concurrency
        if provider.provider_type.lower() == "ollama":
            return 1
        return settings.DEFAULT_PROVIDER_MAX_CONCURRENCY

    def _builtin_model(self, kind: str) -> ResolvedModel | None:
        """Built-in local defaults, used when the catalog has nothing."""
        if kind == ModelKind.CHAT.value:
            return ResolvedModel(
                kind=kind,
                model_name=settings.DEFAULT_CHAT_MODEL_NAME,
                provider_type="ollama",
                base_url=settings.OLLAMA_BASE_URL,
                capabilities=["tools", "streaming"],
                source=TIER_BUILTIN,
            )
        if kind == ModelKind.RERANKER.value:
            return ResolvedModel(
                kind=kind,
                model_name=settings.RANKER_MODEL_NAME,
                provider_type="flashrank",
                default_params={"cache_dir": settings.RANKER_CACHE_DIR},
                source=TIER_BUILTIN,
            )
        if kind == ModelKind.EMBEDDING.value:
            return ResolvedModel(
                kind=kind,
                model_name=settings.DEFAULT_EMBEDDING_MODEL_NAME,
                provider_type="ollama",
                base_url=settings.OLLAMA_BASE_URL,
                vector_dims=settings.DEFAULT_EMBEDDING_MODEL_DIMS,
                source=TIER_BUILTIN,
            )
        return None

    # <----- Listing / preferences ----->

    async def get_available_models(
        self, user_roles: list[str], kind: str | None = None
    ) -> list[tuple[ModelCatalogEntry, ModelProvider]]:
        statement = select(ModelCatalogEntry, ModelProvider).where(
            ModelCatalogEntry.provider_id == ModelProvider.id,
            ModelCatalogEntry.is_active,  # type: ignore[arg-type]
            ModelProvider.is_active,  # type: ignore[arg-type]
        )
        if kind:
            statement = statement.where(ModelCatalogEntry.kind == kind)

        rows = (await self.session.exec(statement)).all()  # type: ignore[call-overload]
        return [
            (entry, provider)
            for entry, provider in rows
            if has_role_access(user_roles, entry.required_role)
        ]

    async def set_user_preference(
        self, user_id: uuid.UUID, model_id: uuid.UUID, user_roles: list[str]
    ) -> UserModelPreference:
        entry = await self.session.get(ModelCatalogEntry, model_id)
        if entry is None or not entry.is_active:
            raise SelectionError(f"Model {model_id} not found or not active.")
        if not has_role_access(user_roles, entry.required_role):
            raise SelectionError(
                f"Model {model_id} requires role '{entry.required_role}'.",
                tier="permission",
            )

        preference = await self.session.get(UserModelPreference, (user_id, entry.kind))
        if preference is None:
            preference = UserModelPreference(
                user_id=user_id, kind=entry.kind, selected_model_id=model_id
            )
        else:
            preference.selected_model_id = model_id

        self.session.add(preference)
        await self.session.commit()
        await self.session.refresh(preference)
        return preference

    async def get_user_preferences(
        self, user_id: uuid.UUID
    ) -> list[UserModelPreference]:
        return list(
            (
                await self.session.exec(
                    select(UserModelPreference).where(
                        UserModelPreference.user_id == user_id
                    )
                )
            ).all()
        )

    # <----- Admin: providers ----->

    async def create_provider(self, data: ProviderCreate) -> ModelProvider:
        existing = await self._provider_by_name(data.name)
        if existing is not None:
            raise ValueError(f"Provider '{data.name}' already exists.")

        provider = ModelProvider(
            name=data.name,
            provider_type=data.provider_type,
            base_url=data.base_url,
            api_key_encrypted=(
                self.crypto.encrypt(data.api_key) if data.api_key else None
            ),
            is_active=data.is_active,
        )
        self.session.add(provider)
        await self.session.commit()
        await self.session.refresh(provider)
        logger.info("Created model provider '%s' (%s).", data.name, data.provider_type)
        return provider

    async def list_providers(self) -> list[ModelProvider]:
        return list((await self.session.exec(select(ModelProvider))).all())

    async def update_provider(
        self, provider_id: uuid.UUID, data: ProviderUpdate
    ) -> ModelProvider:
        provider = await self._get_provider(provider_id)
        changes = data.model_dump(exclude_unset=True)

        if "name" in changes:
            clash = await self._provider_by_name(changes["name"])
            if clash is not None and clash.id != provider_id:
                raise ValueError(f"Provider '{changes['name']}' already exists.")
            provider.name = changes["name"]
        if "base_url" in changes:
            provider.base_url = changes["base_url"]
        if "api_key" in changes:
            provider.api_key_encrypted = (
                self.crypto.encrypt(changes["api_key"]) if changes["api_key"] else None
            )
        if "is_active" in changes:
            provider.is_active = changes["is_active"]

        self.session.add(provider)
        await self.session.commit()
        await self.session.refresh(provider)
        return provider

    async def delete_provider(self, provider_id: uuid.UUID) -> None:
        provider = await self._get_provider(provider_id)
        entries = (
            await self.session.exec(
                select(ModelCatalogEntry).where(
                    ModelCatalogEntry.provider_id == provider_id
                )
            )
        ).all()
        if entries:
            raise ValueError(
                f"Provider '{provider.name}' still has {len(entries)} catalog "
                "entries; delete or move them first."
            )
        await self.session.delete(provider)
        await self.session.commit()
        logger.info("Deleted model provider '%s'.", provider.name)

    async def _get_provider(self, provider_id: uuid.UUID) -> ModelProvider:
        provider = await self.session.get(ModelProvider, provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found.")
        return provider

    async def _provider_by_name(self, name: str) -> ModelProvider | None:
        result = await self.session.exec(
            select(ModelProvider).where(ModelProvider.name == name)
        )
        return result.first()

    # <----- Admin: catalog entries ----->

    async def create_model(self, data: ModelEntryCreate) -> ModelCatalogEntry:
        await self._get_provider(data.provider_id)

        if data.kind == ModelKind.EMBEDDING and not data.vector_dims:
            raise ValueError("Embedding entries must declare vector_dims.")

        entry = ModelCatalogEntry(
            provider_id=data.provider_id,
            kind=data.kind.value if isinstance(data.kind, ModelKind) else data.kind,
            model_name=data.model_name,
            display_name=data.display_name or data.model_name,
            default_params=dict(data.default_params or {}),
            is_active=data.is_active,
            required_role=data.required_role,
            capabilities=list(data.capabilities or []),
            effort_levels=list(data.effort_levels or []),
            vector_dims=data.vector_dims,
        )
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        logger.info(
            "Created catalog entry '%s' (kind=%s).", entry.model_name, entry.kind
        )
        return entry

    async def list_models(self, kind: str | None = None) -> list[ModelCatalogEntry]:
        statement = select(ModelCatalogEntry)
        if kind:
            statement = statement.where(ModelCatalogEntry.kind == kind)
        return list((await self.session.exec(statement)).all())

    async def update_model(
        self, model_id: uuid.UUID, data: ModelEntryUpdate
    ) -> ModelCatalogEntry:
        entry = await self._get_model(model_id)
        changes = data.model_dump(exclude_unset=True)

        for field_name in (
            "model_name",
            "display_name",
            "is_active",
            "required_role",
            "vector_dims",
        ):
            if field_name in changes:
                setattr(entry, field_name, changes[field_name])
        if "default_params" in changes and changes["default_params"] is not None:
            entry.default_params = dict(changes["default_params"])
        if "capabilities" in changes and changes["capabilities"] is not None:
            entry.capabilities = list(changes["capabilities"])

        if entry.kind == ModelKind.EMBEDDING.value and not entry.vector_dims:
            raise ValueError("Embedding entries must declare vector_dims.")

        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def delete_model(self, model_id: uuid.UUID) -> None:
        entry = await self._get_model(model_id)

        preferences = (
            await self.session.exec(
                select(UserModelPreference).where(
                    UserModelPreference.selected_model_id == model_id
                )
            )
        ).all()
        for preference in preferences:
            await self.session.delete(preference)

        defaults = (
            await self.session.exec(
                select(DefaultSettings).where(DefaultSettings.value == str(model_id))
            )
        ).all()
        for default_row in defaults:
            await self.session.delete(default_row)

        await self.session.delete(entry)
        await self.session.commit()
        logger.info("Deleted catalog entry '%s'.", entry.model_name)

    async def get_model(self, model_id: uuid.UUID) -> ModelCatalogEntry:
        """Fetch one catalog entry or raise ValueError (used by routes)."""
        return await self._get_model(model_id)

    async def _get_model(self, model_id: uuid.UUID) -> ModelCatalogEntry:
        entry = await self.session.get(ModelCatalogEntry, model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found.")
        return entry

    # <----- Admin: global defaults ----->

    async def set_default(self, kind: str, model_id: uuid.UUID) -> DefaultSettings:
        entry = await self._get_model(model_id)
        if entry.kind != kind:
            raise ValueError(
                f"Model {model_id} is of kind '{entry.kind}', not '{kind}'."
            )
        if not entry.is_active:
            raise ValueError("Default model must be active.")

        key = f"{DEFAULT_KEY_PREFIX}{kind}"
        default_row = await self.session.get(DefaultSettings, key)
        if default_row is None:
            default_row = DefaultSettings(key=key, value=str(model_id))
        else:
            default_row.value = str(model_id)

        self.session.add(default_row)
        await self.session.commit()
        await self.session.refresh(default_row)
        return default_row

    async def get_defaults(self) -> dict[str, uuid.UUID]:
        rows = (
            await self.session.exec(
                select(DefaultSettings).where(
                    DefaultSettings.key.startswith(DEFAULT_KEY_PREFIX)  # type: ignore[attr-defined]
                )
            )
        ).all()
        return {
            row.key.removeprefix(DEFAULT_KEY_PREFIX): uuid.UUID(row.value)
            for row in rows
        }

    # <----- Health ----->

    async def probe_providers(self) -> list[ModelProvider]:
        """Probe every provider with an endpoint and persist the results."""
        providers = await self.list_providers()
        now = datetime.now(UTC)

        for provider in providers:
            if not provider.is_active:
                continue
            status = await probe_provider(
                provider.provider_type,
                provider.base_url,
                settings.MODEL_PROBE_TIMEOUT,
            )
            provider.health_status = status
            provider.last_probe_at = now
            self.session.add(provider)
            logger.info(
                "Provider '%s' probe result: %s", provider.name, status
            )

        await self.session.commit()
        # The commit above expired the provider instances; refresh them
        # explicitly so callers (e.g. GET /health/models building response
        # models) never trigger a lazy refresh, which raises MissingGreenlet
        # in async context.
        for provider in providers:
            await self.session.refresh(provider)
        return providers

    # <----- Bootstrap ----->

    async def bootstrap(self) -> None:
        """Seed built-in providers/entries/defaults; probe everything."""
        await self._ensure_flashrank()
        await self._maybe_seed_ollama()
        await self.probe_providers()

    async def _ensure_flashrank(self) -> None:
        provider = await self._provider_by_name(FLASHRANK_PROVIDER_NAME)
        if provider is None:
            provider = ModelProvider(
                name=FLASHRANK_PROVIDER_NAME,
                provider_type="flashrank",
                base_url=None,
                api_key_encrypted=None,
                is_active=True,
                health_status="local",
            )
            self.session.add(provider)
            await self.session.flush()
            logger.info("Seeded in-process provider '%s'.", FLASHRANK_PROVIDER_NAME)

        entry = (
            await self.session.exec(
                select(ModelCatalogEntry).where(
                    ModelCatalogEntry.provider_id == provider.id,
                    ModelCatalogEntry.kind == ModelKind.RERANKER.value,
                    ModelCatalogEntry.model_name == settings.RANKER_MODEL_NAME,
                )
            )
        ).first()
        if entry is None:
            entry = ModelCatalogEntry(
                provider_id=provider.id,
                kind=ModelKind.RERANKER.value,
                model_name=settings.RANKER_MODEL_NAME,
                display_name=f"flashrank {settings.RANKER_MODEL_NAME}",
                default_params={"cache_dir": settings.RANKER_CACHE_DIR},
                is_active=True,
                required_role="user",
            )
            self.session.add(entry)
            await self.session.flush()

        await self._set_default_if_missing(ModelKind.RERANKER.value, entry.id)

    async def _maybe_seed_ollama(self) -> None:
        """Seed the built-in local Ollama provider only if it responds."""
        status = await probe_provider(
            "ollama", settings.OLLAMA_BASE_URL, settings.MODEL_PROBE_TIMEOUT
        )
        if status != "up":
            logger.warning(
                "Local Ollama at %s did not respond (%s); skipping seed of the "
                "default chat model.",
                settings.OLLAMA_BASE_URL,
                status,
            )
            return

        provider = await self._provider_by_name(OLLAMA_PROVIDER_NAME)
        if provider is None:
            provider = ModelProvider(
                name=OLLAMA_PROVIDER_NAME,
                provider_type="ollama",
                base_url=settings.OLLAMA_BASE_URL,
                api_key_encrypted=None,
                is_active=True,
            )
            self.session.add(provider)
            await self.session.flush()
            logger.info("Seeded provider '%s' (probed up).", OLLAMA_PROVIDER_NAME)

        entry = (
            await self.session.exec(
                select(ModelCatalogEntry).where(
                    ModelCatalogEntry.provider_id == provider.id,
                    ModelCatalogEntry.kind == ModelKind.CHAT.value,
                    ModelCatalogEntry.model_name == settings.DEFAULT_CHAT_MODEL_NAME,
                )
            )
        ).first()
        if entry is None:
            entry = ModelCatalogEntry(
                provider_id=provider.id,
                kind=ModelKind.CHAT.value,
                model_name=settings.DEFAULT_CHAT_MODEL_NAME,
                display_name=f"{settings.DEFAULT_CHAT_MODEL_NAME} (Local)",
                default_params={"temperature": 0.7},
                is_active=True,
                required_role="user",
                capabilities=["tools", "streaming"],
            )
            self.session.add(entry)
            await self.session.flush()

        await self._set_default_if_missing(ModelKind.CHAT.value, entry.id)
        await self._seed_embedding_entry(provider)
        await self.session.commit()

    async def _seed_embedding_entry(self, provider: ModelProvider) -> None:
        """Seed the default embedding model (PLAN decision #1: bge-m3)."""
        entry = (
            await self.session.exec(
                select(ModelCatalogEntry).where(
                    ModelCatalogEntry.provider_id == provider.id,
                    ModelCatalogEntry.kind == ModelKind.EMBEDDING.value,
                    ModelCatalogEntry.model_name
                    == settings.DEFAULT_EMBEDDING_MODEL_NAME,
                )
            )
        ).first()
        if entry is None:
            entry = ModelCatalogEntry(
                provider_id=provider.id,
                kind=ModelKind.EMBEDDING.value,
                model_name=settings.DEFAULT_EMBEDDING_MODEL_NAME,
                display_name=f"{settings.DEFAULT_EMBEDDING_MODEL_NAME} (Local)",
                is_active=True,
                required_role="user",
                vector_dims=settings.DEFAULT_EMBEDDING_MODEL_DIMS,
            )
            self.session.add(entry)
            await self.session.flush()

        await self._set_default_if_missing(ModelKind.EMBEDDING.value, entry.id)

    async def _set_default_if_missing(self, kind: str, model_id: uuid.UUID) -> None:
        key = f"{DEFAULT_KEY_PREFIX}{kind}"
        existing = await self.session.get(DefaultSettings, key)
        if existing is None:
            self.session.add(DefaultSettings(key=key, value=str(model_id)))
            # Flush, do not commit: a commit would expire the ORM instances the
            # callers keep using (e.g. `provider` in `_maybe_seed_ollama`) and
            # their lazy refresh would raise MissingGreenlet in async context.
            # `bootstrap` commits everything once seeding is done.
            await self.session.flush()


# Dependency
def get_repository(
    session: AsyncSession = Depends(get_session),
) -> ModelRepository:
    return ModelRepository(session)
