from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .model import LLMConfig, LLMProvider


async def init_ollama_db(
    session: AsyncSession,
    provider_name: str = "Local Ollama",
    base_url: str = "http://localhost:11434",
    default_model: str = "llama3.1:8b",
    config_name: str = "Ollama Default",
):
    provider_statement = select(LLMProvider).where(LLMProvider.name == provider_name)
    result = await session.execute(provider_statement)
    provider = result.scalar_one_or_none()

    if not provider:
        provider = LLMProvider(
            name=provider_name, provider_type="ollama", base_url=base_url
        )
        session.add(provider)
        await session.flush()

        print(f"Add Providera: {provider_name} ({base_url})")
    else:
        print(f"ℹ️  Provider '{provider_name}' exist. Skipping...")

    config_statement = select(LLMConfig).where(LLMConfig.config_name == config_name)
    c_result = await session.execute(config_statement)
    config = c_result.scalar_one_or_none()

    if not config:
        config = LLMConfig(
            config_name=config_name,
            provider_id=provider.id,
            model_name=default_model,
            is_default=True,
            is_active=True,
        )
        session.add(config)
        print(f"Created config: {config_name} (Model: {default_model})")
    else:
        print(f"ℹ️  Config '{config_name}' is exist. Skipping...")

    await session.commit()
