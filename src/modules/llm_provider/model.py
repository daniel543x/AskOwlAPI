import uuid
from typing import Any, Dict, List, Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class LLMProvider(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, description="Np. 'OpenRouter Main', 'Local Ollama'")
    provider_type: str = Field(description="Np. 'openai', 'ollama', 'groq'")
    base_url: Optional[str] = Field(default=None)
    api_key_encrypted: Optional[str] = Field(default=None)

    models: List["LLMModelCatalog"] = Relationship(back_populates="provider")


class LLMModelCatalog(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    provider_id: uuid.UUID = Field(foreign_key="llmprovider.id")

    provider: Optional[LLMProvider] = Relationship(back_populates="models")

    model_name: str = Field()
    display_name: str = Field()

    default_params: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    is_active: bool = Field(default=False)

    # For future control access to LLMs by users.
    required_role: str = Field(default="user")


class UserLLMPreference(SQLModel, table=True):
    user_id: uuid.UUID = Field(primary_key=True)
    selected_model_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="llmmodelcatalog.id"
    )


# Table for default model for all function.
# Value -> model ID from LLMModelCatalog
class DefaultSettings(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str = Field()
