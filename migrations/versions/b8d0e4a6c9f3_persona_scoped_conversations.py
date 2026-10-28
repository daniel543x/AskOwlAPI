"""persona-scoped conversations + image blob kind (PR 20)

Revision ID: b8d0e4a6c9f3
Revises: a6c8e0f2b4d7
Create Date: 2026-10-28 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8d0e4a6c9f3'
down_revision: Union[str, Sequence[str], None] = 'a6c8e0f2b4d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persona mode (PLAN Phase 11, PR 20 / decision 10)."""
    op.add_column(
        'conversation',
        sa.Column('persona_id', sa.Uuid(), nullable=True),
    )
    op.create_index(
        op.f('ix_conversation_persona_id'), 'conversation', ['persona_id'],
        unique=False,
    )
    op.create_foreign_key(
        'fk_conversation_persona_id', 'conversation', 'persona',
        ['persona_id'], ['id'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'fk_conversation_persona_id', 'conversation', type_='foreignkey'
    )
    op.drop_index(op.f('ix_conversation_persona_id'), table_name='conversation')
    op.drop_column('conversation', 'persona_id')
