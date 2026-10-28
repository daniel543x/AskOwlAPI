"""research_note table

Revision ID: a6c8e0f2b4d7
Revises: f4a8c2e6b9d1
Create Date: 2026-10-28 18:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'a6c8e0f2b4d7'
down_revision: Union[str, Sequence[str], None] = 'f4a8c2e6b9d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Research notes (PLAN Phase 11, PR 18 / decision 9).

    The DB is the source of truth: report assembly reads the rows, never
    the transient gather blob, so citations survive regeneration.
    """
    op.create_table('researchnote',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('conversation_id', sa.Uuid(), nullable=True),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('task', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('sources_used', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversation.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_researchnote_user_id'), 'researchnote', ['user_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_researchnote_conversation_id'), 'researchnote',
        ['conversation_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_researchnote_conversation_id'), table_name='researchnote')
    op.drop_index(op.f('ix_researchnote_user_id'), table_name='researchnote')
    op.drop_table('researchnote')
