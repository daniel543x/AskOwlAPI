"""chunk scoping column

Revision ID: e5a91c3f47bd
Revises: b7e4d29c81af
Create Date: 2026-10-26 23:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5a91c3f47bd'
down_revision: Union[str, Sequence[str], None] = 'b7e4d29c81af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Chunk-store scoping (PLAN Phase 9, AR-8 task 1).

    `conversation_id`: scope column for WU-5 retrieval scoping (NULL =
    legacy unscoped rows, which stay globally retrievable).

    Note: the AR-8 HNSW ANN index is deliberately NOT created here —
    pgvector requires fixed column dimensions for HNSW/IVFFlat, and
    `chunk.embedding` is intentionally untyped (multiple embedders with
    different dimensions coexist, PLAN decision #1). Revisit together
    with a dims-per-column decision when the store actually grows.
    """
    op.add_column(
        'chunk', sa.Column('conversation_id', sa.Uuid(), nullable=True)
    )
    op.create_index(
        op.f('ix_chunk_conversation_id'), 'chunk', ['conversation_id'],
        unique=False,
    )


def downgrade() -> None:
    """Drop the scope column (rows keep existing)."""
    op.drop_index(op.f('ix_chunk_conversation_id'), table_name='chunk')
    op.drop_column('chunk', 'conversation_id')
