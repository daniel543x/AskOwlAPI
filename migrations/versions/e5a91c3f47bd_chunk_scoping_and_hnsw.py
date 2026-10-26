"""chunk scoping column + ANN index

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
    """Chunk-store hygiene (PLAN Phase 9, AR-8).

    - `conversation_id`: scope column for WU-5 retrieval scoping (NULL =
      legacy unscoped rows, which stay globally retrievable).
    - HNSW index on the embedding column (pgvector >= 0.5, cosine ops):
      keeps vector retrieval flat as the store grows past ~1k chunks.
    """
    op.add_column(
        'chunk', sa.Column('conversation_id', sa.Uuid(), nullable=True)
    )
    op.create_index(
        op.f('ix_chunk_conversation_id'), 'chunk', ['conversation_id'],
        unique=False,
    )
    op.create_index(
        'ix_chunk_embedding_hnsw',
        'chunk',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )


def downgrade() -> None:
    """Drop the ANN index and the scope column (rows keep existing)."""
    op.drop_index('ix_chunk_embedding_hnsw', table_name='chunk')
    op.drop_index(op.f('ix_chunk_conversation_id'), table_name='chunk')
    op.drop_column('chunk', 'conversation_id')
