"""chunk FTS index for the hybrid lexical leg

Revision ID: c3d84b06e9a2
Revises: e5a91c3f47bd
Create Date: 2026-10-26 23:40:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3d84b06e9a2'
down_revision: Union[str, Sequence[str], None] = 'e5a91c3f47bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Lexical-leg index (PLAN Phase 9, WU-4).

    GIN expression index over the same `to_tsvector('simple', content)`
    expression `ChunkRepository.fts_candidates` filters and orders by, so
    the lexical leg stays index-backed.
    """
    op.create_index(
        'ix_chunk_content_fts',
        'chunk',
        ["to_tsvector('simple', content)"],
        unique=False,
        postgresql_using='gin',
    )


def downgrade() -> None:
    """Drop the FTS index (the lexical leg degrades to seq scan)."""
    op.drop_index('ix_chunk_content_fts', table_name='chunk')
