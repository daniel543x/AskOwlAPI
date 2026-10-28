"""add scraped_page cache table

Revision ID: b7e4d29c81af
Revises: c84e1f7a20bd
Create Date: 2026-10-26 22:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'b7e4d29c81af'
down_revision: Union[str, Sequence[str], None] = 'c84e1f7a20bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the per-URL scraped-content cache (PLAN Phase 9, WU-10).

    Keyed by the sha256 of the normalized URL so tracking-param variants
    share one entry; TTL is enforced at read time (PAGE_CACHE_TTL_S).
    """
    op.create_table('scraped_page',
    sa.Column('url_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('url', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('http_status', sa.Integer(), nullable=True),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('url_hash')
    )
    op.create_index(
        op.f('ix_scraped_page_fetched_at'), 'scraped_page', ['fetched_at'],
        unique=False,
    )


def downgrade() -> None:
    """Drop the scraped-page cache (entries are recomputable by refetching)."""
    op.drop_index(op.f('ix_scraped_page_fetched_at'), table_name='scraped_page')
    op.drop_table('scraped_page')
