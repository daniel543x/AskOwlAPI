"""search provider declared source types

Revision ID: c8e1f2a4b6d3
Revises: b7d4e6f90a12
Create Date: 2026-10-28 15:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8e1f2a4b6d3'
down_revision: Union[str, Sequence[str], None] = 'b7d4e6f90a12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Provider-declared source types (PLAN Phase 10, PR 11 / decision 13).

    Existing configs keep working: NULL/empty falls back to the full v1
    set in the app layer, and the server default below matches it.
    """
    op.add_column(
        'searchproviderconfig',
        sa.Column(
            'supported_source_types',
            sa.JSON(),
            nullable=True,
            server_default=sa.text('\'["web", "news", "academic", "social"]\''),
        ),
    )


def downgrade() -> None:
    op.drop_column('searchproviderconfig', 'supported_source_types')
