"""model catalog effort_levels column

Revision ID: a3f9c2d81e64
Revises: e5a91c3f47bd
Create Date: 2026-10-28 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f9c2d81e64'
down_revision: Union[str, Sequence[str], None] = 'e5a91c3f47bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Admin-curated effort levels on catalog entries (PLAN Phase 10, PR 9).

    Non-empty = admin override; empty = probe result or the provider seed
    table answers (precedence: admin override > probe > seed).
    """
    op.add_column(
        'modelcatalogentry',
        sa.Column(
            'effort_levels',
            sa.JSON(),
            nullable=True,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column('modelcatalogentry', 'effort_levels')
