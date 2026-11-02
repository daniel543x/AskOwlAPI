"""persona allowed_source_types (backlog: persona-level source types)

Revision ID: a9d3e7c1b5f8
Revises: f1c5e9a3d7b2
Create Date: 2026-10-31 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a9d3e7c1b5f8'
down_revision: Union[str, Sequence[str], None] = 'f1c5e9a3d7b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'persona',
        sa.Column(
            'allowed_source_types',
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column('persona', 'allowed_source_types')
