"""persona description and tags (webapp API-NOTES #1-2)

Revision ID: d4f8b2e6a9c1
Revises: c3d84b06e9a2
Create Date: 2026-10-31 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4f8b2e6a9c1'
down_revision: Union[str, Sequence[str], None] = 'c3d84b06e9a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'persona',
        sa.Column('description', sa.String(length=300), nullable=True),
    )
    op.add_column(
        'persona',
        sa.Column('tags', sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    op.drop_column('persona', 'tags')
    op.drop_column('persona', 'description')
