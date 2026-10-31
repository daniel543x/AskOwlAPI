"""conversation/persona soft delete columns (WU-16)

Revision ID: e6b4f2a8c9d0
Revises: c2f9e4b8d1a6
Create Date: 2026-10-31 19:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6b4f2a8c9d0'
down_revision: Union[str, Sequence[str], None] = 'c2f9e4b8d1a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversation',
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'persona',
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('persona', 'deleted_at')
    op.drop_column('conversation', 'deleted_at')
