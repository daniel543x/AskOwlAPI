"""persona model (webapp API-NOTES #5)

Revision ID: f1c5e9a3d7b2
Revises: d4f8b2e6a9c1
Create Date: 2026-10-31 09:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1c5e9a3d7b2'
down_revision: Union[str, Sequence[str], None] = 'd4f8b2e6a9c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'persona',
        sa.Column('model', sa.Uuid(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('persona', 'model')
