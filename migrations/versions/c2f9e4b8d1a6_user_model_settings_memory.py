"""usermodelsettings: per-model thinking/effort memory (WU-15)

Revision ID: c2f9e4b8d1a6
Revises: a9d3e7c1b5f8
Create Date: 2026-10-31 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2f9e4b8d1a6'
down_revision: Union[str, Sequence[str], None] = 'a9d3e7c1b5f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'usermodelsettings',
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('model_id', sa.Uuid(), nullable=False),
        sa.Column('thinking', sa.Boolean(), nullable=True),
        sa.Column('effort', sa.String(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id']),
        sa.ForeignKeyConstraint(['model_id'], ['modelcatalogentry.id']),
        sa.PrimaryKeyConstraint('user_id', 'model_id'),
    )


def downgrade() -> None:
    op.drop_table('usermodelsettings')
