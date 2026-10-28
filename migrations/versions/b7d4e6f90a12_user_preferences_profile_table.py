"""user preferences profile table

Revision ID: b7d4e6f90a12
Revises: a3f9c2d81e64
Create Date: 2026-10-28 16:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'b7d4e6f90a12'
down_revision: Union[str, Sequence[str], None] = 'a3f9c2d81e64'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Per-user preference defaults (PLAN Phase 10, PR 10 / decision 12).

    One row per user; the whole row is the profile. Stored values are
    clamped against the catalog on save; asks resolve request > profile >
    model default.
    """
    op.create_table('userpreferences',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('default_model_id', sa.Uuid(), nullable=True),
    sa.Column('default_mode', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('thinking', sa.Boolean(), nullable=True),
    sa.Column('effort', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('source_types', sa.JSON(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['default_model_id'], ['modelcatalogentry.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('user_id')
    )


def downgrade() -> None:
    op.drop_table('userpreferences')
