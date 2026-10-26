"""add refresh_token table

Revision ID: c84e1f7a20bd
Revises: 3bdb45a92f6d
Create Date: 2026-10-26 10:12:44.310952

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'c84e1f7a20bd'
down_revision: Union[str, Sequence[str], None] = '3bdb45a92f6d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the refresh-token session table (PLAN Phase 8).

    Hash-only storage; one row per issued session, parent-linked on
    single-use rotation. Superseded rows are purged by the app's
    retention sweep after REFRESH_TOKEN_RETENTION_DAYS.
    """
    op.create_table('refreshtoken',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('token_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('parent_id', sa.Uuid(), nullable=True),
    sa.Column('client_label', sqlmodel.sql.sqltypes.AutoString(length=200), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('rotated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['parent_id'], ['refreshtoken.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_refreshtoken_token_hash'), 'refreshtoken', ['token_hash'], unique=True)
    op.create_index(op.f('ix_refreshtoken_user_id'), 'refreshtoken', ['user_id'], unique=False)


def downgrade() -> None:
    """Drop the refresh-token table (all sessions are forgotten)."""
    op.drop_index(op.f('ix_refreshtoken_user_id'), table_name='refreshtoken')
    op.drop_index(op.f('ix_refreshtoken_token_hash'), table_name='refreshtoken')
    op.drop_table('refreshtoken')
