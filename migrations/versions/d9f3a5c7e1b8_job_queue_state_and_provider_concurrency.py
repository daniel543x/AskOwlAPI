"""job queue state + provider concurrency

Revision ID: d9f3a5c7e1b8
Revises: c8e1f2a4b6d3
Create Date: 2026-10-28 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'd9f3a5c7e1b8'
down_revision: Union[str, Sequence[str], None] = 'c8e1f2a4b6d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Resource gates + job queue (PLAN Phase 10, PR 12 / decision 19).

    `job` IS the queue state (status transitions persisted by the
    JobQueue); `modelprovider.max_concurrency` sizes the per-resource
    gate (NULL = type default: Ollama 1, cloud DEFAULT_PROVIDER_MAX_CONCURRENCY).
    """
    op.add_column(
        'modelprovider',
        sa.Column('max_concurrency', sa.Integer(), nullable=True),
    )
    op.create_table('job',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('kind', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('status', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('queue_position', sa.Integer(), nullable=True),
    sa.Column('conversation_id', sa.Uuid(), nullable=True),
    sa.Column('payload', sa.JSON(), nullable=True),
    sa.Column('result_summary', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('error', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('seen', sa.Boolean(), nullable=False),
    sa.Column('retries', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_user_id'), 'job', ['user_id'], unique=False)
    op.create_index(op.f('ix_job_kind'), 'job', ['kind'], unique=False)
    op.create_index(op.f('ix_job_status'), 'job', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_job_status'), table_name='job')
    op.drop_index(op.f('ix_job_kind'), table_name='job')
    op.drop_index(op.f('ix_job_user_id'), table_name='job')
    op.drop_table('job')
    op.drop_column('modelprovider', 'max_concurrency')
