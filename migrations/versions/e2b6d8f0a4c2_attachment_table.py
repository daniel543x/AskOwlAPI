"""attachment table

Revision ID: e2b6d8f0a4c2
Revises: d9f3a5c7e1b8
Create Date: 2026-10-28 17:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'e2b6d8f0a4c2'
down_revision: Union[str, Sequence[str], None] = 'd9f3a5c7e1b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Uploaded text files (PLAN Phase 10, PR 15 / decision 14).

    Metadata-only row; the parsed text lives in conversation-scoped
    chunks (decision 21).
    """
    op.create_table('attachment',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('conversation_id', sa.Uuid(), nullable=False),
    sa.Column('filename', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('size_bytes', sa.Integer(), nullable=False),
    sa.Column('extracted_chars', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversation.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_attachment_user_id'), 'attachment', ['user_id'], unique=False
    )
    op.create_index(
        op.f('ix_attachment_conversation_id'), 'attachment',
        ['conversation_id'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_attachment_conversation_id'), table_name='attachment')
    op.drop_index(op.f('ix_attachment_user_id'), table_name='attachment')
    op.drop_table('attachment')
