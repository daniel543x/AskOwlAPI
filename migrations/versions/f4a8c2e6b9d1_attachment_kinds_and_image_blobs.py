"""attachment kinds + image blobs

Revision ID: f4a8c2e6b9d1
Revises: e2b6d8f0a4c2
Create Date: 2026-10-28 17:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'f4a8c2e6b9d1'
down_revision: Union[str, Sequence[str], None] = 'e2b6d8f0a4c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Image attachments (PLAN Phase 10, PR 16 / decision 14).

    `attachment.kind` separates chunk-backed text uploads from blob-backed
    images; images go to the synthesis LLM as content blocks and never to
    the search engine path.
    """
    op.add_column(
        'attachment',
        sa.Column(
            'kind',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default=sa.text("'text'"),
        ),
    )
    op.create_table('attachmentblob',
    sa.Column('attachment_id', sa.Uuid(), nullable=False),
    sa.Column('data', sa.LargeBinary(), nullable=False),
    sa.ForeignKeyConstraint(['attachment_id'], ['attachment.id'], ),
    sa.PrimaryKeyConstraint('attachment_id')
    )


def downgrade() -> None:
    op.drop_table('attachmentblob')
    op.drop_column('attachment', 'kind')
