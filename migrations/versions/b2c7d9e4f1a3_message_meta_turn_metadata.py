"""message.meta turn metadata column

Revision ID: b2c7d9e4f1a3
Revises: e6b4f2a8c9d0
Create Date: 2026-11-01 19:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c7d9e4f1a3'
down_revision: Union[str, Sequence[str], None] = 'e6b4f2a8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Turn metadata on stored user messages (webapp API-NOTES #11/#12).

    The user message of every persisted turn carries a small meta object
    ({mode, mode_label, model, model_label, model_provider}) so the UI can
    restore its "mode · model" caption after a reload. Nullable: legacy
    rows read back as NULL and assistant messages are always NULL.
    """
    op.add_column('message', sa.Column('meta', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('message', 'meta')
