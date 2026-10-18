"""rename pro role to research

Revision ID: 3bdb45a92f6d
Revises: 0131be5df7bb
Create Date: 2026-10-18 17:37:45.746218

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3bdb45a92f6d'
down_revision: Union[str, Sequence[str], None] = '0131be5df7bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename the seeded `pro` role to `research` (decision #4, 2026-10-18).

    Data-only migration: user assignments reference role.id, so renaming
    the row keeps every existing elevation intact.
    """
    op.execute(
        "UPDATE role SET name = 'research', "
        "description = 'Elevated role: research and agent modes.' "
        "WHERE name = 'pro'"
    )


def downgrade() -> None:
    """Restore the `pro` role name."""
    op.execute(
        "UPDATE role SET name = 'pro', "
        "description = 'Elevated role: research and agent modes.' "
        "WHERE name = 'research'"
    )
