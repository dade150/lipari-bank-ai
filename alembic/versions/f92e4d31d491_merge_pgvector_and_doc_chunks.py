"""merge pgvector and doc_chunks

Revision ID: f92e4d31d491
Revises: xxxx_add_pgvector, g5_doc_chunks
Create Date: 2026-09-14 14:35:43.480748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f92e4d31d491'
down_revision: Union[str, Sequence[str], None] = ('xxxx_add_pgvector', 'g5_doc_chunks')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
