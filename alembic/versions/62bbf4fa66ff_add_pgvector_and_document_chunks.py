"""add_pgvector_and_document_chunks

Revision ID: xxxx_add_pgvector
Revises: <revisione_precedente>
Create Date: 2024-X-X
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector # <- Import fondamentale

# revision identifiers, used by Alembic.
revision: str = 'xxxx_add_pgvector'
down_revision: Union[str, None] = 'cc64b69443fd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Abilita l'estensione su PostgreSQL
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Crea la tabella
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("document_id", sa.String, nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("chunk_metadata", sa.JSON, default={}),
    )

    # Crea l'indice HNSW per ricerche vettoriali ultra-veloci
    op.execute("CREATE INDEX ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)")

def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.execute("DROP EXTENSION IF EXISTS vector")