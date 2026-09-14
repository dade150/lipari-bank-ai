"""Add document_chunks table with pgvector

Revision ID: g5_doc_chunks
Revises: cc64b69443fd
Create Date: 2026-09-14
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "g5_doc_chunks"
down_revision = "cc64b69443fd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("document_id", sa.String(), index=True),
        sa.Column("chunk_index", sa.Integer()),
        sa.Column("content", sa.Text()),
        sa.Column("embedding", Vector(1536)),
        sa.Column("chunk_metadata", sa.JSON(), server_default="{}"),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
