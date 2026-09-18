"""add bank accounts and transactions tables

Revision ID: e3f90b8dd558
Revises: f92e4d31d491
Create Date: 2026-09-17 16:36:44.353899

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'e3f90b8dd558'
down_revision: Union[str, Sequence[str], None] = 'f92e4d31d491'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'bank_accounts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('account_type', sa.String(), nullable=False),
        sa.Column('balance', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('currency', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_bank_accounts_user_id'), 'bank_accounts', ['user_id'], unique=False,
    )
    op.create_table(
        'transactions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('account_id', sa.String(), nullable=False),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['bank_accounts.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_transactions_account_id'), 'transactions', ['account_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_transactions_account_id'), table_name='transactions')
    op.drop_table('transactions')
    op.drop_index(op.f('ix_bank_accounts_user_id'), table_name='bank_accounts')
    op.drop_table('bank_accounts')
