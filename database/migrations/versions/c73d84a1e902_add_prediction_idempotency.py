
"""Add idempotency keys and request fingerprints to fraud predictions.

Revision ID: c73d84a1e902
Revises: a91f7c2d5b60
Create Date: 2026-10-22

A unique index on the nullable idempotency_key column prevents two persisted
predictions from using the same non-null request key.

Existing records remain intact, with NULL in the new fields.
"""

from alembic import op
import sqlalchemy as sa


revision = "c73d84a1e902"
down_revision = "a91f7c2d5b60"
branch_labels = None
depends_on = None


TABLE_NAME = "fraud_predictions"
IDEMPOTENCY_INDEX = "uq_fraud_predictions_idempotency_key"


def upgrade() -> None:
    """Add idempotency metadata and enforce key uniqueness."""
    op.add_column(
        TABLE_NAME,
        sa.Column(
            "idempotency_key",
            sa.String(length=128),
            nullable=True,
        ),
    )

    op.add_column(
        TABLE_NAME,
        sa.Column(
            "request_fingerprint",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.create_index(
        IDEMPOTENCY_INDEX,
        TABLE_NAME,
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    """Remove idempotency metadata and its unique index."""
    op.drop_index(
        IDEMPOTENCY_INDEX,
        table_name=TABLE_NAME,
    )

    op.drop_column(
        TABLE_NAME,
        "request_fingerprint",
    )

    op.drop_column(
        TABLE_NAME,
        "idempotency_key",
    )
