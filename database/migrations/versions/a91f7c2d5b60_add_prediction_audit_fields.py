"""Add prediction audit fields and preserve prediction history.

Revision ID: a91f7c2d5b60
Revises: 8d93170f6a48
Create Date: 2026-10-19

Changes:
    - Add calibration_method, decision_reason, raw_model_margin, and scored_at.
    - Replace the unique transaction_id index or constraint with a
      non-unique index so multiple predictions can be retained per transaction.
    - Leave historical audit fields nullable when their original values
      cannot be reconstructed.

Existing prediction rows and their original created_at values are preserved.
"""

from alembic import op
import sqlalchemy as sa


revision = "a91f7c2d5b60"
down_revision = "8d93170f6a48"
branch_labels = None
depends_on = None


TABLE_NAME = "fraud_predictions"
TRANSACTION_INDEX_NAME = "ix_fraud_predictions_transaction_id"
SCORED_AT_INDEX_NAME = "ix_fraud_predictions_scored_at"


def _replace_unique_transaction_index() -> None:
    """Replace uniqueness enforcement on transaction_id with a normal index.

    The existing schema may represent uniqueness as a unique index or a unique
    constraint. Inspect the actual PostgreSQL schema so the migration handles
    either representation.
    """
    bind = op.get_bind()

    inspector = sa.inspect(bind)

    for constraint in inspector.get_unique_constraints(TABLE_NAME):
        columns = constraint.get("column_names") or []

        if columns == ["transaction_id"]:
            op.drop_constraint(
                constraint["name"],
                TABLE_NAME,
                type_="unique",
            )

    # Refresh the inspector after dropping any unique constraint.
    inspector = sa.inspect(bind)

    for index in inspector.get_indexes(TABLE_NAME):
        columns = index.get("column_names") or []

        if columns == ["transaction_id"] and index.get("unique", False):
            op.drop_index(
                index["name"],
                table_name=TABLE_NAME,
            )

    inspector = sa.inspect(bind)

    existing_transaction_indexes = [
        index
        for index in inspector.get_indexes(TABLE_NAME)
        if (index.get("column_names") or []) == ["transaction_id"]
        and not index.get("unique", False)
    ]

    if not existing_transaction_indexes:
        op.create_index(
            TRANSACTION_INDEX_NAME,
            TABLE_NAME,
            ["transaction_id"],
            unique=False,
        )


def upgrade() -> None:
    """Apply audit-field additions and allow prediction history."""
    op.add_column(
        TABLE_NAME,
        sa.Column(
            "calibration_method",
            sa.String(length=30),
            nullable=True,
        ),
    )

    op.add_column(
        TABLE_NAME,
        sa.Column(
            "decision_reason",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.add_column(
        TABLE_NAME,
        sa.Column(
            "raw_model_margin",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        TABLE_NAME,
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.create_index(
        SCORED_AT_INDEX_NAME,
        TABLE_NAME,
        ["scored_at"],
        unique=False,
    )

    _replace_unique_transaction_index()


def downgrade() -> None:
    """Revert the audit fields and restore transaction-level uniqueness.

    A downgrade is refused if multiple prediction records exist for any
    transaction. Such history cannot be represented by the original schema.
    """
    bind = op.get_bind()

    duplicate = bind.execute(
        sa.text(
            """
            SELECT transaction_id
            FROM fraud_predictions
            GROUP BY transaction_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()

    if duplicate is not None:
        raise RuntimeError(
            "Cannot downgrade: multiple predictions exist for at least one "
            "transaction. Remove or archive the extra prediction records "
            "before restoring uniqueness."
        )

    inspector = sa.inspect(bind)

    for index in inspector.get_indexes(TABLE_NAME):
        columns = index.get("column_names") or []

        if columns == ["transaction_id"] and not index.get("unique", False):
            op.drop_index(
                index["name"],
                table_name=TABLE_NAME,
            )

    inspector = sa.inspect(bind)

    if any(
        index["name"] == SCORED_AT_INDEX_NAME
        for index in inspector.get_indexes(TABLE_NAME)
    ):
        op.drop_index(
            SCORED_AT_INDEX_NAME,
            table_name=TABLE_NAME,
        )

    op.drop_column(TABLE_NAME, "scored_at")
    op.drop_column(TABLE_NAME, "raw_model_margin")
    op.drop_column(TABLE_NAME, "decision_reason")
    op.drop_column(TABLE_NAME, "calibration_method")

    op.create_index(
        TRANSACTION_INDEX_NAME,
        TABLE_NAME,
        ["transaction_id"],
        unique=True,
    )