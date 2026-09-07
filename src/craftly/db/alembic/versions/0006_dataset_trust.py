"""Dataset trust, independent review, and promotion evidence.

Revision ID: 0006_dataset_trust
Revises: 0005_agent_collaboration
Create Date: 2026-07-18
"""

from __future__ import annotations

from pathlib import Path

from alembic import op


revision = "0006_dataset_trust"
down_revision = "0005_agent_collaboration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path("src/craftly/db/migrations/0006_dataset_trust.sql")
    op.execute(sql_path.read_text(encoding="utf-8"))


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
