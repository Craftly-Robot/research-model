"""Durable agent collaboration runtime.

Revision ID: 0005_agent_collaboration
Revises: 0004_training_workspace
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0005_agent_collaboration"
down_revision = "0004_training_workspace"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        Path("src/craftly/db/migrations/0005_agent_collaboration.sql").read_text(encoding="utf-8")
    )


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
