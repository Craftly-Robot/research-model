"""Task retry columns.

Revision ID: 0003_task_retry
Revises: 0002_data_platform
"""

from __future__ import annotations

from pathlib import Path

from alembic import op


revision = "0003_task_retry"
down_revision = "0002_data_platform"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(Path("src/craftly/db/migrations/0003_task_retry.sql").read_text(encoding="utf-8"))


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
