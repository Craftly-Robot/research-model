"""Data platform schema.

Revision ID: 0002_data_platform
Revises: 0001_initial
Create Date: 2026-07-10
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0002_data_platform"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path("src/craftly/db/migrations/0002_data_platform.sql")
    op.execute(sql_path.read_text(encoding="utf-8"))


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
