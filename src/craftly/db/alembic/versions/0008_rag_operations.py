"""Lease-based Hybrid RAG jobs and outbox delivery.

Revision ID: 0008_rag_operations
Revises: 0007_hybrid_rag
Create Date: 2026-07-21
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0008_rag_operations"
down_revision = "0007_hybrid_rag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        Path("src/craftly/db/migrations/0008_rag_operations.sql").read_text(encoding="utf-8")
    )


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
