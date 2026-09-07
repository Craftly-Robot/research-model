"""Multi-tenant, generation-based Hybrid RAG persistence.

Revision ID: 0007_hybrid_rag
Revises: 0006_dataset_trust
Create Date: 2026-07-21
"""

from __future__ import annotations

from pathlib import Path

from alembic import op


revision = "0007_hybrid_rag"
down_revision = "0006_dataset_trust"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(Path("src/craftly/db/migrations/0007_hybrid_rag.sql").read_text(encoding="utf-8"))


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
