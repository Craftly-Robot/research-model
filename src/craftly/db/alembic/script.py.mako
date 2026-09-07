"""${message}"""

from __future__ import annotations

from alembic import op


revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    pass


def downgrade() -> None:
    raise RuntimeError("Craftly production migrations are forward-only")
