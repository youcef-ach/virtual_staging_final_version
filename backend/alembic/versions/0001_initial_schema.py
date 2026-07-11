"""Initial schema — all tables from PLAN.md section 4.

This first migration bootstraps the schema from the ORM metadata so there is a
single source of truth at project start. Every subsequent migration must use
explicit ``op.*`` operations.

Revision ID: 0001
Revises:
Create Date: 2026-07-11
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    import app.models  # noqa: F401  (register all tables)
    from app.db import Base

    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    import app.models  # noqa: F401
    from app.db import Base

    Base.metadata.drop_all(bind=op.get_bind())
