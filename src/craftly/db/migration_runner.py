"""Postgres migration runner for Craftly production deployments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    sql: str
    checksum: str


def load_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[Migration]:
    migrations = []
    for path in sorted(migrations_dir.glob("*.sql")):
        sql = path.read_text(encoding="utf-8-sig")
        migrations.append(
            Migration(
                version=path.stem,
                path=path,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    return migrations


def expand_psql_includes(sql: str, *, base_dir: Path | None = None) -> str:
    base = base_dir or Path.cwd()
    output: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\i "):
            include_path = Path(stripped[3:].strip())
            if not include_path.is_absolute():
                include_path = (base / include_path).resolve()
            output.append(include_path.read_text(encoding="utf-8-sig"))
        else:
            output.append(line)
    return "\n".join(output)


async def apply_migrations(database_url: str, *, dry_run: bool = False) -> dict[str, Any]:
    import asyncpg

    migrations = load_migrations()
    applied: list[str] = []
    async with asyncpg.create_pool(database_url, min_size=1, max_size=2) as pool:
        async with pool.acquire() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  version text PRIMARY KEY,
                  checksum text NOT NULL,
                  applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            rows = await connection.fetch("SELECT version, checksum FROM schema_migrations")
            existing = {row["version"]: row["checksum"] for row in rows}
            for migration in migrations:
                if migration.version in existing:
                    if existing[migration.version] != migration.checksum:
                        raise RuntimeError(f"migration checksum mismatch: {migration.version}")
                    continue
                if dry_run:
                    applied.append(migration.version)
                    continue
                async with connection.transaction():
                    sql = expand_psql_includes(migration.sql, base_dir=Path.cwd())
                    await connection.execute(sql)
                    await connection.execute(
                        "INSERT INTO schema_migrations(version, checksum) VALUES($1, $2)",
                        migration.version,
                        migration.checksum,
                    )
                    applied.append(migration.version)
    return {"status": "ok", "applied": applied, "migration_count": len(migrations), "dry_run": dry_run}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Craftly Postgres migrations.")
    parser.add_argument("--database-url", default=os.environ.get("CRAFTLY_DATABASE_URL", ""))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("CRAFTLY_DATABASE_URL or --database-url is required")
    result = asyncio.run(apply_migrations(args.database_url, dry_run=args.dry_run))
    print(result)


if __name__ == "__main__":
    main()

