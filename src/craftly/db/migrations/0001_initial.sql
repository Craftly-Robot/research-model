-- Craftly production baseline schema.
-- Keep this migration append-only after release.

CREATE TABLE IF NOT EXISTS schema_migrations (
  version text PRIMARY KEY,
  checksum text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now()
);

\i src/craftly/db/schema.sql

