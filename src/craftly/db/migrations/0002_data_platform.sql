-- Craftly production data platform schema.

CREATE TABLE IF NOT EXISTS data_sources (
  id uuid PRIMARY KEY,
  name text NOT NULL UNIQUE,
  category text NOT NULL,
  license text NOT NULL,
  allowed_domains text[] NOT NULL DEFAULT '{}',
  seed_urls text[] NOT NULL DEFAULT '{}',
  status text NOT NULL DEFAULT 'approved',
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
  id uuid PRIMARY KEY,
  dataset_id text NOT NULL,
  version_id text NOT NULL,
  manifest_uri text NOT NULL,
  tokenizer_uri text,
  shard_manifest_uri text,
  clean_rows integer NOT NULL DEFAULT 0,
  token_count bigint NOT NULL DEFAULT 0,
  contamination_hits integer NOT NULL DEFAULT 0,
  extracted_tasks integer NOT NULL DEFAULT 0,
  status text NOT NULL,
  result_json jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(dataset_id, version_id)
);

CREATE TABLE IF NOT EXISTS data_quality_events (
  id uuid PRIMARY KEY,
  dataset_id text NOT NULL,
  version_id text,
  event_type text NOT NULL,
  severity text NOT NULL,
  source_uri text,
  content_hash text,
  message text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset_created ON dataset_versions(dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_events_dataset ON data_quality_events(dataset_id, event_type, created_at DESC);

