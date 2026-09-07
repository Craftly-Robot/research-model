-- Durable source, review, and dataset-promotion evidence.

CREATE TABLE IF NOT EXISTS data_source_snapshots (
  id uuid PRIMARY KEY,
  source_id text NOT NULL,
  source_family text NOT NULL,
  immutable_revision text NOT NULL,
  registry_sha256 text NOT NULL,
  license_evidence_sha256 text NOT NULL,
  legal_approval_sha256 text NOT NULL,
  snapshot_sha256 text NOT NULL,
  status text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(source_id, immutable_revision, snapshot_sha256)
);

CREATE TABLE IF NOT EXISTS dataset_review_items (
  id uuid PRIMARY KEY,
  content_hash text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  source_id text NOT NULL,
  data_type text NOT NULL,
  high_value boolean NOT NULL DEFAULT false,
  status text NOT NULL DEFAULT 'pending',
  version integer NOT NULL DEFAULT 1,
  payload jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(content_hash, source_snapshot_sha256)
);

CREATE TABLE IF NOT EXISTS dataset_review_assignments (
  id uuid PRIMARY KEY,
  review_item_id uuid NOT NULL REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  reviewer_id text NOT NULL,
  reviewer_slot smallint NOT NULL CHECK (reviewer_slot IN (1, 2)),
  status text NOT NULL DEFAULT 'claimed',
  claimed_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  UNIQUE(review_item_id, reviewer_id),
  UNIQUE(review_item_id, reviewer_slot)
);

CREATE TABLE IF NOT EXISTS dataset_review_decisions (
  id uuid PRIMARY KEY,
  review_item_id uuid NOT NULL REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  reviewer_id text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('approve', 'reject')),
  rationale text NOT NULL,
  content_hash text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  evidence jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(review_item_id, reviewer_id)
);

CREATE TABLE IF NOT EXISTS dataset_review_adjudications (
  id uuid PRIMARY KEY,
  review_item_id uuid NOT NULL REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  adjudicator_id text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('approve', 'reject')),
  rationale text NOT NULL,
  evidence jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(review_item_id)
);

CREATE TABLE IF NOT EXISTS dataset_promotion_events (
  id uuid PRIMARY KEY,
  dataset_id text NOT NULL,
  version_id text NOT NULL,
  actor_id text NOT NULL,
  manifest_sha256 text NOT NULL,
  policy_sha256 text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('promoted', 'rejected')),
  evidence jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(dataset_id, version_id, manifest_sha256)
);

CREATE INDEX IF NOT EXISTS idx_dataset_review_items_status_created
  ON dataset_review_items(status, created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_review_items_source
  ON dataset_review_items(source_id, data_type, status);
CREATE INDEX IF NOT EXISTS idx_dataset_review_assignments_reviewer
  ON dataset_review_assignments(reviewer_id, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_dataset_review_decisions_item
  ON dataset_review_decisions(review_item_id, created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_promotion_dataset
  ON dataset_promotion_events(dataset_id, version_id, created_at);

CREATE TABLE IF NOT EXISTS dataset_dedup_exact (
  dataset_version text NOT NULL,
  content_hash text NOT NULL,
  PRIMARY KEY(dataset_version, content_hash)
);

CREATE TABLE IF NOT EXISTS dataset_dedup_structure (
  dataset_version text NOT NULL,
  structure_hash text NOT NULL,
  PRIMARY KEY(dataset_version, structure_hash)
);

CREATE TABLE IF NOT EXISTS dataset_dedup_lineage (
  dataset_version text NOT NULL,
  lineage_hash text NOT NULL,
  PRIMARY KEY(dataset_version, lineage_hash)
);

CREATE TABLE IF NOT EXISTS dataset_dedup_fingerprints (
  id bigserial PRIMARY KEY,
  dataset_version text NOT NULL,
  simhash_hex text NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_dedup_bands (
  dataset_version text NOT NULL,
  band_key text NOT NULL,
  fingerprint_id bigint NOT NULL REFERENCES dataset_dedup_fingerprints(id) ON DELETE CASCADE,
  PRIMARY KEY(dataset_version, band_key, fingerprint_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_dedup_bands_lookup
  ON dataset_dedup_bands(dataset_version, band_key);
