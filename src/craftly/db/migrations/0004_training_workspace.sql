-- Craftly training workspace and scale-control schema.

CREATE TABLE IF NOT EXISTS training_profiles (
  profile_id text NOT NULL,
  version integer NOT NULL CHECK (version > 0),
  profile_hash text NOT NULL CHECK (length(profile_hash) = 64),
  profile_json jsonb NOT NULL,
  enabled boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(profile_id, version),
  UNIQUE(profile_hash)
);

CREATE TABLE IF NOT EXISTS training_jobs (
  id uuid PRIMARY KEY,
  owner_id text NOT NULL,
  idempotency_key text NOT NULL,
  profile_id text NOT NULL,
  profile_version integer NOT NULL,
  spec_hash text NOT NULL CHECK (length(spec_hash) = 64),
  spec_json jsonb NOT NULL,
  status text NOT NULL CHECK (status IN (
    'validating','queued','provisioning','running','checkpointing',
    'evaluating','succeeded','failed','blocked','cancelled'
  )),
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  event_sequence bigint NOT NULL DEFAULT 0 CHECK (event_sequence >= 0),
  archived_event_sequence bigint NOT NULL DEFAULT 0 CHECK (archived_event_sequence >= 0),
  current_attempt_id uuid,
  scheduler_binding jsonb NOT NULL DEFAULT '{}',
  failure_code text,
  failure_detail text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(owner_id, idempotency_key),
  FOREIGN KEY(profile_id, profile_version) REFERENCES training_profiles(profile_id, version)
);

CREATE TABLE IF NOT EXISTS training_attempts (
  id uuid PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
  attempt_number integer NOT NULL CHECK (attempt_number > 0),
  scheduler text NOT NULL CHECK (scheduler IN ('notebook','kubernetes','kubernetes_pytorch','slurm')),
  scheduler_binding jsonb NOT NULL DEFAULT '{}',
  checkpoint_uri text,
  status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  UNIQUE(job_id, attempt_number)
);

ALTER TABLE training_jobs
  DROP CONSTRAINT IF EXISTS training_jobs_current_attempt_id_fkey;
ALTER TABLE training_jobs
  ADD CONSTRAINT training_jobs_current_attempt_id_fkey
  FOREIGN KEY(current_attempt_id) REFERENCES training_attempts(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS training_artifacts (
  id uuid PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
  attempt_id uuid REFERENCES training_attempts(id) ON DELETE SET NULL,
  kind text NOT NULL CHECK (kind IN ('spec','log','checkpoint','evaluation','report','dataset','tokenizer')),
  uri text NOT NULL,
  sha256 text NOT NULL CHECK (length(sha256) = 64),
  size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
  promoted boolean NOT NULL DEFAULT false,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(job_id, uri)
);

CREATE TABLE IF NOT EXISTS checkpoint_versions (
  id uuid PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
  attempt_id uuid REFERENCES training_attempts(id) ON DELETE SET NULL,
  step bigint NOT NULL CHECK (step >= 0),
  manifest_uri text NOT NULL,
  manifest_sha256 text NOT NULL CHECK (length(manifest_sha256) = 64),
  dataset_sha256 text NOT NULL CHECK (length(dataset_sha256) = 64),
  tokenizer_sha256 text NOT NULL CHECK (length(tokenizer_sha256) = 64),
  topology_json jsonb NOT NULL,
  metrics_json jsonb NOT NULL DEFAULT '{}',
  reload_verified boolean NOT NULL DEFAULT false,
  promoted boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(job_id, step)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
  id uuid PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
  checkpoint_id uuid REFERENCES checkpoint_versions(id) ON DELETE SET NULL,
  status text NOT NULL,
  report_uri text NOT NULL,
  report_sha256 text NOT NULL CHECK (length(report_sha256) = 64),
  decision text NOT NULL,
  metrics_json jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS service_accounts (
  id uuid PRIMARY KEY,
  name text NOT NULL UNIQUE,
  scopes text[] NOT NULL,
  token_prefix text NOT NULL,
  token_hash text NOT NULL UNIQUE CHECK (length(token_hash) = 64),
  enabled boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  last_used_at timestamptz
);

CREATE TABLE IF NOT EXISTS service_account_sessions (
  id uuid PRIMARY KEY,
  service_account_id uuid NOT NULL REFERENCES service_accounts(id) ON DELETE CASCADE,
  refresh_hash text NOT NULL UNIQUE CHECK (length(refresh_hash) = 64),
  expires_at timestamptz NOT NULL,
  revoked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_audit_events (
  id uuid PRIMARY KEY,
  actor_id text NOT NULL,
  job_id uuid REFERENCES training_jobs(id) ON DELETE SET NULL,
  action text NOT NULL,
  outcome text NOT NULL,
  request_id text,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_event_ingress (
  attempt_id uuid NOT NULL REFERENCES training_attempts(id) ON DELETE CASCADE,
  rank integer NOT NULL CHECK (rank >= 0),
  source_sequence bigint NOT NULL CHECK (source_sequence >= 0),
  assigned_sequence bigint NOT NULL CHECK (assigned_sequence > 0),
  received_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(attempt_id, rank, source_sequence),
  UNIQUE(attempt_id, assigned_sequence)
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created ON training_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_training_jobs_owner_created ON training_jobs(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_attempts_job ON training_attempts(job_id, attempt_number);
CREATE INDEX IF NOT EXISTS idx_training_artifacts_job_kind ON training_artifacts(job_id, kind, created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_job_step ON checkpoint_versions(job_id, step DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_job_created ON evaluation_runs(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_sessions_expiry ON service_account_sessions(expires_at) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_training_audit_job_created ON training_audit_events(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_event_ingress_received ON training_event_ingress(received_at);
