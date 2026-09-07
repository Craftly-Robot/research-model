-- Craftly MVP production Postgres schema.
-- Local development uses src/craftly/db/local_store.py with equivalent SQLite tables.

CREATE TABLE IF NOT EXISTS organizations (
  id uuid PRIMARY KEY,
  name text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS organization_members (
  organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  role text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS projects (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  name text NOT NULL,
  repo_path text NOT NULL,
  default_branch text NOT NULL DEFAULT 'main',
  index_status text NOT NULL DEFAULT 'not_indexed',
  active_index_revision uuid,
  index_error text,
  last_indexed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_members (
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  role text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  session_id uuid REFERENCES sessions(id) ON DELETE SET NULL,
  prompt text NOT NULL,
  mode text NOT NULL,
  status text NOT NULL,
  model_profile text NOT NULL,
  confidence numeric(5,4),
  summary text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_graphs (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  run_id uuid REFERENCES runs(id) ON DELETE CASCADE,
  goal text NOT NULL,
  status text NOT NULL,
  graph_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tasks (
  id uuid PRIMARY KEY,
  task_graph_id uuid NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  run_id uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  kind text NOT NULL,
  title text NOT NULL,
  status text NOT NULL,
  depends_on uuid[] NOT NULL DEFAULT '{}',
  input_json jsonb NOT NULL DEFAULT '{}',
  output_json jsonb NOT NULL DEFAULT '{}',
  attempt integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 2,
  lease_owner text,
  lease_expires_at timestamptz,
  cancel_requested boolean NOT NULL DEFAULT false,
  error text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workspace_files (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  path text NOT NULL,
  language text,
  content_hash text NOT NULL,
  size_bytes integer NOT NULL,
  index_revision uuid,
  indexed_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(project_id, path)
);

CREATE TABLE IF NOT EXISTS code_chunks (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_id uuid NOT NULL REFERENCES workspace_files(id) ON DELETE CASCADE,
  path text NOT NULL,
  language text,
  start_line integer NOT NULL,
  end_line integer NOT NULL,
  symbol_name text,
  kind text NOT NULL,
  chunk_hash text NOT NULL,
  token_count integer NOT NULL,
  content text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  index_revision uuid,
  indexed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS patches (
  id uuid PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  run_id uuid REFERENCES runs(id) ON DELETE SET NULL,
  status text NOT NULL,
  diff text NOT NULL,
  files_changed text[] NOT NULL DEFAULT '{}',
  backup_json jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  applied_at timestamptz,
  rolled_back_at timestamptz
);

CREATE TABLE IF NOT EXISTS evaluations (
  id uuid PRIMARY KEY,
  benchmark text NOT NULL,
  model_profile text NOT NULL,
  status text NOT NULL,
  total integer NOT NULL DEFAULT 0,
  resolved integer NOT NULL DEFAULT 0,
  score numeric(8,5),
  report_path text,
  result_json jsonb NOT NULL DEFAULT '{}',
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_entries (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
  kind text NOT NULL,
  content text NOT NULL,
  source_run_id uuid REFERENCES runs(id) ON DELETE SET NULL,
  relevance numeric(5,4) NOT NULL DEFAULT 0.5,
  success_rate numeric(5,4) NOT NULL DEFAULT 0.5,
  usage_count integer NOT NULL DEFAULT 0,
  last_used_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_index_revisions (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source_revision text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  chunker_version text NOT NULL,
  status text NOT NULL,
  manifest jsonb NOT NULL DEFAULT '{}',
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  committed_at timestamptz
);

CREATE TABLE IF NOT EXISTS rag_index_jobs (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  revision_id uuid REFERENCES rag_index_revisions(id) ON DELETE SET NULL,
  idempotency_key text NOT NULL,
  status text NOT NULL,
  attempt integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 3,
  cancel_requested boolean NOT NULL DEFAULT false,
  request_json jsonb NOT NULL DEFAULT '{}',
  result_json jsonb NOT NULL DEFAULT '{}',
  lease_owner text,
  lease_expires_at timestamptz,
  available_at timestamptz NOT NULL DEFAULT now(),
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  UNIQUE (organization_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS rag_outbox_events (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  revision_id uuid NOT NULL REFERENCES rag_index_revisions(id) ON DELETE CASCADE,
  kind text NOT NULL,
  payload jsonb NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  attempt integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 5,
  lease_owner text,
  lease_expires_at timestamptz,
  error text,
  available_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  delivered_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_one_building_revision
  ON rag_index_revisions(project_id) WHERE status = 'building';

CREATE TABLE IF NOT EXISTS learning_candidates (
  id uuid PRIMARY KEY,
  project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
  run_id uuid REFERENCES runs(id) ON DELETE SET NULL,
  patch_id uuid REFERENCES patches(id) ON DELETE SET NULL,
  kind text NOT NULL,
  status text NOT NULL,
  prompt text NOT NULL,
  chosen text NOT NULL,
  verification_json jsonb NOT NULL DEFAULT '{}',
  score numeric(8,5),
  exported_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_messages (
  id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task_graph_id uuid NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  task_id uuid REFERENCES tasks(id) ON DELETE SET NULL,
  correlation_id uuid NOT NULL,
  sender_role text NOT NULL CHECK (sender_role IN ('architect', 'coder', 'tester', 'security_reviewer', 'critic', 'verifier', 'orchestrator')),
  recipient_role text NOT NULL CHECK (recipient_role IN ('architect', 'coder', 'tester', 'security_reviewer', 'critic', 'verifier', 'orchestrator', 'broadcast')),
  kind text NOT NULL CHECK (kind IN ('proposal', 'evidence', 'challenge', 'review', 'decision')),
  payload_json jsonb NOT NULL,
  evidence_refs text[] NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS blackboard_entries (
  id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task_graph_id uuid NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  entry_key text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('fact', 'artifact', 'decision', 'question', 'evidence')),
  value_json jsonb NOT NULL,
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  immutable boolean NOT NULL DEFAULT false,
  verified boolean NOT NULL DEFAULT false,
  source_message_id uuid REFERENCES agent_messages(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(run_id, entry_key),
  CHECK (kind <> 'evidence' OR immutable)
);

CREATE TABLE IF NOT EXISTS failure_records (
  id uuid PRIMARY KEY,
  project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
  run_id uuid REFERENCES runs(id) ON DELETE SET NULL,
  task_id uuid REFERENCES tasks(id) ON DELETE SET NULL,
  signature text NOT NULL,
  cluster_key text NOT NULL,
  raw_error text NOT NULL,
  root_cause text,
  patch_id uuid REFERENCES patches(id) ON DELETE SET NULL,
  verification_ref text,
  status text NOT NULL DEFAULT 'observed' CHECK (status IN ('observed', 'linked', 'verified')),
  occurrence_count integer NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
  dataset_candidate_id uuid REFERENCES learning_candidates(id) ON DELETE SET NULL,
  metadata_json jsonb NOT NULL DEFAULT '{}',
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_project_status ON runs(project_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_graph_status ON tasks(task_graph_id, status);
CREATE INDEX IF NOT EXISTS idx_workspace_files_project_path ON workspace_files(project_id, path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_project_path ON code_chunks(project_id, path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_project_symbol ON code_chunks(project_id, symbol_name);
CREATE INDEX IF NOT EXISTS idx_memory_project_kind ON memory_entries(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_learning_status ON learning_candidates(status);
CREATE INDEX IF NOT EXISTS idx_messages_run_created ON agent_messages(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_correlation ON agent_messages(correlation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_blackboard_run_kind ON blackboard_entries(run_id, kind);
CREATE INDEX IF NOT EXISTS idx_failures_cluster ON failure_records(project_id, cluster_key);

