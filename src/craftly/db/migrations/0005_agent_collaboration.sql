-- Durable concurrent-agent collaboration, blackboard, and failure intelligence.

ALTER TABLE tasks ADD COLUMN IF NOT EXISTS lease_owner text;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT false;

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

CREATE INDEX IF NOT EXISTS idx_messages_run_created ON agent_messages(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_correlation ON agent_messages(correlation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_blackboard_run_kind ON blackboard_entries(run_id, kind);
CREATE INDEX IF NOT EXISTS idx_failures_cluster ON failure_records(project_id, cluster_key);
