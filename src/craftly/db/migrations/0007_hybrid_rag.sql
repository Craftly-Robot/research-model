-- Multi-tenant, generation-based Hybrid RAG persistence.

CREATE TABLE IF NOT EXISTS organizations (
  id uuid PRIMARY KEY,
  name text NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS organization_members (
  organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS project_members (
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('owner', 'editor', 'viewer')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, user_id)
);

INSERT INTO organizations(id, name)
VALUES ('00000000-0000-0000-0000-000000000001', 'Local Development Organization')
ON CONFLICT (id) DO NOTHING;

ALTER TABLE projects ADD COLUMN IF NOT EXISTS organization_id uuid;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS active_index_revision uuid;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS index_error text;
UPDATE projects SET organization_id = '00000000-0000-0000-0000-000000000001' WHERE organization_id IS NULL;
ALTER TABLE projects ALTER COLUMN organization_id SET NOT NULL;
DO $$ BEGIN
  ALTER TABLE projects ADD CONSTRAINT projects_organization_fk
    FOREIGN KEY (organization_id) REFERENCES organizations(id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

ALTER TABLE workspace_files ADD COLUMN IF NOT EXISTS index_revision uuid;
ALTER TABLE code_chunks ADD COLUMN IF NOT EXISTS index_revision uuid;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS organization_id uuid;
UPDATE memory_entries m SET organization_id = COALESCE(
  (SELECT p.organization_id FROM projects p WHERE p.id = m.project_id),
  '00000000-0000-0000-0000-000000000001'
) WHERE organization_id IS NULL;
ALTER TABLE memory_entries ALTER COLUMN organization_id SET NOT NULL;
DO $$ BEGIN
  ALTER TABLE memory_entries ADD CONSTRAINT memory_entries_organization_fk
    FOREIGN KEY (organization_id) REFERENCES organizations(id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS rag_index_revisions (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source_revision text NOT NULL,
  source_snapshot_sha256 text NOT NULL CHECK (source_snapshot_sha256 ~ '^[0-9a-f]{64}$'),
  chunker_version text NOT NULL,
  status text NOT NULL CHECK (status IN ('building', 'committed', 'failed', 'superseded')),
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
  status text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'dead_letter')),
  attempt integer NOT NULL DEFAULT 0 CHECK (attempt >= 0),
  max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 20),
  cancel_requested boolean NOT NULL DEFAULT false,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS rag_outbox_events (
  id uuid PRIMARY KEY,
  organization_id uuid NOT NULL REFERENCES organizations(id),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  revision_id uuid NOT NULL REFERENCES rag_index_revisions(id) ON DELETE CASCADE,
  kind text NOT NULL,
  payload jsonb NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'delivering', 'delivered', 'dead_letter')),
  attempt integer NOT NULL DEFAULT 0 CHECK (attempt >= 0),
  available_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  delivered_at timestamptz
);

DO $$ BEGIN
  ALTER TABLE projects ADD CONSTRAINT projects_active_index_revision_fk
    FOREIGN KEY (active_index_revision) REFERENCES rag_index_revisions(id) DEFERRABLE INITIALLY DEFERRED;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  ALTER TABLE workspace_files ADD CONSTRAINT workspace_files_index_revision_fk
    FOREIGN KEY (index_revision) REFERENCES rag_index_revisions(id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  ALTER TABLE code_chunks ADD CONSTRAINT code_chunks_index_revision_fk
    FOREIGN KEY (index_revision) REFERENCES rag_index_revisions(id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS idx_projects_org ON projects(organization_id, id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_org_project ON memory_entries(organization_id, project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_revisions_project ON rag_index_revisions(organization_id, project_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_one_building_revision ON rag_index_revisions(project_id) WHERE status = 'building';
CREATE INDEX IF NOT EXISTS idx_rag_jobs_status ON rag_index_jobs(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_rag_outbox_pending ON rag_outbox_events(status, available_at);

ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_index_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_index_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_outbox_events ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY projects_tenant_isolation ON projects USING (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  ) WITH CHECK (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY memory_entries_tenant_isolation ON memory_entries USING (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  ) WITH CHECK (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY rag_revisions_tenant_isolation ON rag_index_revisions USING (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  ) WITH CHECK (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY rag_jobs_tenant_isolation ON rag_index_jobs USING (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  ) WITH CHECK (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY rag_outbox_tenant_isolation ON rag_outbox_events USING (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  ) WITH CHECK (
    organization_id = NULLIF(current_setting('craftly.organization_id', true), '')::uuid
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
