-- Lease-based Hybrid RAG jobs and durable vector-sync delivery.

ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS request_json jsonb NOT NULL DEFAULT '{}';
ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS result_json jsonb NOT NULL DEFAULT '{}';
ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS lease_owner text;
ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS available_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE rag_index_jobs ADD COLUMN IF NOT EXISTS finished_at timestamptz;

ALTER TABLE rag_outbox_events ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 5;
ALTER TABLE rag_outbox_events ADD COLUMN IF NOT EXISTS lease_owner text;
ALTER TABLE rag_outbox_events ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE rag_outbox_events ADD COLUMN IF NOT EXISTS error text;

DROP INDEX IF EXISTS idx_rag_jobs_status;
CREATE INDEX idx_rag_jobs_claim
  ON rag_index_jobs(status, available_at, created_at)
  WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS idx_rag_jobs_lease
  ON rag_index_jobs(lease_expires_at)
  WHERE status = 'running';

DROP INDEX IF EXISTS idx_rag_outbox_pending;
CREATE INDEX idx_rag_outbox_claim
  ON rag_outbox_events(status, available_at, created_at)
  WHERE status IN ('pending', 'delivering');
CREATE INDEX IF NOT EXISTS idx_rag_outbox_lease
  ON rag_outbox_events(lease_expires_at)
  WHERE status = 'delivering';

ALTER TABLE rag_index_jobs DROP CONSTRAINT IF EXISTS rag_index_jobs_status_check;
ALTER TABLE rag_index_jobs ADD CONSTRAINT rag_index_jobs_status_check
  CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'dead_letter'));

ALTER TABLE rag_outbox_events DROP CONSTRAINT IF EXISTS rag_outbox_events_status_check;
ALTER TABLE rag_outbox_events ADD CONSTRAINT rag_outbox_events_status_check
  CHECK (status IN ('pending', 'delivering', 'delivered', 'dead_letter'));

CREATE OR REPLACE FUNCTION craftly_claim_rag_index_job(
  p_worker_id text,
  p_lease_seconds integer DEFAULT 120
) RETURNS SETOF rag_index_jobs
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  IF p_worker_id !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$' THEN
    RAISE EXCEPTION 'invalid RAG worker identifier';
  END IF;
  IF p_lease_seconds < 10 OR p_lease_seconds > 900 THEN
    RAISE EXCEPTION 'RAG lease must be between 10 and 900 seconds';
  END IF;
  RETURN QUERY
  WITH candidate AS (
    SELECT id FROM rag_index_jobs
    WHERE (
      (status = 'queued' AND available_at <= now()) OR
      (status = 'running' AND lease_expires_at <= now())
    )
      AND cancel_requested = false
      AND attempt < max_attempts
    ORDER BY available_at, created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
  )
  UPDATE rag_index_jobs AS job
  SET status = 'running', attempt = job.attempt + 1,
      lease_owner = p_worker_id,
      lease_expires_at = now() + (p_lease_seconds * interval '1 second'),
      updated_at = now(), error = NULL
  FROM candidate
  WHERE job.id = candidate.id
  RETURNING job.*;
END;
$$;

CREATE OR REPLACE FUNCTION craftly_claim_rag_outbox(
  p_worker_id text,
  p_lease_seconds integer DEFAULT 120
) RETURNS SETOF rag_outbox_events
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  IF p_worker_id !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$' THEN
    RAISE EXCEPTION 'invalid RAG worker identifier';
  END IF;
  IF p_lease_seconds < 10 OR p_lease_seconds > 900 THEN
    RAISE EXCEPTION 'RAG lease must be between 10 and 900 seconds';
  END IF;
  RETURN QUERY
  WITH candidate AS (
    SELECT id FROM rag_outbox_events
    WHERE (
      (status = 'pending' AND available_at <= now()) OR
      (status = 'delivering' AND lease_expires_at <= now())
    ) AND attempt < max_attempts
    ORDER BY available_at, created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
  )
  UPDATE rag_outbox_events AS event
  SET status = 'delivering', attempt = event.attempt + 1,
      lease_owner = p_worker_id,
      lease_expires_at = now() + (p_lease_seconds * interval '1 second'),
      error = NULL
  FROM candidate
  WHERE event.id = candidate.id
  RETURNING event.*;
END;
$$;

REVOKE ALL ON FUNCTION craftly_claim_rag_index_job(text, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION craftly_claim_rag_outbox(text, integer) FROM PUBLIC;
