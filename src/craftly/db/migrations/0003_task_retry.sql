-- Add durable TaskGraph retry counters.

ALTER TABLE tasks ADD COLUMN IF NOT EXISTS attempt integer NOT NULL DEFAULT 0;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 2;
