CREATE TABLE IF NOT EXISTS dataset_sync_checkpoints (
  run_id uuid NOT NULL,
  mode text NOT NULL CHECK (mode IN ('bootstrap','incremental')),
  source_key text NOT NULL,
  sort text NOT NULL,
  last_successful_page integer NOT NULL DEFAULT 0,
  watermark timestamptz,
  status text NOT NULL CHECK (status IN ('running','complete','partial','retryable_failure','invalid_source')),
  stop_reason text,
  retryable boolean NOT NULL DEFAULT false,
  started_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, mode, source_key, sort)
);

CREATE INDEX IF NOT EXISTS idx_dataset_sync_checkpoints_latest
  ON dataset_sync_checkpoints(mode, source_key, sort, updated_at DESC);

CREATE TABLE IF NOT EXISTS kaggle_dataset_sync_state (
  source_id text PRIMARY KEY,
  last_updated timestamptz,
  current_version_number bigint,
  detail_pending boolean NOT NULL DEFAULT false,
  detail_error text,
  updated_at timestamptz NOT NULL DEFAULT now()
);

