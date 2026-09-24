CREATE TABLE IF NOT EXISTS resources (
  id uuid PRIMARY KEY,
  source text NOT NULL,
  source_id text NOT NULL,
  resource_type text NOT NULL CHECK (resource_type IN ('dataset','notebook')),
  title text NOT NULL,
  url text NOT NULL,
  description text,
  tags text[] NOT NULL DEFAULT '{}',
  author text,
  updated_at timestamptz,
  votes bigint,
  subtitle text,
  license text,
  downloads bigint,
  usability_rating double precision,
  size_bytes bigint,
  language text,
  last_run_at timestamptz,
  dataset_sources text[] NOT NULL DEFAULT '{}',
  indexed_at timestamptz NOT NULL DEFAULT now(),
  last_synced_at timestamptz NOT NULL DEFAULT now(),
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','unavailable','deleted')),
  UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_resources_resource_type ON resources(resource_type);
CREATE INDEX IF NOT EXISTS idx_resources_status ON resources(status);
CREATE INDEX IF NOT EXISTS idx_resources_last_synced_at ON resources(last_synced_at);

