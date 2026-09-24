CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resource_embeddings (
  resource_id uuid PRIMARY KEY REFERENCES resources(id) ON DELETE CASCADE,
  model_name text NOT NULL,
  model_revision text NOT NULL,
  embedding_dimension integer NOT NULL CHECK (embedding_dimension = 384),
  content_hash text NOT NULL,
  embedding vector(384) NOT NULL,
  embedded_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_resource_embeddings_embedding_hnsw
  ON resource_embeddings USING hnsw (embedding vector_cosine_ops);
