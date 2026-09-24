from __future__ import annotations

import os
from uuid import UUID

import pytest

from kagglekit.semantic_search import (
    MODEL_DIMENSION,
    ModelMismatchError,
    PostgresSemanticSearchRepository,
    SemanticSearchService,
)


DATABASE_URL = os.getenv("AGENTDS2_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="AGENTDS2_TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


class FixedBackend:
    model_name = "semantic-postgres-test"
    model_revision = "revision-1"
    dimension = MODEL_DIMENSION

    def embed_query(self, query):
        return [1.0] + [0.0] * (MODEL_DIMENSION - 1)

    def embed_documents(self, documents):
        return [[1.0] + [0.0] * (MODEL_DIMENSION - 1) for _ in documents]


@pytest.fixture()
def repository():
    import psycopg

    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    ids = [
        UUID("30000000-0000-0000-0000-000000000001"),
        UUID("30000000-0000-0000-0000-000000000002"),
        UUID("30000000-0000-0000-0000-000000000003"),
    ]
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM resources WHERE source = 'semantic-search-test'")
        cursor.executemany(
            """
            INSERT INTO resources
              (id, source, source_id, resource_type, status, title, tags, url)
            VALUES (%s, 'semantic-search-test', %s, %s, %s, %s, '{}', %s)
            """,
            [
                (ids[0], "active-dataset", "dataset", "active", "Housing", "https://x/1"),
                (ids[1], "inactive-dataset", "dataset", "unavailable", "Hidden", "https://x/2"),
                (ids[2], "active-notebook", "notebook", "active", "Notebook", "https://x/3"),
            ],
        )
    repo = PostgresSemanticSearchRepository(DATABASE_URL)
    backend = FixedBackend()
    repo.ensure_model_compatible(backend, rebuild=True)
    repo.upsert_embeddings(
        [
            (ids[0], "a", [1.0] + [0.0] * (MODEL_DIMENSION - 1)),
            (ids[1], "b", [1.0] + [0.0] * (MODEL_DIMENSION - 1)),
            (ids[2], "c", [0.5, 0.5] + [0.0] * (MODEL_DIMENSION - 2)),
        ],
        backend,
    )
    yield repo, backend
    repo.close()
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM resources WHERE source = 'semantic-search-test'")
    connection.close()


def test_real_pg_score_direction_active_filter_resource_filter_dedup_and_pagination(repository):
    repo, backend = repository
    service = SemanticSearchService(repo, backend)
    datasets = service.search("housing", resource_type="dataset")
    notebooks = service.search("housing", resource_type="notebook")
    assert [row.source_id for row in datasets] == ["active-dataset"]
    assert [row.source_id for row in notebooks] == ["active-notebook"]
    assert datasets[0].semantic_score > notebooks[0].semantic_score
    assert len({row.id for row in datasets}) == len(datasets)
    assert service.search("housing", resource_type="dataset", limit=1, offset=1) == []


def test_real_pg_detects_model_mismatch(repository):
    repo, backend = repository
    backend.model_revision = "revision-2"
    with pytest.raises(ModelMismatchError, match="rebuild"):
        repo.ensure_model_compatible(backend, rebuild=False)
