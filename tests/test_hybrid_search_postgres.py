from __future__ import annotations

import os
from uuid import UUID

import pytest

from kagglekit.hybrid_search import HybridSearchService
from kagglekit.keyword_search import KeywordSearchService, PostgresKeywordSearchRepository
from kagglekit.semantic_search import (
    MODEL_DIMENSION,
    PostgresSemanticSearchRepository,
    SemanticSearchService,
)


DATABASE_URL = os.getenv("AGENTDS2_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="AGENTDS2_TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


class FixedBackend:
    model_name = "hybrid-postgres-test"
    model_revision = "revision-1"
    dimension = MODEL_DIMENSION

    def embed_query(self, query):
        return [1.0] + [0.0] * (MODEL_DIMENSION - 1)

    def embed_documents(self, documents):
        return [[1.0] + [0.0] * (MODEL_DIMENSION - 1) for _ in documents]


def test_real_postgres_hybrid_filter_dedup_scores_and_pagination():
    import psycopg

    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    ids = [UUID(int=401), UUID(int=402), UUID(int=403)]
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM resources WHERE source = 'hybrid-search-test'")
        cursor.executemany(
            """
            INSERT INTO resources
              (id, source, source_id, resource_type, status, title, tags, url)
            VALUES (%s, 'hybrid-search-test', %s, %s, 'active', %s, '{}', %s)
            """,
            [
                (ids[0], "both", "dataset", "hybridtoken housing", "https://x/401"),
                (ids[1], "semantic-only", "dataset", "Unrelated words", "https://x/402"),
                (ids[2], "notebook", "notebook", "hybridtoken", "https://x/403"),
            ],
        )
    keyword_repo = PostgresKeywordSearchRepository(DATABASE_URL)
    semantic_repo = PostgresSemanticSearchRepository(DATABASE_URL)
    backend = FixedBackend()
    try:
        semantic_repo.ensure_model_compatible(backend, rebuild=True)
        semantic_repo.upsert_embeddings(
            [
                (ids[0], "a", [1.0] + [0.0] * (MODEL_DIMENSION - 1)),
                (ids[1], "b", [0.9, 0.1] + [0.0] * (MODEL_DIMENSION - 2)),
                (ids[2], "c", [1.0] + [0.0] * (MODEL_DIMENSION - 1)),
            ],
            backend,
        )
        service = HybridSearchService(
            KeywordSearchService(keyword_repo), SemanticSearchService(semantic_repo, backend)
        )
        results = service.search(
            ["hybridtoken"], "housing", resource_type="dataset", limit=10,
            keyword_top_k=10, semantic_top_k=10,
        )
        assert [row.source_id for row in results[:2]] == ["both", "semantic-only"]
        assert len({row.id for row in results}) == len(results)
        assert results[0].keyword_rank == results[0].semantic_rank == 1
        assert results[1].keyword_rank is None and results[1].semantic_rank == 2
        assert service.search(
            ["hybridtoken"], "housing", resource_type="dataset", limit=1, offset=1,
            keyword_top_k=10, semantic_top_k=10,
        )[0].source_id == "semantic-only"
    finally:
        keyword_repo.close()
        semantic_repo.close()
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM resources WHERE source = 'hybrid-search-test'")
        connection.close()
