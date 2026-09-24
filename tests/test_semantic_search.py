from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import UUID

import pytest

from kagglekit.semantic_search import (
    DESCRIPTION_MAX_CHARS,
    MODEL_DIMENSION,
    BackfillStats,
    EmbeddingCandidate,
    EmbeddingDimensionError,
    FastEmbedBackend,
    ModelInitializationError,
    ModelMismatchError,
    PostgresSemanticSearchRepository,
    SemanticBackfillService,
    SemanticSearchError,
    SemanticSearchService,
    build_semantic_document,
    semantic_content_hash,
    vector_literal,
)


ID1 = UUID("20000000-0000-0000-0000-000000000001")
ID2 = UUID("20000000-0000-0000-0000-000000000002")


class FakeBackend:
    model_name = "test-model"
    model_revision = "fixed-revision"
    dimension = MODEL_DIMENSION

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, documents):
        self.document_calls.append(list(documents))
        return [[float(index + 1)] + [0.0] * (MODEL_DIMENSION - 1) for index, _ in enumerate(documents)]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0] + [0.0] * (MODEL_DIMENSION - 1)


class FakeRepository:
    def __init__(self, candidates=(), results=()):
        self.candidates = list(candidates)
        self.results = list(results)
        self.ensure_calls = []
        self.upserts = []
        self.search_calls = []

    def ensure_model_compatible(self, backend, *, rebuild):
        self.ensure_calls.append((backend.model_name, backend.model_revision, backend.dimension, rebuild))

    def embedding_candidates(self):
        return self.candidates

    def upsert_embeddings(self, rows, backend):
        self.upserts.extend(rows)

    def search(self, vector, backend, *, resource_type, limit, offset):
        self.search_calls.append((vector, resource_type, limit, offset))
        return self.results[offset : offset + limit]


def candidate(resource_id=ID1, *, title="House prices", description=None, tags=None, existing_hash=None):
    return EmbeddingCandidate(resource_id, title, None, tags or [], description, existing_hash)


def test_semantic_document_has_only_supported_fields_and_labels():
    document = build_semantic_document(
        title="Housing", subtitle="Sales", tags=["real estate", "prices"], description="Rows"
    )
    assert document == "title: Housing\nsubtitle: Sales\ntags: real estate, prices\ndescription: Rows"


def test_semantic_document_is_null_safe():
    assert build_semantic_document(title="Housing", subtitle=None, tags=None, description=None) == "title: Housing"


def test_semantic_document_normalizes_unicode_and_whitespace():
    document = build_semantic_document(title="Ａ  \n B", subtitle=None, tags=[], description=None)
    assert document == "title: A B"


def test_semantic_document_turns_tags_into_text():
    assert "tags: car, vehicle prices" in build_semantic_document(
        title="x", subtitle=None, tags=[" car ", "vehicle\nprices"], description=None
    )


def test_description_is_truncated():
    document = build_semantic_document(
        title="x", subtitle=None, tags=[], description="z" * (DESCRIPTION_MAX_CHARS + 50)
    )
    assert document.endswith("z" * DESCRIPTION_MAX_CHARS)
    assert len(document.split("description: ", 1)[1]) == DESCRIPTION_MAX_CHARS


def test_content_hash_is_stable_and_sensitive_to_changes():
    assert semantic_content_hash("same") == semantic_content_hash("same")
    assert semantic_content_hash("same") != semantic_content_hash("changed")


def test_vector_literal_validates_dimension():
    with pytest.raises(EmbeddingDimensionError, match="384"):
        vector_literal([1.0])


def test_vector_literal_rejects_non_finite_values():
    with pytest.raises(EmbeddingDimensionError, match="non-finite"):
        vector_literal([float("nan")] + [0.0] * (MODEL_DIMENSION - 1))


def test_vector_literal_is_pgvector_text():
    result = vector_literal([1.0] + [0.0] * (MODEL_DIMENSION - 1))
    assert result.startswith("[1,0,0,") and result.endswith("]")


def test_backfill_embeds_missing_rows():
    repository = FakeRepository([candidate()])
    backend = FakeBackend()
    stats = SemanticBackfillService(repository, backend).run()
    assert (stats.total, stats.embedded, stats.skipped) == (1, 1, 0)
    assert repository.upserts[0][0] == ID1


def test_backfill_skips_unchanged_content_hash():
    item = candidate()
    item = replace(item, existing_hash=semantic_content_hash(item.document()))
    repository = FakeRepository([item])
    backend = FakeBackend()
    stats = SemanticBackfillService(repository, backend).run()
    assert (stats.embedded, stats.skipped) == (0, 1)
    assert backend.document_calls == []


def test_backfill_regenerates_changed_content():
    repository = FakeRepository([candidate(existing_hash=semantic_content_hash("old"))])
    stats = SemanticBackfillService(repository, FakeBackend()).run()
    assert stats.embedded == 1


def test_backfill_batches_rows_and_preserves_ids():
    repository = FakeRepository([candidate(ID1), candidate(ID2, title="Cars")])
    backend = FakeBackend()
    SemanticBackfillService(repository, backend, batch_size=1).run()
    assert len(backend.document_calls) == 2
    assert [row[0] for row in repository.upserts] == [ID1, ID2]


def test_backfill_forwards_explicit_rebuild():
    repository = FakeRepository()
    SemanticBackfillService(repository, FakeBackend()).run(rebuild=True)
    assert repository.ensure_calls[0][-1] is True


def test_backfill_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        SemanticBackfillService(FakeRepository(), FakeBackend(), batch_size=0)


def test_backfill_rejects_wrong_backend_batch_count():
    backend = FakeBackend()
    backend.embed_documents = lambda documents: []
    with pytest.raises(SemanticSearchError, match="batch size"):
        SemanticBackfillService(FakeRepository([candidate()]), backend).run()


def test_search_embeds_normalized_query_and_forwards_defaults():
    repository = FakeRepository()
    backend = FakeBackend()
    SemanticSearchService(repository, backend).search("  中国\n房价  ")
    assert backend.query_calls == ["中国 房价"]
    assert repository.search_calls[0][1:] == (None, 20, 0)


def test_search_forwards_resource_filter_and_pagination():
    repository = FakeRepository()
    SemanticSearchService(repository, FakeBackend()).search(
        "car prices", resource_type="dataset", limit=5, offset=2
    )
    assert repository.search_calls[0][1:] == ("dataset", 5, 2)


@pytest.mark.parametrize("query", ["", "  ", "---"])
def test_search_rejects_empty_or_low_information_punctuation(query):
    with pytest.raises(ValueError, match="query"):
        SemanticSearchService(FakeRepository(), FakeBackend()).search(query)


@pytest.mark.parametrize("resource_type", ["Dataset", "model", ""])
def test_search_rejects_invalid_resource_type(resource_type):
    with pytest.raises(ValueError, match="resource_type"):
        SemanticSearchService(FakeRepository(), FakeBackend()).search(
            "housing", resource_type=resource_type
        )


@pytest.mark.parametrize("limit", [0, 101])
def test_search_limit_has_safe_bounds(limit):
    with pytest.raises(ValueError, match="limit"):
        SemanticSearchService(FakeRepository(), FakeBackend()).search("housing", limit=limit)


def test_search_rejects_negative_offset():
    with pytest.raises(ValueError, match="offset"):
        SemanticSearchService(FakeRepository(), FakeBackend()).search("housing", offset=-1)


def test_migration_enables_pgvector_table_cosine_hnsw_and_dimension():
    sql = (Path(__file__).parents[1] / "migrations" / "004_semantic_search.sql").read_text()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS resource_embeddings" in sql
    assert "vector(384)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql


def test_backfill_stats_serializes_timings():
    assert BackfillStats(2, 1, 1, 0.5, 500.0).to_dict()["average_ms_per_embedding"] == 500.0


def test_model_initialization_failure_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ModelInitializationError, match="pinned embedding model"):
        FastEmbedBackend(cache_dir=tmp_path).embed_query("housing")


def test_semantic_cli_contract_parses_query_filter_and_pagination():
    from kagglekit.cli import build_parser

    args = build_parser().parse_args(
        [
            "search", "semantic", "--query", "中国房价", "--resource-type", "dataset",
            "--limit", "7", "--offset", "2",
        ]
    )
    assert (args.search_mode, args.query, args.resource_type, args.limit, args.offset) == (
        "semantic", "中国房价", "dataset", 7, 2,
    )


def test_semantic_backfill_cli_contract_parses_rebuild():
    from kagglekit.cli import build_parser

    args = build_parser().parse_args(["semantic-backfill", "--batch-size", "8", "--rebuild"])
    assert (args.batch_size, args.rebuild) == (8, True)


def test_database_connection_failure_is_not_swallowed(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=fail))
    with pytest.raises(RuntimeError, match="database unavailable"):
        PostgresSemanticSearchRepository("postgresql://invalid")
