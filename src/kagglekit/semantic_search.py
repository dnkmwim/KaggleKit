from __future__ import annotations

import hashlib
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence
from uuid import UUID


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REPOSITORY = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
MODEL_DIMENSION = 384
DESCRIPTION_MAX_CHARS = 1200
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
VALID_RESOURCE_TYPES = frozenset({"dataset", "notebook"})


class SemanticSearchError(RuntimeError):
    pass


class ModelInitializationError(SemanticSearchError):
    pass


class ModelMismatchError(SemanticSearchError):
    pass


class EmbeddingDimensionError(SemanticSearchError):
    pass


def normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def build_semantic_document(
    *, title: object | None, subtitle: object | None, tags: Sequence[object] | None,
    description: object | None,
) -> str:
    normalized_tags = [normalize_text(tag) for tag in (tags or [])]
    fields = (
        ("title", normalize_text(title)),
        ("subtitle", normalize_text(subtitle)),
        ("tags", ", ".join(tag for tag in normalized_tags if tag)),
        ("description", normalize_text(description)[:DESCRIPTION_MAX_CHARS]),
    )
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def semantic_content_hash(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def vector_literal(vector: Sequence[float]) -> str:
    if len(vector) != MODEL_DIMENSION:
        raise EmbeddingDimensionError(
            f"expected {MODEL_DIMENSION} dimensions, got {len(vector)}"
        )
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingDimensionError("embedding contains a non-finite value")
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


@dataclass(frozen=True)
class EmbeddingCandidate:
    id: UUID
    title: str
    subtitle: str | None
    tags: list[str]
    description: str | None
    existing_hash: str | None

    def document(self) -> str:
        return build_semantic_document(
            title=self.title,
            subtitle=self.subtitle,
            tags=self.tags,
            description=self.description,
        )


@dataclass(frozen=True)
class BackfillStats:
    total: int
    embedded: int
    skipped: int
    elapsed_seconds: float
    average_ms_per_embedding: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticSearchResult:
    id: UUID
    source: str
    source_id: str
    resource_type: str
    title: str
    subtitle: str | None
    description: str | None
    tags: list[str]
    author: str | None
    url: str
    updated_at: datetime | None
    votes: int | None
    downloads: int | None
    usability_rating: float | None
    size_bytes: int | None
    semantic_score: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["id"] = str(self.id)
        result["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return result


class EmbeddingBackend(Protocol):
    model_name: str
    model_revision: str
    dimension: int

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FastEmbedBackend:
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    dimension = MODEL_DIMENSION

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(
            cache_dir
            or os.getenv("AGENTDS2_MODEL_CACHE", "")
            or Path.home() / ".cache" / "kagglekit" / "models"
        )
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=str(self.cache_dir),
                allow_patterns=(
                    "config.json",
                    "model_optimized.onnx",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "unigram.json",
                ),
            )
            self._model = TextEmbedding(
                model_name=MODEL_NAME,
                specific_model_path=model_path,
            )
            return self._model
        except Exception as exc:
            raise ModelInitializationError(
                f"could not initialize pinned embedding model "
                f"{MODEL_NAME}@{MODEL_REVISION}: {exc}"
            ) from exc

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        vectors = [vector.tolist() for vector in self._get_model().embed(list(documents))]
        self._validate(vectors)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        vectors = [vector.tolist() for vector in self._get_model().query_embed([query])]
        self._validate(vectors)
        return vectors[0]

    def _validate(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            vector_literal(vector)


class SemanticSearchRepository(Protocol):
    def ensure_model_compatible(self, backend: EmbeddingBackend, *, rebuild: bool) -> None: ...

    def embedding_candidates(self) -> list[EmbeddingCandidate]: ...

    def upsert_embeddings(
        self, rows: Sequence[tuple[UUID, str, Sequence[float]]], backend: EmbeddingBackend
    ) -> None: ...

    def search(
        self, vector: Sequence[float], backend: EmbeddingBackend, *,
        resource_type: str | None, limit: int, offset: int,
    ) -> list[SemanticSearchResult]: ...


class PostgresSemanticSearchRepository:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use PostgreSQL") from exc
        self.connection = psycopg.connect(database_url, autocommit=True)

    def close(self) -> None:
        self.connection.close()

    def ensure_model_compatible(
        self, backend: EmbeddingBackend, *, rebuild: bool = False
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT model_name, model_revision, embedding_dimension "
                "FROM resource_embeddings"
            )
            models = set(cursor.fetchall())
            expected = (backend.model_name, backend.model_revision, backend.dimension)
            incompatible = models - {expected}
            if incompatible and not rebuild:
                raise ModelMismatchError(
                    "stored embeddings use a different model; rerun semantic-backfill --rebuild"
                )
            if rebuild:
                cursor.execute("DELETE FROM resource_embeddings")

    def embedding_candidates(self) -> list[EmbeddingCandidate]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.id, r.title, r.subtitle, r.tags, r.description, e.content_hash
                FROM resources AS r
                LEFT JOIN resource_embeddings AS e ON e.resource_id = r.id
                WHERE r.status = 'active' AND r.resource_type = 'dataset'
                ORDER BY r.id
                """
            )
            return [EmbeddingCandidate(*row) for row in cursor.fetchall()]

    def upsert_embeddings(
        self,
        rows: Sequence[tuple[UUID, str, Sequence[float]]],
        backend: EmbeddingBackend,
    ) -> None:
        if not rows:
            return
        values = [
            (
                resource_id,
                backend.model_name,
                backend.model_revision,
                backend.dimension,
                content_hash,
                vector_literal(vector),
            )
            for resource_id, content_hash, vector in rows
        ]
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO resource_embeddings
                  (resource_id, model_name, model_revision, embedding_dimension,
                   content_hash, embedding, embedded_at)
                VALUES (%s, %s, %s, %s, %s, %s::vector, now())
                ON CONFLICT (resource_id) DO UPDATE SET
                  model_name = EXCLUDED.model_name,
                  model_revision = EXCLUDED.model_revision,
                  embedding_dimension = EXCLUDED.embedding_dimension,
                  content_hash = EXCLUDED.content_hash,
                  embedding = EXCLUDED.embedding,
                  embedded_at = now()
                """,
                values,
            )

    def search(
        self,
        vector: Sequence[float],
        backend: EmbeddingBackend,
        *,
        resource_type: str | None,
        limit: int,
        offset: int,
    ) -> list[SemanticSearchResult]:
        self.ensure_model_compatible(backend, rebuild=False)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                WITH query_vector AS (SELECT %s::vector AS embedding)
                SELECT r.id, r.source, r.source_id, r.resource_type, r.title, r.subtitle,
                       r.description, r.tags, r.author, r.url, r.updated_at, r.votes,
                       r.downloads, r.usability_rating, r.size_bytes,
                       (1 - (e.embedding <=> q.embedding))::double precision AS semantic_score
                FROM resource_embeddings AS e
                JOIN resources AS r ON r.id = e.resource_id
                CROSS JOIN query_vector AS q
                WHERE r.status = 'active'
                  AND e.model_name = %s AND e.model_revision = %s
                  AND e.embedding_dimension = %s
                  AND (%s::text IS NULL OR r.resource_type = %s::text)
                ORDER BY e.embedding <=> q.embedding, r.id ASC
                LIMIT %s OFFSET %s
                """,
                (
                    vector_literal(vector), backend.model_name, backend.model_revision,
                    backend.dimension, resource_type, resource_type, limit, offset,
                ),
            )
            return [SemanticSearchResult(*row) for row in cursor.fetchall()]


class SemanticBackfillService:
    def __init__(
        self, repository: SemanticSearchRepository, backend: EmbeddingBackend,
        *, batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.repository = repository
        self.backend = backend
        self.batch_size = batch_size

    def run(self, *, rebuild: bool = False) -> BackfillStats:
        started = time.perf_counter()
        self.repository.ensure_model_compatible(self.backend, rebuild=rebuild)
        candidates = self.repository.embedding_candidates()
        pending: list[tuple[EmbeddingCandidate, str, str]] = []
        for candidate in candidates:
            document = candidate.document()
            content_hash = semantic_content_hash(document)
            if candidate.existing_hash == content_hash:
                continue
            pending.append((candidate, document, content_hash))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.backend.embed_documents([item[1] for item in batch])
            if len(vectors) != len(batch):
                raise SemanticSearchError("embedding backend returned the wrong batch size")
            self.repository.upsert_embeddings(
                [
                    (candidate.id, content_hash, vector)
                    for (candidate, _document, content_hash), vector in zip(batch, vectors)
                ],
                self.backend,
            )

        elapsed = time.perf_counter() - started
        embedded = len(pending)
        return BackfillStats(
            total=len(candidates),
            embedded=embedded,
            skipped=len(candidates) - embedded,
            elapsed_seconds=round(elapsed, 6),
            average_ms_per_embedding=round(elapsed * 1000 / embedded, 3) if embedded else 0.0,
        )


class SemanticSearchService:
    def __init__(
        self, repository: SemanticSearchRepository, backend: EmbeddingBackend
    ) -> None:
        self.repository = repository
        self.backend = backend

    def search(
        self, query: str, *, resource_type: str | None = None,
        limit: int = DEFAULT_LIMIT, offset: int = 0,
    ) -> list[SemanticSearchResult]:
        normalized_query = normalize_text(query)
        if not normalized_query or not any(character.isalnum() for character in normalized_query):
            raise ValueError("semantic query must contain a letter or number")
        if resource_type is not None and resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError("resource_type must be dataset or notebook")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        vector = self.backend.embed_query(normalized_query)
        return self.repository.search(
            vector,
            self.backend,
            resource_type=resource_type,
            limit=limit,
            offset=offset,
        )
