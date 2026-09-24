from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID


DEFAULT_LIMIT = 20
MAX_LIMIT = 100
VALID_RESOURCE_TYPES = frozenset({"dataset", "notebook"})


@dataclass(frozen=True)
class KeywordSearchResult:
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
    keyword_score: float
    matched_queries: list[str]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["id"] = str(self.id)
        result["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return result


class KeywordSearchRepository(Protocol):
    def search(
        self,
        lexical_queries: list[str],
        *,
        resource_type: str | None,
        limit: int,
        offset: int,
    ) -> list[KeywordSearchResult]: ...


class PostgresKeywordSearchRepository:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use PostgreSQL") from exc
        self.connection = psycopg.connect(database_url, autocommit=True)

    def close(self) -> None:
        self.connection.close()

    def search(
        self,
        lexical_queries: list[str],
        *,
        resource_type: str | None,
        limit: int,
        offset: int,
    ) -> list[KeywordSearchResult]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                WITH input_queries AS (
                  SELECT lexical_query, query_order,
                         websearch_to_tsquery('english', lexical_query) AS query
                  FROM unnest(%s::text[]) WITH ORDINALITY AS q(lexical_query, query_order)
                ),
                matches AS (
                  SELECT r.id, r.source, r.source_id, r.resource_type, r.title, r.subtitle,
                         r.description, r.tags, r.author, r.url, r.updated_at, r.votes,
                         r.downloads, r.usability_rating, r.size_bytes,
                         iq.lexical_query, iq.query_order,
                         ts_rank_cd(r.search_document, iq.query) AS query_score
                  FROM resources AS r
                  JOIN input_queries AS iq ON r.search_document @@ iq.query
                  WHERE r.status = 'active'
                    AND (%s::text IS NULL OR r.resource_type = %s::text)
                )
                SELECT id, source, source_id, resource_type, title, subtitle, description,
                       tags, author, url, updated_at, votes, downloads, usability_rating,
                       size_bytes, max(query_score)::double precision AS keyword_score,
                       array_agg(lexical_query ORDER BY query_order) AS matched_queries
                FROM matches
                GROUP BY id, source, source_id, resource_type, title, subtitle, description,
                         tags, author, url, updated_at, votes, downloads, usability_rating,
                         size_bytes
                ORDER BY keyword_score DESC, id ASC
                LIMIT %s OFFSET %s
                """,
                (lexical_queries, resource_type, resource_type, limit, offset),
            )
            return [KeywordSearchResult(*row) for row in cursor.fetchall()]


class KeywordSearchService:
    def __init__(self, repository: KeywordSearchRepository) -> None:
        self.repository = repository

    def search(
        self,
        lexical_queries: list[str],
        *,
        resource_type: str | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[KeywordSearchResult]:
        queries = self._validate_queries(lexical_queries)
        if resource_type is not None and resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError("resource_type must be dataset or notebook")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return self.repository.search(
            queries, resource_type=resource_type, limit=limit, offset=offset
        )

    @staticmethod
    def _validate_queries(lexical_queries: list[str]) -> list[str]:
        if not lexical_queries:
            raise ValueError("at least one lexical query is required")
        queries: list[str] = []
        seen: set[str] = set()
        for query in lexical_queries:
            normalized = query.strip()
            if not normalized or not any(character.isalnum() for character in normalized):
                raise ValueError("lexical queries must contain a letter or number")
            if normalized not in seen:
                seen.add(normalized)
                queries.append(normalized)
        return queries
