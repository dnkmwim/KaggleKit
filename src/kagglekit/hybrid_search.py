from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from .keyword_search import KeywordSearchResult
from .semantic_search import SemanticSearchResult


DEFAULT_RRF_K = 60
DEFAULT_KEYWORD_TOP_K = 20
DEFAULT_SEMANTIC_TOP_K = 20
MAX_LIMIT = 100
VALID_RESOURCE_TYPES = frozenset({"dataset", "notebook"})


@dataclass(frozen=True)
class HybridSearchResult:
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
    hybrid_score: float
    keyword_rank: int | None
    semantic_rank: int | None
    keyword_score: float | None
    semantic_score: float | None
    matched_queries: list[str]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["id"] = str(self.id)
        result["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return result


@dataclass(frozen=True)
class HybridSearchTiming:
    keyword_ms: float
    semantic_ms: float
    fusion_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HybridSearchResponse:
    results: list[HybridSearchResult]
    timing: HybridSearchTiming


class KeywordSearcher(Protocol):
    def search(
        self, lexical_queries: list[str], *, resource_type: str | None,
        limit: int, offset: int,
    ) -> list[KeywordSearchResult]: ...


class SemanticSearcher(Protocol):
    def search(
        self, query: str, *, resource_type: str | None,
        limit: int, offset: int,
    ) -> list[SemanticSearchResult]: ...


def reciprocal_rank_fusion(
    keyword_results: list[KeywordSearchResult],
    semantic_results: list[SemanticSearchResult],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    merged: dict[UUID, dict[str, object]] = {}
    for rank, result in enumerate(keyword_results, start=1):
        entry = merged.setdefault(
            result.id,
            {"resource": result, "keyword_rank": None, "semantic_rank": None,
             "keyword_score": None, "semantic_score": None, "matched_queries": []},
        )
        if entry["keyword_rank"] is None:
            entry["keyword_rank"] = rank
            entry["keyword_score"] = result.keyword_score
            entry["matched_queries"] = list(result.matched_queries)

    for rank, result in enumerate(semantic_results, start=1):
        entry = merged.setdefault(
            result.id,
            {"resource": result, "keyword_rank": None, "semantic_rank": None,
             "keyword_score": None, "semantic_score": None, "matched_queries": []},
        )
        if entry["semantic_rank"] is None:
            entry["semantic_rank"] = rank
            entry["semantic_score"] = result.semantic_score

    fused: list[HybridSearchResult] = []
    for entry in merged.values():
        resource = entry["resource"]
        keyword_rank = entry["keyword_rank"]
        semantic_rank = entry["semantic_rank"]
        hybrid_score = sum(
            1.0 / (rrf_k + rank)
            for rank in (keyword_rank, semantic_rank)
            if isinstance(rank, int)
        )
        fused.append(
            HybridSearchResult(
                id=resource.id,
                source=resource.source,
                source_id=resource.source_id,
                resource_type=resource.resource_type,
                title=resource.title,
                subtitle=resource.subtitle,
                description=resource.description,
                tags=list(resource.tags),
                author=resource.author,
                url=resource.url,
                updated_at=resource.updated_at,
                votes=resource.votes,
                downloads=resource.downloads,
                usability_rating=resource.usability_rating,
                size_bytes=resource.size_bytes,
                hybrid_score=hybrid_score,
                keyword_rank=keyword_rank if isinstance(keyword_rank, int) else None,
                semantic_rank=semantic_rank if isinstance(semantic_rank, int) else None,
                keyword_score=(
                    float(entry["keyword_score"])
                    if entry["keyword_score"] is not None else None
                ),
                semantic_score=(
                    float(entry["semantic_score"])
                    if entry["semantic_score"] is not None else None
                ),
                matched_queries=list(entry["matched_queries"]),
            )
        )

    # Source scores are intentionally absent from the sort key. The best source
    # rank and UUID make ties deterministic without changing the RRF formula.
    return sorted(
        fused,
        key=lambda item: (
            -item.hybrid_score,
            min(rank for rank in (item.keyword_rank, item.semantic_rank) if rank is not None),
            str(item.id),
        ),
    )


class HybridSearchService:
    def __init__(self, keyword_searcher: KeywordSearcher, semantic_searcher: SemanticSearcher) -> None:
        self.keyword_searcher = keyword_searcher
        self.semantic_searcher = semantic_searcher

    def search(
        self,
        lexical_queries: list[str],
        semantic_query: str,
        *,
        resource_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
        rrf_k: int = DEFAULT_RRF_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
    ) -> list[HybridSearchResult]:
        return self.search_with_timing(
            lexical_queries,
            semantic_query,
            resource_type=resource_type,
            limit=limit,
            offset=offset,
            rrf_k=rrf_k,
            keyword_top_k=keyword_top_k,
            semantic_top_k=semantic_top_k,
        ).results

    def search_with_timing(
        self,
        lexical_queries: list[str],
        semantic_query: str,
        *,
        resource_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
        rrf_k: int = DEFAULT_RRF_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
    ) -> HybridSearchResponse:
        self._validate(
            resource_type=resource_type,
            limit=limit,
            offset=offset,
            rrf_k=rrf_k,
            keyword_top_k=keyword_top_k,
            semantic_top_k=semantic_top_k,
        )
        total_started = time.perf_counter()
        started = time.perf_counter()
        keyword_results = self.keyword_searcher.search(
            lexical_queries, resource_type=resource_type, limit=keyword_top_k, offset=0
        )
        keyword_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        semantic_results = self.semantic_searcher.search(
            semantic_query, resource_type=resource_type, limit=semantic_top_k, offset=0
        )
        semantic_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(keyword_results, semantic_results, rrf_k=rrf_k)
        results = fused[offset : offset + limit]
        fusion_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        return HybridSearchResponse(
            results=results,
            timing=HybridSearchTiming(
                keyword_ms=round(keyword_ms, 3),
                semantic_ms=round(semantic_ms, 3),
                fusion_ms=round(fusion_ms, 3),
                total_ms=round(total_ms, 3),
            ),
        )

    @staticmethod
    def _validate(
        *, resource_type: str | None, limit: int, offset: int, rrf_k: int,
        keyword_top_k: int, semantic_top_k: int,
    ) -> None:
        if resource_type is not None and resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError("resource_type must be dataset or notebook")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        for name, value in (
            ("keyword_top_k", keyword_top_k), ("semantic_top_k", semantic_top_k)
        ):
            if not 1 <= value <= MAX_LIMIT:
                raise ValueError(f"{name} must be between 1 and {MAX_LIMIT}")
