from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Iterable

from .hybrid_search import reciprocal_rank_fusion


VALID_LABELS = frozenset({"Relevant", "Partially Relevant", "Irrelevant"})


def validate_query_fixture(queries: list[dict[str, object]]) -> None:
    if len(queries) != 24:
        raise ValueError("hybrid evaluation fixture must contain exactly 24 queries")
    ids = [item.get("query_id") for item in queries]
    if len(set(ids)) != 24:
        raise ValueError("query_id values must be unique")
    categories = Counter(item.get("category") for item in queries)
    if len(categories) != 6 or set(categories.values()) != {4}:
        raise ValueError("fixture must contain six categories with four queries each")
    for item in queries:
        if not isinstance(item.get("raw_query"), str):
            raise ValueError("raw_query must be a string")
        if not isinstance(item.get("semantic_query"), str):
            raise ValueError("semantic_query must be a string")
        lexical = item.get("lexical_queries")
        if not isinstance(lexical, list) or not lexical or not all(
            isinstance(value, str) for value in lexical
        ):
            raise ValueError("lexical_queries must be a non-empty string list")


def compute_metrics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    labelled = [row for row in rows if row.get("relevance_label") in VALID_LABELS]
    grouped: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in labelled:
        grouped.setdefault((row.get("query_id"), row.get("method")), []).append(row)

    by_method: dict[str, dict[str, int | float]] = {}
    for method in ("keyword", "semantic", "hybrid"):
        groups = [
            sorted(items, key=lambda row: int(row["rank"]))
            for (query_id, row_method), items in grouped.items()
            if row_method == method
        ]
        top_fives = [items[:5] for items in groups]
        by_method[method] = {
            "labelled_queries": len(groups),
            "top_1_relevant_rate": (
                sum(bool(items) and items[0]["relevance_label"] == "Relevant" for items in groups)
                / len(groups)
                if groups else 0.0
            ),
            "top_5_relevant_count": sum(
                row["relevance_label"] == "Relevant" for items in top_fives for row in items
            ),
            "top_5_relevant_or_partial_count": sum(
                row["relevance_label"] in {"Relevant", "Partially Relevant"}
                for items in top_fives for row in items
            ),
            "top_5_irrelevant_count": sum(
                row["relevance_label"] == "Irrelevant" for items in top_fives for row in items
            ),
        }

    by_query: dict[object, dict[str, set[object]]] = {}
    labels: dict[tuple[object, object], object] = {}
    for row in labelled:
        query_id = row.get("query_id")
        method = str(row.get("method"))
        resource_id = row.get("resource_id")
        by_query.setdefault(query_id, {}).setdefault(method, set()).add(resource_id)
        labels[(query_id, resource_id)] = row.get("relevance_label")

    overlaps = {"keyword_semantic": 0, "keyword_hybrid": 0, "semantic_hybrid": 0}
    semantic_only_relevant_in_hybrid = 0
    keyword_only_relevant_in_hybrid = 0
    for query_id, methods in by_query.items():
        keyword = methods.get("keyword", set())
        semantic = methods.get("semantic", set())
        hybrid = methods.get("hybrid", set())
        overlaps["keyword_semantic"] += len(keyword & semantic)
        overlaps["keyword_hybrid"] += len(keyword & hybrid)
        overlaps["semantic_hybrid"] += len(semantic & hybrid)
        semantic_only_relevant_in_hybrid += sum(
            labels.get((query_id, resource_id)) == "Relevant"
            for resource_id in (semantic - keyword) & hybrid
        )
        keyword_only_relevant_in_hybrid += sum(
            labels.get((query_id, resource_id)) == "Relevant"
            for resource_id in (keyword - semantic) & hybrid
        )

    return {
        "by_method": by_method,
        "top_5_overlap": overlaps,
        "hybrid_semantic_only_relevant_count": semantic_only_relevant_in_hybrid,
        "hybrid_keyword_only_relevant_count": keyword_only_relevant_in_hybrid,
    }


def run_evaluation(
    queries: list[dict[str, object]],
    keyword_searcher,
    semantic_searcher,
    *,
    resource_type: str | None = "dataset",
    rrf_k: int = 60,
    keyword_top_k: int = 20,
    semantic_top_k: int = 20,
    existing_rows: Iterable[dict[str, object]] = (),
) -> dict[str, object]:
    validate_query_fixture(queries)
    existing_labels = {
        (row.get("query_id"), row.get("method"), row.get("resource_id")): (
            row.get("relevance_label"), row.get("notes")
        )
        for row in existing_rows
        if row.get("relevance_label") in VALID_LABELS
    }
    rows: list[dict[str, object]] = []
    timings: list[dict[str, float | str]] = []
    for query in queries:
        started_total = time.perf_counter()
        started = time.perf_counter()
        keyword_results = keyword_searcher.search(
            query["lexical_queries"], resource_type=resource_type,
            limit=keyword_top_k, offset=0,
        )
        keyword_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        semantic_results = semantic_searcher.search(
            query["semantic_query"], resource_type=resource_type,
            limit=semantic_top_k, offset=0,
        )
        semantic_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        hybrid_results = reciprocal_rank_fusion(
            keyword_results, semantic_results, rrf_k=rrf_k
        )
        fusion_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - started_total) * 1000

        for method, results in (
            ("keyword", keyword_results),
            ("semantic", semantic_results),
            ("hybrid", hybrid_results),
        ):
            for rank, result in enumerate(results[:5], start=1):
                key = (query["query_id"], method, str(result.id))
                label, notes = existing_labels.get(key, (None, ""))
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "category": query["category"],
                        "raw_query": query["raw_query"],
                        "lexical_queries": query["lexical_queries"],
                        "semantic_query": query["semantic_query"],
                        "method": method,
                        "rank": rank,
                        "resource_id": str(result.id),
                        "title": result.title,
                        "relevance_label": label,
                        "notes": notes,
                    }
                )
        timings.append(
            {
                "query_id": str(query["query_id"]),
                "keyword_ms": round(keyword_ms, 3),
                "semantic_ms": round(semantic_ms, 3),
                "fusion_ms": round(fusion_ms, 3),
                "total_ms": round(total_ms, 3),
            }
        )
    return {
        "status": "pending_human_review",
        "parameters": {
            "rrf_k": rrf_k,
            "keyword_top_k": keyword_top_k,
            "semantic_top_k": semantic_top_k,
            "resource_type": resource_type,
        },
        "rows": rows,
        "timings": timings,
        "metrics": compute_metrics(rows),
    }


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
