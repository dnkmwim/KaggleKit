from dataclasses import replace
from uuid import UUID

import pytest

from kagglekit.hybrid_search import HybridSearchService, reciprocal_rank_fusion
from kagglekit.keyword_search import KeywordSearchResult
from kagglekit.semantic_search import SemanticSearchResult


def keyword(resource_id: int, *, score: float = 1.0, title: str | None = None):
    return KeywordSearchResult(
        UUID(int=resource_id), "kaggle", f"owner/{resource_id}", "dataset",
        title or f"Resource {resource_id}", None, None, [], "owner",
        f"https://example.test/{resource_id}", None, None, None, None, None,
        score, ["query"],
    )


def semantic(resource_id: int, *, score: float = 0.5, title: str | None = None):
    return SemanticSearchResult(
        UUID(int=resource_id), "kaggle", f"owner/{resource_id}", "dataset",
        title or f"Resource {resource_id}", None, None, [], "owner",
        f"https://example.test/{resource_id}", None, None, None, None, None, score,
    )


class FakeKeywordSearcher:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def search(self, queries, *, resource_type, limit, offset):
        self.calls.append((queries, resource_type, limit, offset))
        return self.results[:limit]


class FakeSemanticSearcher:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def search(self, query, *, resource_type, limit, offset):
        self.calls.append((query, resource_type, limit, offset))
        return self.results[:limit]


def test_ranks_start_at_one_and_both_sides_are_added():
    results = reciprocal_rank_fusion(
        [keyword(1), keyword(2)], [semantic(2), semantic(3)], rrf_k=60
    )
    by_id = {row.id.int: row for row in results}
    assert by_id[1].keyword_rank == 1 and by_id[1].semantic_rank is None
    assert by_id[3].keyword_rank is None and by_id[3].semantic_rank == 2
    assert by_id[2].keyword_rank == 2 and by_id[2].semantic_rank == 1
    assert by_id[2].hybrid_score == pytest.approx(1 / 62 + 1 / 61)


def test_deduplicates_duplicate_inputs_and_keeps_first_rank():
    results = reciprocal_rank_fusion(
        [keyword(1), keyword(1, score=999)], [semantic(1), semantic(1, score=999)]
    )
    assert len(results) == 1
    assert (results[0].keyword_rank, results[0].semantic_rank) == (1, 1)


def test_stable_tie_breaker_uses_best_rank_then_id():
    results = reciprocal_rank_fusion(
        [keyword(2), keyword(1)], [semantic(3), semantic(4)], rrf_k=60
    )
    assert [row.id.int for row in results] == [2, 3, 1, 4]


def test_source_scores_are_explanation_only_not_rrf_inputs():
    first = reciprocal_rank_fusion([keyword(1, score=0.001)], [semantic(2, score=0.001)])
    second = reciprocal_rank_fusion([keyword(1, score=999)], [semantic(2, score=999)])
    assert [(row.id, row.hybrid_score) for row in first] == [
        (row.id, row.hybrid_score) for row in second
    ]


def test_configurable_k_changes_formula():
    assert reciprocal_rank_fusion([keyword(1)], [], rrf_k=10)[0].hybrid_score == pytest.approx(1 / 11)
    assert reciprocal_rank_fusion([keyword(1)], [], rrf_k=100)[0].hybrid_score == pytest.approx(1 / 101)


def test_empty_keyword_semantic_and_both_empty_are_supported():
    assert reciprocal_rank_fusion([], [semantic(1)])[0].semantic_rank == 1
    assert reciprocal_rank_fusion([keyword(1)], [])[0].keyword_rank == 1
    assert reciprocal_rank_fusion([], []) == []


def test_service_forwards_filter_and_candidate_depth_and_paginates_after_fusion():
    keyword_searcher = FakeKeywordSearcher([keyword(1), keyword(2), keyword(3)])
    semantic_searcher = FakeSemanticSearcher([semantic(3), semantic(4), semantic(5)])
    service = HybridSearchService(keyword_searcher, semantic_searcher)
    page = service.search(
        ["housing"], "房价", resource_type="dataset", limit=2, offset=1,
        keyword_top_k=3, semantic_top_k=2,
    )
    assert keyword_searcher.calls == [(["housing"], "dataset", 3, 0)]
    assert semantic_searcher.calls == [("房价", "dataset", 2, 0)]
    assert len(page) == 2
    assert len({row.id for row in page}) == len(page)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rrf_k": 0}, "rrf_k"),
        ({"keyword_top_k": 0}, "keyword_top_k"),
        ({"semantic_top_k": 101}, "semantic_top_k"),
        ({"limit": 0}, "limit"),
        ({"offset": -1}, "offset"),
        ({"resource_type": "model"}, "resource_type"),
    ],
)
def test_service_validates_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HybridSearchService(FakeKeywordSearcher(), FakeSemanticSearcher()).search(
            ["query"], "query", **kwargs
        )


def test_result_serialization_contains_explanation_fields():
    row = reciprocal_rank_fusion([keyword(1)], [semantic(1)])[0].to_dict()
    assert row["id"] == str(UUID(int=1))
    assert row["keyword_rank"] == row["semantic_rank"] == 1
    assert row["keyword_score"] == 1.0
    assert row["semantic_score"] == 0.5
    assert row["matched_queries"] == ["query"]


def test_cli_contract_parses_hybrid_parameters():
    from kagglekit.cli import build_parser

    args = build_parser().parse_args(
        [
            "search", "hybrid", "--lexical-query", "housing prices",
            "--lexical-query", "property prices", "--semantic-query", "中国房价",
            "--resource-type", "dataset", "--rrf-k", "10",
            "--keyword-top-k", "50", "--semantic-top-k", "10",
            "--limit", "5", "--offset", "2", "--include-timing",
        ]
    )
    assert args.lexical_queries == ["housing prices", "property prices"]
    assert (args.semantic_query, args.rrf_k, args.keyword_top_k, args.semantic_top_k) == (
        "中国房价", 10, 50, 10
    )
    assert (args.limit, args.offset, args.include_timing) == (5, 2, True)
