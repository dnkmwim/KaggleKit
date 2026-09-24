import json
from pathlib import Path

from kagglekit.hybrid_evaluation import compute_metrics, run_evaluation, validate_query_fixture
from test_hybrid_search import FakeKeywordSearcher, FakeSemanticSearcher, keyword, semantic


ROOT = Path(__file__).parents[1]


def test_24_query_fixture_has_six_balanced_categories_and_required_fields():
    queries = json.loads(
        (ROOT / "evaluation" / "hybrid_search_queries.json").read_text(encoding="utf-8")
    )
    validate_query_fixture(queries)


def test_review_template_does_not_pretend_to_have_human_labels():
    template = json.loads(
        (ROOT / "evaluation" / "hybrid_search_results.json").read_text(encoding="utf-8")
    )
    assert template["status"] == "pending_human_review"
    assert all(row.get("relevance_label") is None for row in template["rows"])


def test_metrics_use_only_human_labelled_rows_and_report_overlap():
    rows = [
        {"query_id": "Q", "method": "keyword", "rank": 1, "resource_id": "a", "relevance_label": "Relevant"},
        {"query_id": "Q", "method": "semantic", "rank": 1, "resource_id": "b", "relevance_label": "Relevant"},
        {"query_id": "Q", "method": "hybrid", "rank": 1, "resource_id": "a", "relevance_label": "Relevant"},
        {"query_id": "Q", "method": "hybrid", "rank": 2, "resource_id": "b", "relevance_label": "Relevant"},
        {"query_id": "Q", "method": "hybrid", "rank": 3, "resource_id": "c", "relevance_label": None},
    ]
    metrics = compute_metrics(rows)
    assert metrics["by_method"]["hybrid"]["top_1_relevant_rate"] == 1.0
    assert metrics["by_method"]["hybrid"]["top_5_relevant_count"] == 2
    assert metrics["top_5_overlap"]["keyword_hybrid"] == 1
    assert metrics["hybrid_semantic_only_relevant_count"] == 1
    assert metrics["hybrid_keyword_only_relevant_count"] == 1


def test_fixture_is_runnable_and_preserves_existing_human_labels():
    queries = json.loads(
        (ROOT / "evaluation" / "hybrid_search_queries.json").read_text(encoding="utf-8")
    )
    previous = [
        {
            "query_id": "A01", "method": "keyword", "resource_id": str(keyword(1).id),
            "relevance_label": "Relevant", "notes": "human checked",
        }
    ]
    report = run_evaluation(
        queries, FakeKeywordSearcher([keyword(1)]), FakeSemanticSearcher([semantic(2)]),
        existing_rows=previous,
    )
    assert len(report["rows"]) == 24 * 4
    labelled = next(
        row for row in report["rows"]
        if row["query_id"] == "A01" and row["method"] == "keyword"
    )
    assert (labelled["relevance_label"], labelled["notes"]) == (
        "Relevant", "human checked"
    )


def test_hybrid_evaluation_cli_contract():
    from kagglekit.cli import build_parser

    args = build_parser().parse_args(
        ["hybrid-evaluate", "--rrf-k", "100", "--keyword-top-k", "10"]
    )
    assert (args.command, args.rrf_k, args.keyword_top_k, args.semantic_top_k) == (
        "hybrid-evaluate", 100, 10, 20
    )
