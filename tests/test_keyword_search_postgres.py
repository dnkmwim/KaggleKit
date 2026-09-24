from __future__ import annotations

import os
from uuid import UUID

import pytest

from kagglekit.keyword_search import KeywordSearchService, PostgresKeywordSearchRepository


DATABASE_URL = os.getenv("AGENTDS2_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="AGENTDS2_TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


ROWS = (
    (
        "10000000-0000-0000-0000-000000000001",
        "title-hit",
        "dataset",
        "active",
        "quasarorchid amberquasar cobaltnebula",
        None,
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000002",
        "description-hit",
        "dataset",
        "active",
        "Control resource",
        None,
        "quasarorchid descriptiontoken",
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000003",
        "subtitle-hit",
        "dataset",
        "active",
        "Control resource",
        "subtitletoken",
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000004",
        "tags-hit",
        "dataset",
        "active",
        "Control resource",
        None,
        None,
        ["tagtoken", "two word tag"],
    ),
    (
        "10000000-0000-0000-0000-000000000005",
        "inactive-hit",
        "dataset",
        "unavailable",
        "statustoken",
        None,
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000006",
        "deleted-hit",
        "dataset",
        "deleted",
        "statustoken",
        None,
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000007",
        "notebook-hit",
        "notebook",
        "active",
        "filtertoken",
        None,
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000008",
        "dataset-hit",
        "dataset",
        "active",
        "filtertoken paginationtoken",
        None,
        None,
        [],
    ),
    (
        "10000000-0000-0000-0000-000000000009",
        "pagination-hit",
        "dataset",
        "active",
        "paginationtoken",
        None,
        None,
        [],
    ),
)


@pytest.fixture(scope="module")
def search():
    import psycopg

    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM resources WHERE source = 'keyword-search-test'")
        cursor.executemany(
            """
            INSERT INTO resources
              (id, source, source_id, resource_type, status, title, subtitle,
               description, tags, url)
            VALUES (%s, 'keyword-search-test', %s, %s, %s, %s, %s, %s, %s,
                    'https://example.test/' || %s)
            """,
            [
                (
                    UUID(row[0]),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[1],
                )
                for row in ROWS
            ],
        )
    repository = PostgresKeywordSearchRepository(DATABASE_URL)
    yield KeywordSearchService(repository)
    repository.close()
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM resources WHERE source = 'keyword-search-test'")
    connection.close()


def test_search_document_contains_all_fields_and_handles_nulls_and_text_arrays(search):
    assert [item.source_id for item in search.search(["quasarorchid"])] == ["title-hit", "description-hit"]
    assert [item.source_id for item in search.search(["subtitletoken"])] == ["subtitle-hit"]
    assert [item.source_id for item in search.search(["descriptiontoken"])] == ["description-hit"]
    assert [item.source_id for item in search.search(["tagtoken"])] == ["tags-hit"]
    assert [item.source_id for item in search.search(["two word tag"])] == ["tags-hit"]


def test_title_weight_ranks_above_description(search):
    results = search.search(["quasarorchid"])
    assert results[0].source_id == "title-hit"
    assert results[0].keyword_score > results[1].keyword_score


def test_only_active_resources_are_returned(search):
    assert search.search(["statustoken"]) == []


def test_resource_type_filter_is_applied(search):
    assert [item.source_id for item in search.search(["filtertoken"], resource_type="dataset")] == ["dataset-hit"]
    assert [item.source_id for item in search.search(["filtertoken"], resource_type="notebook")] == ["notebook-hit"]


def test_multiple_queries_or_merge_deduplicates_and_records_matches(search):
    results = search.search(["amberquasar", "cobaltnebula"])
    assert len(results) == 1
    assert results[0].source_id == "title-hit"
    assert results[0].matched_queries == ["amberquasar", "cobaltnebula"]


def test_score_order_pagination_no_result_and_injection_safety(search):
    results = search.search(["paginationtoken"], limit=1, offset=0)
    next_page = search.search(["paginationtoken"], limit=1, offset=1)
    assert len(results) == len(next_page) == 1
    assert results[0].id != next_page[0].id
    assert search.search(["notfoundtokenxyzzy"]) == []
    assert search.search(["quasarorchid'); DROP TABLE resources; --"]) == []
    assert search.search(["quasarorchid"])[0].keyword_score >= search.search(["quasarorchid"])[1].keyword_score
