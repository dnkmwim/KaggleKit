from dataclasses import replace
from uuid import UUID

import pytest

from kagglekit.keyword_search import KeywordSearchResult, KeywordSearchService


RESULT = KeywordSearchResult(
    id=UUID("00000000-0000-0000-0000-000000000001"),
    source="kaggle",
    source_id="owner/item",
    resource_type="dataset",
    title="Housing prices",
    subtitle=None,
    description=None,
    tags=["real estate"],
    author="owner",
    url="https://example.test/item",
    updated_at=None,
    votes=1,
    downloads=2,
    usability_rating=0.9,
    size_bytes=3,
    keyword_score=1.0,
    matched_queries=["housing prices"],
)


class FakeRepository:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, lexical_queries, *, resource_type, limit, offset):
        self.calls.append((lexical_queries, resource_type, limit, offset))
        return self.results[offset : offset + limit]


def test_single_query_is_forwarded_with_defaults():
    repository = FakeRepository([RESULT])
    results = KeywordSearchService(repository).search(["housing prices"])
    assert results == [RESULT]
    assert repository.calls == [(["housing prices"], None, 20, 0)]


def test_multiple_queries_are_or_alternatives_and_duplicates_are_removed():
    repository = FakeRepository([replace(RESULT, matched_queries=["housing prices", "property prices"])])
    results = KeywordSearchService(repository).search(
        ["housing prices", "property prices", "housing prices"]
    )
    assert len(results) == 1
    assert repository.calls[0][0] == ["housing prices", "property prices"]


def test_resource_type_and_pagination_are_forwarded():
    repository = FakeRepository([RESULT, replace(RESULT, source_id="owner/other")])
    KeywordSearchService(repository).search(
        ["prices"], resource_type="dataset", limit=1, offset=1
    )
    assert repository.calls == [(["prices"], "dataset", 1, 1)]


@pytest.mark.parametrize("queries", [[], [""], ["  "], ["---"]])
def test_empty_or_invalid_queries_are_rejected(queries):
    with pytest.raises(ValueError, match="query|queries"):
        KeywordSearchService(FakeRepository()).search(queries)


@pytest.mark.parametrize("resource_type", ["Dataset", "model", ""])
def test_invalid_resource_type_is_rejected(resource_type):
    with pytest.raises(ValueError, match="resource_type"):
        KeywordSearchService(FakeRepository()).search(["classification"], resource_type=resource_type)


@pytest.mark.parametrize("limit", [0, 101])
def test_limit_has_safe_bounds(limit):
    with pytest.raises(ValueError, match="limit"):
        KeywordSearchService(FakeRepository()).search(["classification"], limit=limit)


def test_negative_offset_is_rejected():
    with pytest.raises(ValueError, match="offset"):
        KeywordSearchService(FakeRepository()).search(["classification"], offset=-1)


def test_sql_injection_like_input_is_data_not_sql():
    repository = FakeRepository([])
    query = "classification'); DROP TABLE resources; --"
    assert KeywordSearchService(repository).search([query]) == []
    assert repository.calls[0][0] == [query]


def test_result_serialization_handles_uuid_and_nulls():
    serialized = RESULT.to_dict()
    assert serialized["id"] == "00000000-0000-0000-0000-000000000001"
    assert serialized["updated_at"] is None
    assert serialized["subtitle"] is None
