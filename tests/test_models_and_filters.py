from datetime import datetime, timezone

from kagglekit.dataset_sync.filters import data_science_negative, tag_page_is_valid
from kagglekit.dataset_sync.models import (
    DatasetMetadata,
    VersionState,
    merge_non_empty,
    version_changed,
)


def dataset(**overrides):
    values = {"ref": "owner/item", "title": "Item", "url": "https://example.test/item"}
    values.update(overrides)
    return DatasetMetadata(**values)


def test_tag_exact_validation_and_fallback_rejection():
    assert tag_page_is_valid([dataset(tags=["classification"])], "classification")
    assert not tag_page_is_valid([dataset(tags=["classification data"])], "classification")
    assert not tag_page_is_valid([dataset(tags=["finance"])], "classification")


def test_data_science_negative_filter_is_narrow_and_normalized():
    assert data_science_negative(dataset(title="DATA-SCIENCE: Cheat—Sheets")) == "cheat sheets"
    assert data_science_negative(dataset(title="Data Science Books Collection")) == "data science books"
    assert data_science_negative(dataset(title="Rare Book Sales")) is None
    assert data_science_negative(dataset(title="Climate", description="data science jobs")) is None


def test_list_detail_non_empty_merge_and_null_protection():
    baseline = dataset(description="list description", tags=["classification"], author="owner")
    detail = dataset(title="", url="", description="detail description", tags=[], author=None)
    merged = merge_non_empty(baseline, detail)
    assert merged.description == "detail description"
    assert merged.tags == ["classification"]
    assert merged.author == "owner"
    assert merged.url == baseline.url


def test_version_unchanged_skips_and_changed_requests_detail():
    when = datetime(2026, 9, 17, tzinfo=timezone.utc)
    metadata = dataset(last_updated=when, current_version_number=3)
    assert not version_changed(metadata, VersionState(when, 3, False))
    assert version_changed(metadata, VersionState(when, 2, False))
    assert version_changed(metadata, VersionState(when, 3, True))
    assert version_changed(metadata, None)

