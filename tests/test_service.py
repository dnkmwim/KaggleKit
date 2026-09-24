from datetime import datetime, timedelta, timezone

from kagglekit.dataset_sync.config import SyncConfig
from kagglekit.dataset_sync.kaggle import TransientKaggleError
from kagglekit.dataset_sync.models import DatasetMetadata, PageResult, VersionState
from kagglekit.dataset_sync.repository import Checkpoint, InMemoryRepository
from kagglekit.dataset_sync.service import DatasetSyncService
from kagglekit.dataset_sync.sources import Source


NOW = datetime(2026, 9, 17, tzinfo=timezone.utc)


def item(ref="owner/item", *, title="Item", tags=None, updated=NOW, version=1, subtitle=None):
    return DatasetMetadata(
        ref=ref,
        title=title,
        url=f"https://www.kaggle.com/datasets/{ref}",
        tags=tags or [],
        last_updated=updated,
        current_version_number=version,
        subtitle=subtitle,
    )


class FakeAdapter:
    def __init__(self, pages, details=None):
        self.pages = pages
        self.details = details or {}
        self.list_calls = []
        self.detail_calls = []

    def list_datasets(self, source, sort, page):
        self.list_calls.append((source.key, sort, page))
        value = self.pages.get((source.key, sort, page), [])
        if isinstance(value, Exception):
            raise value
        return PageResult(value, 200, 5)

    def dataset_detail(self, ref):
        self.detail_calls.append(ref)
        value = self.details.get(ref, item(ref, title=""))
        if isinstance(value, Exception):
            raise value
        return value, 0


def service(adapter, repository=None, *, page_limit=20, dry_run=False):
    events = []
    repo = repository or InMemoryRepository()
    sync = DatasetSyncService(
        adapter,
        repo,
        SyncConfig(bootstrap_page_limit=page_limit, request_delay_seconds=0),
        event_sink=events.append,
        dry_run=dry_run,
    )
    return sync, repo, events


def test_ref_dedup_hit_sources_merge_and_source_suppression_not_global_delete():
    shared = item("owner/shared", title="Data Science Jobs", tags=["classification"])
    pages = {
        ("query:data science", "updated", 1): [shared],
        ("tag:classification", "updated", 1): [shared],
    }
    adapter = FakeAdapter(pages)
    sync, repo, events = service(adapter)
    result = sync.run(
        "incremental", (Source("query", "data science"), Source("tag", "classification"))
    )
    assert result.candidates == 1
    ready = [event for event in events if event["event"] == "candidate_ready"]
    assert ready[0]["hit_sources"] == ["tag:classification"]
    suppressed = [event for event in events if event["event"] == "source_suppressed"]
    assert suppressed[0]["matched_negative_term"] == "jobs"
    assert "owner/shared" in repo.resources


def test_duplicate_ref_merges_all_non_suppressed_hit_sources():
    shared = item("owner/shared", tags=["classification"])
    pages = {
        ("query:machine learning", "updated", 1): [shared],
        ("tag:classification", "updated", 1): [shared],
    }
    sync, _, events = service(FakeAdapter(pages))
    sync.run("incremental", (Source("query", "machine learning"), Source("tag", "classification")))
    ready = [event for event in events if event["event"] == "candidate_ready"]
    assert ready[0]["hit_sources"] == ["query:machine learning", "tag:classification"]


def test_invalid_tag_fallback_page_is_rejected():
    pages = {("tag:classification", "updated", 1): [item(tags=["finance"])]}
    sync, repo, events = service(FakeAdapter(pages))
    result = sync.run("incremental", (Source("tag", "classification"),))
    assert result.candidates == 0
    assert not repo.resources
    assert repo.checkpoints[-1].status == "invalid_source"
    assert any(event["event"] == "tag_validation_failed" for event in events)


def test_detail_unchanged_skipped_and_changed_requested():
    repo = InMemoryRepository()
    unchanged = item("owner/same", updated=NOW, version=1)
    changed = item("owner/changed", updated=NOW, version=2)
    for metadata in (unchanged, changed):
        repo.upsert_resource(metadata)
    repo.version_states["owner/same"] = VersionState(NOW, 1, False)
    repo.version_states["owner/changed"] = VersionState(NOW, 1, False)
    pages = {("query:forecasting", "updated", 1): [unchanged, changed]}
    adapter = FakeAdapter(pages)
    sync, _, _ = service(adapter, repo)
    sync.run("incremental", (Source("query", "forecasting"),))
    assert adapter.detail_calls == ["owner/changed"]


def test_detail_failure_keeps_existing_metadata_and_marks_pending():
    repo = InMemoryRepository()
    existing = item("owner/item", version=1)
    existing.description = "keep me"
    repo.upsert_resource(existing)
    repo.version_states[existing.ref] = VersionState(NOW, 1, False)
    incoming = item("owner/item", version=2)
    adapter = FakeAdapter(
        {("query:forecasting", "updated", 1): [incoming]},
        {incoming.ref: TimeoutError("detail timeout")},
    )
    sync, _, _ = service(adapter, repo)
    sync.run("incremental", (Source("query", "forecasting"),))
    assert repo.resources[incoming.ref].description == "keep me"
    assert repo.version_states[incoming.ref].detail_pending


def test_transient_failure_does_not_advance_page_or_watermark():
    repo = InMemoryRepository()
    old_watermark = NOW - timedelta(days=1)
    repo.save_checkpoint(
        Checkpoint(
            run_id=__import__("uuid").uuid4(), mode="incremental", source_key="query:forecasting",
            sort="updated", last_successful_page=4, watermark=old_watermark,
            status="complete",
        )
    )
    error = TransientKaggleError("timeout", retry_count=4)
    adapter = FakeAdapter({("query:forecasting", "updated", 1): error})
    sync, _, _ = service(adapter, repo)
    sync.run("incremental", (Source("query", "forecasting"),))
    failed = repo.checkpoints[-1]
    assert failed.status == "retryable_failure"
    assert failed.last_successful_page == 0
    assert failed.watermark == old_watermark
    assert adapter.list_calls == [("query:forecasting", "updated", 1)]


def test_incremental_stops_after_two_stale_full_pages():
    repo = InMemoryRepository()
    source = Source("query", "forecasting")
    watermark = NOW
    repo.save_checkpoint(
        Checkpoint(
            run_id=__import__("uuid").uuid4(), mode="incremental", source_key=source.key,
            sort="updated", last_successful_page=2, watermark=watermark, status="complete",
        )
    )
    page1 = [item(f"owner/a{i}", updated=NOW - timedelta(days=1)) for i in range(20)]
    page2 = [item(f"owner/b{i}", updated=NOW - timedelta(days=2)) for i in range(20)]
    for metadata in page1 + page2:
        repo.upsert_resource(metadata)
        repo.version_states[metadata.ref] = VersionState(metadata.last_updated, 1, False)
    adapter = FakeAdapter({(source.key, "updated", 1): page1, (source.key, "updated", 2): page2})
    sync, _, _ = service(adapter, repo)
    sync.run("incremental", (source,))
    assert adapter.list_calls == [(source.key, "updated", 1), (source.key, "updated", 2)]
    assert repo.checkpoints[-1].stop_reason == "two_stale_pages"


def test_detail_pending_does_not_count_as_changed_for_incremental_stop():
    repo = InMemoryRepository()
    source = Source("query", "forecasting")
    repo.save_checkpoint(
        Checkpoint(
            run_id=__import__("uuid").uuid4(), mode="incremental", source_key=source.key,
            sort="updated", last_successful_page=2, watermark=NOW, status="complete",
        )
    )
    page1 = [item(f"owner/pending-a{i}", updated=NOW - timedelta(days=1)) for i in range(20)]
    page2 = [item(f"owner/pending-b{i}", updated=NOW - timedelta(days=2)) for i in range(20)]
    for metadata in page1 + page2:
        repo.upsert_resource(metadata)
        repo.version_states[metadata.ref] = VersionState(metadata.last_updated, 1, True)
    adapter = FakeAdapter({(source.key, "updated", 1): page1, (source.key, "updated", 2): page2})
    sync, _, _ = service(adapter, repo)
    sync.run("incremental", (source,))
    assert len(adapter.list_calls) == 2
    assert repo.checkpoints[-1].stop_reason == "two_stale_pages"


def test_bootstrap_page_limit_is_partial_not_complete():
    source = Source("query", "forecasting")
    full = [item(f"owner/{i}") for i in range(20)]
    pages = {(source.key, "votes", 1): full, (source.key, "updated", 1): full}
    sync, repo, _ = service(FakeAdapter(pages), page_limit=1)
    sync.run("bootstrap", (source,))
    statuses = [checkpoint.status for checkpoint in repo.checkpoints]
    assert statuses == ["partial", "partial"]
    assert all(checkpoint.stop_reason == "configured_page_limit" for checkpoint in repo.checkpoints)


def test_upsert_is_idempotent_and_preserves_application_id():
    repo = InMemoryRepository()
    metadata = item()
    repo.upsert_resource(metadata)
    first_id = repo.resource_ids[metadata.ref]
    metadata.title = "Updated"
    repo.upsert_resource(metadata)
    assert len(repo.resources) == 1
    assert repo.resource_ids[metadata.ref] == first_id
    assert repo.resources[metadata.ref].title == "Updated"


def test_dry_run_performs_no_repository_writes():
    source = Source("query", "forecasting")
    sync, repo, _ = service(FakeAdapter({(source.key, "updated", 1): [item()]}), dry_run=True)
    result = sync.run("incremental", (source,))
    assert result.candidates == 1
    assert not repo.resources
    assert not repo.checkpoints
