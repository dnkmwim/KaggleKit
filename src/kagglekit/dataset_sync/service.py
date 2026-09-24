from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Callable
from uuid import UUID, uuid4

from .config import SyncConfig
from .filters import data_science_negative, tag_page_is_valid
from .kaggle import KaggleAdapter, TransientKaggleError
from .models import Candidate, DatasetMetadata, VersionState, merge_non_empty, version_changed
from .repository import Checkpoint, Repository
from .sources import Source


OBSERVED_PAGE_SIZE = 20
EventSink = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: UUID
    candidates: int
    resources_upserted: int
    detail_requested: int
    detail_failed: int


class DatasetSyncService:
    def __init__(
        self,
        adapter: KaggleAdapter,
        repository: Repository,
        config: SyncConfig,
        *,
        event_sink: EventSink | None = None,
        dry_run: bool = False,
    ) -> None:
        self.adapter = adapter
        self.repository = repository
        self.config = config
        self.event_sink = event_sink or (lambda event: print(json.dumps(event, ensure_ascii=False, default=str)))
        self.dry_run = dry_run

    def run(self, mode: str, sources: tuple[Source, ...], *, run_id: UUID | None = None) -> RunResult:
        if mode not in {"bootstrap", "incremental"}:
            raise ValueError(f"Unsupported mode: {mode}")
        active_run_id = run_id or uuid4()
        candidates: dict[str, Candidate] = {}
        sorts = ("votes", "updated") if mode == "bootstrap" else ("updated",)
        for source in sources:
            for sort in sorts:
                self._scan_pass(active_run_id, mode, source, sort, candidates)
        upserted, requested, failed = self._persist_candidates(active_run_id, mode, candidates)
        self._emit(
            event="run_complete", run_id=active_run_id, mode=mode,
            candidate_count=len(candidates), resources_upserted=upserted,
            detail_requested_count=requested, detail_failed_count=failed,
        )
        return RunResult(active_run_id, len(candidates), upserted, requested, failed)

    def _scan_pass(
        self,
        run_id: UUID,
        mode: str,
        source: Source,
        sort: str,
        candidates: dict[str, Candidate],
    ) -> None:
        page = 1
        last_successful_page = 0
        old_page_streak = 0
        prior_watermark = self.repository.latest_watermark(source.key, sort) if mode == "incremental" else None
        observed_watermark = prior_watermark
        while True:
            try:
                result = self.adapter.list_datasets(source, sort, page)
            except TransientKaggleError as exc:
                self._checkpoint(
                    run_id, mode, source, sort, last_successful_page, prior_watermark,
                    "retryable_failure", "transient_error", True,
                )
                self._emit(
                    event="page_failed", run_id=run_id, mode=mode, source=source.key, sort=sort,
                    page=page, http_status=exc.http_status, retry_count=exc.retry_count,
                    stop_reason="retryable_failure",
                )
                return

            raw_count = len(result.items)
            valid_items = [item for item in result.items if item.valid_for_resource()]
            if source.kind == "tag" and not tag_page_is_valid(valid_items, source.value):
                self._checkpoint(
                    run_id, mode, source, sort, last_successful_page, prior_watermark,
                    "invalid_source", "tag_exact_validation_failed", False,
                )
                self._emit(
                    event="tag_validation_failed", run_id=run_id, mode=mode, source=source.key,
                    sort=sort, page=page, http_status=result.http_status,
                    latency_ms=result.latency_ms, raw_result_count=raw_count,
                    valid_result_count=0, retry_count=result.retry_count,
                    stop_reason="tag_exact_validation_failed",
                )
                return

            states = self.repository.get_version_states([item.ref for item in valid_items])
            filtered_count = 0
            new_count = 0
            duplicate_count = 0
            accepted_for_staleness: list[DatasetMetadata] = []
            for item in valid_items:
                if source.key == "query:data science":
                    matched = data_science_negative(item)
                    if matched:
                        filtered_count += 1
                        self._emit(
                            event="source_suppressed", run_id=run_id, mode=mode,
                            ref=item.ref, source=source.key, matched_negative_term=matched,
                        )
                        continue
                accepted_for_staleness.append(item)
                candidate = candidates.get(item.ref)
                if candidate is None:
                    candidates[item.ref] = Candidate(item, {source.key})
                    new_count += 1
                else:
                    candidate.metadata = _prefer_fresher(candidate.metadata, item)
                    candidate.hit_sources.add(source.key)
                    duplicate_count += 1

            page_new_or_changed = sum(
                _candidate_changed(item, states.get(item.ref)) for item in accepted_for_staleness
            )

            page_dates = [item.last_updated for item in valid_items if item.last_updated]
            if page_dates:
                newest = max(page_dates)
                observed_watermark = max(filter(None, (observed_watermark, newest)), default=newest)
            all_not_newer = bool(
                prior_watermark
                and valid_items
                and len(page_dates) == len(valid_items)
                and all(value <= prior_watermark for value in page_dates)
            )
            if mode == "incremental" and page_new_or_changed == 0 and all_not_newer:
                old_page_streak += 1
            else:
                old_page_streak = 0

            last_successful_page = page
            self._checkpoint(
                run_id, mode, source, sort, last_successful_page, prior_watermark,
                "running", None, False,
            )
            self._emit(
                event="page_processed", run_id=run_id, mode=mode, source=source.key, sort=sort,
                page=page, http_status=result.http_status, latency_ms=result.latency_ms,
                raw_result_count=raw_count, valid_result_count=len(accepted_for_staleness),
                new_candidate_count=new_count, duplicate_count=duplicate_count,
                detail_requested_count=0, detail_failed_count=0,
                filtered_data_science_source_count=filtered_count,
                retry_count=result.retry_count, stop_reason=None,
            )

            stop_reason: str | None = None
            status = "complete"
            if raw_count == 0:
                stop_reason = "empty_page"
            elif raw_count < OBSERVED_PAGE_SIZE:
                stop_reason = "short_page"
            elif mode == "incremental" and old_page_streak >= 2:
                stop_reason = "two_stale_pages"
            elif mode == "bootstrap" and page >= self.config.bootstrap_page_limit:
                stop_reason = "configured_page_limit"
                status = "partial"

            if stop_reason:
                self._checkpoint(
                    run_id, mode, source, sort, last_successful_page, observed_watermark,
                    status, stop_reason, False,
                )
                self._emit(
                    event="pass_stopped", run_id=run_id, mode=mode, source=source.key,
                    sort=sort, page=page, stop_reason=stop_reason, status=status,
                )
                return
            page += 1

    def _persist_candidates(
        self, run_id: UUID, mode: str, candidates: dict[str, Candidate]
    ) -> tuple[int, int, int]:
        refs = list(candidates)
        states = self.repository.get_version_states(refs)
        existing = self.repository.get_resources(refs)
        upserted = requested = failed = 0
        for ref, candidate in candidates.items():
            list_metadata = candidate.metadata
            merged = merge_non_empty(existing[ref], list_metadata) if ref in existing else list_metadata
            pending = False
            detail_error: str | None = None
            if version_changed(list_metadata, states.get(ref)):
                requested += 1
                try:
                    detail, retry_count = self.adapter.dataset_detail(ref)
                    merged = merge_non_empty(merged, detail)
                    self._emit(
                        event="detail_succeeded", run_id=run_id, mode=mode, ref=ref,
                        retry_count=retry_count,
                    )
                except Exception as exc:  # detail failures do not roll back a valid list candidate
                    failed += 1
                    pending = True
                    detail_error = f"{type(exc).__name__}: {exc}"[:1000]
                    self._emit(
                        event="detail_failed", run_id=run_id, mode=mode, ref=ref,
                        error=detail_error,
                    )
            if not self.dry_run:
                self.repository.upsert_resource(merged)
                self.repository.save_version_state(
                    list_metadata, detail_pending=pending, detail_error=detail_error
                )
            upserted += 1
            self._emit(
                event="candidate_ready", run_id=run_id, mode=mode, ref=ref,
                hit_sources=sorted(candidate.hit_sources), detail_pending=pending,
                dry_run=self.dry_run,
            )
        return upserted, requested, failed

    def _checkpoint(
        self,
        run_id: UUID,
        mode: str,
        source: Source,
        sort: str,
        page: int,
        watermark: datetime | None,
        status: str,
        stop_reason: str | None,
        retryable: bool,
    ) -> None:
        if not self.dry_run:
            self.repository.save_checkpoint(
                Checkpoint(
                    run_id=run_id, mode=mode, source_key=source.key, sort=sort,
                    last_successful_page=page, watermark=watermark, status=status,
                    stop_reason=stop_reason, retryable=retryable,
                )
            )

    def _emit(self, **event: object) -> None:
        self.event_sink(event)


def _prefer_fresher(left: DatasetMetadata, right: DatasetMetadata) -> DatasetMetadata:
    if left.last_updated and right.last_updated and right.last_updated < left.last_updated:
        return merge_non_empty(right, left)
    return merge_non_empty(left, right)


def _candidate_changed(metadata: DatasetMetadata, state: VersionState | None) -> bool:
    if state is None:
        return True
    return (
        metadata.last_updated != state.last_updated
        or metadata.current_version_number != state.current_version_number
    )
