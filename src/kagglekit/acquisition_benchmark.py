from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Protocol

from .dataset_sync.config import SyncConfig
from .dataset_sync.kaggle import KaggleAdapter
from .dataset_sync.sources import Source


PHASE1_FIELDS = (
    "ref",
    "title",
    "url",
    "description",
    "subtitle",
    "tags",
    "author",
    "last_updated",
    "votes",
    "downloads",
    "usability_rating",
    "size_bytes",
    "license",
)


class DetailFetcher(Protocol):
    def fetch(self, ref: str) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SampleResult:
    ref: str
    status: str
    latency_ms: int
    populated_fields: int = 0
    total_fields: int = len(PHASE1_FIELDS)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MethodResult:
    method: str
    requested: int
    succeeded: int
    failed: int
    wall_time_ms: int
    throughput_per_second: float
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    field_completeness: float
    samples: tuple[SampleResult, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["samples"] = [asdict(item) for item in self.samples]
        return value


class RestDetailFetcher:
    def __init__(self, adapter: KaggleAdapter) -> None:
        self.adapter = adapter

    def fetch(self, ref: str) -> dict[str, Any]:
        item, _ = self.adapter.dataset_detail(ref)
        return {field: getattr(item, field) for field in PHASE1_FIELDS}


class KaggleCliDetailFetcher:
    """Fetch official dataset-metadata.json without downloading dataset files."""

    def __init__(
        self,
        executable: str = "kaggle",
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.executable = executable
        self.runner = runner

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def fetch(self, ref: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="kagglekit-kaggle-metadata-") as directory:
            completed = self.runner(
                [self.executable, "datasets", "metadata", ref, "--path", directory],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(message or f"Kaggle CLI exited {completed.returncode}")
            path = Path(directory) / "dataset-metadata.json"
            if not path.exists():
                matches = list(Path(directory).glob("*metadata*.json"))
                if not matches:
                    raise RuntimeError("Kaggle CLI did not create a metadata JSON file")
                path = matches[0]
            raw = json.loads(path.read_text(encoding="utf-8"))
        return _normalize_cli_metadata(ref, raw)


def run_method(
    method: str,
    refs: list[str],
    fetcher_factory: Callable[[], DetailFetcher],
    *,
    workers: int = 1,
) -> MethodResult:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    if workers == 1:
        samples = tuple(_fetch_one(ref, fetcher_factory()) for ref in refs)
    else:
        indexed: dict[int, SampleResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_one, ref, fetcher_factory()): index
                for index, ref in enumerate(refs)
            }
            for future in as_completed(futures):
                indexed[futures[future]] = future.result()
        samples = tuple(indexed[index] for index in range(len(refs)))
    wall_time_ms = round((time.perf_counter() - started) * 1000)
    successful = [item for item in samples if item.status == "success"]
    latencies = sorted(item.latency_ms for item in successful)
    total_present = sum(item.populated_fields for item in successful)
    total_possible = len(successful) * len(PHASE1_FIELDS)
    seconds = wall_time_ms / 1000
    return MethodResult(
        method=method,
        requested=len(refs),
        succeeded=len(successful),
        failed=len(refs) - len(successful),
        wall_time_ms=wall_time_ms,
        throughput_per_second=round(len(successful) / seconds, 3) if seconds else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        field_completeness=round(total_present / total_possible, 4) if total_possible else 0.0,
        samples=samples,
    )


def benchmark(
    refs: list[str],
    methods: list[str],
    *,
    workers: int,
    config: SyncConfig | None = None,
) -> dict[str, Any]:
    clean_refs = list(dict.fromkeys(ref.strip() for ref in refs if ref.strip()))
    if not clean_refs:
        raise ValueError("At least one dataset ref is required")
    active_config = config or SyncConfig.from_env()
    factories: dict[str, tuple[Callable[[], DetailFetcher], int]] = {
        "rest-sequential": (lambda: RestDetailFetcher(KaggleAdapter(active_config)), 1),
        "rest-concurrent": (lambda: RestDetailFetcher(KaggleAdapter(active_config)), workers),
        "kaggle-cli": (lambda: KaggleCliDetailFetcher(), 1),
    }
    unsupported = sorted(set(methods) - set(factories))
    if unsupported:
        raise ValueError(f"Unknown benchmark methods: {', '.join(unsupported)}")
    results: list[MethodResult] = []
    skipped: list[dict[str, str]] = []
    for method in methods:
        factory, method_workers = factories[method]
        fetcher = factory()
        if method == "kaggle-cli" and not fetcher.available():  # type: ignore[attr-defined]
            skipped.append({"method": method, "reason": "kaggle executable not installed"})
            continue
        results.append(run_method(method, clean_refs, factory, workers=method_workers))
    ranked = sorted(
        (item for item in results if item.failed == 0 and item.field_completeness >= 0.80),
        key=lambda item: (-item.throughput_per_second, -item.field_completeness),
    )
    return {
        "schema_version": 1,
        "sample_refs": clean_refs,
        "environment": {
            "authenticated": bool(os.getenv("KAGGLE_API_TOKEN")),
            "concurrent_workers": workers,
        },
        "methods": [item.to_dict() for item in results],
        "skipped": skipped,
        "recommendation": ranked[0].method if ranked else None,
        "recommendation_rule": (
            "zero failures and at least 80% Phase 1 field completeness; "
            "then highest throughput, then field completeness"
        ),
        "notes": [
            "kagglehub is excluded because it is download/cache oriented and does not provide equivalent catalog discovery metadata benchmarking.",
            "Benchmark detail acquisition separately from common candidate discovery.",
        ],
    }


def discover_refs(
    source_key: str,
    sample_size: int,
    *,
    config: SyncConfig | None = None,
) -> list[str]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    try:
        kind, value = source_key.split(":", 1)
    except ValueError as exc:
        raise ValueError("source must use tag:<value> or query:<value>") from exc
    if kind not in {"tag", "query"} or not value.strip():
        raise ValueError("source must use tag:<value> or query:<value>")
    adapter = KaggleAdapter(config or SyncConfig.from_env())
    refs: list[str] = []
    seen: set[str] = set()
    page = 1
    source = Source(kind, value.strip())
    while len(refs) < sample_size:
        result = adapter.list_datasets(source, "updated", page)
        if not result.items:
            break
        for item in result.items:
            if item.valid_for_resource() and item.ref not in seen:
                seen.add(item.ref)
                refs.append(item.ref)
                if len(refs) == sample_size:
                    break
        if len(result.items) < 20:
            break
        page += 1
    return refs


def _fetch_one(ref: str, fetcher: DetailFetcher) -> SampleResult:
    started = time.perf_counter()
    try:
        value = fetcher.fetch(ref)
        latency = round((time.perf_counter() - started) * 1000)
        populated = sum(_present(value.get(field)) for field in PHASE1_FIELDS)
        return SampleResult(ref, "success", latency, populated)
    except Exception as exc:
        latency = round((time.perf_counter() - started) * 1000)
        return SampleResult(ref, "error", latency, error=f"{type(exc).__name__}: {exc}"[:1000])


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _normalize_cli_metadata(ref: str, raw: dict[str, Any]) -> dict[str, Any]:
    licenses = raw.get("licenses") or []
    first_license = licenses[0] if licenses else None
    license_name = first_license.get("name") if isinstance(first_license, dict) else first_license
    tags = raw.get("keywords") or raw.get("tags") or []
    return {
        "ref": raw.get("id") or ref,
        "title": raw.get("title"),
        "url": raw.get("url") or f"https://www.kaggle.com/datasets/{ref}",
        "description": raw.get("description"),
        "subtitle": raw.get("subtitle"),
        "tags": tags,
        "author": raw.get("ownerName") or raw.get("ownerRef") or ref.split("/", 1)[0],
        "last_updated": raw.get("lastUpdated"),
        "votes": raw.get("voteCount"),
        "downloads": raw.get("downloadCount"),
        "usability_rating": raw.get("usabilityRating"),
        "size_bytes": raw.get("totalBytes") or raw.get("size"),
        "license": license_name or raw.get("licenseName"),
    }
