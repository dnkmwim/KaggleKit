import json
import subprocess

from kagglekit.acquisition_benchmark import (
    KaggleCliDetailFetcher,
    PHASE1_FIELDS,
    _percentile,
    benchmark,
    discover_refs,
    run_method,
)
from kagglekit.dataset_sync.config import SyncConfig


class FakeFetcher:
    def __init__(self, fail_ref=None):
        self.fail_ref = fail_ref

    def fetch(self, ref):
        if ref == self.fail_ref:
            raise TimeoutError("test timeout")
        return {field: f"{ref}-{field}" for field in PHASE1_FIELDS}


def test_run_method_reports_throughput_completeness_and_failures():
    result = run_method(
        "fake", ["a/one", "b/two", "c/three"], lambda: FakeFetcher("b/two"), workers=2
    )
    assert result.requested == 3
    assert result.succeeded == 2
    assert result.failed == 1
    assert result.field_completeness == 1.0
    assert [sample.ref for sample in result.samples] == ["a/one", "b/two", "c/three"]
    assert result.samples[1].status == "error"


def test_nearest_rank_percentile():
    assert _percentile([], 0.95) is None
    assert _percentile([10, 20, 30, 40], 0.50) == 20
    assert _percentile([10, 20, 30, 40], 0.95) == 40


def test_kaggle_cli_reads_metadata_without_dataset_download(tmp_path):
    def runner(command, **kwargs):
        output = command[command.index("--path") + 1]
        payload = {
            "id": "owner/sample",
            "title": "Sample",
            "description": "Description",
            "keywords": ["classification"],
            "licenses": [{"name": "CC0-1.0"}],
        }
        from pathlib import Path

        Path(output, "dataset-metadata.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = KaggleCliDetailFetcher(runner=runner).fetch("owner/sample")
    assert result["ref"] == "owner/sample"
    assert result["license"] == "CC0-1.0"
    assert result["tags"] == ["classification"]


def test_benchmark_recommendation_requires_field_completeness(monkeypatch):
    class SparseCli:
        def available(self):
            return True

        def fetch(self, ref):
            return {"ref": ref, "title": "Only two fields"}

    monkeypatch.setattr("kagglekit.acquisition_benchmark.KaggleCliDetailFetcher", SparseCli)
    report = benchmark(
        ["owner/sample"],
        ["kaggle-cli"],
        workers=1,
        config=SyncConfig(request_delay_seconds=0),
    )
    assert report["methods"][0]["field_completeness"] < 0.80
    assert report["recommendation"] is None


def test_discover_refs_uses_live_list_shape_without_duplicates(monkeypatch):
    from kagglekit.dataset_sync.models import DatasetMetadata, PageResult

    class FakeAdapter:
        def __init__(self, config):
            pass

        def list_datasets(self, source, sort, page):
            values = [
                DatasetMetadata(ref="a/one", title="One", url="https://one"),
                DatasetMetadata(ref="a/one", title="One", url="https://one"),
                DatasetMetadata(ref="b/two", title="Two", url="https://two"),
            ]
            return PageResult(values, 200, 1)

    monkeypatch.setattr("kagglekit.acquisition_benchmark.KaggleAdapter", FakeAdapter)
    assert discover_refs("tag:classification", 2) == ["a/one", "b/two"]
