from __future__ import annotations

import io

import pytest

from kagglekit.dataset_markdown.builder import DatasetMarkdownBuilder, DatasetResource
from kagglekit.dataset_markdown.models import DatasetFile


class RangeIgnoringClient:
    def __init__(self, name: str, body: bytes):
        self.name, self.body = name, body

    def list_files(self, ref):
        return [DatasetFile(self.name, len(self.body))], None

    def get(self, ref, path, *, byte_range=None, cap=None, deadline=None):
        assert len(self.body) <= cap
        return 200, {"content-length": str(len(self.body))}, self.body


def _build(name: str, body: bytes):
    resource = DatasetResource("owner/slug", "Dataset", "https://www.kaggle.com/datasets/owner/slug")
    return DatasetMarkdownBuilder(RangeIgnoringClient(name, body)).build(resource)


def test_csv_range_ignored_small_file_falls_back():
    markdown, inspected = _build("data.csv", b"id,label\n1,2\n")
    assert inspected[0].status == "completed"
    assert [field.name for field in inspected[0].fields] == ["id", "label"]
    assert "## Fields" in markdown


def test_parquet_range_ignored_small_file_falls_back():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    stream = io.BytesIO()
    pq.write_table(pa.table({"id": [1], "label": ["a"]}), stream)
    _, inspected = _build("data.parquet", stream.getvalue())
    assert inspected[0].status == "completed"
    assert [field.name for field in inspected[0].fields] == ["id", "label"]
