from __future__ import annotations

import io
import json
import sqlite3
import zipfile

import pytest

from kagglekit.dataset_markdown.builder import DatasetMarkdownBuilder, DatasetResource
from kagglekit.dataset_markdown.config import RuntimeGuards
from kagglekit.dataset_markdown.inspector import KaggleFileClient, SchemaInspector
from kagglekit.dataset_markdown.models import DatasetFile


class FakeClient:
    def __init__(self, files: dict[str, bytes], wrapped: set[str] = frozenset()):
        self.files = files
        self.wrapped = wrapped
        self.calls: list[tuple[str, str | None]] = []

    def list_files(self, ref):
        return [DatasetFile(path, len(data)) for path, data in self.files.items()], None

    def get(self, ref, path, *, byte_range=None, cap=None, deadline=None):
        self.calls.append((path, byte_range))
        data = self.files[path]
        if path in self.wrapped:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(path, data)
            data = buffer.getvalue()
        if byte_range is None:
            return 200, {"content-length": str(len(data))}, data
        if byte_range.startswith("-"):
            n = int(byte_range[1:])
            return 206, {"content-range": f"bytes {len(data)-n}-{len(data)-1}/{len(data)}"}, data[-n:]
        start, end = map(int, byte_range.split("-"))
        chunk = data[start:end + 1]
        return 206, {"content-range": f"bytes {start}-{start+len(chunk)-1}/{len(data)}"}, chunk


def resource(ref="owner/slug"):
    return DatasetResource(ref, "Official title", f"https://www.kaggle.com/datasets/{ref}",
                           description="## Author heading\nSource facts.", tags=("classification",))


def test_csv_exact_names_bom_multiline_and_yaml():
    data = b'\xef\xbb\xbf"first\nline",id,id,\n1,2,3,4\n'
    builder = DatasetMarkdownBuilder(FakeClient({"sample.csv": data}))
    markdown, results = builder.build(resource())
    assert results[0].status == "completed"
    assert [field.name for field in results[0].fields] == ["first\nline", "id", "id", ""]
    assert results[0].warnings == ["duplicate_field_name", "empty_field_name"]
    assert "schema_version: \"dataset-md-v1.0\"" in markdown
    assert "### Author heading" in markdown
    assert "Original URL: https://www.kaggle.com/datasets/owner/slug" in markdown
    assert "No fields" not in markdown


def test_zip_wrapped_csv_is_not_treated_as_raw_range():
    client = FakeClient({"wrapped.csv": b"alpha,beta\n1,2\n"}, wrapped={"wrapped.csv"})
    markdown, results = DatasetMarkdownBuilder(client).build(resource())
    assert results[0].status == "completed"
    assert results[0].representation == "zip"
    assert [field.name for field in results[0].fields] == ["alpha", "beta"]
    assert ("wrapped.csv", None) in client.calls
    assert "alpha" in markdown


def test_arff_native_type_and_no_data_row_inference():
    data = b"@RELATION x\n@ATTRIBUTE 'job title' STRING\n@ATTRIBUTE age NUMERIC\n@DATA\nfoo,10\n"
    _, results = DatasetMarkdownBuilder(FakeClient({"data.arff": data})).build(resource())
    assert results[0].status == "completed"
    assert [(x.name, x.type_raw) for x in results[0].fields] == [("job title", "STRING"), ("age", "NUMERIC")]


def test_sqlite_table_xinfo_allowlist():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE items(id INTEGER, base TEXT, generated TEXT GENERATED ALWAYS AS (base || 'x') VIRTUAL)")
    connection.execute("CREATE VIEW item_view AS SELECT id FROM items")
    try:
        connection.execute("CREATE VIRTUAL TABLE search USING fts5(body)")
    except sqlite3.OperationalError:
        pass
    data = connection.serialize()
    connection.close()
    _, results = DatasetMarkdownBuilder(FakeClient({"data.sqlite": data})).build(resource())
    assert results[0].status == "completed"
    assert [sub.name for sub in results[0].subresources] == ["items"]
    assert [field.name for field in results[0].subresources[0].fields] == ["id", "base", "generated"]


def test_xlsx_instruction_sheet_is_partial():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "Data"
    workbook.active.append(["id", "name"])
    instruction = workbook.create_sheet("Instructions")
    instruction.append(["Read first"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    _, results = DatasetMarkdownBuilder(FakeClient({"sample.xlsx": buffer.getvalue()})).build(resource())
    assert results[0].status == "partial"
    assert [(s.name, s.status) for s in results[0].subresources] == [("Data", "completed"), ("Instructions", "skipped")]


def test_parquet_footer_schema_only():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buffer = io.BytesIO()
    pq.write_table(pa.table({"id": [1], "label": ["a"]}), buffer)
    client = FakeClient({"sample.parquet": buffer.getvalue()})
    _, results = DatasetMarkdownBuilder(client).build(resource())
    assert results[0].status == "completed"
    assert [field.name for field in results[0].fields] == ["id", "label"]
    assert ("sample.parquet", None) not in client.calls


def test_budget_skip_keeps_base_markdown():
    guards = RuntimeGuards(max_expanded_bytes_per_file=8)
    markdown, results = DatasetMarkdownBuilder(FakeClient({"data.sqlite": b"SQLite format 3\x00" + b"x" * 100}), guards).build(resource())
    assert results[0].status == "skipped" and results[0].reason == "file_too_large"
    assert "## Source" in markdown and "## Fields" not in markdown
    assert "No fields" not in markdown


def test_reader_entry_uses_normalized_resource():
    class Reader:
        def get(self, source_id):
            assert source_id == "owner/slug"
            return resource()
    markdown, _ = DatasetMarkdownBuilder(FakeClient({})).build_from_reader(Reader(), "owner/slug")
    assert "source_id: \"owner/slug\"" in markdown
