from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace

import pytest

from kagglekit.dataset_markdown.builder import DatasetMarkdownBuilder, DatasetResource, PostgresDatasetReader
from kagglekit.dataset_markdown.models import DatasetFile


RESOURCE = DatasetResource("owner/slug", "Dataset", "https://www.kaggle.com/datasets/owner/slug")


class BytesClient:
    def __init__(self, name: str, body: bytes):
        self.name, self.body = name, body

    def list_files(self, ref):
        return [DatasetFile(self.name, len(self.body))], None

    def get(self, ref, path, *, byte_range=None, cap=None, deadline=None):
        if byte_range:
            return 206, {"content-range": f"bytes 0-{len(self.body)-1}/{len(self.body)}"}, self.body
        return 200, {"content-length": str(len(self.body))}, self.body


def test_arff_stops_before_invalid_data_bytes():
    data = b"@relation x\n@attribute id NUMERIC\n@data\n\xff\n"
    markdown, inspected = DatasetMarkdownBuilder(BytesClient("sample.arff", data)).build(RESOURCE)
    assert inspected[0].status == "completed"
    assert inspected[0].fields[0].type_raw == "NUMERIC"
    assert "## Fields" in markdown


def test_xlsx_merged_candidate_is_no_schema():
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["id", "name"])
    sheet.merge_cells("A1:A2")
    stream = io.BytesIO()
    book.save(stream)
    markdown, inspected = DatasetMarkdownBuilder(BytesClient("sample.xlsx", stream.getvalue())).build(RESOURCE)
    assert inspected[0].status == "skipped"
    assert inspected[0].reason == "no_schema"
    assert "## Fields" not in markdown


def test_xlsx_does_not_search_past_instruction_row():
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Instructions"])
    sheet.append(["id", "name"])
    stream = io.BytesIO()
    book.save(stream)
    _, inspected = DatasetMarkdownBuilder(BytesClient("sample.xlsx", stream.getvalue())).build(RESOURCE)
    assert inspected[0].status == "skipped"
    assert inspected[0].reason == "no_schema"


def test_postgres_reader_filters_active_dataset_and_is_read_only(monkeypatch):
    rows = ModuleType("psycopg.rows")
    rows.dict_row = object()

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            assert "resource_type='dataset'" in query
            assert "status='active'" in query
            assert params == ("owner/slug",)

        def fetchone(self):
            return {"source_id": "owner/slug", "title": "Dataset",
                    "url": "https://www.kaggle.com/datasets/owner/slug", "tags": ["x"]}

    class Connection:
        read_only = False

        def cursor(self):
            return Cursor()

        def close(self):
            pass

    connection = Connection()
    module = ModuleType("psycopg")
    module.connect = lambda *args, **kwargs: connection
    monkeypatch.setitem(sys.modules, "psycopg", module)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    reader = PostgresDatasetReader("postgresql://local/test")
    assert connection.read_only
    assert reader.get("owner/slug").tags == ("x",)
    reader.close()
