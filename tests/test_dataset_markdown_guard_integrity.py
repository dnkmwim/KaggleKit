from __future__ import annotations

import io

import pytest

from kagglekit.dataset_markdown.builder import DatasetMarkdownBuilder, DatasetResource
from kagglekit.dataset_markdown.config import RuntimeGuards
from kagglekit.dataset_markdown.models import DatasetFile


class Client:
    def __init__(self, path, body):
        self.path, self.body = path, body

    def list_files(self, ref):
        return [DatasetFile(self.path, len(self.body))], None

    def get(self, ref, path, *, byte_range=None, cap=None, deadline=None):
        return (206 if byte_range else 200), {}, self.body


RESOURCE = DatasetResource("owner/slug", "Dataset", "https://www.kaggle.com/datasets/owner/slug")


def test_direct_over_limit_fields_are_not_retained_or_rendered():
    markdown, inspected = DatasetMarkdownBuilder(Client("file.csv", b"a,b,c\n"),
                                                RuntimeGuards(max_fields_per_subresource=2)).build(RESOURCE)
    assert inspected[0].status == "skipped"
    assert inspected[0].reason == "budget_exceeded"
    assert inspected[0].fields == []
    assert "## Fields" not in markdown


def test_one_xlsx_sheet_over_limit_yields_partial_not_false_fields():
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    book.active.title = "small"
    book.active.append(["a", "b"])
    book.create_sheet("wide").append(["a", "b", "c"])
    stream = io.BytesIO()
    book.save(stream)
    markdown, inspected = DatasetMarkdownBuilder(Client("file.xlsx", stream.getvalue()),
                                                RuntimeGuards(max_fields_per_subresource=2)).build(RESOURCE)
    assert inspected[0].status == "partial"
    assert inspected[0].reason == "budget_exceeded"
    assert [(sub.name, sub.status) for sub in inspected[0].subresources] == [
        ("small", "completed"), ("wide", "skipped")]
    assert "#### small" in markdown and "#### wide" not in markdown
