"""Bounded, read-only Kaggle file schema inspection, outside Dataset Sync."""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import time
import zipfile
from pathlib import PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import RuntimeGuards
from .models import DatasetFile, Field, FileInspection, Subresource


class InspectionStop(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class KaggleFileClient:
    BASE = "https://www.kaggle.com/api/v1"

    def __init__(self, guards: RuntimeGuards):
        self.guards = guards
        self.transferred = 0

    def _request(self, url: str, *, byte_range: str | None = None, cap: int | None = None,
                 deadline: float | None = None) -> tuple[int, dict[str, str], bytes]:
        remaining = self.guards.max_total_transfer_bytes_per_dataset - self.transferred
        if remaining <= 0:
            raise InspectionStop("skipped", "budget_exceeded")
        limit = min(cap or self.guards.max_full_download_bytes_per_file, remaining)
        timeout = self.guards.request_timeout
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise InspectionStop("skipped", "timeout")
        headers = {"User-Agent": "KaggleKit-DatasetMarkdown/0.1"}
        if byte_range:
            headers["Range"] = f"bytes={byte_range}"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                if not response.url.startswith("https://"):
                    raise InspectionStop("failed", "unavailable")
                # Read one extra byte only to prove the hard cap was reached.
                body = response.read(limit + 1)
                self.transferred += len(body)
                if len(body) > limit:
                    raise InspectionStop("skipped", "budget_exceeded")
                return response.status, {k.lower(): v for k, v in response.headers.items()}, body
        except HTTPError as exc:
            raise InspectionStop("failed", "unavailable" if exc.code == 404 else "inspection_error") from exc
        except (TimeoutError, URLError) as exc:
            raise InspectionStop("failed", "timeout" if isinstance(exc, TimeoutError) else "unavailable") from exc

    def list_files(self, ref: str) -> tuple[list[DatasetFile], str | None]:
        self.transferred = 0
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", ref):
            return [], "unavailable"
        files: list[DatasetFile] = []
        token: str | None = None
        try:
            for _ in range(20):
                query = {"pageSize": "200"}
                if token:
                    query["pageToken"] = token
                url = f"{self.BASE}/datasets/list/{ref}?{urlencode(query)}"
                _, _, body = self._request(url, cap=1_000_000)
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("datasetFiles"), list):
                    return files, "inspection_error"
                for item in payload["datasetFiles"]:
                    path = item.get("name")
                    if isinstance(path, str) and path:
                        size = item.get("totalBytes")
                        files.append(DatasetFile(path, size if isinstance(size, int) and size >= 0 else None))
                token = payload.get("nextPageTokenNullable") or None
                if not token:
                    return files, None
            return files, "budget_exceeded"
        except (InspectionStop, ValueError, TypeError) as exc:
            return files, exc.reason if isinstance(exc, InspectionStop) else "inspection_error"

    def file_url(self, ref: str, path: str) -> str:
        return f"{self.BASE}/datasets/download/{ref}/{quote(path, safe='/')}"

    def get(self, ref: str, path: str, *, byte_range: str | None = None,
            cap: int | None = None, deadline: float | None = None) -> tuple[int, dict[str, str], bytes]:
        return self._request(self.file_url(ref, path), byte_range=byte_range, cap=cap, deadline=deadline)


def _format(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    return {".csv": "csv", ".tsv": "tsv", ".arff": "arff", ".xlsx": "xlsx",
            ".sqlite": "sqlite", ".db": "sqlite", ".sqlite3": "sqlite",
            ".parquet": "parquet", ".zip": "archive", ".gz": "archive",
            ".gzip": "archive", ".tar": "archive", ".7z": "archive"}.get(
                suffix, suffix[1:] if suffix else "unknown")

def _logical_header(raw: bytes, delimiter: str, complete: bool) -> tuple[list[Field], list[str]]:
    quoted = False
    end = None
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == 34:
            if quoted and index + 1 < len(raw) and raw[index + 1] == 34:
                index += 2
                continue
            quoted = not quoted
        elif char == 10 and not quoted:
            end = index + 1
            break
        index += 1
    if end is None:
        if not complete or quoted:
            raise InspectionStop("skipped", "header_too_large")
        end = len(raw)
    try:
        text = raw[:end].decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InspectionStop("skipped", "unsupported_encoding") from exc
    try:
        row = next(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True))
    except (csv.Error, StopIteration) as exc:
        raise InspectionStop("failed", "inspection_error") from exc
    if len(row) < 1:
        raise InspectionStop("skipped", "no_schema")
    warnings = []
    if len(row) != len(set(row)):
        warnings.append("duplicate_field_name")
    if "" in row:
        warnings.append("empty_field_name")
    return [Field(i, name) for i, name in enumerate(row)], warnings


def _arff_header(raw: bytes, complete: bool) -> list[Field]:
    marker = re.search(rb"(?im)^[ \t]*@data(?:[ \t]|\r?$)", raw)
    if marker:
        raw = raw[:marker.end()]
    elif not complete:
        raise InspectionStop("skipped", "header_too_large")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InspectionStop("skipped", "unsupported_encoding") from exc
    fields: list[Field] = []
    found_data = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if re.match(r"(?i)^@data(?:\s|$)", stripped):
            found_data = True
            break
        if re.match(r"(?i)^@attribute\s", stripped):
            match = re.match(r"(?is)^@attribute\s+(?:'((?:\\.|[^'])*)'|\"((?:\\.|[^\"])*)\"|(\S+))\s+(.+)$", stripped)
            if not match:
                raise InspectionStop("failed", "inspection_error")
            name = next(value for value in match.groups()[:3] if value is not None)
            fields.append(Field(len(fields), name, match.group(4).strip(), "native_schema"))
    if not found_data:
        raise InspectionStop("skipped", "header_too_large" if not complete else "no_schema")
    if not fields:
        raise InspectionStop("skipped", "no_schema")
    return fields


class SchemaInspector:
    def __init__(self, client: KaggleFileClient, guards: RuntimeGuards):
        self.client, self.guards = client, guards

    def inspect_dataset(self, ref: str, files: list[DatasetFile]) -> list[FileInspection]:
        results = []
        attempted = 0
        for item in files:
            result = FileInspection(item.path, _format(item.path), item.size)
            results.append(result)
            if result.format not in {"csv", "tsv", "arff", "xlsx", "sqlite", "parquet"}:
                result.status, result.reason = "skipped", "archive" if result.format == "archive" else "unsupported_format"
                continue
            if attempted >= self.guards.max_files_inspected_per_dataset:
                result.status, result.reason = "skipped", "budget_exceeded"
                continue
            attempted += 1
            deadline = time.monotonic() + self.guards.inspection_timeout_per_file
            try:
                self._inspect(ref, result, deadline)
            except InspectionStop as exc:
                result.fields = []
                for sub in result.subresources:
                    if sub.status == "not_run":
                        sub.fields = []
                        sub.status, sub.reason = exc.status, exc.reason
                result.status = "partial" if any(s.status == "completed" for s in result.subresources) else exc.status
                result.reason = exc.reason
            except Exception:
                result.fields = []
                for sub in result.subresources:
                    if sub.status == "not_run":
                        sub.fields = []
                        sub.status, sub.reason = "failed", "inspection_error"
                result.status = "partial" if any(s.status == "completed" for s in result.subresources) else "failed"
                result.reason = "inspection_error"
        return results

    def _inspect(self, ref: str, result: FileInspection, deadline: float) -> None:
        fmt = result.format
        if fmt in {"csv", "tsv", "arff"}:
            status, headers, body = self.client.get(ref, result.path, byte_range=f"0-{self.guards.max_header_bytes - 1}",
                                                    cap=self.guards.max_full_download_bytes_per_file, deadline=deadline)
            result.representation = "zip" if body.startswith(b"PK\x03\x04") else "raw"
            if result.representation == "zip":
                body = (self._zip_member(body, result.path, read_limit=self.guards.max_header_bytes)
                        if status == 200 else self._wrapped(ref, result, headers, deadline, header_only=True))
                complete = result.declared_size is not None and result.declared_size <= len(body)
            else:
                complete = status == 200 and len(body) <= self.guards.max_header_bytes
                complete = complete or (result.declared_size is not None and len(body) >= result.declared_size)
            body = body[:self.guards.max_header_bytes]
            if fmt == "arff":
                result.fields = _arff_header(body, complete)
            else:
                result.fields, result.warnings = _logical_header(body, "," if fmt == "csv" else "\t", complete)
            self._check_fields(result.fields)
            result.status = "completed"
            return
        if fmt == "parquet":
            self._parquet(ref, result, deadline)
            return
        if result.declared_size is not None and result.declared_size > self.guards.max_expanded_bytes_per_file:
            raise InspectionStop("skipped", "file_too_large")
        status, headers, body = self.client.get(ref, result.path,
                                                cap=self.guards.max_full_download_bytes_per_file, deadline=deadline)
        if status != 200:
            raise InspectionStop("failed", "inspection_error")
        if fmt == "xlsx":
            result.representation = "xlsx" if body.startswith(b"PK\x03\x04") else "other"
            self._xlsx(body, result, deadline)
        else:
            result.representation = "zip" if body.startswith(b"PK\x03\x04") else "sqlite" if body.startswith(b"SQLite format 3\x00") else "other"
            if result.representation == "zip":
                body = self._zip_member(body, result.path)
            if not body.startswith(b"SQLite format 3\x00"):
                raise InspectionStop("failed", "inspection_error")
            self._sqlite(body, result, deadline)

    def _wrapped(self, ref: str, result: FileInspection, headers: dict[str, str], deadline: float,
                 header_only: bool = False) -> bytes:
        content_range = headers.get("content-range", "")
        try:
            compressed_size = int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError) as exc:
            raise InspectionStop("skipped", "budget_exceeded") from exc
        if compressed_size > self.guards.max_full_download_bytes_per_file:
            raise InspectionStop("skipped", "file_too_large")
        status, _, body = self.client.get(ref, result.path,
                                          cap=self.guards.max_full_download_bytes_per_file, deadline=deadline)
        if status != 200 or len(body) != compressed_size:
            raise InspectionStop("failed", "inspection_error")
        return self._zip_member(body, result.path,
                                read_limit=self.guards.max_header_bytes if header_only else None)

    def _zip_member(self, body: bytes, path: str, read_limit: int | None = None) -> bytes:
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                matches = [m for m in members if m.filename == path or m.filename == PurePosixPath(path).name]
                if len(members) != 1 or len(matches) != 1 or ".." in PurePosixPath(matches[0].filename).parts:
                    raise InspectionStop("skipped", "archive")
                member = matches[0]
                if member.file_size > self.guards.max_expanded_bytes_per_file:
                    raise InspectionStop("skipped", "file_too_large")
                with archive.open(member) as stream:
                    expanded = stream.read(min(member.file_size, read_limit or self.guards.max_expanded_bytes_per_file) +
                                           (0 if read_limit else 1))
                if len(expanded) > self.guards.max_expanded_bytes_per_file:
                    raise InspectionStop("skipped", "budget_exceeded")
                return expanded
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise InspectionStop("failed", "inspection_error") from exc

    def _parquet(self, ref: str, result: FileInspection, deadline: float) -> None:
        # A 206 response may be a ZIP byte range; verify the representation first.
        cap = self.guards.max_full_download_bytes_per_file
        status, headers, first = self.client.get(ref, result.path, byte_range="0-3", cap=cap, deadline=deadline)
        if first.startswith(b"PK"):
            result.representation = "zip"
            body = self._zip_member(first, result.path) if status == 200 else self._wrapped(ref, result, headers, deadline)
            self._parquet_bytes(body, result)
            return
        result.representation = "parquet" if first.startswith(b"PAR1") else "other"
        if not first.startswith(b"PAR1"):
            raise InspectionStop("failed", "unsupported_parquet_variant")
        if status == 200:
            if result.declared_size is None or len(first) != result.declared_size:
                raise InspectionStop("skipped", "range_unavailable")
            self._parquet_bytes(first, result)
            return
        if status != 206 or len(first) != 4:
            raise InspectionStop("skipped", "range_unavailable")
        first_total = headers.get("content-range", "").rsplit("/", 1)[-1]
        status, tail_headers, trailer = self.client.get(ref, result.path, byte_range="-8", cap=cap, deadline=deadline)
        if status == 200:
            if result.declared_size is None or len(trailer) != result.declared_size:
                raise InspectionStop("skipped", "range_unavailable")
            self._parquet_bytes(trailer, result)
            return
        if status != 206 or tail_headers.get("content-range", "").rsplit("/", 1)[-1] != first_total:
            raise InspectionStop("skipped", "range_unavailable")
        if len(trailer) != 8:
            raise InspectionStop("failed", "inspection_error")
        if trailer[-4:] == b"PARE":
            raise InspectionStop("skipped", "unsupported_encryption")
        if trailer[-4:] != b"PAR1":
            raise InspectionStop("failed", "unsupported_parquet_variant")
        footer_size = int.from_bytes(trailer[:4], "little")
        if footer_size > self.guards.max_footer_bytes:
            raise InspectionStop("skipped", "footer_too_large")
        if footer_size <= 0:
            raise InspectionStop("failed", "inspection_error")
        status, footer_headers, footer = self.client.get(ref, result.path, byte_range=f"-{footer_size + 8}",
                                                          cap=cap, deadline=deadline)
        if status == 200:
            if result.declared_size is None or len(footer) != result.declared_size:
                raise InspectionStop("skipped", "range_unavailable")
            self._parquet_bytes(footer, result)
            return
        if (status != 206 or footer_headers.get("content-range", "").rsplit("/", 1)[-1] != first_total
                or len(footer) != footer_size + 8 or footer[-8:] != trailer):
            raise InspectionStop("skipped", "range_unavailable")
        self._parse_parquet(b"PAR1" + footer, result)

    def _parquet_bytes(self, body: bytes, result: FileInspection) -> None:
        if not body.startswith(b"PAR1") or len(body) < 12:
            raise InspectionStop("failed", "unsupported_parquet_variant")
        trailer = body[-8:]
        if trailer[-4:] == b"PARE":
            raise InspectionStop("skipped", "unsupported_encryption")
        footer_size = int.from_bytes(trailer[:4], "little")
        if trailer[-4:] != b"PAR1" or footer_size > self.guards.max_footer_bytes:
            raise InspectionStop("skipped", "footer_too_large")
        self._parse_parquet(b"PAR1" + body[-footer_size - 8:], result)

    def _parse_parquet(self, sparse: bytes, result: FileInspection) -> None:
        try:
            import pyarrow.parquet as pq
            schema = pq.read_metadata(io.BytesIO(sparse)).schema
            result.fields = [Field(i, schema.column(i).name, str(schema.column(i)), "native_schema")
                             for i in range(len(schema))]
        except (ImportError, OSError, ValueError) as exc:
            raise InspectionStop("failed", "unsupported_parquet_variant") from exc
        self._check_fields(result.fields)
        result.status = "completed"

    def _xlsx(self, body: bytes, result: FileInspection, deadline: float) -> None:
        try:
            import openpyxl
            from openpyxl.utils.cell import range_boundaries
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                if "[Content_Types].xml" not in archive.namelist():
                    raise InspectionStop("failed", "inspection_error")
                xml = [m for m in archive.infolist() if m.filename.endswith(".xml")]
                total = sum(m.file_size for m in xml)
                if total > self.guards.max_xlsx_xml_bytes or any(
                    m.file_size > self.guards.max_xlsx_xml_member_bytes or
                    m.file_size / max(m.compress_size, 1) > self.guards.max_xlsx_compression_ratio for m in xml
                ):
                    raise InspectionStop("skipped", "budget_exceeded")
                workbook = openpyxl.load_workbook(io.BytesIO(body), read_only=True, data_only=False)
                try:
                    sheets = workbook.worksheets
                    for index, sheet in enumerate(sheets):
                        if time.monotonic() > deadline:
                            result.subresources.append(Subresource(sheet.title, "skipped", "timeout"))
                            continue
                        if index >= self.guards.max_subresources_per_file:
                            result.subresources.append(Subresource(sheet.title, "skipped", "budget_exceeded"))
                            continue
                        sub = Subresource(sheet.title)
                        result.subresources.append(sub)
                        row_number = None
                        row_values = None
                        for row_index, cells in enumerate(sheet.iter_rows(
                                min_row=1, max_row=self.guards.max_xlsx_header_rows,
                                max_col=self.guards.max_fields_per_subresource + 1,
                                values_only=False), start=1):
                            values = [cell.value for cell in cells]
                            if any(value is not None and value != "" for value in values):
                                row_number, row_values = row_index, values
                                break
                        if row_values is None:
                            sub.status, sub.reason = "skipped", "no_schema"
                            continue
                        while row_values and (row_values[-1] is None or row_values[-1] == ""):
                            row_values.pop()
                        labels = [value for value in row_values if value is not None and value != ""]
                        if (len(labels) < 2 or len(labels) != len(row_values) or
                            any(not isinstance(x, str) or len(x) > self.guards.max_field_label_chars for x in labels) or
                            len(labels) != len(set(labels))):
                            sub.status, sub.reason = "skipped", "no_schema"
                            continue
                        path = getattr(sheet, "_worksheet_path", f"xl/worksheets/sheet{index + 1}.xml")
                        if path in archive.namelist():
                            xml_data = archive.read(path)
                            for merged in re.findall(rb'<mergeCell\s+[^>]*ref="([A-Z]+[0-9]+:[A-Z]+[0-9]+)"', xml_data):
                                min_col, min_row, max_col, max_row = range_boundaries(merged.decode("ascii"))
                                if min_row <= row_number <= max_row and min_col <= len(labels) and max_col >= 1:
                                    sub.status, sub.reason = "skipped", "no_schema"
                                    break
                        if sub.status == "skipped":
                            continue
                        sub.fields = [Field(i, label) for i, label in enumerate(labels)]
                        if len(sub.fields) > self.guards.max_fields_per_subresource:
                            sub.fields = []
                            sub.status, sub.reason = "skipped", "budget_exceeded"
                            continue
                        sub.status = "completed"
                finally:
                    workbook.close()
        except (zipfile.BadZipFile, OSError, ValueError, ImportError) as exc:
            raise InspectionStop("failed", "inspection_error") from exc
        self._combine(result)

    def _sqlite(self, body: bytes, result: FileInspection, deadline: float) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.deserialize(body)
            connection.execute("PRAGMA query_only=ON")
            objects = [(name, kind) for schema, name, kind, *_ in connection.execute("PRAGMA table_list")
                       if schema == "main" and kind == "table" and not name.startswith("sqlite_")]
            for index, (name, _) in enumerate(objects):
                if time.monotonic() > deadline:
                    result.subresources.append(Subresource(name, "skipped", "timeout"))
                    continue
                if index >= self.guards.max_subresources_per_file:
                    result.subresources.append(Subresource(name, "skipped", "budget_exceeded"))
                    continue
                sub = Subresource(name)
                result.subresources.append(sub)
                escaped = name.replace('"', '""')
                rows = connection.execute(f'PRAGMA table_xinfo("{escaped}")').fetchall()
                sub.fields = [Field(row[0], row[1], row[2] or None, "native_schema" if row[2] else None)
                              for row in rows if row[6] in (0, 2, 3)]
                if len(sub.fields) > self.guards.max_fields_per_subresource:
                    sub.fields = []
                    sub.status, sub.reason = "skipped", "budget_exceeded"
                    continue
                sub.status = "completed" if sub.fields else "skipped"
                sub.reason = None if sub.fields else "no_schema"
        except sqlite3.DatabaseError as exc:
            raise InspectionStop("failed", "inspection_error") from exc
        finally:
            connection.close()
        self._combine(result)

    def _check_fields(self, fields: list[Field]) -> None:
        if len(fields) > self.guards.max_fields_per_subresource:
            raise InspectionStop("skipped", "budget_exceeded")

    @staticmethod
    def _combine(result: FileInspection) -> None:
        statuses = [sub.status for sub in result.subresources]
        reasons = {sub.reason for sub in result.subresources}
        reason = next((value for value in ("timeout", "budget_exceeded", "inspection_error", "no_schema")
                       if value in reasons), next((value for value in reasons if value), "no_schema"))
        if not statuses:
            result.status, result.reason = "skipped", "no_schema"
        elif all(status == "completed" for status in statuses):
            result.status, result.reason = "completed", None
        elif "completed" in statuses:
            result.status, result.reason = "partial", reason
        else:
            result.status, result.reason = ("failed" if "failed" in statuses else "skipped"), reason
