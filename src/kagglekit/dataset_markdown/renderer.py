"""Deterministic Dataset Markdown V1 rendering from normalized, sourced facts."""

from __future__ import annotations

from datetime import datetime
import json
import re
from typing import TYPE_CHECKING

from .config import RuntimeGuards
from .models import DatasetFile, Field, FileInspection

if TYPE_CHECKING:
    from .builder import DatasetResource


def _yaml_value(value: object) -> str:
    # JSON scalars/arrays/objects are a YAML 1.2 subset and preserve escaping.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _display(value: str, limit: int) -> str:
    text = value if value else "(empty)"
    if len(text) > limit:
        text = text[:limit] + "…"
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "↵").replace("\n", "↵")


def _description(text: str) -> str:
    """Display-only heading demotion; original description stays in PostgreSQL."""
    source = text.splitlines()
    kinds: dict[int, tuple[int, str]] = {}
    fenced = False
    fence_char = ""
    index = 0
    while index < len(source):
        line = source[index]
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)[0]
            if not fenced:
                fenced, fence_char = True, marker
            elif marker == fence_char:
                fenced = False
        if not fenced and not fence:
            atx = re.match(r"^(#{1,6})(\s+.*)$", line)
            if atx:
                kinds[index] = (len(atx.group(1)), "atx")
            elif line.strip() and index + 1 < len(source):
                setext = re.match(r"^\s*(=+|-+)\s*$", source[index + 1])
                if setext:
                    kinds[index] = (1 if setext.group(1)[0] == "=" else 2, "setext")
                    index += 1
        index += 1
    shift = max(0, 3 - min((level for level, _ in kinds.values()), default=3))
    output = []
    index = 0
    while index < len(source):
        line = source[index]
        if index in kinds:
            level, kind = kinds[index]
            marker = "#" * min(level + shift, 6)
            if kind == "setext":
                output.append(marker + " " + line.strip())
                index += 2
                continue
            atx = re.match(r"^(#{1,6})(\s+.*)$", line)
            output.append(marker + atx.group(2))
        else:
            output.append(line)
        index += 1
    return "\n".join(output)


def _field_table(fields: list[Field], guards: RuntimeGuards, warnings: list[str]) -> list[str]:
    lines = ["| Ordinal | Field | Native type |", "| ---: | --- | --- |"]
    for field in fields:
        if len(field.name) > guards.max_field_label_chars and "display_name_truncated" not in warnings:
            warnings.append("display_name_truncated")
        display_type = field.type_raw or ""
        if display_type.startswith("<ParquetColumnSchema>"):
            physical = re.search(r"physical_type: ([^\n]+)", display_type)
            logical = re.search(r"logical_type: ([^\n]+)", display_type)
            if physical:
                display_type = physical.group(1)
                if logical and logical.group(1) != "None":
                    display_type += " (" + logical.group(1) + ")"
        native = _display(display_type, guards.max_field_label_chars) if display_type else ""
        lines.append(f"| {field.ordinal} | {_display(field.name, guards.max_field_label_chars)} | {native} |")
    return lines


def render_markdown(resource: DatasetResource, files: list[DatasetFile],
                    inspections: list[FileInspection], guards: RuntimeGuards,
                    list_error: str | None = None) -> str:
    identity: dict[str, object] = {
        "schema_version": "dataset-md-v1.0", "source": "kaggle", "source_id": resource.source_id,
        "resource_type": "dataset", "title": resource.title, "original_url": resource.url,
    }
    optional = {
        "subtitle": resource.subtitle, "author": resource.author,
        "updated_at": resource.updated_at.isoformat() if isinstance(resource.updated_at, datetime) else None,
        "license": resource.license, "downloads": resource.downloads, "votes": resource.votes,
        "usability_rating": resource.usability_rating, "size_bytes": resource.size_bytes,
        "tags": list(resource.tags) if resource.tags else None,
    }
    identity.update({key: value for key, value in optional.items() if value is not None})
    if files and list_error is None:
        formats = sorted({item.format for item in inspections})
        identity["files"] = {
            "count": len(files), "formats": formats, "format_detection": "path_extension",
            "has_directory_structure": any("/" in item.path for item in files),
            "unknown_file_count": sum(item.format == "unknown" for item in inspections),
            "examples": [item.path for item in files[:guards.max_file_examples]],
        }
    lines = ["---", *[f"{key}: {_yaml_value(value)}" for key, value in identity.items()], "---", "",
             f"# {resource.title.replace(chr(10), ' ')}", ""]
    if resource.subtitle:
        lines += ["> " + resource.subtitle.replace("\n", " "), ""]
    if resource.description:
        lines += ["## Description", "", _description(resource.description), ""]
    if files or list_error:
        lines += ["## Files", ""]
        if list_error:
            lines += [f"File list unavailable ({list_error}); schema inspection not run.", ""]
        else:
            lines += [f"- File count: {len(files)}", "- Formats (path extension): " + ", ".join(sorted({item.format for item in inspections})),
                      f"- Directory structure: {'yes' if any('/' in item.path for item in files) else 'no'}",
                      f"- Unknown format count: {sum(item.format == 'unknown' for item in inspections)}"]
            for item in files[:guards.max_file_examples]:
                lines.append("- Example: `" + item.path.replace("`", "\\`") + "`")
            lines.append("")
            for item in inspections[:guards.max_file_examples]:
                if item.status != "completed":
                    lines.append("- Inspection `" + item.path.replace("`", "\\`") + "`: " + item.status +
                                 (f" ({item.reason})" if item.reason else ""))
            if lines[-1] != "":
                lines.append("")
    sections = []
    for item in inspections:
        if item.status == "completed" and item.fields:
            sections += [f"### {_display(item.path, 500)}", "", *_field_table(item.fields, guards, item.warnings), ""]
        usable = [sub for sub in item.subresources if sub.status == "completed" and sub.fields]
        if usable:
            if not item.fields:
                sections += [f"### {_display(item.path, 500)}", ""]
            for sub in usable:
                sections += [f"#### {_display(sub.name, 500)}", "", *_field_table(sub.fields, guards, sub.warnings), ""]
    if sections:
        lines += ["## Fields", "", *sections]
    lines += ["## Source", "", "- Platform: Kaggle", f"- Original URL: {resource.url}", ""]
    return "\n".join(lines)
