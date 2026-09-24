from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


@dataclass(slots=True)
class DatasetMetadata:
    ref: str
    title: str
    url: str
    kaggle_id: int | None = None
    subtitle: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    last_updated: datetime | None = None
    current_version_number: int | None = None
    votes: int | None = None
    downloads: int | None = None
    usability_rating: float | None = None
    size_bytes: int | None = None
    license: str | None = None
    is_private: bool = False

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "DatasetMetadata":
        tag_values: list[str] = []
        for tag in raw.get("tags") or raw.get("keywords") or []:
            value = tag.get("ref") or tag.get("name") if isinstance(tag, dict) else tag
            if value:
                tag_values.append(str(value))
        owner = raw.get("ownerName") or raw.get("ownerRef") or raw.get("creatorName")
        license_value = raw.get("licenseName")
        if not license_value and raw.get("licenses"):
            first = raw["licenses"][0]
            license_value = first.get("name") if isinstance(first, dict) else first
        ref = str(raw.get("ref") or raw.get("id") or "")
        url = str(raw.get("url") or (f"https://www.kaggle.com/datasets/{ref}" if ref else ""))
        return cls(
            ref=ref,
            kaggle_id=raw.get("id") if isinstance(raw.get("id"), int) else None,
            title=str(raw.get("title") or ""),
            subtitle=_text(raw.get("subtitle")),
            url=url,
            tags=tag_values,
            author=_text(owner),
            last_updated=parse_datetime(raw.get("lastUpdated") or raw.get("updatedAt")),
            current_version_number=_int(raw.get("currentVersionNumber")),
            votes=_int(_first_present(raw, "voteCount", "votes")),
            downloads=_int(_first_present(raw, "downloadCount", "downloads")),
            usability_rating=_float(raw.get("usabilityRating")),
            size_bytes=_int(_first_present(raw, "totalBytes", "size")),
            license=_text(license_value),
            description=_text(raw.get("description")),
            is_private=bool(raw.get("isPrivate", False)),
        )

    def valid_for_resource(self) -> bool:
        return bool(self.ref and self.title and self.url and not self.is_private)


@dataclass(slots=True)
class Candidate:
    metadata: DatasetMetadata
    hit_sources: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class VersionState:
    last_updated: datetime | None
    current_version_number: int | None
    detail_pending: bool = False


@dataclass(slots=True)
class PageResult:
    items: list[DatasetMetadata]
    http_status: int
    latency_ms: int
    retry_count: int = 0


def merge_non_empty(base: DatasetMetadata, override: DatasetMetadata | None) -> DatasetMetadata:
    values = {item.name: getattr(base, item.name) for item in fields(DatasetMetadata)}
    if override is not None:
        for item in fields(DatasetMetadata):
            value = getattr(override, item.name)
            if _non_empty(value):
                values[item.name] = value
    return DatasetMetadata(**values)


def version_changed(metadata: DatasetMetadata, state: VersionState | None) -> bool:
    if state is None or state.detail_pending:
        return True
    return (
        metadata.last_updated != state.last_updated
        or metadata.current_version_number != state.current_version_number
    )


def _non_empty(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None
