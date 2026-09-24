from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["not_run", "completed", "partial", "skipped", "failed"]


@dataclass
class Field:
    ordinal: int
    name: str
    type_raw: str | None = None
    type_source: str | None = None


@dataclass
class Subresource:
    name: str
    status: Status = "not_run"
    reason: str | None = None
    fields: list[Field] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FileInspection:
    path: str
    format: str
    declared_size: int | None = None
    status: Status = "not_run"
    reason: str | None = None
    representation: str | None = None
    fields: list[Field] = field(default_factory=list)
    subresources: list[Subresource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetFile:
    path: str
    size: int | None
