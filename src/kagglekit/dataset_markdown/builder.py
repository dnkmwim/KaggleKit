from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .config import RuntimeGuards
from .inspector import KaggleFileClient, SchemaInspector
from .models import FileInspection
from .renderer import render_markdown


@dataclass(frozen=True)
class DatasetResource:
    source_id: str
    title: str
    url: str
    description: str | None = None
    subtitle: str | None = None
    tags: tuple[str, ...] = ()
    author: str | None = None
    updated_at: datetime | None = None
    votes: int | None = None
    downloads: int | None = None
    usability_rating: float | None = None
    size_bytes: int | None = None
    license: str | None = None


class DatasetReader(Protocol):
    def get(self, source_id: str) -> DatasetResource | None: ...


class PostgresDatasetReader:
    """Read-only access to the existing normalized resources table."""

    def __init__(self, database_url: str):
        import psycopg
        from psycopg.rows import dict_row

        self.connection = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
        self.connection.read_only = True

    def get(self, source_id: str) -> DatasetResource | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT source_id,title,url,description,subtitle,tags,author,updated_at,"
                "votes,downloads,usability_rating,size_bytes,license FROM resources "
                "WHERE source='kaggle' AND resource_type='dataset' AND status='active' "
                "AND source_id=%s",
                (source_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        row["tags"] = tuple(row["tags"] or ())
        return DatasetResource(**row)

    def close(self) -> None:
        self.connection.close()


class DatasetMarkdownBuilder:
    def __init__(self, client: KaggleFileClient | None = None, guards: RuntimeGuards | None = None):
        self.guards = guards or RuntimeGuards()
        self.client = client or KaggleFileClient(self.guards)

    def build(self, resource: DatasetResource) -> tuple[str, list[FileInspection]]:
        if not resource.source_id or not resource.title or not resource.url:
            raise ValueError("Dataset resource lacks required identity")
        if resource.url != f"https://www.kaggle.com/datasets/{resource.source_id}":
            raise ValueError("Dataset URL does not match official Kaggle source_id")
        inspector = SchemaInspector(self.client, self.guards)
        files, list_error = self.client.list_files(resource.source_id)
        inspections = inspector.inspect_dataset(resource.source_id, files) if list_error is None else []
        return render_markdown(resource, files, inspections, self.guards, list_error), inspections

    def build_from_reader(self, reader: DatasetReader, source_id: str) -> tuple[str, list[FileInspection]]:
        resource = reader.get(source_id)
        if resource is None:
            raise LookupError(f"Active Kaggle Dataset not found: {source_id}")
        return self.build(resource)
