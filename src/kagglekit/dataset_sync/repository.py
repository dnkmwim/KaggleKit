from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from .models import DatasetMetadata, VersionState


class Repository(Protocol):
    def get_version_states(self, refs: list[str]) -> dict[str, VersionState]: ...
    def get_resources(self, refs: list[str]) -> dict[str, DatasetMetadata]: ...
    def upsert_resource(self, metadata: DatasetMetadata) -> None: ...
    def save_version_state(self, metadata: DatasetMetadata, *, detail_pending: bool, detail_error: str | None) -> None: ...
    def latest_watermark(self, source_key: str, sort: str) -> datetime | None: ...
    def save_checkpoint(self, checkpoint: "Checkpoint") -> None: ...


class Checkpoint:
    def __init__(
        self,
        *,
        run_id: UUID,
        mode: str,
        source_key: str,
        sort: str,
        last_successful_page: int,
        watermark: datetime | None,
        status: str,
        stop_reason: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.run_id = run_id
        self.mode = mode
        self.source_key = source_key
        self.sort = sort
        self.last_successful_page = last_successful_page
        self.watermark = watermark
        self.status = status
        self.stop_reason = stop_reason
        self.retryable = retryable


class InMemoryRepository:
    def __init__(self) -> None:
        self.resources: dict[str, DatasetMetadata] = {}
        self.resource_ids: dict[str, UUID] = {}
        self.version_states: dict[str, VersionState] = {}
        self.checkpoints: list[Checkpoint] = []
        self.upsert_count = 0

    def get_version_states(self, refs: list[str]) -> dict[str, VersionState]:
        return {ref: self.version_states[ref] for ref in refs if ref in self.version_states}

    def get_resources(self, refs: list[str]) -> dict[str, DatasetMetadata]:
        return {ref: self.resources[ref] for ref in refs if ref in self.resources}

    def upsert_resource(self, metadata: DatasetMetadata) -> None:
        self.resource_ids.setdefault(metadata.ref, uuid4())
        self.resources[metadata.ref] = metadata
        self.upsert_count += 1

    def save_version_state(self, metadata: DatasetMetadata, *, detail_pending: bool, detail_error: str | None) -> None:
        self.version_states[metadata.ref] = VersionState(
            metadata.last_updated, metadata.current_version_number, detail_pending
        )

    def latest_watermark(self, source_key: str, sort: str) -> datetime | None:
        values = [
            item.watermark
            for item in self.checkpoints
            if item.source_key == source_key and item.sort == sort and item.status == "complete" and item.watermark
        ]
        return max(values) if values else None

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        key = (checkpoint.run_id, checkpoint.mode, checkpoint.source_key, checkpoint.sort)
        self.checkpoints = [
            item
            for item in self.checkpoints
            if (item.run_id, item.mode, item.source_key, item.sort) != key
        ]
        self.checkpoints.append(checkpoint)


class PostgresRepository:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use PostgreSQL") from exc
        # Reads must not leave an implicit outer transaction open around the
        # explicit write transactions below; otherwise close() rolls them back.
        self.connection = psycopg.connect(database_url, autocommit=True)

    def close(self) -> None:
        self.connection.close()

    def get_version_states(self, refs: list[str]) -> dict[str, VersionState]:
        if not refs:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT source_id, last_updated, current_version_number, detail_pending "
                "FROM kaggle_dataset_sync_state WHERE source_id = ANY(%s)", (refs,)
            )
            return {row[0]: VersionState(row[1], row[2], row[3]) for row in cursor.fetchall()}

    def get_resources(self, refs: list[str]) -> dict[str, DatasetMetadata]:
        if not refs:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT source_id,title,url,subtitle,description,tags,author,updated_at,votes,downloads,"
                "usability_rating,size_bytes,license FROM resources "
                "WHERE source='kaggle' AND source_id = ANY(%s)", (refs,)
            )
            result = {}
            for row in cursor.fetchall():
                result[row[0]] = DatasetMetadata(
                    ref=row[0], title=row[1], url=row[2], subtitle=row[3], description=row[4],
                    tags=list(row[5]), author=row[6], last_updated=row[7], votes=row[8],
                    downloads=row[9], usability_rating=row[10], size_bytes=row[11], license=row[12],
                )
            return result

    def upsert_resource(self, metadata: DatasetMetadata) -> None:
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO resources
                  (id,source,source_id,resource_type,title,url,description,tags,author,updated_at,
                   votes,subtitle,license,downloads,usability_rating,size_bytes,indexed_at,last_synced_at,status)
                VALUES
                  (%s,'kaggle',%s,'dataset',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now(),'active')
                ON CONFLICT (source,source_id) DO UPDATE SET
                  title=EXCLUDED.title,url=EXCLUDED.url,description=EXCLUDED.description,tags=EXCLUDED.tags,
                  author=EXCLUDED.author,updated_at=EXCLUDED.updated_at,votes=EXCLUDED.votes,
                  subtitle=EXCLUDED.subtitle,license=EXCLUDED.license,downloads=EXCLUDED.downloads,
                  usability_rating=EXCLUDED.usability_rating,size_bytes=EXCLUDED.size_bytes,
                  last_synced_at=now(),status='active'
                """,
                (
                    uuid4(), metadata.ref, metadata.title, metadata.url, metadata.description,
                    metadata.tags, metadata.author, metadata.last_updated, metadata.votes,
                    metadata.subtitle, metadata.license, metadata.downloads,
                    metadata.usability_rating, metadata.size_bytes,
                ),
            )

    def save_version_state(self, metadata: DatasetMetadata, *, detail_pending: bool, detail_error: str | None) -> None:
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO kaggle_dataset_sync_state
                  (source_id,last_updated,current_version_number,detail_pending,detail_error,updated_at)
                VALUES (%s,%s,%s,%s,%s,now())
                ON CONFLICT (source_id) DO UPDATE SET
                  last_updated=EXCLUDED.last_updated,
                  current_version_number=EXCLUDED.current_version_number,
                  detail_pending=EXCLUDED.detail_pending,
                  detail_error=EXCLUDED.detail_error,
                  updated_at=now()
                """,
                (metadata.ref, metadata.last_updated, metadata.current_version_number, detail_pending, detail_error),
            )

    def latest_watermark(self, source_key: str, sort: str) -> datetime | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT max(watermark) FROM dataset_sync_checkpoints "
                "WHERE source_key=%s AND sort=%s AND status='complete'", (source_key, sort)
            )
            return cursor.fetchone()[0]

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO dataset_sync_checkpoints
                  (run_id,mode,source_key,sort,last_successful_page,watermark,status,stop_reason,retryable,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (run_id,mode,source_key,sort) DO UPDATE SET
                  last_successful_page=EXCLUDED.last_successful_page,
                  watermark=EXCLUDED.watermark,status=EXCLUDED.status,
                  stop_reason=EXCLUDED.stop_reason,retryable=EXCLUDED.retryable,updated_at=now()
                """,
                (
                    checkpoint.run_id, checkpoint.mode, checkpoint.source_key, checkpoint.sort,
                    checkpoint.last_successful_page, checkpoint.watermark, checkpoint.status,
                    checkpoint.stop_reason, checkpoint.retryable,
                ),
            )
