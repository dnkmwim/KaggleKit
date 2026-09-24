from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

from .builder import DatasetMarkdownBuilder, DatasetResource, PostgresDatasetReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one Dataset Markdown V1 document, without Dataset Sync writes")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-id", help="Read one active Kaggle Dataset from PostgreSQL")
    source.add_argument("--metadata-json", type=Path, help="Read one existing normalized Dataset metadata JSON")
    parser.add_argument("--output", type=Path, help="Write Markdown here; otherwise print to stdout")
    parser.add_argument("--inspection-json", type=Path, help="Optional exact canonical inspection sidecar")
    args = parser.parse_args()
    builder = DatasetMarkdownBuilder()
    if args.metadata_json:
        raw = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if isinstance(raw.get("updated_at"), str):
            raw["updated_at"] = datetime.fromisoformat(raw["updated_at"].replace("Z", "+00:00"))
        raw["tags"] = tuple(raw.get("tags") or ())
        resource = DatasetResource(**raw)
        markdown, inspections = builder.build(resource)
    else:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            parser.error("DATABASE_URL is required with --source-id")
        reader = PostgresDatasetReader(database_url)
        try:
            markdown, inspections = builder.build_from_reader(reader, args.source_id)
        finally:
            reader.close()
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    if args.inspection_json:
        args.inspection_json.write_text(json.dumps([item.to_dict() for item in inspections],
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
