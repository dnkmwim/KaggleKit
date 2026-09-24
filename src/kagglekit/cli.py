from __future__ import annotations

import argparse
import json
import os

from .acquisition_benchmark import benchmark, discover_refs
from .dataset_sync.config import SyncConfig
from .dataset_sync.kaggle import KaggleAdapter
from .dataset_sync.repository import InMemoryRepository, PostgresRepository
from .dataset_sync.service import DatasetSyncService
from .dataset_sync.sources import select_sources
from .hybrid_evaluation import load_json, run_evaluation, write_json
from .hybrid_search import HybridSearchService
from .keyword_search import KeywordSearchService, PostgresKeywordSearchRepository
from .semantic_search import (
    FastEmbedBackend,
    PostgresSemanticSearchRepository,
    SemanticBackfillService,
    SemanticSearchError,
    SemanticSearchService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kagglekit")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset_sync = commands.add_parser("dataset-sync", help="Sync Kaggle Dataset candidates")
    modes = dataset_sync.add_subparsers(dest="mode", required=True)
    for mode in ("bootstrap", "incremental"):
        item = modes.add_parser(mode)
        item.add_argument("--source", help="One frozen source, e.g. tag:classification")
        item.add_argument("--dry-run", action="store_true", help="Call Kaggle but do not write PostgreSQL")
    smoke = modes.add_parser("smoke", help="Fetch one live page without database writes")
    smoke.add_argument("--source", default="tag:classification")
    search = commands.add_parser("search", help="Search local Resource Metadata")
    search_modes = search.add_subparsers(dest="search_mode", required=True)
    keyword = search_modes.add_parser("keyword", help="PostgreSQL full-text keyword search")
    keyword.add_argument("--query", action="append", required=True, dest="queries")
    keyword.add_argument("--resource-type", choices=("dataset", "notebook"))
    keyword.add_argument("--limit", type=int, default=20)
    keyword.add_argument("--offset", type=int, default=0)
    semantic = search_modes.add_parser("semantic", help="PostgreSQL pgvector semantic search")
    semantic.add_argument("--query", required=True)
    semantic.add_argument("--resource-type", choices=("dataset", "notebook"), default="dataset")
    semantic.add_argument("--limit", type=int, default=20)
    semantic.add_argument("--offset", type=int, default=0)
    hybrid = search_modes.add_parser("hybrid", help="RRF keyword + semantic search")
    hybrid.add_argument("--lexical-query", action="append", required=True, dest="lexical_queries")
    hybrid.add_argument("--semantic-query", required=True)
    hybrid.add_argument("--resource-type", choices=("dataset", "notebook"), default="dataset")
    hybrid.add_argument("--limit", type=int, default=20)
    hybrid.add_argument("--offset", type=int, default=0)
    hybrid.add_argument("--rrf-k", type=int, default=60)
    hybrid.add_argument("--keyword-top-k", type=int, default=20)
    hybrid.add_argument("--semantic-top-k", type=int, default=20)
    hybrid.add_argument("--include-timing", action="store_true")
    backfill = commands.add_parser("semantic-backfill", help="Build dataset semantic embeddings")
    backfill.add_argument("--batch-size", type=int, default=32)
    backfill.add_argument("--rebuild", action="store_true")
    evaluate = commands.add_parser("hybrid-evaluate", help="Generate the 24-query review table")
    evaluate.add_argument("--queries", default="evaluation/hybrid_search_queries.json")
    evaluate.add_argument("--output", default="evaluation/hybrid_search_results.json")
    evaluate.add_argument("--resource-type", choices=("dataset", "notebook"), default="dataset")
    evaluate.add_argument("--rrf-k", type=int, default=60)
    evaluate.add_argument("--keyword-top-k", type=int, default=20)
    evaluate.add_argument("--semantic-top-k", type=int, default=20)
    acquisition = commands.add_parser(
        "acquisition-benchmark", help="Compare Kaggle metadata acquisition methods"
    )
    acquisition_sample = acquisition.add_mutually_exclusive_group(required=True)
    acquisition_sample.add_argument("--ref", action="append", dest="refs")
    acquisition_sample.add_argument(
        "--source", help="Discover a live Kaggle sample, e.g. tag:classification"
    )
    acquisition.add_argument("--sample-size", type=int, default=40)
    acquisition.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=("rest-sequential", "rest-concurrent", "kaggle-cli"),
    )
    acquisition.add_argument("--workers", type=int, default=4)
    acquisition.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "acquisition-benchmark":
        try:
            refs = args.refs or discover_refs(args.source, args.sample_size)
            report = benchmark(
                refs,
                args.methods or ["rest-sequential", "rest-concurrent", "kaggle-cli"],
                workers=args.workers,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"output": output, "recommendation": report["recommendation"]}))
        return 0
    if args.command == "hybrid-evaluate":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise SystemExit("DATABASE_URL is required for hybrid evaluation")
        keyword_repository = PostgresKeywordSearchRepository(database_url)
        semantic_repository = PostgresSemanticSearchRepository(database_url)
        try:
            try:
                queries = load_json(args.queries)
                previous = load_json(args.output) if os.path.exists(args.output) else {"rows": []}
                report = run_evaluation(
                    queries,
                    KeywordSearchService(keyword_repository),
                    SemanticSearchService(semantic_repository, FastEmbedBackend()),
                    resource_type=args.resource_type,
                    rrf_k=args.rrf_k,
                    keyword_top_k=args.keyword_top_k,
                    semantic_top_k=args.semantic_top_k,
                    existing_rows=previous.get("rows", []),
                )
                write_json(args.output, report)
            except (ValueError, SemanticSearchError, OSError, json.JSONDecodeError) as exc:
                raise SystemExit(str(exc)) from exc
            print(json.dumps({"output": args.output, "rows": len(report["rows"])}, ensure_ascii=True))
            return 0
        finally:
            keyword_repository.close()
            semantic_repository.close()

    if args.command == "semantic-backfill":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise SystemExit("DATABASE_URL is required for semantic backfill")
        repository = PostgresSemanticSearchRepository(database_url)
        try:
            try:
                stats = SemanticBackfillService(
                    repository, FastEmbedBackend(), batch_size=args.batch_size
                ).run(rebuild=args.rebuild)
            except (ValueError, SemanticSearchError) as exc:
                raise SystemExit(str(exc)) from exc
            print(json.dumps(stats.to_dict(), ensure_ascii=True))
            return 0
        finally:
            repository.close()

    if args.command == "search":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise SystemExit("DATABASE_URL is required for search")
        if args.search_mode == "semantic":
            repository = PostgresSemanticSearchRepository(database_url)
            try:
                try:
                    results = SemanticSearchService(repository, FastEmbedBackend()).search(
                        args.query,
                        resource_type=args.resource_type,
                        limit=args.limit,
                        offset=args.offset,
                    )
                except (ValueError, SemanticSearchError) as exc:
                    raise SystemExit(str(exc)) from exc
                print(json.dumps([result.to_dict() for result in results], ensure_ascii=True))
                return 0
            finally:
                repository.close()

        if args.search_mode == "hybrid":
            keyword_repository = PostgresKeywordSearchRepository(database_url)
            semantic_repository = PostgresSemanticSearchRepository(database_url)
            try:
                try:
                    response = HybridSearchService(
                        KeywordSearchService(keyword_repository),
                        SemanticSearchService(semantic_repository, FastEmbedBackend()),
                    ).search_with_timing(
                        args.lexical_queries,
                        args.semantic_query,
                        resource_type=args.resource_type,
                        limit=args.limit,
                        offset=args.offset,
                        rrf_k=args.rrf_k,
                        keyword_top_k=args.keyword_top_k,
                        semantic_top_k=args.semantic_top_k,
                    )
                except (ValueError, SemanticSearchError) as exc:
                    raise SystemExit(str(exc)) from exc
                payload: object = [result.to_dict() for result in response.results]
                if args.include_timing:
                    payload = {"results": payload, "timing": response.timing.to_dict()}
                print(json.dumps(payload, ensure_ascii=True))
                return 0
            finally:
                keyword_repository.close()
                semantic_repository.close()

        repository = PostgresKeywordSearchRepository(database_url)
        try:
            try:
                results = KeywordSearchService(repository).search(
                    args.queries,
                    resource_type=args.resource_type,
                    limit=args.limit,
                    offset=args.offset,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            # ASCII-safe JSON remains valid across Windows console code pages
            # even when Kaggle metadata contains emoji or non-GBK characters.
            print(json.dumps([result.to_dict() for result in results], ensure_ascii=True))
            return 0
        finally:
            repository.close()

    config = SyncConfig.from_env()
    sources = select_sources(args.source)
    adapter = KaggleAdapter(config)
    if args.mode == "smoke":
        result = adapter.list_datasets(sources[0], "updated", 1)
        valid = sum(item.valid_for_resource() for item in result.items)
        print(
            f"source={sources[0].key} status={result.http_status} "
            f"count={len(result.items)} valid={valid} latency_ms={result.latency_ms}"
        )
        return 0

    dry_run = bool(args.dry_run)
    database_url = os.getenv("DATABASE_URL")
    if dry_run:
        repository = InMemoryRepository()
    elif database_url:
        repository = PostgresRepository(database_url)
    else:
        raise SystemExit("DATABASE_URL is required unless --dry-run is used")

    try:
        result = DatasetSyncService(adapter, repository, config, dry_run=dry_run).run(
            args.mode, sources
        )
        print(f"run_id={result.run_id} candidates={result.candidates} upserted={result.resources_upserted}")
        return 0
    finally:
        close = getattr(repository, "close", None)
        if close:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
