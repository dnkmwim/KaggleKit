"""Fetch one public Kaggle metadata page without writing a database."""

from kagglekit.dataset_sync.config import SyncConfig
from kagglekit.dataset_sync.kaggle import KaggleAdapter
from kagglekit.dataset_sync.sources import Source


def main() -> None:
    result = KaggleAdapter(SyncConfig.from_env()).list_datasets(
        Source("tag", "classification"), "updated", 1
    )
    for item in result.items[:5]:
        print(f"{item.ref}\t{item.title}\t{item.url}")


if __name__ == "__main__":
    main()

