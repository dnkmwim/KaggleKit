from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class SyncConfig:
    bootstrap_page_limit: int = 20
    retry_max_attempts: int = 5
    timeout_seconds: float = 30.0
    backoff_base_seconds: float = 2.0
    rate_limit_backoff_seconds: float = 180.0
    request_delay_seconds: float = 0.8

    @classmethod
    def from_env(cls) -> "SyncConfig":
        config = cls(
            bootstrap_page_limit=int(os.getenv("DATASET_SYNC_BOOTSTRAP_PAGE_LIMIT", "20")),
            retry_max_attempts=int(os.getenv("DATASET_SYNC_RETRY_MAX_ATTEMPTS", "5")),
            timeout_seconds=float(os.getenv("DATASET_SYNC_TIMEOUT_SECONDS", "30")),
            backoff_base_seconds=float(os.getenv("DATASET_SYNC_BACKOFF_BASE_SECONDS", "2")),
            rate_limit_backoff_seconds=float(os.getenv("DATASET_SYNC_RATE_LIMIT_BACKOFF_SECONDS", "180")),
            request_delay_seconds=float(os.getenv("DATASET_SYNC_REQUEST_DELAY_SECONDS", "0.8")),
        )
        if config.bootstrap_page_limit < 1 or config.retry_max_attempts < 1:
            raise ValueError("Page limit and retry attempts must be positive")
        if min(
            config.timeout_seconds,
            config.backoff_base_seconds,
            config.rate_limit_backoff_seconds,
            config.request_delay_seconds,
        ) < 0:
            raise ValueError("Timing settings cannot be negative")
        return config
