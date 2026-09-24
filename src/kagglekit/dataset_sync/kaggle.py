from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import random
import ssl
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import SyncConfig
from .models import DatasetMetadata, PageResult
from .sources import Source


class TransientKaggleError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None, retry_count: int = 0):
        super().__init__(message)
        self.http_status = http_status
        self.retry_count = retry_count


@dataclass(frozen=True, slots=True)
class JsonResponse:
    payload: Any
    status: int
    latency_ms: int


Transport = Callable[[str, float], JsonResponse]


class KaggleAdapter:
    BASE_URL = "https://www.kaggle.com/api/v1"

    def __init__(
        self,
        config: SyncConfig,
        *,
        transport: Transport | None = None,
        token: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        active_token = token if token is not None else os.getenv("KAGGLE_API_TOKEN")
        self.transport = transport or (
            lambda url, timeout: _stdlib_transport(url, timeout, token=active_token)
        )
        self.sleep = sleep
        self.random_value = random_value

    def list_datasets(self, source: Source, sort: str, page: int) -> PageResult:
        params = {"page": page, "sortby": sort}
        params["tagids" if source.kind == "tag" else "search"] = source.value
        response, retry_count = self._get(f"{self.BASE_URL}/datasets/list?{urlencode(params)}")
        if not isinstance(response.payload, list):
            raise ValueError("Kaggle Dataset list returned a non-list payload")
        items = [DatasetMetadata.from_api(item) for item in response.payload]
        return PageResult(items, response.status, response.latency_ms, retry_count)

    def dataset_detail(self, ref: str) -> tuple[DatasetMetadata, int]:
        response, retry_count = self._get(f"{self.BASE_URL}/datasets/view/{quote(ref, safe='/')}")
        if not isinstance(response.payload, dict):
            raise ValueError("Kaggle Dataset detail returned a non-object payload")
        return DatasetMetadata.from_api(response.payload), retry_count

    def _get(self, url: str) -> tuple[JsonResponse, int]:
        last_error: BaseException | None = None
        for attempt in range(1, self.config.retry_max_attempts + 1):
            if self.config.request_delay_seconds:
                self.sleep(self.config.request_delay_seconds)
            try:
                return self.transport(url, self.config.timeout_seconds), attempt - 1
            except HTTPError as exc:
                if exc.code != 429 and not 500 <= exc.code <= 599:
                    raise
                last_error = exc
                if attempt < self.config.retry_max_attempts:
                    self.sleep(
                        self._retry_delay(
                            attempt, exc.headers.get("Retry-After"), rate_limited=exc.code == 429
                        )
                    )
            except (TimeoutError, URLError, ssl.SSLError, ConnectionError) as exc:
                last_error = exc
                if attempt < self.config.retry_max_attempts:
                    self.sleep(self._retry_delay(attempt, None, rate_limited=False))
        status = last_error.code if isinstance(last_error, HTTPError) else None
        raise TransientKaggleError(
            f"Kaggle request exhausted retries: {last_error}",
            http_status=status,
            retry_count=self.config.retry_max_attempts - 1,
        ) from last_error

    def _retry_delay(self, attempt: int, retry_after: str | None, *, rate_limited: bool) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    value = parsedate_to_datetime(retry_after)
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    return max(0.0, (value - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        exponential = self.config.backoff_base_seconds * (2 ** (attempt - 1))
        if rate_limited:
            exponential = max(exponential, self.config.rate_limit_backoff_seconds)
        return exponential + self.random_value()


def _stdlib_transport(url: str, timeout: float, *, token: str | None = None) -> JsonResponse:
    started = time.perf_counter()
    headers = {"Accept": "application/json", "User-Agent": "KaggleKit/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        latency_ms = int((time.perf_counter() - started) * 1000)
        return JsonResponse(payload=payload, status=response.status, latency_ms=latency_ms)
