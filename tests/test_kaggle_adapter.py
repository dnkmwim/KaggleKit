from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from kagglekit.dataset_sync.config import SyncConfig
from kagglekit.dataset_sync.kaggle import (
    JsonResponse,
    KaggleAdapter,
    TransientKaggleError,
    _stdlib_transport,
)
from kagglekit.dataset_sync.sources import Source


def test_fixture_list_and_detail_parsing(fixture_json):
    tag_page = fixture_json("tag_page.json")
    detail = fixture_json("detail_response.json")

    def transport(url, timeout):
        payload = detail if "/view/" in url else tag_page
        return JsonResponse(payload, 200, 12)

    adapter = KaggleAdapter(SyncConfig(request_delay_seconds=0), transport=transport)
    page = adapter.list_datasets(Source("tag", "classification"), "updated", 1)
    item, retries = adapter.dataset_detail("owner/shared")
    assert page.items[0].ref == "owner/shared"
    assert page.items[0].tags == ["classification"]
    assert item.description == "Detailed description"
    assert retries == 0


def test_saved_empty_short_and_invalid_fixtures_are_parsed(fixture_json):
    payloads = [
        fixture_json("empty_page.json"),
        fixture_json("short_page.json"),
        fixture_json("invalid_tag_fallback_page.json"),
    ]

    def transport(url, timeout):
        return JsonResponse(payloads.pop(0), 200, 1)

    adapter = KaggleAdapter(SyncConfig(request_delay_seconds=0), transport=transport)
    source = Source("tag", "classification")
    assert adapter.list_datasets(source, "updated", 1).items == []
    assert len(adapter.list_datasets(source, "updated", 2).items) == 1
    assert adapter.list_datasets(source, "updated", 3).items[0].tags == ["finance"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_429_and_5xx_are_retried_then_exhausted(status):
    calls = []

    def transport(url, timeout):
        calls.append(url)
        raise HTTPError(url, status, "transient", {"Retry-After": "0"}, BytesIO())

    adapter = KaggleAdapter(
        SyncConfig(retry_max_attempts=3, request_delay_seconds=0, backoff_base_seconds=0),
        transport=transport,
        sleep=lambda _: None,
        random_value=lambda: 0,
    )
    with pytest.raises(TransientKaggleError) as caught:
        adapter.list_datasets(Source("query", "forecasting"), "updated", 1)
    assert len(calls) == 3
    assert caught.value.http_status == status
    assert caught.value.retry_count == 2


def test_429_without_retry_after_uses_configured_minutes_scale_delay():
    delays = []
    calls = 0

    def transport(url, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(url, 429, "limited", {}, BytesIO())
        return JsonResponse([], 200, 1)

    adapter = KaggleAdapter(
        SyncConfig(
            retry_max_attempts=2,
            request_delay_seconds=0,
            backoff_base_seconds=2,
            rate_limit_backoff_seconds=180,
        ),
        transport=transport,
        sleep=delays.append,
        random_value=lambda: 0,
    )
    adapter.list_datasets(Source("query", "forecasting"), "updated", 1)
    assert delays == [180]


def test_stdlib_transport_adds_bearer_token(monkeypatch):
    observed = {}

    class Response:
        status = 200

        def read(self):
            return b"[]"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout):
        observed["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr("kagglekit.dataset_sync.kaggle.urlopen", fake_urlopen)
    _stdlib_transport("https://example.invalid", 1, token="YOUR_API_KEY")
    assert observed["authorization"] == "Bearer YOUR_API_KEY"
