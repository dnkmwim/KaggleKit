import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture_json():
    root = Path(__file__).parent / "fixtures"

    def load(name: str):
        return json.loads((root / name).read_text(encoding="utf-8"))

    return load

