"""Mark the heavy tests so a quick run can skip them: `pytest -m "not slow"`."""

from __future__ import annotations

import pytest

SLOW_PREFIXES = ("test_family_", "test_rules_", "test_extra_families", "test_admission")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name.startswith(SLOW_PREFIXES):
            item.add_marker(pytest.mark.slow)
