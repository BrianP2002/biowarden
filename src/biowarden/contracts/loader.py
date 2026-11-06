from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from biowarden.contracts.schema import ToolContract

PACKAGED = ("mockfold_tfoldlike", "mockfold_boltzlike")


def load_contract(path: str | Path) -> ToolContract:
    return ToolContract.model_validate(yaml.safe_load(Path(path).read_text()))


def load_packaged(name: str) -> ToolContract:
    text = (resources.files("biowarden.contracts") / "data" / f"{name}.yaml").read_text()
    return ToolContract.model_validate(yaml.safe_load(text))


def contract_json_schema() -> dict[str, Any]:
    return ToolContract.model_json_schema()
