"""Which programs the layer watches and where the real ones live: derived from the loaded specs."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from biowarden.engine.spec import Spec

# Programs that read data files and so can trigger a fact rule, whatever the data format.
INTERPRETERS = (
    "awk",
    "gawk",
    "mawk",
    "python3",
    "python",
    "python-sc",
    "python-struct",
    "python-bio",
    "perl",
    "Rscript",
)


def call_rule_tools(specs: Iterable[Spec]) -> list[str]:
    """Tools named by call rules (the programs whose own arguments are checked)."""
    return sorted({t for s in specs for r in s.rules if r.kind == "call" for t in r.applies.tools})


def watched_tools(specs: Iterable[Spec], extra: Iterable[str] = ()) -> list[str]:
    """Every program a shim should stand in for: call-rule tools, the tools fact rules name, the
    interpreters when any fact rule exists, and whatever the user asked to watch."""
    specs = list(specs)
    names = set(call_rule_tools(specs)) | set(extra)
    fact_rules = [r for s in specs for r in s.rules if r.kind == "fact"]
    for rule in fact_rules:
        names |= set(rule.applies.tools)
    if fact_rules:
        names |= set(INTERPRETERS)
    return sorted(names)


def path_dirs(path: str | None = None) -> list[str]:
    value = os.environ.get("PATH", "") if path is None else path
    return [d for d in value.split(os.pathsep) if d and os.path.isdir(d)]


def find_binaries(names: Iterable[str], dirs: Sequence[str]) -> dict[str, Path]:
    """name -> first executable of that name in `dirs`; names that are not installed are absent."""
    found: dict[str, Path] = {}
    for name in names:
        for directory in dirs:
            candidate = Path(directory) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                found[name] = candidate
                break
    return found


def import_python_rules(dirs: Iterable[str | Path]) -> list[str]:
    """Import every `*.py` in `dirs` so its `@checker` functions register. This executes the files:
    only use it on directories you trust (the CLIs ask for --trust-python)."""
    loaded: list[str] = []
    for directory in dirs:
        for file in sorted(Path(directory).glob("*.py")):
            name = f"tbm_user_rules_{abs(hash(str(file.resolve())))}_{file.stem}"
            if name in sys.modules:
                continue
            module_spec = importlib.util.spec_from_file_location(name, file)
            if module_spec is None or module_spec.loader is None:
                continue
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
            loaded.append(file.name)
    return loaded
