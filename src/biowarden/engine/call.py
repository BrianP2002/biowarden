"""One tool invocation as the engine sees it."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from biowarden.engine.argv import first_positional

DECLARE_FLAG = "--declare"
DECLARE_ENV = "BIOWARDEN_DECLARE"


class DeclarationError(ValueError):
    """A `--declare` marker that is not of the form key=value."""


def _parse_pair(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip() or not value.strip():
        raise DeclarationError(f"declaration must look like key=value, got {text!r}")
    return key.strip(), value.strip()


def has_declare_marker(argv: Sequence[str]) -> bool:
    """True when `argv` holds a `--declare` marker, well formed or not."""
    return any(a == DECLARE_FLAG or a.startswith(DECLARE_FLAG + "=") for a in argv)


def split_declarations(argv: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Remove `--declare k=v` / `--declare=k=v` markers from `argv` and return them separately."""
    clean: list[str] = []
    declared: dict[str, str] = {}
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        if token == DECLARE_FLAG:
            if i + 1 >= len(items):
                raise DeclarationError("--declare needs a key=value argument")
            key, value = _parse_pair(items[i + 1])
            declared[key] = value
            i += 2
            continue
        if token.startswith(DECLARE_FLAG + "="):
            key, value = _parse_pair(token[len(DECLARE_FLAG) + 1 :])
            declared[key] = value
            i += 1
            continue
        clean.append(token)
        i += 1
    return clean, declared


@dataclass(frozen=True)
class Call:
    """`tool` is the executable's basename; `argv` excludes it and excludes declaration markers."""

    tool: str
    argv: tuple[str, ...]
    cwd: Path
    declared: Mapping[str, str] = field(default_factory=dict)
    # words of earlier commands in the same command string (and here-document bodies): only used to
    # find data files a pipeline stage reads without naming them, never part of the call's identity
    context: tuple[str, ...] = ()
    # the trial's work directory when a sandbox sits between the agent and this host: a file the
    # call names is only read when it really lies below it (None: any file may be read)
    root: Path | None = None

    @property
    def subcommand(self) -> str | None:
        return first_positional(self.argv)

    def call_hash(self) -> str:
        """Identifies "the same call" for the retry cap; declarations are part of the call."""
        payload = json.dumps(
            [self.tool, list(self.argv), sorted(self.declared.items()), str(self.cwd)]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_argv(
        cls,
        tool: str,
        argv: Sequence[str],
        cwd: Path | str,
        env: Mapping[str, str] | None = None,
        root: Path | None = None,
    ) -> Call:
        """Build a call from a raw command line. Markers in argv override the environment."""
        clean, from_argv = split_declarations(argv)
        declared: dict[str, str] = {}
        raw_env = (env if env is not None else os.environ).get(DECLARE_ENV, "")
        for part in filter(None, (p.strip() for p in raw_env.split(";"))):
            key, value = _parse_pair(part)
            declared[key] = value
        declared.update(from_argv)
        return cls(Path(tool).name, tuple(clean), Path(cwd).absolute(), declared, root=root)
