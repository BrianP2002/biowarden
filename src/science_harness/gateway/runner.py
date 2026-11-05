"""Run the mockfold CLI as a subprocess, the way any wrapped bio tool would be run."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from science_harness.bench.fixtures.mockfold import CONFIDENCE_NAME, PDB_NAME
from science_harness.contracts.request import Request

_MODULE = "science_harness.bench.fixtures.mockfold"


@dataclass(frozen=True)
class ToolRun:
    exit_code: int
    stdout: str
    stderr: str
    pdb: Path | None
    confidence: Path | None


class MockfoldRunner:
    def __init__(self, backend: str, faults: Sequence[str] = ()) -> None:
        self.backend = backend
        self.faults = tuple(faults)
        self._version: str | None = None

    def _python(self, *args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", "-m", _MODULE, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def version(self) -> str:
        """The version the tool reports about itself (`mockfold --version`)."""
        if self._version is None:
            proc = self._python("--version", timeout=30)
            words = proc.stdout.split()
            self._version = words[-1] if proc.returncode == 0 and words else "unknown"
        return self._version

    def run(self, request: Request, out_dir: Path) -> ToolRun:
        out_dir.mkdir(parents=True, exist_ok=True)
        request_path = out_dir / "request.json"
        request_path.write_text(json.dumps(request.to_tool_json()))
        args = ["predict", "--backend", self.backend, "--request", str(request_path)]
        args += ["--out-dir", str(out_dir)]
        for fault in self.faults:
            args += ["--fault", fault]
        proc = self._python(*args)
        pdb, confidence = out_dir / PDB_NAME, out_dir / CONFIDENCE_NAME
        return ToolRun(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            pdb=pdb if pdb.exists() else None,
            confidence=confidence if confidence.exists() else None,
        )
