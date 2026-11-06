"""Shared test data and small helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from biowarden.contracts import Entity, ToolContract, load_packaged

# tFold-TCR role code -> (label, sequence); sequences are made up but distinct.
ROLES = {
    "B": ("tcr_beta", "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEY"),
    "A": ("tcr_alpha", "KEVEQNSGPLSVPEGAIASLNCTYSDRGSQS"),
    "D": ("tcr_delta", "AQSVTQPDARVTVSEGASLQLRCKYSYSATP"),
    "G": ("tcr_gamma", "QKVQQSPESLIVPEGGTATLTCNYSDSHG"),
    "M": ("mhc", "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSD"),
    "N": ("b2m", "IQRTPKIQVYSRHPAENGKSNFLNCYVSGF"),
    "P": ("peptide", "GILGFVFTL"),
}

TOOL_ORDER = "BAMNP"  # the order tfoldlike assembles in
SWAPPED = "ABMNP"  # alpha before beta, the order a pipeline built around other tools would use


def entities(order: str) -> list[Entity]:
    return [Entity(role=r, label=ROLES[r][0], sequence=ROLES[r][1]) for r in order]


def run_mockfold(
    tmp_path: Path, backend: str, ents: list[Entity], faults: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess[str], Path]:
    out_dir = tmp_path / f"raw-{backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    request = out_dir / "request.json"
    request.write_text(json.dumps({"entities": [e.model_dump() for e in ents]}))
    cmd = [sys.executable, "-I", "-m", "biowarden.bench.fixtures.mockfold", "predict"]
    cmd += ["--backend", backend, "--request", str(request), "--out-dir", str(out_dir)]
    for fault in faults:
        cmd += ["--fault", fault]
    return subprocess.run(cmd, capture_output=True, text=True, check=False), out_dir


def read_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def variant(backend: str, mutate: Any) -> ToolContract:
    """A packaged contract with ``mutate(data)`` applied to its dict form."""
    data = load_packaged(f"mockfold_{backend}").model_dump(mode="json")
    mutate(data)
    return ToolContract.model_validate(data)


# Rule admission runs over this many seeds. Routine runs use 40; the full 300-seed calibration that
# decides promotion is `python -I -m biowarden.bench.admission` (or BIOWARDEN_ADMISSION_SEEDS=300).
ADMISSION_SEEDS = int(os.environ.get("BIOWARDEN_ADMISSION_SEEDS", "40"))
FULL_ADMISSION = ADMISSION_SEEDS >= 300
MIN_RIGHT_CALLS = max(1, 200 * ADMISSION_SEEDS // 300)  # the gate's 200 correct calls, scaled


def promotion_is_consistent(stats: Any) -> bool:
    """A rule promotes exactly when it has enough correct calls (200, and 200 undeclared ones
    for a conditional rule) with the false-block bound under 2%; fewer calls must stay warn."""
    entry = stats.promotion_entry()
    enough = stats.right_declared.n >= 200 and (
        not stats.conditional or stats.cells[("right", False)].n >= 200
    )
    return entry["mode"] == ("block" if enough and stats.admitted() else "warn")


def gate_seeds(default: str) -> range:
    """Seeds for the family gates; BIOWARDEN_GATE_SEEDS="1000-1004" re-runs held-out seeds."""
    text = os.environ.get("BIOWARDEN_GATE_SEEDS", default)
    low, _, high = text.partition("-")
    return range(int(low), int(high or low) + 1)


def console_blocks(text: str) -> list[list[str]]:
    """The fenced ```console blocks of a Markdown file, each as logical lines.

    A line that starts with two spaces continues the line before it (the README wraps long output).
    """
    blocks = []
    for body in re.findall(r"```console\n(.*?)```", text, flags=re.S):
        lines: list[str] = []
        for line in body.splitlines():
            if line.startswith("  ") and lines:
                lines[-1] += " " + line.strip()
            else:
                lines.append(line)
        blocks.append(lines)
    return blocks
