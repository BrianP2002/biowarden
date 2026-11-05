"""Read tool outputs with gemmi into plain records the oracles can compare."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi


class OutputError(Exception):
    """The tool's output is missing or cannot be parsed."""


@dataclass(frozen=True)
class ChainRecord:
    chain_id: str
    sequence: str


@dataclass(frozen=True)
class ParsedOutput:
    chains: tuple[ChainRecord, ...]
    n_atoms: int
    coords_finite: bool
    plddt: tuple[float, ...]  # CA B-factors, which is where predictors put pLDDT
    confidence: dict[str, Any] | None


def _one_letter(residue_name: str) -> str:
    info = gemmi.find_tabulated_residue(residue_name)
    code = info.one_letter_code.upper() if info else " "
    return code if code.strip() else "X"


def parse_output(pdb_path: Path, confidence_path: Path | None = None) -> ParsedOutput:
    try:
        structure = gemmi.read_structure(str(pdb_path))
    except Exception as exc:
        raise OutputError(f"cannot read {pdb_path.name}: {exc}") from exc
    if len(structure) == 0:
        raise OutputError(f"{pdb_path.name} has no model")

    chains: list[ChainRecord] = []
    plddt: list[float] = []
    n_atoms, finite = 0, True
    for chain in structure[0]:
        chains.append(ChainRecord(chain.name, "".join(_one_letter(r.name) for r in chain)))
        for residue in chain:
            for atom in residue:
                n_atoms += 1
                finite = finite and all(
                    math.isfinite(v) for v in (atom.pos.x, atom.pos.y, atom.pos.z)
                )
                if atom.name == "CA":
                    plddt.append(atom.b_iso)

    confidence = None
    if confidence_path is not None and confidence_path.exists():
        try:
            confidence = json.loads(confidence_path.read_text())
        except json.JSONDecodeError as exc:
            raise OutputError(f"cannot read {confidence_path.name}: {exc}") from exc
    return ParsedOutput(tuple(chains), n_atoms, finite, tuple(plddt), confidence)
