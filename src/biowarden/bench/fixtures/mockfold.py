"""mockfold: a CPU-only stand-in for structure predictors.

It reproduces the tFold-TCR class of silent failure (incident T01) without any
GPU, weights or licence: the process exits 0 and writes a valid PDB, but the chain order and
labelling follow the tool's own convention instead of the caller's.

The fixture does not import anything from the rest of the package; it is the "tool" under test.
"""

from __future__ import annotations

import argparse
import json
import math
import string
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
BACKENDS = ("tfoldlike", "boltzlike")
FAULTS = ("dg_overwrites_ba", "silent_missing_n")

PDB_NAME = "complex.pdb"
CONFIDENCE_NAME = "confidence.json"

_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip

_TFOLD_ROLES = ("B", "A", "D", "G", "M", "N", "P")


class MockfoldError(Exception):
    """A loud, well-behaved failure (the CLI exits with status 2)."""


@dataclass(frozen=True)
class _Chain:
    chain_id: str
    sequence: str


@dataclass(frozen=True)
class PredictOutput:
    pdb: Path
    confidence: Path
    chain_ids: tuple[str, ...]


def _read_entities(request: dict[str, Any]) -> list[dict[str, str]]:
    entities = request.get("entities")
    if not isinstance(entities, list) or not entities:
        raise MockfoldError("request needs a non-empty 'entities' list")
    for entity in entities:
        sequence = entity.get("sequence", "") if isinstance(entity, dict) else ""
        if not sequence or any(ch not in _THREE for ch in sequence):
            raise MockfoldError(f"unsupported sequence: {sequence!r}")
    return entities


def _assemble_tfoldlike(entities: list[dict[str, str]], faults: frozenset[str]) -> list[_Chain]:
    by_role: dict[str, str] = {}
    for entity in entities:
        role = entity.get("role", "")
        if role not in _TFOLD_ROLES:
            raise MockfoldError(f"unknown role code {role!r}")
        if role in by_role:
            raise MockfoldError(f"duplicate role code {role!r}")
        by_role[role] = entity["sequence"]

    if ("B" in by_role) != ("A" in by_role) or ("D" in by_role) != ("G" in by_role):
        raise MockfoldError("TCR chains come in pairs: B with A, D with G")
    has_ab, has_gd = "B" in by_role, "D" in by_role
    if not (has_ab or has_gd):
        raise MockfoldError("no TCR chains given")
    if has_ab and has_gd and "dg_overwrites_ba" not in faults:
        raise MockfoldError("alpha/beta and gamma/delta chains cannot be combined")
    if "M" not in by_role or "P" not in by_role:
        raise MockfoldError("roles M and P are required")
    if "N" not in by_role and "silent_missing_n" not in faults:
        raise MockfoldError("role N (beta2m) is missing")

    ligand = ("D", "G") if has_gd else ("B", "A")  # D:G wins when both pairs are present
    order = [*ligand, "M", *(["N"] if "N" in by_role else []), "P"]
    return [_Chain(role, by_role[role]) for role in order]


def _assemble_boltzlike(entities: list[dict[str, str]]) -> list[_Chain]:
    if len(entities) > len(string.ascii_uppercase):
        raise MockfoldError("too many entities")
    return [
        _Chain(string.ascii_uppercase[i], entity["sequence"]) for i, entity in enumerate(entities)
    ]


def _plddt(slot: int, residue: int) -> float:
    return float(60 + (7 * slot + 3 * residue) % 35)  # 60..94, a pure function of position


def _ca_position(slot: int, residue: int) -> tuple[float, float, float]:
    angle = math.radians(100.0 * residue)
    return (
        round(12.0 * slot + 2.3 * math.cos(angle), 3),
        round(2.3 * math.sin(angle), 3),
        round(1.5 * residue, 3),
    )


def _pdb_text(chains: Sequence[_Chain], scale: float) -> str:
    lines: list[str] = []
    serial = 1
    for slot, chain in enumerate(chains):
        for residue, aa in enumerate(chain.sequence):
            x, y, z = _ca_position(slot, residue)
            lines.append(
                f"ATOM  {serial:5d}  CA  {_THREE[aa]:>3s} {chain.chain_id}{residue + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{_plddt(slot, residue) * scale:6.2f}"
                f"           C  "
            )
            serial += 1
        lines.append(f"TER   {serial:5d}      {_THREE[chain.sequence[-1]]:>3s} {chain.chain_id}")
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def predict(
    request: dict[str, Any],
    backend: str,
    out_dir: str | Path,
    faults: Iterable[str] = (),
) -> PredictOutput:
    """Write complex.pdb and confidence.json for ``request`` into ``out_dir``."""
    if backend not in BACKENDS:
        raise MockfoldError(f"unknown backend {backend!r}")
    fault_set = frozenset(faults)
    if unknown := fault_set - set(FAULTS):
        raise MockfoldError(f"unknown fault(s): {sorted(unknown)}")
    if fault_set and backend != "tfoldlike":
        raise MockfoldError("faults only apply to the tfoldlike backend")

    entities = _read_entities(request)
    if backend == "tfoldlike":
        chains, scale = _assemble_tfoldlike(entities, fault_set), 0.01
    else:
        chains, scale = _assemble_boltzlike(entities), 1.0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdb_path, confidence_path = out / PDB_NAME, out / CONFIDENCE_NAME
    pdb_path.write_text(_pdb_text(chains, scale))

    per_chain = {
        chain.chain_id: round(
            sum(_plddt(slot, i) for i in range(len(chain.sequence))) / len(chain.sequence) * scale,
            4,
        )
        for slot, chain in enumerate(chains)
    }
    total = sum(len(chain.sequence) for chain in chains)
    mean = sum(
        _plddt(slot, i) for slot, chain in enumerate(chains) for i in range(len(chain.sequence))
    )
    confidence_path.write_text(
        json.dumps({"plddt_mean": round(mean / total * scale, 4), "chain_plddt": per_chain}) + "\n"
    )
    return PredictOutput(pdb_path, confidence_path, tuple(chain.chain_id for chain in chains))


_EPILOG = """\
backends:
  tfoldlike  Entity roles are tFold-TCR role codes: B=TCR beta, A=TCR alpha, D=delta, G=gamma,
             M=MHC heavy chain, N=beta2m, P=peptide. The complex is assembled as B:A + M:N + P
             (D:G + M:N + P for gamma/delta) in that fixed order, whatever the order of the
             request. Output chains carry the role codes as IDs and are written in the
             assembled order. pLDDT is written on a 0-1 scale.
  boltzlike  Roles are ignored. The request order is kept; chains are named A, B, C, ... in that
             order. pLDDT is written on a 0-100 scale.

faults (tfoldlike only; without them the tool stops with an error instead):
  dg_overwrites_ba  a request holding both B,A and D,G is assembled from D:G alone
  silent_missing_n  a request without N builds the complex without beta2m
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mockfold", description="Toy complex predictor.")
    parser.add_argument("--version", action="version", version=f"mockfold {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "predict",
        help="write a toy complex structure",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("--backend", choices=BACKENDS, required=True)
    run.add_argument("--request", type=Path, required=True, help="JSON with an 'entities' list")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--fault", action="append", default=[], choices=FAULTS, dest="faults")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        request = json.loads(args.request.read_text())
        result = predict(request, args.backend, args.out_dir, args.faults)
    except (MockfoldError, OSError, json.JSONDecodeError) as exc:
        print(f"mockfold: error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {result.pdb} {result.confidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
