"""Additional task families, one module each. A module exports SPEC (a FamilySpec).

Each family is derived from a documented incident or tool behaviour (``source``) and must pass the
generic gates in tests/test_extra_families.py: the reference solver passes every twin and tier, the
naive solver passes the clean twin and fails the faulted one with exactly the recorded trap answer,
the twins differ only in ``twin_files``, the truth never appears in the workspace, and workspaces
are byte-reproducible.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.tools import ToolEnv

# Registered modules; one that does not exist yet is skipped, so names can be added ahead of time.
MODULES = (
    "f05_depth_coords",
    "f06_dup_depth",
    "f07_regions_vs_targets",
    "f08_reads_vs_alignments",
    "f09_multiallelic",
    "f10_indel_bed",
    "f11_gff_exons",
    "f12_strand_seq",
    "f13_phred_offset",
    "f14_id_version",
    "f15_log_scale",
    "f16_tss_upstream",
    # wrapper-script faults (locus wrapper)
    "f17_wrap_filter",
    "f18_wrap_truncate",
    "f19_wrap_stderr",
    "f20_wrap_drop",
    # the user's wording disagrees with the data (locus task_statement)
    "f21_stmt_base",
    "f22_stmt_encoding",
    "f23_stmt_strand",
    "f24_stmt_units",
    # stale documentation (locus documentation)
    "f25_doc_flag",
    "f26_doc_default",
    "f27_doc_output",
    "f28_doc_format",
)

Solver = Callable[[Task, Path, ToolEnv], dict[str, Any]]


@dataclass(frozen=True)
class FamilySpec:
    id: str  # "F5" ... "F16"
    slot: str  # the convention slot the trap lives in
    source: str  # incident id (GEN-/STR-/SCD-) or documentation the trap comes from
    build: Callable[..., Task]  # (seed, twin, work, tools, tier, route) -> Task
    reference: Solver  # handles the trap; returns the dict an agent would write to result.json
    naive: Solver  # takes the obvious route and falls into the trap on the faulted twin
    exposed: Callable[[list[str], Path], bool]  # did the agent's commands take the trapped route?
    twin_files: frozenset[str]  # file names that may differ between the clean and faulted twin
    locus: str = ""  # where the error lives (bench/tasks/locus.py); required for new families
    tiers: tuple[str, ...] = ("explicit", "implicit")
    routes: tuple[str, ...] = ("free", "forced")


def _module_names() -> list[str]:
    """Registered names plus every `fNN_*.py` file here, so a new family is one file."""
    found = {path.stem for path in Path(__file__).parent.glob("f[0-9]*_*.py")}
    return sorted({*MODULES, *found}, key=lambda name: (int(name.split("_")[0][1:]), name))


def load() -> dict[str, FamilySpec]:
    specs: dict[str, FamilySpec] = {}
    for name in _module_names():
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
            continue
        specs[module.SPEC.id] = module.SPEC
    return specs
