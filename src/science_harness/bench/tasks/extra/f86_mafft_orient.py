"""F86 mafft_orient: MAFFT (and MUSCLE 5) align sequences as written; a reverse-complemented
member is not flipped unless `--adjustdirection` is given.

`mafft seqs.fasta` assumes every sequence is on the same strand. A member that is the reverse
complement of the others is aligned as unrelated sequence: exit 0, an alignment of the right
number of rows, almost no conserved column, and the row simply full of gaps and mismatches.
`--adjustdirection` (or `--adjustdirectionaccurately`) flips such sequences, and renames them with
a `_R_` prefix (`_R_seq3`), which breaks any later join on the sequence names. MUSCLE 5
(`muscle -align`) has no such option.

Verified here (MAFFT v7.526, MUSCLE 5.3; 8 DNA sequences of 300 bases, 92% identical, two of them
reverse-complemented): `mafft --quiet` alignment of 357 columns with 64 conserved columns;
`--auto` 417 columns, 41 conserved; `--adjustdirection` and `--adjustdirectionaccurately` 300
columns, 163 conserved, rows `_R_seq3` and `_R_seq6`; `--reorder` changes the row order, plain
runs keep the input order. `muscle -align` gave an alignment with the two sequences misaligned in
the same way, and writes its rows in guide-tree order (not the input order; there is no `-stable`
option in 5.3: "Unknown option stable"). On the generated sets of this family (seeds 0-59, 8-10
sequences of 240-320 bases, substitutions only) `--adjustdirectionaccurately` returns exactly the
analytic count of conserved columns.

Twins: the same sequences. Clean: all on the same strand. Faulted: 2-3 of them are written as
their reverse complement. The answer is the number of alignment columns where all sequences carry
the same base (no gaps), i.e. the count on the correctly oriented sequences.

  explicit  the README says some sequences may be listed as the reverse complement (faulted) or
            that all are on the same strand (clean)
  implicit  the README only says the file holds copies of one gene from several isolates
  forced    the task sentence dictates mafft
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f77_blastn_task import (
    TOOLS_NOTE,
    fasta_text,
    rand_dna,
    result,
    revcomp,
    run_ok,
    segments,
    substitute,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"seqs.fasta", "README.md"})


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f86-{seed}")
    root = rand_dna(rng, rng.randint(240, 320))
    n = rng.randint(8, 10)
    members = [
        (f"isolate{i + 1:02d}", substitute(rng, root, rng.uniform(0.03, 0.05))) for i in range(n)
    ]
    flipped = set(rng.sample(range(n), rng.randint(2, 3)))
    return {"members": members, "flipped": flipped}


def written(plan: dict[str, Any], twin: str) -> list[tuple[str, str]]:
    return [
        (name, revcomp(seq) if twin == "faulted" and i in plan["flipped"] else seq)
        for i, (name, seq) in enumerate(plan["members"])
    ]


def conserved_columns(alignment: str) -> int:
    """Columns where every sequence carries the same base (no gap), case ignored."""
    rows: list[str] = []
    for line in alignment.splitlines():
        if line.startswith(">"):
            rows.append("")
        elif rows:
            rows[-1] += line.strip().upper()
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("not an alignment")
    return sum(len(set(col)) == 1 and col[0] != "-" for col in zip(*rows, strict=True))


def align(tools: ToolEnv, work: Path, *options: str) -> int:
    proc = run_ok(tools, ["mafft", "--quiet", "--thread", "1", *options, "seqs.fasta"], work)
    return conserved_columns(proc.stdout)


@functools.lru_cache(maxsize=256)
def auto_columns(tools: ToolEnv, fasta: str) -> int:
    """What `mafft --auto` gives for this file (the recorded naive answer)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "seqs.fasta").write_text(fasta)
        return align(tools, work, "--auto")


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    text = fasta_text(written(plan, twin))
    (work / "seqs.fasta").write_text(text)
    size = len(plan["members"][0][1])
    lines = [
        f"seqs.fasta: {len(plan['members'])} copies of one gene (DNA, about {size} bases) from "
        "different isolates; the copies differ by substitutions only (no insertions or "
        "deletions).",
    ]
    if tier == "explicit":
        lines.append(
            "Some sequences are listed as the reverse complement of the others."
            if twin == "faulted"
            else "All sequences are on the same strand."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    root_cols = len(plan["members"][0][1])
    expected = sum(len({seq[i] for _, seq in plan["members"]}) == 1 for i in range(root_cols))
    naive = expected if twin == "clean" else auto_columns(tools, text)
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{len(plan['flipped'])} of {len(plan['members'])} sequences are the reverse "
            "complement of the others",
            "mafft and muscle align sequences as written unless mafft gets --adjustdirection",
        )
    prompt = common.prompt_for(
        route,
        "align the sequences in seqs.fasta and report how many alignment columns are fully "
        "conserved (the same base in every sequence, no gap)",
        "use mafft to align the sequences in seqs.fasta and report how many alignment columns "
        "are fully conserved (the same base in every sequence, no gap)",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F86", seed, twin, tier, route)
    return Task(task_id, "F86", "strand", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = align(tools, work, "--adjustdirectionaccurately")
    return result(n, "mafft --adjustdirectionaccurately, then conserved columns")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(align(tools, work, "--auto"), "mafft --auto, then conserved columns")


# Exposure: a mafft or muscle alignment command without --adjustdirection[accurately]. muscle has
# no such option, so any muscle -align / -super5 command is exposed. Judged per command.
MAFFT = re.compile(r"(?<![\w-])mafft(?:-\w+)?\b")
MUSCLE = re.compile(r"(?<![\w-])muscle\b")
ADJUST = re.compile(r"--adjustdirection(?:accurately)?\b")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if MUSCLE.search(segment) and re.search(r"-(?:align|super5)\b", segment):
                return True
            if MAFFT.search(segment) and not ADJUST.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F86",
    slot="strand",
    source="MAFFT --adjustdirection (sequences are aligned as written); MUSCLE 5 has no equivalent",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
