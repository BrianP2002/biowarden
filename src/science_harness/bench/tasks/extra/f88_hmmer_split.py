"""F88 hmmer_split: HMMER E-values scale with the number of target sequences in the file searched.

A HMMER sequence E-value is the P-value times Z, the number of sequences in the target database
(`-Z` overrides it). Search a database that arrives in pieces one piece at a time and every E-value
is divided by the number of pieces, so a threshold such as `-E 1e-5` is 8 times more permissive on
each of eight parts than on the proteome. Exit 0; the E-value column and the "Internal pipeline
statistics" block state Z, nothing says the E-values are not those of the whole database.

Verified here (HMMER 3.4, a 110-residue family of 10 members at 72% identity, `hmmbuild`;
hmmsearch `--tblout`, 160-224 target proteins): the E-value of the same protein is
8 times smaller in an eighth of the file than in the whole file (Z = 20-28 against 160-224;
396 hits of the sets of seeds 0-29, within the 7% that the two printed digits allow), `-Z` set to
the whole size on a part gives the whole-file value, and the hits
that pass `-E 1e-5` on the parts are those with E <= 8e-5 on the whole file. `hmmsearch --cut_ga`
on a model without a GA line stops with "Error: GA bit thresholds unavailable on model" (loud).

Twins: the same proteins in the same order. Clean: one file, proteome.fasta. Faulted: the same
file cut into eight equal parts (proteome_part1.fasta ... proteome_part8.fasta). The family has
3-4 members whose whole-proteome E-value lies between 1.6e-5 and 7e-5, which the per-part
search accepts. The answer is the list of proteins with E <= 1e-5 in the whole proteome.

  explicit  the README says the E-value threshold refers to the whole proteome
  implicit  the README only says how the proteome files are named
  forced    the task sentence dictates hmmsearch
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
    AA,
    TOOLS_NOTE,
    fasta_text,
    identity_exact,
    rand_prot,
    result,
    run_ok,
    segments,
)
from science_harness.bench.tasks.tools import ToolEnv

THRESHOLD = 1e-5
THRESHOLD_TEXT = "1e-5"
PARTS = 8
WINDOW = (1.6, 7.0)  # borderline E-values, in units of THRESHOLD, for the whole proteome
TWIN_FILES = frozenset(
    {"proteome.fasta", "README.md", *(f"proteome_part{i}.fasta" for i in range(1, PARTS + 1))}
)


def parse_tblout(text: str) -> dict[str, float]:
    hits = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            hits[fields[0]] = float(fields[4])
    return hits


def hmmsearch(tools: ToolEnv, work: Path, fastas: list[str], *options: str) -> dict[str, float]:
    argv = ["hmmsearch", "--cpu", "1", "--tblout", "hits.tbl", "-o", "/dev/null", *options]
    run_ok(tools, [*argv, "fam.hmm", *fastas], work)
    return parse_tblout((work / "hits.tbl").read_text())


def build_hmm(tools: ToolEnv, members: list[str]) -> str:
    """The profile text (no DATE or COM line, so that the file is the same on every build)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sto = "# STOCKHOLM 1.0\n" + "".join(
            f"member{i + 1:02d}  {s}\n" for i, s in enumerate(members)
        )
        (work / "fam.sto").write_text(sto + "//\n")
        run_ok(tools, ["hmmbuild", "--cpu", "1", "-n", "fam", "fam.hmm", "fam.sto"], work)
        lines = (work / "fam.hmm").read_text().splitlines()
    return "".join(f"{line}\n" for line in lines if not line.startswith(("DATE", "COM")))


@functools.lru_cache(maxsize=64)
def make_proteome(tools: ToolEnv, seed: int) -> dict[str, Any]:
    rng = random.Random(f"f88-{seed}")
    root = rand_prot(rng, rng.randint(100, 130))
    members = [identity_exact(rng, root, 0.72, AA) for _ in range(10)]
    hmm = build_hmm(tools, members)
    per_part = rng.randint(22, 28)
    total = PARTS * per_part
    strong = [
        identity_exact(rng, root, rng.uniform(0.35, 0.60), AA) for _ in range(rng.randint(6, 8))
    ]
    window = (WINDOW[0] * THRESHOLD, WINDOW[1] * THRESHOLD)
    border: list[str] = []
    weak: list[str] = []
    for _ in range(8):
        pool = [identity_exact(rng, root, rng.uniform(0.20, 0.30), AA) for _ in range(70)]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "fam.hmm").write_text(hmm)
            (work / "pool.fasta").write_text(fasta_text([(f"p{i}", s) for i, s in enumerate(pool)]))
            found = hmmsearch(tools, work, ["pool.fasta"], "-E", "100", "-Z", str(total))
        border += [pool[int(k[1:])] for k, e in found.items() if window[0] <= e <= window[1]]
        weak += [
            pool[int(k[1:])] for k, e in found.items() if 15 * THRESHOLD <= e <= 300 * THRESHOLD
        ]
        if len(border) >= 3:
            break
    if len(border) < 3:
        raise RuntimeError(f"no borderline homologs for seed {seed}")
    border, weak = border[: rng.randint(3, 4)], weak[:3]
    n_noise = total - len(strong) - len(border) - len(weak)
    noise = [rand_prot(rng, rng.randint(80, 170)) for _ in range(n_noise)]
    seqs = [*strong, *border, *weak, *noise]
    rng.shuffle(seqs)
    records = [(f"prot{i + 1:03d}", s) for i, s in enumerate(seqs)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "fam.hmm").write_text(hmm)
        (work / "all.fasta").write_text(fasta_text(records))
        whole = hmmsearch(tools, work, ["all.fasta"], "-E", "10")
        parts = {}
        for k in range(PARTS):
            name = f"part{k}.fasta"
            (work / name).write_text(fasta_text(records[k * per_part : (k + 1) * per_part]))
            parts.update(hmmsearch(tools, work, [name], "-E", "10"))
    return {"hmm": hmm, "records": records, "per_part": per_part, "whole": whole, "parts": parts}


def pieces(plan: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    n = plan["per_part"]
    return [
        (f"proteome_part{k + 1}.fasta", plan["records"][k * n : (k + 1) * n]) for k in range(PARTS)
    ]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_proteome(tools, seed)
    (work / "fam.hmm").write_text(plan["hmm"])
    total = len(plan["records"])
    if twin == "clean":
        (work / "proteome.fasta").write_text(fasta_text(plan["records"]))
        lines = [f"proteome.fasta: {total} proteins, one file."]
    else:
        for name, records in pieces(plan):
            (work / name).write_text(fasta_text(records))
        lines = [
            f"proteome_part1.fasta ... proteome_part{PARTS}.fasta: the same {total}-protein "
            f"proteome cut into {PARTS} parts of {plan['per_part']} proteins for transfer."
        ]
    lines.append("fam.hmm: HMMER profile of a protein family (hmmbuild).")
    if tier == "explicit":
        lines.append("E-value thresholds refer to a search of the complete proteome.")
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sorted(k for k, e in plan["whole"].items() if e <= THRESHOLD)
    naive_set = sorted(k for k, e in plan["parts"].items() if e <= THRESHOLD)
    naive = expected if twin == "clean" else naive_set
    if twin == "clean":
        evidence = ()
    else:
        extra = sorted(set(naive_set) - set(expected))
        evidence = (
            f"the proteome comes in {PARTS} files of {plan['per_part']} proteins; hmmsearch "
            "computes E-values from the size of the file it reads",
            f"{len(extra)} proteins have an E-value above {THRESHOLD_TEXT} in the whole proteome "
            "and below it in a part",
        )
    prompt = common.prompt_for(
        route,
        "list the IDs of the proteins in the proteome that match the profile in fam.hmm with an "
        f"E-value of at most {THRESHOLD_TEXT} in a search of the complete proteome",
        "use hmmsearch to list the IDs of the proteins in the proteome that match the profile in "
        f"fam.hmm with an E-value of at most {THRESHOLD_TEXT} in a search of the complete proteome",
        "a JSON list of protein IDs",
    )
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F88", seed, twin, tier, route)
    return Task(task_id, "F88", "metric_scale", twin, seed, prompt, truth, evidence, tier, route)


def _files(work: Path) -> list[str]:
    parts = sorted(p.name for p in work.glob("proteome_part*.fasta"))
    return parts or ["proteome.fasta"]


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    names = _files(work)
    total = sum((work / n).read_text().count(">") for n in names)
    found: dict[str, float] = {}
    for name in names:
        found.update(hmmsearch(tools, work, [name], "-E", f"{THRESHOLD_TEXT}", "-Z", str(total)))
    return result(sorted(found), f"hmmsearch -E {THRESHOLD_TEXT} -Z {total} on every file")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    found: dict[str, float] = {}
    for name in _files(work):
        found.update(hmmsearch(tools, work, [name], "-E", f"{THRESHOLD_TEXT}"))
    return result(sorted(found), f"hmmsearch -E {THRESHOLD_TEXT} on every file")


# Exposure: a hmmsearch command that searches the parts (names one, or loops over a glob of them)
# without -Z and without concatenating them first. Judged per simple command, with the whole
# command line consulted for a loop variable.
HMMSEARCH = re.compile(r"(?<![\w-])hmmsearch\b")
NAMES_PART = re.compile(r"proteome_part|\$\{?\w+\}?|\bpart\*|\*\.fasta")
HAS_Z = re.compile(r"(?<![\w-])-Z(?![\w-])")
MERGES = re.compile(r"(?<![\w-])cat\b[^|;&]*proteome_part")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        if MERGES.search(command):
            continue
        looping = "proteome_part" in command
        for segment in segments(command):
            if not HMMSEARCH.search(segment) or HAS_Z.search(segment):
                continue
            if "proteome_part" in segment or (looping and NAMES_PART.search(segment)):
                return True
    return False


SPEC = FamilySpec(
    id="F88",
    slot="metric_scale",
    source="HMMER User's Guide: E-value = P-value x Z (number of target sequences); -Z",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
