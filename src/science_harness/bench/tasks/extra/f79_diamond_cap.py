"""F79 diamond_cap: `diamond blastp` reports at most 25 target sequences per query by default.

DIAMOND's `--max-target-seqs` / `-k` defaults to 25 (`diamond blastp --help`: "maximum number of
target sequences to report alignments for (default=25)"; 0 reports all). Nothing marks the cut:
the tabular file simply ends at 25 subjects per query, best first, and the summary line on stderr
reads "Reported 25 pairwise alignments, 25 HSPs." as for any other run. A query with 40 homologs in
the database therefore looks like a query with 25, and the weakest homologs are the ones missing.

Verified here (DIAMOND 2.2.8, one 200-residue query, 40 homologs at 55-95% identity plus 30
unrelated proteins): default 25 rows; `-k 100` 39 rows; `-k 0` 39 rows (the default mode does not
find one of the 40). On the generated sets of this family (seeds 0-59): the default call returns
exactly 25 subjects whenever the family has more than 25 members, the last row is then the 25th
best member and not the weakest one, `-k 0` returns every member; exit status 0 and no warning
either way.

Twins: the same query, the same weakest homolog (70% identical to the query, 14-27 points below
every other member) and the same kind of database. Clean: 12-20 more family members (the weakest
member ranks inside the first 25). Faulted: 32-44 more members, so the weakest ranks last and falls
outside the cap. The answer is the ID of that weakest member.

  explicit  the README says how many sequences of db.fasta belong to the query's family
  implicit  the README only says db.fasta holds the query's family and unrelated proteins
  forced    the task sentence dictates diamond blastp
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

EVALUE = "1e-5"
CAP = 25
SAFE_CAP = 100
TWIN_FILES = frozenset({"db.fasta", "README.md"})


def make_plan(seed: int, twin: str) -> dict[str, Any]:
    rng = random.Random(f"f79-{seed}")
    root = rand_prot(rng, rng.randint(170, 220))
    weakest = (
        f"P{rng.randint(10000, 99999)}",
        identity_exact(rng, root, 0.70, AA),
    )
    # the twin-specific draws come after everything shared
    n_more = rng.randint(32, 44) if twin == "faulted" else rng.randint(12, 20)
    n_other = rng.randint(22, 30)
    used = {weakest[0]}
    records = [weakest]
    while len(records) < 1 + n_more:
        name = f"P{rng.randint(10000, 99999)}"
        if name not in used:
            used.add(name)
            records.append((name, identity_exact(rng, root, rng.uniform(0.84, 0.97), AA)))
    while len(records) < 1 + n_more + n_other:
        name = f"P{rng.randint(10000, 99999)}"
        if name not in used:
            used.add(name)
            records.append((name, rand_prot(rng, rng.randint(150, 230))))
    rng.shuffle(records)
    return {"root": root, "records": records, "weakest": weakest[0], "n_family": 1 + n_more}


def ranked_hits(tools: ToolEnv, work: Path, *options: str) -> list[tuple[str, float]]:
    """(subject, bit score) of every db.fasta hit below EVALUE, best first (own temp database)."""
    base = ["diamond", "makedb", "--in", "db.fasta", "-d", "db", "--threads", "1", "--quiet"]
    run_ok(tools, base, work)
    argv = ["diamond", "blastp", "-q", "query.fasta", "-d", "db", "-o", "hits.tsv"]
    argv += ["-e", EVALUE, "--threads", "1", "--quiet", "-f", "6", "sseqid", "bitscore", *options]
    run_ok(tools, argv, work)
    rows = [r.split("\t") for r in (work / "hits.tsv").read_text().splitlines() if r.strip()]
    best: dict[str, float] = {}
    for subject, score in rows:
        best[subject] = max(best.get(subject, 0.0), float(score))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def weakest_of(hits: list[tuple[str, float]]) -> str:
    return min(hits, key=lambda item: (item[1], item[0]))[0]


@functools.lru_cache(maxsize=256)
def default_weakest(tools: ToolEnv, query: str, db: str) -> str:
    """The weakest hit among the rows the default call returns (the recorded naive answer)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "query.fasta").write_text(query)
        (work / "db.fasta").write_text(db)
        return weakest_of(ranked_hits(tools, work))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed, twin)
    query, db = fasta_text([("query", plan["root"])]), fasta_text(plan["records"])
    (work / "query.fasta").write_text(query)
    (work / "db.fasta").write_text(db)
    lines = [
        "query.fasta: one protein (the query).",
        f"db.fasta: {len(plan['records'])} proteins: members of the query's family and "
        "unrelated proteins.",
    ]
    if tier == "explicit":
        lines.append(
            f"{plan['n_family']} of the proteins in db.fasta are members of the query's family "
            "(70-97% identical to the query)."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = [plan["weakest"]]
    if twin == "clean":
        naive, evidence = expected, ()
    else:
        naive = [default_weakest(tools, query, db)]
        evidence = (
            f"db.fasta holds {plan['n_family']} family members for one query, more than DIAMOND's "
            "default of 25 target sequences per query",
        )
    prompt = common.prompt_for(
        route,
        f"find the most distant homolog of the query protein in db.fasta: among the proteins "
        f"with a sequence-search hit below an E-value of {EVALUE}, the one with the lowest bit "
        "score",
        f"use diamond blastp to find the most distant homolog of the query protein in db.fasta: "
        f"among the proteins with a hit below an E-value of {EVALUE}, the one with the lowest "
        "bit score",
        "a JSON list holding the one protein ID",
    )
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F79", seed, twin, tier, route)
    return Task(task_id, "F79", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    hits = ranked_hits(tools, work, "--max-target-seqs", "0")
    return result([weakest_of(hits)], "diamond blastp --max-target-seqs 0, lowest bit score")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    hits = ranked_hits(tools, work)
    return result([weakest_of(hits)], "diamond blastp with default options, lowest bit score")


# Exposure: a diamond blastp/blastx search that keeps the default cap, or sets a cap below the
# number of sequences in db.fasta (0 means all hits). Judged per command.
DIAMOND = re.compile(r"(?<![\w-])diamond\s+(?:blastp|blastx)\b")
CAP_OPTION = re.compile(r"(?<![\w-])(?:-k|--max-target-seqs)(?:[ =]+|(?=\d))['\"]?(\d+)")


def exposed(commands: list[str], work: Path) -> bool:
    db = work / "db.fasta"
    size = db.read_text().count(">") if db.is_file() else SAFE_CAP  # families have 33-45 members
    for command in commands:
        for segment in segments(command):
            if not DIAMOND.search(segment):
                continue
            given = CAP_OPTION.findall(segment)
            cap = int(given[-1]) if given else CAP
            if cap == 0 or cap >= size:
                continue
            return True
    return False


SPEC = FamilySpec(
    id="F79",
    slot="record_count",
    source="DIAMOND --max-target-seqs default 25 (GEN-36; diamond issue 687)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
