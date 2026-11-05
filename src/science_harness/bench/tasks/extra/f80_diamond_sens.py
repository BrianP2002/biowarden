"""F80 diamond_sens: DIAMOND's default mode misses homologs below about 50% identity.

`diamond blastp` without a sensitivity flag runs the `--fast` seed settings, tuned for hits above
roughly 90% identity at scale. Distant homologs (35-50% identity, a typical ortholog across
phyla) fall through the seeding stage, so the query simply has no row in the output: exit 0, no
warning, and the stderr summary reads "1 queries aligned" or "0 queries aligned" like any other run.
`--ultra-sensitive` finds them, and so does BLAST's blastp; the intermediate modes (`--sensitive`,
`--more-sensitive`, `--very-sensitive`) still miss a partner here and there (see below).

Verified here (DIAMOND 2.2.8, BLAST+ 2.17.0; twelve 180-residue queries, each with one mutated
partner in a database of 42 proteins; queries found, E-value limit 0.001):
    identity of the partner   30 35 40 45 50 55 60 70 80 | 30 35 40
    default                   -  -  -  -  x  -  x  x  x |  -  -  -      4 of 12 queries
    --sensitive               -  x  x  x  x  x  x  x  x |  -  x  x     10 of 12
    --ultra-sensitive         x  x  x  x  x  x  x  x  x |  x  x  x     12 of 12
    BLAST+ blastp (E 0.001)   all 12
and on the generated sets of this family (seeds 0-29): the default call misses every partner below
about 50% identity and none above 75%. Partners missed on the faulted twins of seeds 0-9 and
1000-1004 (15 sets, 144 homologous queries), and the number of sets in which every partner is
found:
    mode                default  --fast  --mid  --sensitive  --more  --very  --ultra
    partners missed        75      82     46        19         19       3       0
    sets complete           0       0      1         2          2      13      15
so only `--ultra-sensitive` is a safe route: `exposed()` counts every other mode as the trap.

Twins: the same decoys and the same partners in the database. Clean: every homologous query is
75-92% identical to its partner. Faulted: about 60% of them are 35-41% identical to their partner.

  explicit  the README gives the identity range of the query-partner pairs
  implicit  the README only says the queries come from other species
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
    TOOLS_NOTE,
    fasta_text,
    identity_exact,
    rand_prot,
    result,
    run_ok,
    segments,
)
from science_harness.bench.tasks.tools import ToolEnv

AA = "ACDEFGHIKLMNPQRSTVWY"
TWIN_FILES = frozenset({"queries.fasta", "README.md"})


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f80-{seed}")
    n_hom = rng.randint(8, 11)
    homologs, db, remote = [], [], []
    for _ in range(n_hom):
        length = rng.randint(170, 210)
        partner = rand_prot(rng, length)
        close = identity_exact(rng, partner, rng.uniform(0.75, 0.92), AA)
        far = identity_exact(rng, partner, rng.uniform(0.35, 0.41), AA)
        homologs.append((close, far))
        db.append(partner)
        remote.append(rng.random() < 0.6)
    decoys = [rand_prot(rng, rng.randint(170, 210)) for _ in range(3)]
    db += [rand_prot(rng, rng.randint(150, 230)) for _ in range(24)]
    rng.shuffle(db)
    layout = [("h", i) for i in range(n_hom)] + [("d", i) for i in range(len(decoys))]
    rng.shuffle(layout)  # file order; the IDs are neutral (prot01, ...)
    return {"homologs": homologs, "remote": remote, "decoys": decoys, "db": db, "layout": layout}


def query_records(plan: dict[str, Any], twin: str) -> list[tuple[str, str]]:
    records = []
    for pos, (kind, i) in enumerate(plan["layout"]):
        if kind == "h":
            close, far = plan["homologs"][i]
            seq = far if twin == "faulted" and plan["remote"][i] else close
        else:
            seq = plan["decoys"][i]
        records.append((f"prot{pos + 1:02d}", seq))
    return records


def homolog_ids(plan: dict[str, Any], remote_only: bool = False) -> list[str]:
    return sorted(
        f"prot{pos + 1:02d}"
        for pos, (kind, i) in enumerate(plan["layout"])
        if kind == "h" and (not remote_only or plan["remote"][i])
    )


@functools.lru_cache(maxsize=256)
def default_hits(tools: ToolEnv, queries: str, db: str) -> list[str]:
    """The default-mode answer for these files, run in a temporary directory (recorded naive)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "queries.fasta").write_text(queries)
        (work / "db.fasta").write_text(db)
        return found(tools, work)


def found(tools: ToolEnv, work: Path, *options: str) -> list[str]:
    """Queries with a diamond hit at the default E-value limit (0.001)."""
    run_ok(
        tools,
        ["diamond", "makedb", "--in", "db.fasta", "-d", "db", "--threads", "1", "--quiet"],
        work,
    )
    argv = ["diamond", "blastp", "-q", "queries.fasta", "-d", "db", "-o", "hits.tsv"]
    argv += ["--threads", "1", "--quiet", *options]
    run_ok(tools, argv, work)
    rows = (work / "hits.tsv").read_text().splitlines()
    return sorted({row.split("\t")[0] for row in rows if row.strip()})


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "queries.fasta").write_text(fasta_text(query_records(plan, twin)))
    db = [(f"ref{k + 1:03d}", seq) for k, seq in enumerate(plan["db"])]
    (work / "db.fasta").write_text(fasta_text(db))
    lines = [
        "queries.fasta: proteins from several species; three of them have no counterpart in "
        "the database.",
        f"db.fasta: {len(plan['db'])} reference proteins.",
    ]
    if tier == "explicit":
        lines.append(
            "Each query that has a counterpart is 75-92% identical to it."
            if twin == "clean"
            else "Most queries that have a counterpart are only 35-41% identical to it."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = homolog_ids(plan)
    if twin == "clean":
        naive, evidence = expected, ()
    else:
        far = homolog_ids(plan, remote_only=True)
        # what the default mode really returns (it finds a few of the 35-41% pairs by chance)
        naive = default_hits(
            tools, (work / "queries.fasta").read_text(), (work / "db.fasta").read_text()
        )
        evidence = (
            f"{len(far)} of {len(expected)} counterparts are 35-41% identical to their query",
            "DIAMOND's default mode is tuned for close homologs",
        )
    prompt = common.prompt_for(
        route,
        "list the IDs of the proteins in queries.fasta that have a homologous protein in "
        "db.fasta (sequence search hit with an E-value below 0.001)",
        "use diamond blastp to list the IDs of the proteins in queries.fasta that have a "
        "homologous protein in db.fasta (hit with an E-value below 0.001)",
        "a JSON list of protein IDs",
    )
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F80", seed, twin, tier, route)
    return Task(task_id, "F80", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(found(tools, work, "--ultra-sensitive"), "diamond blastp --ultra-sensitive")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(found(tools, work), "diamond blastp with default options")


# Exposure: a diamond blastp/blastx search in any mode but --ultra-sensitive (the default, --faster,
# --fast, --mid-sensitive, --sensitive, --more-sensitive and --very-sensitive miss partners above).
DIAMOND = re.compile(r"(?<![\w-])diamond\s+(?:blastp|blastx)\b")
SENSITIVE = re.compile(r"--ultra-sensitive\b")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        DIAMOND.search(segment) and not SENSITIVE.search(segment)
        for command in commands
        for segment in segments(command)
    )


SPEC = FamilySpec(
    id="F80",
    slot="default_param",
    source="DIAMOND sensitivity modes (default fast; --sensitive ... --ultra-sensitive), manual",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
