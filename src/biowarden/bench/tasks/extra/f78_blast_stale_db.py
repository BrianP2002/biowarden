"""F78 blast_stale_db: a BLAST database is a snapshot; it does not follow the FASTA it came from.

`makeblastdb -in proteins.fasta` writes proteins.fasta.phr/.pin/.psq (+ .pjs) next to the FASTA, and
`blastp -db proteins.fasta` reads THOSE files. When proteins are appended to the FASTA afterwards
(a new release, a merged file) the database knows nothing about them: the search runs without error
and without warning on the old sequences, and a query whose true partner is new gets its best
remaining paralog instead.

Verified here (BLAST+ 2.17.0): a database built from a 40-protein FASTA and a FASTA that has since
grown to 47; `blastdbcmd -db proteins.fasta -info` says "40 sequences" (and so does the
`number-of-sequences` field of proteins.fasta.pjs) while `grep -c '>' proteins.fasta` says 47;
`blastp -db proteins.fasta` exits 0 with an empty stderr. Rebuilding with `makeblastdb` (or
`blastp -subject proteins.fasta`) sees all 47. The database title is the FASTA file name, which is
how the two are matched even when the database files are called something else.

Twins: the same FASTA, the same queries. Clean: the database was built from the current FASTA.
Faulted: it was built before the last 6-8 proteins were appended; each of those is a paralog
(70-80% identical) of an older protein, and 3-5 of the queries come from the new proteins. The
answer is, for every query in file order, the ID of its best hit by bit score.

The database is built with `-blastdb_version 4` and its two timestamps are normalised, so that the
workspace is byte-reproducible.

  explicit  the README says whether the database was built before the last proteins were added
  implicit  the README only says that a BLAST database of proteins.fasta is in the folder
  forced    the task sentence dictates blastp
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f77_blastn_task import (
    AA,
    TOOLS_NOTE,
    fasta_text,
    identity_exact,
    rand_prot,
    result,
    run_ok,
    segments,
)
from biowarden.bench.tasks.tools import ToolEnv

DB_SUFFIXES = (".phr", ".pin", ".psq", ".pjs")
FIXED_DATE = b"Jan 01, 2026 12:00 AM"
DATE = re.compile(rb"[A-Z][a-z]{2} \d{2}, \d{4} [ \d]\d:\d{2} [AP]M")
STAMP = re.compile(rb'"last-updated": "[^"]*"')
TWIN_FILES = frozenset({"README.md", *(f"proteins.fasta{s}" for s in DB_SUFFIXES)})


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f78-{seed}")
    n_old = rng.randint(34, 46)
    n_new = rng.randint(6, 8)
    old = [(f"PRT{i + 1:04d}", rand_prot(rng, rng.randint(120, 200))) for i in range(n_old)]
    parents = rng.sample(range(n_old), n_new)
    new = [
        (f"PRT{n_old + i + 1:04d}", identity_exact(rng, old[p][1], rng.uniform(0.70, 0.80), AA))
        for i, p in enumerate(parents)
    ]
    records = old + new
    n_from_new = rng.randint(3, 5)
    n_from_old = rng.randint(4, 5)
    origins = rng.sample(range(n_old, n_old + n_new), n_from_new)
    origins += rng.sample(range(n_old), n_from_old)
    rng.shuffle(origins)
    queries = [
        (f"query{i + 1:02d}", identity_exact(rng, records[o][1], 0.94, AA))
        for i, o in enumerate(origins)
    ]
    return {
        "records": records,
        "n_old": n_old,
        "queries": queries,
        "best": [records[o][0] for o in origins],
    }


@functools.lru_cache(maxsize=128)
def database_files(tools: ToolEnv, fasta: str) -> tuple[tuple[str, bytes], ...]:
    """The BLAST database files of `fasta` named proteins.fasta, with the dates normalised."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "proteins.fasta").write_text(fasta)
        argv = ["makeblastdb", "-in", "proteins.fasta", "-dbtype", "prot", "-blastdb_version", "4"]
        run_ok(tools, argv, work)
        files = []
        for suffix in DB_SUFFIXES:
            data = (work / f"proteins.fasta{suffix}").read_bytes()
            data = DATE.sub(FIXED_DATE, data)
            data = STAMP.sub(b'"last-updated": "2026-01-01T00:00:00"', data)
            files.append((f"proteins.fasta{suffix}", data))
        return tuple(files)


def best_hits(tools: ToolEnv, work: Path, db_options: list[str]) -> list[str]:
    argv = ["blastp", "-query", "queries.fasta", *db_options, "-outfmt", "6 qseqid sseqid bitscore"]
    proc = run_ok(tools, [*argv, "-num_threads", "1"], work)
    best: dict[str, tuple[float, str]] = {}
    for line in proc.stdout.splitlines():
        query, subject, score = line.split("\t")
        if query not in best or float(score) > best[query][0]:
            best[query] = (float(score), subject)
    order = [
        line[1:].split()[0]
        for line in (work / "queries.fasta").read_text().splitlines()
        if line.startswith(">")
    ]
    return [best[q][1] if q in best else "none" for q in order]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    records = plan["records"]
    (work / "proteins.fasta").write_text(fasta_text(records))
    (work / "queries.fasta").write_text(fasta_text(plan["queries"]))
    built_from = records if twin == "clean" else records[: plan["n_old"]]
    for name, data in database_files(tools, fasta_text(built_from)):
        (work / name).write_bytes(data)
    lines = [
        f"proteins.fasta: {len(records)} reference proteins (current release).",
        "proteins.fasta.phr/.pin/.psq/.pjs: the BLAST protein database of proteins.fasta "
        "(makeblastdb).",
        f"queries.fasta: {len(plan['queries'])} proteins to be identified.",
    ]
    if tier == "explicit":
        lines.append(
            f"The database was built before the last {len(records) - plan['n_old']} proteins "
            "were appended to proteins.fasta."
            if twin == "faulted"
            else "The database was built from the current proteins.fasta."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = plan["best"]
    if twin == "clean":
        naive, evidence = expected, ()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            for name in (
                "queries.fasta",
                *(n for n, _ in database_files(tools, fasta_text(built_from))),
            ):
                (scratch / name).write_bytes((work / name).read_bytes())
            naive = best_hits(tools, scratch, ["-db", "proteins.fasta"])
        evidence = (
            f"proteins.fasta holds {len(records)} proteins but its BLAST database only "
            f"{plan['n_old']} (blastdbcmd -db proteins.fasta -info)",
        )
    prompt = common.prompt_for(
        route,
        "for every query in queries.fasta, in file order, report the ID of its best hit in "
        "proteins.fasta (highest bit score)",
        "use blastp to report, for every query in queries.fasta, in file order, the ID of its "
        "best hit in proteins.fasta (highest bit score)",
        "a JSON list of protein IDs, one per query",
    )
    truth = {"kind": "str_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F78", seed, twin, tier, route)
    return Task(task_id, "F78", "index_state", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["makeblastdb", "-in", "proteins.fasta", "-dbtype", "prot", "-out", "fresh"]
    run_ok(tools, argv, work)
    hits = best_hits(tools, work, ["-db", "fresh"])
    return result(hits, "makeblastdb of the current proteins.fasta, then blastp")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    hits = best_hits(tools, work, ["-db", "proteins.fasta"])
    return result(hits, "blastp against the existing database proteins.fasta")


# Exposure: a BLAST search against the shipped database (`-db proteins.fasta`, or its prefix) in a
# command that does not rebuild the database first. Judged per command.
BLAST = re.compile(r"(?<![\w-])(?:blastp|blastx|tblastn|psiblast)\b")
SHIPPED = re.compile(r"-db[ =]+['\"]?(?:\./)?proteins\.fasta(?![\w.])")
REBUILD = re.compile(r"(?<![\w-])makeblastdb\b")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        if REBUILD.search(command):
            continue
        for segment in segments(command):
            if BLAST.search(segment) and SHIPPED.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F78",
    slot="index_state",
    source="BLAST+ makeblastdb: the database is a snapshot of the FASTA at build time",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="data",
)
