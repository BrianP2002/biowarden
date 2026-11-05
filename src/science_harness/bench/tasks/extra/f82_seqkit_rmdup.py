"""F82 seqkit_rmdup: `seqkit rmdup -s` treats a sequence and its reverse complement as duplicates.

`seqkit rmdup --by-seq` removes records with identical sequences, and for nucleotides it compares
both strands: `-P, --only-positive-strand  only considering the positive strand when comparing by
sequence`. Without `-P` a guide RNA, a primer or a barcode that is listed as the reverse
complement of another entry is dropped as a "duplicate". Exit 0; the only message is an INFO line
on stderr with the number of removed records.

Verified here (seqkit v2.14.0): six records (a1 ACGTACGTAAGGCCTTAA, its reverse complement a3, an
exact copy a4, a case variant, two unrelated): `rmdup -s` kept 4 records (a3 and a4 removed,
"2 duplicated records removed"), `rmdup -s -P` kept 5 (only the copy removed), `rmdup -s -i` also
removed the lower-case copy. On the generated sets of this family (seeds 0-29): `rmdup -s` leaves
one record fewer than there are distinct sequences for every reverse-complement pair.

Twins: the same number of records, the same exact duplicates. Clean: the extra entries are
unrelated guides. Faulted: 6-12 entries are the reverse complement of another entry (a design
export that lists both strands). The answer is the number of distinct sequences as written.

  explicit  the README says sequences are compared exactly as written, 5' to 3'
  implicit  the README only says the file is a guide library
  forced    the task sentence dictates seqkit rmdup
"""

from __future__ import annotations

import random
import re
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
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"guides.fasta", "README.md"})


def make_plan(seed: int, twin: str) -> list[tuple[str, str]]:
    rng = random.Random(f"f82-{seed}")
    uniq: list[str] = []
    n_unique = rng.randint(40, 60)
    while len(uniq) < n_unique:
        s = rand_dna(rng, 20)
        if s != revcomp(s) and s not in uniq and revcomp(s) not in uniq:
            uniq.append(s)
    n_exact = rng.randint(6, 12)
    n_extra = rng.randint(6, 12)
    sequences = list(uniq)
    sequences += [rng.choice(uniq) for _ in range(n_exact)]  # exact duplicates
    parents = rng.sample(uniq, n_extra)  # distinct parents: the extra entries are all distinct
    for parent in parents:
        if twin == "faulted":
            sequences.append(revcomp(parent))
        else:
            while True:
                s = rand_dna(rng, 20)
                if s != revcomp(s) and s not in uniq and revcomp(s) not in uniq:
                    break
            sequences.append(s)
            uniq.append(s)
    rng.shuffle(sequences)
    return [(f"guide{i + 1:03d}", s) for i, s in enumerate(sequences)]


def distinct_after(tools: ToolEnv, work: Path, *options: str) -> int:
    argv = ["seqkit", "rmdup", "-s", *options, "guides.fasta"]
    out = run_ok(tools, argv, work).stdout
    return out.count(">")


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    records = make_plan(seed, twin)
    (work / "guides.fasta").write_text(fasta_text(records, width=80))
    lines = [
        f"guides.fasta: {len(records)} guide RNA spacers (20 bases, DNA letters) of a CRISPR "
        "library design export; some guides are listed more than once.",
    ]
    if tier == "explicit":
        lines.append(
            "Guides are compared exactly as written, 5' to 3': a guide and the reverse "
            "complement of another guide are two different guides."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    seqs = [s for _, s in records]
    expected = len(set(seqs))
    both_strands = len({min(s, revcomp(s)) for s in seqs})
    prompt = common.prompt_for(
        route,
        "count the distinct guide sequences in guides.fasta",
        "use seqkit rmdup to count the distinct guide sequences in guides.fasta",
        "a JSON integer",
    )
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{expected - both_strands} guides are the reverse complement of another guide",
            "seqkit rmdup -s compares both strands unless -P is given",
        )
    truth = {"kind": "int", "expected": expected, "naive": both_strands}
    task_id = common.task_id("F82", seed, twin, tier, route)
    return Task(task_id, "F82", "strand", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = distinct_after(tools, work, "-P")
    return result(n, "seqkit rmdup -s -P (same strand only)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(distinct_after(tools, work), "seqkit rmdup -s (both strands)")


# Exposure: a seqkit rmdup by sequence (-s, --by-seq) without -P / --only-positive-strand.
# Judged per command; `rmdup` by name (the default) never compares sequences.
RMDUP = re.compile(r"(?<![\w-])seqkit\s+(?:[^|;&]*\s)?rmdup\b")
BY_SEQ = re.compile(r"(?<![\w-])(?:-[A-Za-z]*s[A-Za-z]*|--by-seq)(?![\w-])")
POSITIVE = re.compile(r"(?<![\w-])(?:-[A-Za-z]*P[A-Za-z]*|--only-positive-strand)(?![\w-])")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if RMDUP.search(segment) and BY_SEQ.search(segment) and not POSITIVE.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F82",
    slot="strand",
    source="seqkit rmdup --by-seq: -P/--only-positive-strand (both strands compared by default)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
