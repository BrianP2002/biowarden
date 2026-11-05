"""F77 blastn_task: `blastn` defaults to megablast, which needs an exact 28-base seed.

BLAST+ blastn has four tasks. The default is `megablast`: word size 28, match/mismatch 1/-2, built
for near-identical sequences (same species, assemblies). A query that is a diverged copy of a
reference region (an ortholog from a related species at 80-90% identity) rarely holds 28 exact
bases in a row, so megablast reports no hit at all, with exit 0 and an empty stderr. `-task blastn`
(word size 11, match/mismatch 2/-3) or `-task dc-megablast` finds the same region with a strong
E-value.

Verified here (BLAST+ 2.17.0, 120-base queries cut from a 4 kb reference, `-subject`/`-db`):
    default blastn          identity 100, 95: found; 90: one of two; 85, 80, 75, 70: none
    -task blastn            found at every identity from 100 down to 70
    default blastn          exact 20-base and 25-base queries: no hit; exact 35-base queries: found
    -task blastn            exact 20- and 25-base queries found (the `blastn-short` task also does)
and on the generated sets of this family (seeds 0-29): see tests/test_family_seq.py.

Twins: the same reference and the same decoys. Clean: the homologous queries are 97-100%
identical to their reference region (the default finds them). Faulted: about 70% of them are
diverged copies (substitutions at roughly 11% of the positions, no exact run of 23 bases or more),
the rest stay near-identical, so the default call returns a plausible, shorter list.

  explicit  the README says how diverged the queries are (97-100%, or 85-90%)
  implicit  the README only says the queries come from a related species
  forced    the task sentence dictates blastn

This module also holds the sequence helpers the other `seq` families share.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

# ---------------------------------------------------------------------------------------------
# shared helpers for the seq families (F77-F88): tiny synthetic sequences and reads
# ---------------------------------------------------------------------------------------------

DNA = "ACGT"
AA = "ACDEFGHIKLMNPQRSTVWY"
_COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")
TOOLS_NOTE = (
    "Tools on the PATH: seqtk, seqkit, fastp, cutadapt, trimmomatic, BLAST+ (makeblastdb, blastn, "
    "blastp), diamond, mafft, muscle and HMMER (hmmbuild, hmmsearch, hmmscan). diamond needs an "
    "explicit --threads (for example --threads 4): without it, it starts one thread per core and "
    "stops in the memory-limited shell."
)
SEGMENT = re.compile(r"&&|\|\||;|\n|\|")


def rand_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(DNA) for _ in range(n))


def rand_prot(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def substitute(rng: random.Random, seq: str, rate: float, alphabet: str = DNA) -> str:
    """Substitute each position with probability `rate` (never by itself)."""
    out = list(seq)
    for i, base in enumerate(seq):
        if rng.random() < rate:
            out[i] = rng.choice([a for a in alphabet if a != base])
    return "".join(out)


def identity_exact(rng: random.Random, seq: str, identity: float, alphabet: str = DNA) -> str:
    """Substitute exactly round((1 - identity) * len) positions."""
    out = list(seq)
    for i in rng.sample(range(len(seq)), round(len(seq) * (1 - identity))):
        out[i] = rng.choice([a for a in alphabet if a != seq[i]])
    return "".join(out)


def break_runs(rng: random.Random, original: str, copy: str, max_run: int) -> str:
    """Add substitutions until no stretch of `max_run` or more bases equals the original."""
    out = list(copy)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == original[j]:
            j += 1
        if j - i >= max_run:
            at = i + max_run // 2 + rng.randint(-2, 2)
            out[at] = rng.choice([a for a in DNA if a != original[at]])
            i = at + 1
        else:
            i = j + 1
    return "".join(out)


def wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def fasta_text(records: list[tuple[str, str]], width: int = 60) -> str:
    return "".join(f">{name}\n{wrap(seq, width)}\n" for name, seq in records)


def fastq_text(records: list[tuple[str, str, str]]) -> str:
    return "".join(f"@{name}\n{seq}\n+\n{qual}\n" for name, seq, qual in records)


def phred(scores: list[int]) -> str:
    return "".join(chr(q + 33) for q in scores)


def run_ok(tools: ToolEnv, argv: list[str], cwd: Path, stdin: str | None = None) -> Any:
    proc = tools.run(argv, cwd, stdin=stdin)
    common.check(proc, argv)
    return proc


def segments(command: str) -> list[str]:
    """The simple commands of a shell line: split at ; && || | and newlines."""
    return [s for s in SEGMENT.split(command) if s.strip()]


def result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


# ---------------------------------------------------------------------------------------------
# F77
# ---------------------------------------------------------------------------------------------

EVALUE = "1e-5"
TWIN_FILES = frozenset({"queries.fasta", "README.md"})


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f77-{seed}")
    genome = {
        "chr1": rand_dna(rng, rng.randint(1800, 2400)),
        "chr2": rand_dna(rng, rng.randint(1400, 2000)),
    }
    n_hom = rng.randint(7, 10)
    homologs = []
    for _ in range(n_hom):
        contig = rng.choice(list(genome))
        length = rng.randint(110, 150)
        start = rng.randint(0, len(genome[contig]) - length)
        region = genome[contig][start : start + length]
        near = substitute(rng, region, rng.uniform(0.0, 0.03))
        far = break_runs(rng, region, substitute(rng, region, 0.11), 23)
        homologs.append((near, far, rng.random() < 0.7))
    decoys = [rand_dna(rng, rng.randint(110, 150)) for _ in range(3)]
    layout = [("h", i) for i in range(n_hom)] + [("d", i) for i in range(len(decoys))]
    rng.shuffle(layout)  # file order; the IDs are neutral (seq01, seq02, ...)
    return {"genome": genome, "homologs": homologs, "decoys": decoys, "layout": layout}


def query_records(plan: dict[str, Any], twin: str) -> list[tuple[str, str]]:
    records = []
    for pos, (kind, i) in enumerate(plan["layout"]):
        if kind == "h":
            near, far, diverged = plan["homologs"][i]
            seq = far if twin == "faulted" and diverged else near
        else:
            seq = plan["decoys"][i]
        records.append((f"seq{pos + 1:02d}", seq))
    return records


def homolog_ids(plan: dict[str, Any], diverged_only: bool = False) -> list[str]:
    out = []
    for pos, (kind, i) in enumerate(plan["layout"]):
        if kind == "h" and (not diverged_only or plan["homologs"][i][2]):
            out.append(f"seq{pos + 1:02d}")
    return sorted(out)


def hits(tools: ToolEnv, work: Path, *options: str) -> list[str]:
    """Query IDs with a hit in ref.fasta at E < EVALUE, for `blastn <options>`."""
    argv = ["blastn", "-query", "queries.fasta", "-subject", "ref.fasta", "-evalue", EVALUE]
    argv += [*options, "-outfmt", "6 qseqid", "-num_threads", "1"]
    proc = run_ok(tools, argv, work)
    return sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "ref.fasta").write_text(fasta_text(list(plan["genome"].items())))
    (work / "queries.fasta").write_text(fasta_text(query_records(plan, twin)))
    lines = [
        "ref.fasta: the reference sequence of a species, two contigs (chr1, chr2).",
        "queries.fasta: gene fragments (110-150 bases) cloned from a related species, plus three "
        "decoy sequences that have no counterpart in the reference.",
    ]
    if tier == "explicit":
        lines.append(
            "The gene fragments are 97-100% identical to their reference region."
            if twin == "clean"
            else "Most gene fragments are only 85-90% identical to their reference region."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = homolog_ids(plan)
    if twin == "clean":
        naive = expected
        evidence = ()
    else:
        far = homolog_ids(plan, diverged_only=True)
        naive = sorted(set(expected) - set(far))
        evidence = (
            f"{len(far)} of {len(expected)} gene fragments are 85-90% identical to the reference "
            "with no exact run of 23 bases",
            "blastn's default task (megablast) needs an exact 28-base seed",
        )
    prompt = common.prompt_for(
        route,
        f"list the IDs of the sequences in queries.fasta that have a BLAST hit in ref.fasta with "
        f"an E-value below {EVALUE}",
        f"use blastn to list the IDs of the sequences in queries.fasta that have a hit in "
        f"ref.fasta with an E-value below {EVALUE}",
        "a JSON list of sequence IDs",
    )
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F77", seed, twin, tier, route)
    return Task(task_id, "F77", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    found = hits(tools, work, "-task", "blastn")
    return result(found, "blastn -task blastn (word size 11) against ref.fasta")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(hits(tools, work), "blastn with the default task (megablast)")


# Exposure: a blastn command that runs megablast (no -task, or -task megablast) without lowering
# the word size to 16 or less. Judged per command; tblastn/tblastx/blastp are not exposures.
BLASTN = re.compile(r"(?<![\w-])blastn\b")
TASK = re.compile(r"-task[ =]+['\"]?([\w-]+)")
WORD = re.compile(r"-word_size[ =]+(\d+)")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if not BLASTN.search(segment):
                continue
            task = TASK.findall(segment)
            if task and task[-1] != "megablast":
                continue
            word = WORD.findall(segment)
            if word and int(word[-1]) <= 16:
                continue
            return True
    return False


SPEC = FamilySpec(
    id="F77",
    slot="default_param",
    source="BLAST+ blastn default task megablast (word size 28); BLAST+ manual, blastn tasks",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
