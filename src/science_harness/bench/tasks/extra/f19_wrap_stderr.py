"""F19 wrap_stderr (warning_channel, locus wrapper): the lab's sequence-extraction wrapper sends the
stderr of bedtools to /dev/null, and the one message that mattered says an interval was skipped.

Source GEN-07 / GEN-08 (records dropped for a contig name that does not match, with no output
that says so) and bedtools getfasta itself. Verified on bedtools 2.31.1: for a BED line whose
contig is not in the FASTA index, `bedtools getfasta -fi ref.fa -bed targets.bed -fo -` prints on
stderr `WARNING. chromosome (1) was not found in the FASTA file. Skipping.`, leaves that interval
out of the FASTA on stdout, and exits 0. With `2>/dev/null` the only trace is a FASTA with fewer
records than the BED has lines.

Twin axis (the one family where a data file differs too): the clean BED names every contig as ref.fa
does (chr1, chr2) and the clean wrapper passes stderr through, so nothing is skipped and nothing
needs saying. In the faulted twin the last 8-10 lines of targets.bed (a second panel appended to the
file) name the same intervals as 1 and 2, and the wrapper ends in `2>/dev/null`. Sequences,
intervals and their order are the same, so the truth is the same. Task: count the intervals whose
sequence contains the EcoRI site GAATTC (a palindrome, so strand does not matter). 3-5 of the
second-panel intervals contain it and are missed in the faulted twin (8-13 hits in all). Decoys in
both panels: intervals that cut the site at one end, or hold a one-base variant of it.

Exposure: the wrapper ran on targets.bed itself (a file derived from it is a different argument, and
so is any run after an earlier command rewrote targets.bed in place) with no safe marker in the same
command: a direct bedtools call, or a rewrite of the wrapper or of targets.bed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f17_wrap_filter import (
    has_edit,
    result,
    run_wrapper,
    runs_tool,
    wrapper_calls,
    write_script,
)
from science_harness.bench.tasks.tools import ToolEnv

WRAPPER = "lab-getseq.sh"
BED = "targets.bed"
MOTIF = "GAATTC"
CONTIGS = {"chr1": (900, 1200), "chr2": (800, 1100)}
TWIN_FILES = frozenset({WRAPPER, BED, "README.md"})

HEAD = """#!/bin/sh
# lab-getseq.sh: reference sequence of every interval in a BED file, as FASTA.
# usage: ./lab-getseq.sh BED
set -eu
[ $# -ge 1 ] || { echo "usage: $0 BED" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
"""
CALL = 'bedtools getfasta -fi "${LAB_REF:-$here/ref.fa}" -bed "$1" -fo -'
WRAPPER_CLEAN = HEAD + f"exec {CALL}\n"
WRAPPER_FAULTED = HEAD + f"# keep the job logs quiet\nexec {CALL} 2>/dev/null\n"


@dataclass(frozen=True)
class Interval:
    chrom: str
    start0: int
    end0: int

    def bed(self, alias: bool) -> str:
        chrom = self.chrom.removeprefix("chr") if alias else self.chrom
        return f"{chrom}\t{self.start0}\t{self.end0}\n"


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    first: list[Interval]  # the first panel, sorted
    second: list[Interval]  # the second panel, appended to the file (renamed in the faulted twin)

    @property
    def intervals(self) -> list[Interval]:
        return self.first + self.second

    def sequence(self, iv: Interval) -> str:
        return self.genome[iv.chrom][iv.start0 : iv.end0]


def _intervals(rnd: random.Random, contigs: dict[str, int]) -> list[Interval]:
    out: list[Interval] = []
    for chrom, n in contigs.items():
        cursor = rnd.randint(30, 60)
        while True:
            length = rnd.randint(24, 60)
            if cursor + length + 30 > n:
                break
            out.append(Interval(chrom, cursor, cursor + length))
            cursor += length + rnd.randint(16, 40)
    return out


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f19-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    seqs = {c: "".join(rnd.choice("ACGT") for _ in range(n)) for c, n in contigs.items()}
    for chrom, seq in seqs.items():  # no accidental site: every one in the file is planted
        while MOTIF in seq:
            seq = seq.replace(MOTIF, "GAGTTC")
        seqs[chrom] = seq
    intervals = _intervals(rnd, contigs)
    n_second = rnd.randint(8, 10)
    second_ids = set(rnd.sample(range(len(intervals)), n_second))
    first = [iv for i, iv in enumerate(intervals) if i not in second_ids]
    second = [iv for i, iv in enumerate(intervals) if i in second_ids]
    # which intervals get the site, a cut site, a one-base variant: counted per panel
    pick_second = rnd.sample(second, rnd.randint(3, 5))
    pick_first = rnd.sample(first, rnd.randint(5, 8))
    hits = pick_first + pick_second
    rest = [iv for iv in first + second if iv not in hits]
    rnd.shuffle(rest)
    edits: list[tuple[str, int, str]] = []  # (contig, 0-based start of the edit, new bases)
    for iv in hits:
        at = iv.start0 + rnd.randint(0, iv.end0 - iv.start0 - len(MOTIF))
        edits.append((iv.chrom, at, MOTIF))
    for iv, kind in zip(rest[:6], ("tail", "head", "tail", "head", "near", "near"), strict=False):
        if kind == "tail":  # GAATT inside the interval, the final C just outside it
            edits.append((iv.chrom, iv.end0 - len(MOTIF) + 1, MOTIF))
        elif kind == "head":  # the G just outside, AATTC inside
            edits.append((iv.chrom, iv.start0 - 1, MOTIF))
        else:
            at = iv.start0 + rnd.randint(0, iv.end0 - iv.start0 - len(MOTIF))
            edits.append((iv.chrom, at, rnd.choice(("GAATTG", "GGATTC", "GAACTC"))))
    for chrom, at, bases in edits:
        seq = seqs[chrom]
        seqs[chrom] = seq[:at] + bases + seq[at + len(bases) :]
    return Plan(seqs, first, second)


def count_sites(plan: Plan, intervals: list[Interval]) -> int:
    return sum(MOTIF in plan.sequence(iv) for iv in intervals)


def truth_counts(plan: Plan) -> tuple[int, int]:
    """(intervals holding the site, those among the first panel: all that the faulted run sees)."""
    return count_sites(plan, plan.intervals), count_sites(plan, plan.first)


def bed_text(plan: Plan, twin: str) -> str:
    alias = twin == "faulted"
    return "".join(iv.bed(False) for iv in plan.first) + "".join(
        iv.bed(alias) for iv in plan.second
    )


def build_f19(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """warning_channel: a skipped interval reported only on a stderr the wrapper discards."""
    plan = make_plan(seed)
    common.write_fasta(work, plan.genome, tools)
    (work / BED).write_text(bed_text(plan, twin))
    faulted = twin == "faulted"
    write_script(work, WRAPPER, WRAPPER_FAULTED if faulted else WRAPPER_CLEAN)
    line = (
        f"{WRAPPER}: the lab's sequence extractor around bedtools getfasta "
        f"(usage: ./{WRAPPER} BED; it reads ref.fa and prints FASTA)."
    )
    if tier == "explicit":
        line += (
            " It discards the stderr of bedtools."
            if faulted
            else " It passes the messages of bedtools through."
        )
    readme = [
        "ref.fa (with ref.fa.fai): reference sequence, contigs chr1 and chr2.",
        f"{BED}: target intervals in BED format (chrom, start, end), one per line.",
        line,
    ]
    (work / "README.md").write_text(common.readme(readme))
    source = f"{WRAPPER} source: exec bedtools getfasta ... 2>/dev/null"
    naming = f"{BED} names some contigs 1 and 2, ref.fa names them chr1 and chr2"
    if faulted:
        evidence = (f"README: {WRAPPER} discards the stderr of bedtools", source, naming)
        evidence = evidence if tier == "explicit" else evidence[1:]
    else:
        evidence = (
            (f"README: {WRAPPER} passes the messages of bedtools through",)
            if tier == "explicit"
            else ()
        )
    prompt = common.prompt_for(
        route,
        f"count the intervals in {BED} whose reference sequence in ref.fa contains the EcoRI site "
        f"{MOTIF} (the lab usually extracts interval sequences with ./{WRAPPER})",
        f"use the lab's ./{WRAPPER} wrapper to extract the reference sequence of each interval in "
        f"{BED}, and count the intervals whose sequence contains the EcoRI site {MOTIF}",
        "a JSON integer",
    )
    total, first_only = truth_counts(plan)
    truth = {"kind": "int", "expected": total, "naive": first_only if faulted else total}
    task_id = common.task_id("F19", seed, twin, tier, route)
    return Task(task_id, "F19", "warning_channel", twin, seed, prompt, truth, evidence, tier, route)


def fasta_records(text: str) -> list[str]:
    records: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            records.append("")
        elif records:
            records[-1] += line.strip()
    return records


def reference_f19(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    known = {line.split("\t")[0] for line in (work / "ref.fa.fai").read_text().splitlines()}
    lines, renamed = [], 0
    for line in (work / BED).read_text().splitlines():
        chrom, rest = line.split("\t", 1)
        if chrom not in known and f"chr{chrom}" in known:
            chrom, renamed = f"chr{chrom}", renamed + 1
        lines.append(f"{chrom}\t{rest}")
    argv = ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "stdin", "-fo", "-"]
    proc = tools.run(argv, work, stdin="\n".join(lines) + "\n")
    count = sum(MOTIF in seq for seq in fasta_records(proc.stdout))
    flags = [f"{renamed} intervals name a contig that is not in ref.fa (1/2); renamed to chr1/chr2"]
    return result(
        count,
        "bedtools getfasta called directly after matching contig names",
        flags if renamed else [],
    )


def naive_f19(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = run_wrapper(tools, work, WRAPPER, BED)
    count = sum(MOTIF in seq for seq in fasta_records(proc.stdout))
    return result(count, f"./{WRAPPER} {BED}, sites counted in the FASTA it prints")


def exposed_f19(commands: list[str], work: Path) -> bool:
    bed_rewritten = False
    for command in commands:
        on_bed = any(
            Path(w).name == BED for words in wrapper_calls(command, WRAPPER) for w in words
        )
        rewrites_bed = has_edit(command, BED)
        safe = runs_tool(command, "bedtools") or has_edit(command, WRAPPER) or rewrites_bed
        if on_bed and not safe and not bed_rewritten:
            return True
        bed_rewritten = bed_rewritten or rewrites_bed
    return False


SPEC = FamilySpec(
    id="F19",
    slot="warning_channel",
    source="GEN-07 / GEN-08 (records dropped for contig names, no message); bedtools getfasta",
    build=build_f19,
    reference=reference_f19,
    naive=naive_f19,
    exposed=exposed_f19,
    twin_files=TWIN_FILES,
    locus="wrapper",
)
