"""F8 reads_vs_alignments (record_count, samtools flag semantics): records are not reads.

The task asks how many distinct reads map to each contig, a read counting once however many
alignment records it has there. Clean twin: every read has one record, so any record count is the
read count. Faulted twin: the same primary records plus secondary (0x100) and supplementary (0x800)
records of reads that are already on the contig. `samtools idxstats` and `samtools view -c` count
alignment records, so both are too high on every contig, with exit 0 and no warning.

Extra records always sit on their read's own contig, so "reads on a contig" does not depend on
whether a read is counted by its primary record or by any of its records. The count of distinct
read names among mapped records, per contig, is the truth; `samtools view -c -F 0x900` agrees.
Every record has MAPQ 60, so a mapping-quality filter does not hide the trap.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

READ_LEN = 40
SECONDARY, SUPPLEMENTARY, REVERSE = 0x100, 0x800, 0x10
NON_PRIMARY = SECONDARY | SUPPLEMENTARY
CONTIGS = {"chr1": (800, 1000), "chr2": (700, 900), "chr3": (600, 800)}  # total <= 2.7 kb

IDXSTATS = re.compile(r"\bsamtools\s+idxstats\b")
# idxstats prints contig, length, mapped records, unmapped records; cut down to its first column it
# is only a list of contigs (the reference solver reads it so) and shows no count
CONTIG_LIST = re.compile(
    r"\bsamtools\s+idxstats\b(?:2>&1|[^|;&\n])*\|\s*(?:(?:grep|head|tail|sed|sort)\b[^|;&\n]*\|\s*)*"
    r"(?:cut\s+(?:-d\s*\S+\s+)?-f\s*1(?![\d,-])|awk\b[^|;&\n]*\{\s*print\s+\$1\s*\})"
)
VIEW_COUNT = re.compile(r"\bsamtools\s+view\b[^|;&]*?\s(?:-[A-Za-z]*c[A-Za-z0-9]*|--count)(?=\s|$)")
VIEW_PIPE_COUNT = re.compile(r"\bsamtools\s+view\b.*\|.*(?:\bwc\s+-l\b|\buniq\s+-c\b)")
FLAG_FILTER = re.compile(r"(?:\s-[A-Za-z]*F|\s--excl-flags)(?:\s+|=)?(0[xX][0-9a-fA-F]+|\d+)")
DEDUPE = re.compile(r"\bsort\s+(?:-\w*u\w*|--unique)\b|\buniq(?!\s+-c)\b|!\s*\w+\[")
SEGMENT = re.compile(r"&&|\|\||;|\n")


@dataclass(frozen=True)
class Rec:
    name: str
    flag: int
    chrom: str
    pos: int  # 1-based
    cigar: str
    seq_from: int  # the record carries bases [seq_from:] of the read (hard clips drop the rest)


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    reads: dict[str, str]  # read name -> sequence
    primary: list[Rec]
    extra: list[Rec]  # secondary and supplementary records


def plan(seed: int) -> Plan:
    rnd = random.Random(f"f08-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    reads: dict[str, str] = {}
    primary: list[Rec] = []
    extra: list[Rec] = []
    for chrom, n in contigs.items():
        names = []
        for _ in range(rnd.randint(14, 20)):
            name = f"read{len(reads):04d}"
            reads[name] = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
            names.append(name)
        split = {name: rnd.randint(12, READ_LEN - 12) for name in rnd.sample(names, 4)}
        chimeric = list(split)
        for name in names:
            strand = REVERSE if rnd.random() < 0.5 else 0
            clip = split.get(name)
            cigar = f"{clip}M{READ_LEN - clip}S" if clip else f"{READ_LEN}M"
            primary.append(Rec(name, strand, chrom, rnd.randint(1, n - READ_LEN + 1), cigar, 0))
        flags = {r.name: r.flag for r in primary if r.chrom == chrom}
        # supplementary: the clipped tail aligned elsewhere on the contig; the first chimeric
        # read gets two of them, so a read can have several extra records
        for name in chimeric[: rnd.randint(2, 3)] + chimeric[:1]:
            clip = split[name]
            tail = f"{clip}H{READ_LEN - clip}M"
            pos = rnd.randint(1, n - READ_LEN)
            extra.append(Rec(name, flags[name] | SUPPLEMENTARY, chrom, pos, tail, clip))
        # secondary: the whole read at another locus; the first chimeric read gets one as well
        for name in [chimeric[0], *rnd.sample(names, rnd.randint(2, 3))]:
            pos = rnd.randint(1, n - READ_LEN)
            extra.append(Rec(name, flags[name] | SECONDARY, chrom, pos, f"{READ_LEN}M", 0))
    return Plan(contigs, reads, primary, extra)


def records(p: Plan, twin: str) -> list[Rec]:
    return p.primary + (p.extra if twin == "faulted" else [])


def counts(p: Plan, twin: str) -> tuple[dict[str, int], dict[str, int]]:
    """(distinct reads per contig, alignment records per contig)."""
    recs = records(p, twin)
    reads = {c: len({r.name for r in recs if r.chrom == c}) for c in p.contigs}
    rows = {c: sum(r.chrom == c for r in recs) for c in p.contigs}
    return reads, rows


def sam_text(p: Plan, twin: str) -> str:
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in p.contigs.items()
    )
    rows = []
    for r in records(p, twin):
        seq = p.reads[r.name][r.seq_from :]
        rows.append(
            f"{r.name}\t{r.flag}\t{r.chrom}\t{r.pos}\t60\t{r.cigar}\t*\t0\t0\t{seq}\t"
            + "I" * len(seq)
        )
    return header + "\n".join(rows) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    p = plan(seed)
    common.write_bam(work, sam_text(p, twin), tools)
    explicit = tier == "explicit"
    if twin == "clean":
        status = (
            "Every read has exactly one alignment record (primary alignments only)."
            if explicit
            else "Only primary alignments are kept."
        )
        evidence = (
            ("README states that every read has exactly one alignment record",)
            if explicit
            else ("README says only primary alignments are kept",)
        )
    else:
        status = (
            "Some reads have more than one alignment record (secondary 0x100 and supplementary "
            "0x800 records)."
            if explicit
            else "Secondary and supplementary alignments are kept."
        )
        evidence = (
            (
                "README states that some reads have secondary (0x100) and supplementary (0x800) "
                "records",
            )
            if explicit
            else ("README says secondary and supplementary alignments are kept",)
        ) + ("`samtools flagstat reads.bam` reports secondary and supplementary records",)
    readme = [
        "reads.bam (with reads.bam.bai): single-end reads aligned to chr1, chr2 and chr3, "
        "coordinate-sorted. " + status
    ]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "for each contig, report how many distinct reads in reads.bam map to it, counting a "
        "read once however many alignment records it has on that contig",
        "use `samtools view -c` to report, for each contig, how many distinct reads in "
        "reads.bam map to it, counting a read once however many alignment records it has on "
        "that contig",
        "a JSON object mapping each contig name to its read count",
    )
    reads, rows = counts(p, twin)
    truth = {"kind": "dict_num", "expected": reads, "naive": rows, "tol": 0}
    task_id = common.task_id("F8", seed, twin, tier, route)
    return Task(task_id, "F8", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _contigs(work: Path, tools: ToolEnv) -> list[str]:
    proc = tools.run(["samtools", "idxstats", "reads.bam"], work)
    return [line.split("\t")[0] for line in proc.stdout.splitlines() if not line.startswith("*")]


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["samtools", "view", "-F", "4", "reads.bam"], work)
    names: dict[str, set[str]] = {c: set() for c in _contigs(work, tools)}
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        names[fields[2]].add(fields[0])
    return _result(
        {c: len(s) for c, s in names.items()}, "distinct read names among mapped records"
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["samtools", "idxstats", "reads.bam"], work)
    rows = (line.split("\t") for line in proc.stdout.splitlines() if not line.startswith("*"))
    return _result({r[0]: int(r[2]) for r in rows}, "samtools idxstats mapped counts")


def _flag_value(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError:  # "004": leading zeros are decimal for samtools
        return int(text)


def _shows_idxstats_counts(segment: str) -> bool:
    """samtools idxstats whose output is not cut down to the contig names."""
    return bool(IDXSTATS.search(segment)) and not CONTIG_LIST.search(segment)


def _counts_records(segment: str) -> bool:
    """A samtools view count that does not leave the secondary and supplementary records out."""
    if not (VIEW_COUNT.search(segment) or VIEW_PIPE_COUNT.search(segment)):
        return False
    if DEDUPE.search(segment):
        return False
    return not any(
        _flag_value(v) & NON_PRIMARY == NON_PRIMARY for v in FLAG_FILTER.findall(segment)
    )


def exposed(commands: list[str], work: Path) -> bool:
    """True iff a command counted alignment records per contig.

    Trapped: samtools idxstats, unless its output is cut down to the contig names (cut -f1,
    awk '{print $1}'); samtools view with -c, or piped into wc -l or uniq -c, unless its -F value
    covers 0x900 (primary only) or the pipeline removes repeated read names (sort -u, uniq, an awk
    seen[] filter). Counting in Python without deduplication is not seen.
    """
    for command in commands:
        for segment in SEGMENT.split(command):
            if _shows_idxstats_counts(segment) or _counts_records(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F8",
    slot="record_count",
    source="samtools flag semantics (secondary 0x100, supplementary 0x800)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "README.md"}),
)
