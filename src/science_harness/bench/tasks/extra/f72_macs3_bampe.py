"""F72 macs3_bampe (locus tool_default): MACS3 reads a paired-end BAM as single-end tags.

`macs3 callpeak -f BAM` (and the default `-f AUTO`, which detects BAM) runs in single-end mode on a
paired-end BAM: the log says `Paired-End mode is off`, the second mate of every pair is ignored and
the first mate is extended by `--extsize` (200 with `--nomodel`) instead of using the real fragment.
`-f BAMPE` uses the fragments. GEN-26, macs3-project/MACS#486 ("the 2nd read in the pair will be
ignored"); nothing warns, exit 0, the log has INFO lines only.

Verified on MACS3 3.0.4 (synthetic 50 kb genome, 2 contigs, fragments of 85 +- 8 bases, 55-70
fragments per binding site, `-g 50000 -q 0.01`): 1200 records give `total tags in treatment: 600`
(flags 99 and 83 are counted, 147 and 163 are not); `-f BAMPE` calls 8 peaks for 4 loci of two
binding sites each, 250 bases apart; `-f BAM --nomodel --extsize 200` calls 4 (each pair of sites
joins into one peak, 260-440 bases wide) and `--extsize 147` too; with the sites 420 bases apart
both modes call 8. The twins hold the same loci and the same reads; only the distance between the
two sites of each locus differs (faulted 215-250 bases, clean 400-460).

  explicit  the README says the reads are paired-end
  implicit  the README says nothing about the layout; the flags (0x1, proper pairs) show it
  forced    the task sentence dictates `macs3 callpeak`

Truth: two peaks per locus. `macs3 callpeak -f BAMPE` gave exactly that, and the default format the
naive count, for seeds 0-29 in both twins (a one-off sweep; the tests repeat it for seeds 0-3).
"""

from __future__ import annotations

import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import normalise
from science_harness.bench.tasks.tools import ToolEnv

QVALUE = "0.01"
READ_LEN = 50
FRAG_MEAN, FRAG_SD = 85, 8
SITE_SD = 12
GAP_CLEAN = (400, 460)
GAP_FAULTED = (215, 250)
BG_EXCLUDE = 600  # no background fragment this close to a locus centre
BG_FRAGMENTS = 140
CONTIG_BOUNDS = {"chr1": (28_000, 32_000), "chr2": (18_000, 22_000)}
TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai"})
GENOME_LINE = (
    "genome.sizes: contig names and lengths, tab-separated. "
    "The effective genome size is their total."
)


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    loci: tuple[tuple[str, int], ...]  # (contig, centre0)
    gap: int
    pairs: tuple[
        tuple[str, int, int, bool], ...
    ]  # (contig, start1, fragment length, mate1 forward)

    @property
    def genome_size(self) -> int:
        return sum(self.contigs.values())

    @property
    def expected(self) -> int:
        return 2 * len(self.loci)

    @property
    def merged(self) -> int:
        return len(self.loci)


def _loci(rnd: random.Random, contigs: dict[str, int], n: int) -> tuple[tuple[str, int], ...]:
    chosen: list[tuple[str, int]] = []
    names = list(contigs)
    while len(chosen) < n:
        chrom = rnd.choices(names, weights=[contigs[c] for c in names])[0]
        centre = rnd.randint(2000, contigs[chrom] - 2000)
        if all(c != chrom or abs(centre - x) >= 4000 for c, x in chosen):
            chosen.append((chrom, centre))
    order = {c: i for i, c in enumerate(names)}
    return tuple(sorted(chosen, key=lambda cp: (order[cp[0]], cp[1])))


def make_plan(seed: int, twin: str) -> Plan:
    layout = random.Random(f"f72-{seed}-layout")
    contigs = {c: layout.randint(lo, hi) for c, (lo, hi) in CONTIG_BOUNDS.items()}
    loci = _loci(layout, contigs, 3 + seed % 3)
    gap_rnd = random.Random(f"f72-{seed}-gap-{twin}")
    gap = gap_rnd.randint(*(GAP_CLEAN if twin == "clean" else GAP_FAULTED))
    rnd = random.Random(f"f72-{seed}-reads")  # the same draws in both twins
    pairs: list[tuple[str, int, int, bool]] = []

    def fragment() -> int:
        return max(
            READ_LEN + 10, min(FRAG_MEAN + 3 * FRAG_SD, round(rnd.gauss(FRAG_MEAN, FRAG_SD)))
        )

    for chrom, centre in loci:
        for sign in (-1, 1):
            for _ in range(rnd.randint(55, 70)):
                length = fragment()
                middle = centre + sign * gap // 2 + round(rnd.gauss(0, SITE_SD))
                pairs.append((chrom, max(1, middle - length // 2), length, rnd.random() < 0.5))
    # background: one fragment per stratum of the genome (no pile-ups), none near a locus
    total = sum(contigs.values())
    width = total // BG_FRAGMENTS
    offset = 0
    for chrom, size in contigs.items():
        for lo in range(offset, offset + size - width, width):
            length = fragment()
            start = lo - offset + rnd.randint(1, width - 1)
            forward = rnd.random() < 0.5
            near = any(c == chrom and abs(start + length // 2 - x) < BG_EXCLUDE for c, x in loci)
            if not near and start + length < size:
                pairs.append((chrom, start, length, forward))
        offset += size
    pairs.sort()
    return Plan(contigs, loci, gap, tuple(pairs))


def sam_text(plan: Plan, seed: int) -> str:
    rnd = random.Random(f"f72-{seed}-seq")
    header = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in plan.contigs.items()
    )
    qual = "I" * READ_LEN
    lines: list[str] = []
    for i, (chrom, start, length, forward) in enumerate(plan.pairs):
        name = f"frag{i + 1:04d}"
        left, right = start, start + length - READ_LEN
        seq1 = "".join(rnd.choices("ACGT", k=READ_LEN))
        seq2 = "".join(rnd.choices("ACGT", k=READ_LEN))
        flags = (99, 147) if forward else (163, 83)  # mate 1 forward / mate 1 reverse
        for pos, mate_pos, flag, tlen, seq in (
            (left, right, flags[0], length, seq1),
            (right, left, flags[1], -length, seq2),
        ):
            lines.append(
                f"{name}\t{flag}\t{chrom}\t{pos}\t60\t{READ_LEN}M\t=\t{mate_pos}\t{tlen}\t{seq}\t{qual}"
            )
    return header + "\n".join(lines) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed, twin)
    common.write_bam(work, sam_text(plan, seed), tools, "reads")
    (work / "genome.sizes").write_text("".join(f"{c}\t{n}\n" for c, n in plan.contigs.items()))
    reads_line = (
        "reads.bam (with reads.bam.bai): aligned reads of a transcription-factor profiling assay."
    )
    if tier == "explicit":
        reads_line += " Paired-end sequencing; the fragments are short (about 85 bases)."
    (work / "README.md").write_text(common.readme([reads_line, GENOME_LINE]))
    if twin == "faulted":
        evidence = (
            f"the two binding sites of each locus are {plan.gap} bases apart and the reads are "
            "paired-end (flag 0x1): single-end mode extends each first mate by 200 bases and "
            "joins the two sites",
        )
    else:
        evidence = (
            f"the two binding sites of each locus are {plan.gap} bases apart: wider than the "
            "single-end extension, so both modes find every peak",
        )
    if tier == "explicit":
        evidence = ("README states paired-end sequencing", *evidence)
    prompt = common.prompt_for(
        route,
        f"call peaks in reads.bam with MACS3 at a q-value cutoff of {QVALUE} and report how many "
        "peaks it calls",
        f"use `macs3 callpeak` with a q-value cutoff of {QVALUE} on reads.bam and report how "
        "many peaks it calls",
        "a JSON integer",
    )
    truth = {
        "kind": "int",
        "expected": plan.expected,
        "naive": plan.expected if twin == "clean" else plan.merged,
    }
    task_id = common.task_id("F72", seed, twin, tier, route)
    return Task(task_id, "F72", "read_layout", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers ------------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def genome_size(work: Path) -> int:
    rows = (line.split("\t") for line in (work / "genome.sizes").read_text().splitlines())
    return sum(int(r[1]) for r in rows)


def count_peaks(work: Path, tools: ToolEnv, *options: str, name: str = "run") -> int:
    """Number of peaks of `macs3 callpeak -t reads.bam <options>` (output goes elsewhere)."""
    out = Path(tempfile.mkdtemp(prefix="f72-"))
    try:
        argv = ["macs3", "callpeak", "-t", "reads.bam", "-n", name, "--outdir", str(out)]
        argv += ["-g", str(genome_size(work)), "-q", QVALUE, *options]
        proc = tools.run(argv, work)
        if proc.returncode != 0:
            raise RuntimeError(f"macs3 failed: {proc.stderr[-400:]}")
        text = (out / f"{name}_peaks.narrowPeak").read_text()
        return len([line for line in text.splitlines() if line.strip()])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def is_paired(work: Path, tools: ToolEnv) -> bool:
    proc = tools.run(["samtools", "view", "-c", "-f", "1", "reads.bam"], work)
    return int(proc.stdout.strip() or 0) > 0


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    if is_paired(work, tools):
        count = count_peaks(work, tools, "-f", "BAMPE")
        flags = ["paired-end BAM: MACS3 needs -f BAMPE (the default reads one mate per pair)"]
        return _result(count, "macs3 callpeak -f BAMPE", flags)
    count = count_peaks(work, tools, "-f", "BAM", "--nomodel", "--extsize", "200")
    return _result(count, "macs3 callpeak -f BAM --nomodel --extsize 200")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    count = count_peaks(work, tools, "--nomodel", "--extsize", "200")
    return _result(count, "macs3 callpeak with its default format (AUTO), --nomodel --extsize 200")


_FORMAT = re.compile(r"(?:^|\s)(?:-f\s*|--format(?:\s+|=))(\w+)")
_SPLIT = re.compile(r"[|;&\n]|&&|\|\|")
_CALLPEAK = re.compile(r"\bmacs3?\b.*\bcallpeak\b", re.IGNORECASE)


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff some `macs3 callpeak` runs with a single-end format on this paired-end BAM:
    no `-f`/`--format`, or AUTO, BAM, SAM or BED. `-f BAMPE` (or `-fBAMPE`, BEDPE) is the safe
    route. Checked command by command, so a corrected rerun does not hide the first trapped run;
    `python -m MACS3 callpeak` is the same program."""
    for command in commands:
        for part in _SPLIT.split(normalise(command)):
            if not _CALLPEAK.search(part):
                continue
            found = _FORMAT.search(part)
            if found is None or found.group(1).upper() not in ("BAMPE", "BEDPE", "FRAG"):
                return True
    return False


SPEC = FamilySpec(
    id="F72",
    slot="read_layout",
    source="GEN-26 (macs3-project/MACS#486: -f BAM reads one mate of a pair)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
