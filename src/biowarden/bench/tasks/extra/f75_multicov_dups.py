"""F75 multicov_dups (default_param, tool default): bedtools subcommands disagree on duplicates.

`bedtools multicov` skips reads flagged as duplicates (0x400) and QC-fail (0x200) unless `-D` / `-F`
are given ("Default counts non-duplicates only"). Its sibling subcommands `coverage`, `intersect
-c`, `genomecov -ibam` and `window` count every alignment record whatever the flag. The same BAM and
the same regions therefore give different read counts depending on the subcommand, and
neither says so: exit 0, empty stderr.

Verified here (bedtools 2.31.1, samtools 1.24): six records overlapping chr1 90-200 (flags 0, 1024,
1024, 0, 512, 256 with MAPQ 0 for the last): `samtools view -c -L` 6, `coverage -counts` 6,
`intersect -c` 6, `genomecov -ibam -bg` depth up to 6, `multicov` 3 (the 1024 and the 512 records
are skipped; the 256 record with MAPQ 0 is counted), `multicov -D -F` 6.

Task: for each region, the number of reads that are not marked as PCR duplicates. The twins hold
the same non-duplicate reads; the faulted twin adds duplicate-marked copies (flag 1024, same start
as a kept read) to about 60 percent of the regions. The expected answer is therefore the same in
both twins and `bedtools coverage -counts` gives the right answer only on the clean one. Every
read, copy or not, is named r0000, r0001, ... in a random order: only the flag marks a duplicate.

  explicit  the README states that duplicates are flagged 0x400 and not removed
  implicit  the README only says that duplicates were marked
  forced    the sentence asks for `bedtools coverage`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import pipelines
from biowarden.bench.tasks.extra.f64_picard_dups import read_names
from biowarden.bench.tasks.tools import ToolEnv

CHROMS = {"chr1": 10000, "chr2": 8000}
READ = 50
TOL = 0.5


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f75-{seed}")
    regions: list[tuple[str, int, int, str]] = []
    n = 0
    for chrom, count in (("chr1", rng.randint(6, 7)), ("chr2", rng.randint(4, 5))):
        cursor = 300
        for _ in range(count):
            n += 1
            start = cursor + rng.randint(300, 600)
            end = start + rng.randint(300, 700)
            regions.append((chrom, start, end, f"region_{n:02d}"))
            cursor = end
    reads: list[tuple[str, int, str]] = []  # (chrom, pos0, name), not duplicates
    for chrom, s, e, _name in regions:
        for _ in range(rng.randint(8, 30)):
            reads.append((chrom, rng.randint(s - 40, e - 10), f"r{len(reads):04d}"))
    for _ in range(40):
        chrom = rng.choice(list(CHROMS))
        reads.append((chrom, rng.randint(50, CHROMS[chrom] - 100), f"r{len(reads):04d}"))
    dups: list[tuple[str, int, str]] = []
    for chrom, s, e, _name in regions:
        if rng.random() < 0.6:
            mine = [r for r in reads if r[0] == chrom and r[1] < e and r[1] + READ > s]
            for source in rng.sample(mine, min(len(mine), rng.randint(3, 10))):
                dups.append((chrom, source[1], f"d{len(dups):04d}"))
    return {"regions": regions, "reads": reads, "dups": dups, "seed": seed}


def count_overlaps(reads: list[tuple[str, int, str]], region: tuple[str, int, int, str]) -> int:
    chrom, s, e, _ = region
    return sum(1 for c, p, _ in reads if c == chrom and p < e and p + READ > s)


def sam_text(plan: dict[str, Any], twin: str) -> str:
    rows = [(c, p, n, 0) for c, p, n in plan["reads"]]
    if twin == "faulted":
        rows += [(c, p, n, 1024) for c, p, n in plan["dups"]]
    rows.sort(key=lambda r: (list(CHROMS).index(r[0]), r[1], r[2]))
    # one name pattern for every read (the plan's r/d prefixes only order the draw)
    names = read_names(f"f75-names-{plan['seed']}", len(rows))
    header = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in CHROMS.items()
    )
    body = "".join(
        f"{name}\t{flag}\t{c}\t{p + 1}\t60\t{READ}M\t*\t0\t0\t{'A' * READ}\t{'I' * READ}\n"
        for name, (c, p, _n, flag) in zip(names, rows, strict=True)
    )
    return header + body


def coverage_counts(tools: ToolEnv, work: Path) -> dict[str, int]:
    argv = ["bedtools", "coverage", "-a", "regions.bed", "-b", "sample.bam", "-counts"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return {
        line.split("\t")[3]: int(line.split("\t")[4])
        for line in proc.stdout.splitlines()
        if line.strip()
    }


def multicov_counts(tools: ToolEnv, work: Path) -> dict[str, int]:
    argv = ["bedtools", "multicov", "-bams", "sample.bam", "-bed", "regions.bed"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return {
        line.split("\t")[3]: int(line.split("\t")[4])
        for line in proc.stdout.splitlines()
        if line.strip()
    }


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    common.write_bam(work, sam_text(plan, twin), tools, "sample")
    (work / "regions.bed").write_text(
        "".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in plan["regions"])
    )
    lines = [
        "sample.bam (with sample.bam.bai): aligned reads; PCR duplicates were marked, not removed.",
        "regions.bed: target regions, BED4 (chrom, start, end, name), sorted.",
    ]
    if tier == "explicit":
        lines.append("A read marked as a PCR duplicate has SAM flag 0x400 set.")
    (work / "README.md").write_text(common.readme(lines))
    expected = {r[3]: count_overlaps(plan["reads"], r) for r in plan["regions"]}
    naive = expected if twin == "clean" else coverage_counts(tools, work)
    evidence: tuple[str, ...] = (
        ()
        if twin == "clean"
        else (
            f"{len(plan['dups'])} reads in sample.bam carry flag 0x400: bedtools coverage counts "
            "them, multicov does not",
        )
    )
    what = (
        "the number of reads of sample.bam that are not marked as PCR duplicates and overlap it "
        "by at least one base"
    )
    prompt = common.prompt_for(
        route,
        f"report, for each region in regions.bed, {what}",
        f"use `bedtools coverage` to report, for each region in regions.bed, {what}",
        "a JSON object mapping each region name to its read count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": TOL}
    task_id = common.task_id("F75", seed, twin, tier, route)
    return Task(task_id, "F75", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(multicov_counts(tools, work), "bedtools multicov (skips flag 0x400 by default)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(coverage_counts(tools, work), "bedtools coverage -counts")


COUNTER = re.compile(r"\bbedtools\s+(?:coverage|intersect|genomecov|window)\b")
FILTERED = re.compile(r"-[A-Za-z]*F\s*(?:1024|0x400|0x4\d\d|\d{4})|--rf|markdup\s+-r|\bmulticov\b")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a pipeline counts the reads of sample.bam with coverage, intersect, genomecov or
    window and has no duplicate filter (-F 1024, -F 0x400, also inside a bundle such as -bF); a
    filter in another statement of the command does not cover the raw BAM."""
    for command in commands:
        for pipeline in pipelines(command):
            if COUNTER.search(pipeline) and "sample.bam" in pipeline:
                if not FILTERED.search(pipeline):
                    return True
    return False


SPEC = FamilySpec(
    id="F75",
    slot="default_param",
    source="bedtools multicov skips duplicate and QC-fail reads by default, coverage does not",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"sample.bam", "sample.bam.bai"}),
    locus="tool_default",
)
