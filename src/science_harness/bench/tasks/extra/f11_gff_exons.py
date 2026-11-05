"""F11 gff_exons (index_base): the exon intervals come as BED (clean) or GFF3 (faulted).

GFF3 is 1-based with both ends included; BED is 0-based with the end excluded. Same exons in both
twins: the GFF3 lines carry start0 + 1 and end0. bedtools reads GFF3 natively and gets the count
right; the obvious hand conversion `awk '{print $1,$4,$5}'` keeps the 1-based start as if it were
0-based and silently drops every variant on the first base of an exon (exit 0, no stderr).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.instance import Instance, make_instance
from science_harness.bench.tasks.tools import ToolEnv

VCF_LINE = "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2, contigs chr1 and chr2"
AWK_CONVERT = '$3=="exon"{print $1,$4,$5}'  # the obvious conversion: forgets start - 1


def gff_text(inst: Instance) -> str:
    """The BED regions as GFF3 exon features: 1-based start, end included."""
    rows = (
        f"{r.chrom}\tcurated\texon\t{r.start0 + 1}\t{r.end0}\t.\t+\t.\tID=exon{i:02d}\n"
        for i, r in enumerate(inst.regions)
    )
    return "##gff-version 3\n" + "".join(rows)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    inst = make_instance(seed)
    common.write_vcf(work, inst, tools)
    if twin == "clean":
        (work / "exons.bed").write_text(common.bed_text(inst))
        stated = " BED intervals: start is 0-based, end is exclusive."
        line = "exons.bed: exon annotation in BED format."
        evidence = (
            ("exons.bed is BED; README states 0-based start, end exclusive",)
            if tier == "explicit"
            else ("exons.bed has a .bed extension",)
        )
    else:
        (work / "exons.gff3").write_text(gff_text(inst))
        stated = " GFF3 coordinates: start and end are 1-based, both included."
        line = "exons.gff3: exon annotation in GFF3 format."
        evidence = (
            ("README states GFF3 start and end are 1-based, both included",)
            if tier == "explicit"
            else ("exons.gff3 is GFF3, a 1-based closed format; the README names the format",)
        )
    readme = [VCF_LINE + ".", line + (stated if tier == "explicit" else "")]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "count the variants in variants.vcf.gz that fall inside the exons",
        "use `bedtools intersect` on the exon annotation (converted to BED first if it is not "
        "BED already) to count the variants in variants.vcf.gz that fall inside the exons",
        "a JSON integer",
    )
    expected = len(inst.in_regions())
    # the 1-based GFF start read as a 0-based BED start: the first base of each exon is lost
    shifted = sum(
        any(r.chrom == v.chrom and r.start0 + 1 <= v.pos - 1 < r.end0 for r in inst.regions)
        for v in inst.variants
    )
    truth = {"kind": "int", "expected": expected, "naive": expected if twin == "clean" else shifted}
    task_id = common.task_id("F11", seed, twin, tier, route)
    return Task(task_id, "F11", "index_base", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _count(stdout: str) -> int:
    return sum(1 for line in stdout.splitlines() if line.strip())


def _intersect(work: Path, tools: ToolEnv, exons: str, stdin: str | None = None) -> int:
    argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", exons, "-u"]
    return _count(tools.run(argv, work, stdin=stdin).stdout)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    exons = "exons.bed" if (work / "exons.bed").exists() else "exons.gff3"
    return _result(_intersect(work, tools, exons), "bedtools intersect (reads GFF3 natively)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    if (work / "exons.bed").exists():
        return _result(_intersect(work, tools, "exons.bed"), "bedtools intersect on exons.bed")
    converted = tools.run(["awk", "-v", "OFS=\t", AWK_CONVERT, "exons.gff3"], work).stdout
    count = _intersect(work, tools, "stdin", converted)
    return _result(count, "awk converted the GFF3 to BED, then bedtools intersect")


# Exposure: a command that turns exons.gff3 into intervals with awk or cut and writes or pipes them
# on, without subtracting 1 from the start. Reading the GFF3 with bedtools directly is not exposed.
GFF_FILE = re.compile(r"exons\.gff3?\b")
CONVERT = re.compile(r"\bawk\b[^|;&]*\$4(?!\d)|\bcut\b[^|;&]*-f\s*['\"]?[0-9,-]*\b4\b")
SUBTRACT = re.compile(r"[\w)\]](?:-1|\s-\s1)(?!\d)")
SINK = re.compile(r"\bbedtools\b|>")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        GFF_FILE.search(c) and CONVERT.search(c) and SINK.search(c) and not SUBTRACT.search(c)
        for c in commands
    )


SPEC = FamilySpec(
    id="F11",
    slot="index_base",
    source="GFF3 is 1-based closed, BED is 0-based half-open (cf. GEN-05)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"exons.bed", "exons.gff3", "README.md"}),
)
