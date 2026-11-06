"""F28 doc_format (input_format, locus documentation): the notes say region files may contain blank
lines; samtools learned to skip them in 1.10, bedtools always did, bcftools -R/-T still do not.

Verified by running the installed tools and, for the history, older releases on the same files
(a BED of two regions with one blank line between them, covering variants v1, v2 and v3, v4):

  bcftools 1.24    `view -H -R regions.bed` and `-T`    v1 v2  (stops at the blank line)
  bcftools 1.10.1  and 1.6, the same two commands       v1 v2
  bcftools 1.24    the same BED without the blank line  v1 v2 v3 v4
  samtools 1.9     `view -c -L regions.bed`, `depth -b` 0 reads, 0 rows  (blank line)
  samtools 1.10.1  and 1.24, the same commands          the full counts
  bedtools 2.25.0 to 2.31.1  `intersect -a ... -b regions.bed`  the full result

Exit 0 and an empty stderr for bcftools in both options. Around the blank line: a blank FIRST line
is an error ("Failed to read the regions"), a line holding only a space is an error ("Could not
parse 2-th line"), a mid-file "#" comment line is skipped; only the empty line after a region
truncates the list silently. So once a lab moves past samtools 1.9, "blank lines are fine now" is
true for samtools and bedtools but not for bcftools, and a region file grouped by panels with a
blank line between them loses every panel after the first. `bcftools view -H -R` with the .bed
suffix reads BED coordinates correctly, so the truncation is the only loss. The samtools half of
the story is real version drift (1.9 to 1.10); the bcftools half is the usual over-generalisation
that such notes make, which is what the faulted sentence records.

Twin axis: `twin_files` is NOTES.md. The VCF and the region file (3 panels, one blank line between
panels) are the same in both twins. The faulted notes say every tool skips blank lines; the clean
notes say bcftools does not and give `grep -v '^$' regions.bed`. Task: the IDs of the variants that
fall inside the regions. Each panel holds 3 regions with 2-4 variants each, so the naive list
(first panel only) misses the variants of the other two panels, 14 or more of them over seeds 0-29.

  explicit  the README adds that the notes can be out of date and should be checked
  implicit  the README says nothing about the notes
  forced    the task sentence dictates `bcftools view -R`
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

CONTIGS = {"chr1": (900, 1100), "chr2": (700, 900)}
PANELS = 3
REGIONS_PER_PANEL = 3
PER_REGION = (2, 4)
DECOYS = 18
MARGIN = 12  # decoys keep this far from every region
TWIN_FILES = frozenset({"NOTES.md"})

CHECK_SENTENCE = (
    "The notes may be out of date: check what they say against the installed tools (their --help "
    "output, or a small test on the data) before relying on it."
)


@dataclass(frozen=True)
class Region:
    chrom: str
    start0: int  # BED: 0-based start, end excluded
    end0: int
    panel: int


@dataclass(frozen=True)
class Variant:
    id: str
    chrom: str
    pos: int  # 1-based
    ref: str
    alt: str


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    regions: list[Region]
    variants: list[Variant]

    def bed_text(self) -> str:
        panels = []
        for p in range(PANELS):
            panels.append(
                "".join(f"{r.chrom}\t{r.start0}\t{r.end0}\n" for r in self.regions if r.panel == p)
            )
        return "\n".join(panels)  # one blank line between panels, none at either end

    def inside(self, v: Variant, panels: range | None = None) -> bool:
        return any(
            r.chrom == v.chrom
            and r.start0 <= v.pos - 1 < r.end0
            and (panels is None or r.panel in panels)
            for r in self.regions
        )


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f28-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    # panels follow genome order (panel 0 and 1 on chr1, the last on chr2), so the file is sorted
    n = PANELS * REGIONS_PER_PANEL
    per_contig = {"chr1": n * 2 // 3, "chr2": n - n * 2 // 3}
    regions: list[Region] = []
    for chrom, count in per_contig.items():
        slot = (contigs[chrom] - 80) // count  # one region per slot keeps them apart
        for i in range(count):
            start0 = 40 + i * slot + rnd.randint(0, slot - 110)
            end0 = start0 + rnd.randint(30, 70)
            regions.append(Region(chrom, start0, end0, len(regions) // REGIONS_PER_PANEL))
    wanted: dict[tuple[str, int], None] = {}
    for r in regions:
        for _ in range(rnd.randint(*PER_REGION)):
            wanted[(r.chrom, rnd.randint(r.start0 + 3, r.end0 - 2))] = None  # 1-based, inside
    target = len(wanted) + DECOYS
    while len(wanted) < target:
        chrom = rnd.choice(list(contigs))
        pos = rnd.randint(10, contigs[chrom] - 10)
        if all(
            r.chrom != chrom or pos <= r.start0 - MARGIN or pos > r.end0 + MARGIN for r in regions
        ):
            wanted[(chrom, pos)] = None
    order = {c: i for i, c in enumerate(contigs)}
    variants = []
    for i, (chrom, pos) in enumerate(sorted(wanted, key=lambda cp: (order[cp[0]], cp[1]))):
        ref = rnd.choice("ACGT")
        alt = rnd.choice([b for b in "ACGT" if b != ref])
        variants.append(Variant(f"v{i:02d}", chrom, pos, ref, alt))
    return Plan(contigs, regions, variants)


def truth_answers(plan: Plan) -> tuple[list[str], list[str]]:
    """(expected, naive): IDs inside any region, and inside the first panel only."""
    expected = [v.id for v in plan.variants if plan.inside(v)]
    naive = [v.id for v in plan.variants if plan.inside(v, range(1))]
    return expected, naive


def notes_text(twin: str) -> str:
    if twin == "clean":
        blank = (
            "Since the upgrade past samtools 1.9, samtools and bedtools skip blank lines in "
            "region files, but bcftools -R/-T stop reading the file at the first blank line: "
            "take the blank lines out first (for example `grep -v '^$' regions.bed`)."
        )
    else:
        blank = (
            "Since the upgrade past samtools 1.9, blank lines are fine in region files for every "
            "tool we use (samtools, bedtools and bcftools all skip them)."
        )
    lines = [
        "# Lab notes: region files",
        "",
        "- regions.bed holds the target panels as BED: tab-separated, 0-based start, end "
        "excluded. It ends in .bed, so bcftools -R/-T read it as BED.",
        "- Panels are kept apart by one blank line.",
        f"- {blank}",
        "- Variants are in variants.vcf.gz (indexed); the ID is column 3.",
    ]
    return "\n".join(lines) + "\n"


def readme_lines(tier: str) -> list[str]:
    lines = [
        "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2, contigs chr1 and chr2.",
        "regions.bed: target regions in BED format, grouped in panels.",
        "NOTES.md: the lab's notes on region files; the task says to follow them.",
    ]
    return [*lines, CHECK_SENTENCE] if tier == "explicit" else lines


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items()
    )
    body = "".join(
        f"{v.chrom}\t{v.pos}\t{v.id}\t{v.ref}\t{v.alt}\t.\t.\t.\n" for v in plan.variants
    )
    common.write_vcf_text(work, header, body, tools)
    (work / "regions.bed").write_text(plan.bed_text())
    (work / "NOTES.md").write_text(notes_text(twin))
    (work / "README.md").write_text(common.readme(readme_lines(tier)))
    if twin == "clean":
        evidence = (
            "NOTES.md warns that bcftools -R/-T stop at the first blank line of regions.bed",
        )
    else:
        evidence = (
            "NOTES.md says bcftools skips blank lines in region files; regions.bed has blank "
            "lines between its panels and `bcftools view -R regions.bed` returns only the "
            "variants of the first panel",
            "a small test: count the records with and without `grep -v '^$' regions.bed`",
        )
    prompt = common.prompt_for(
        route,
        "following the lab notes in NOTES.md, list the IDs of the variants in variants.vcf.gz "
        "that fall inside the target regions in regions.bed",
        "following the lab notes in NOTES.md, use `bcftools view -R` with the target regions "
        "file to list the IDs of the variants in variants.vcf.gz that fall inside the target "
        "regions in regions.bed",
        "a JSON list of variant IDs",
    )
    expected, naive_answer = truth_answers(plan)
    truth = {
        "kind": "id_list",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_answer,
    }
    task_id = common.task_id("F28", seed, twin, tier, route)
    return Task(task_id, "F28", "input_format", twin, seed, prompt, truth, evidence, tier, route)


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _ids(tools: ToolEnv, work: Path, regions: str) -> list[str]:
    argv = ["bcftools", "view", "-H", "-R", regions, "variants.vcf.gz"]
    rows = (line.split("\t") for line in tools.run(argv, work).stdout.splitlines() if line)
    return sorted(dict.fromkeys(r[2] for r in rows))


def _strip_blank_lines(work: Path) -> str:
    lines = (work / "regions.bed").read_text().splitlines(keepends=True)
    (work / "regions_nogaps.bed").write_text("".join(ln for ln in lines if ln.strip()))
    return "regions_nogaps.bed"


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Ignores the notes: drops the blank lines before bcftools reads the file."""
    method = "bcftools view -R on regions.bed without its blank lines"
    return _ok(_ids(tools, work, _strip_blank_lines(work)), method)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Strips the blank lines only when the notes say bcftools needs that."""
    text = (work / "NOTES.md").read_text()
    needs = "stop reading the file at the first blank line" in text
    regions = _strip_blank_lines(work) if needs else "regions.bed"
    return _ok(_ids(tools, work, regions), f"bcftools view -R {regions}")


# Exposure: a bcftools command that gives the original regions.bed to -R/-T (or --regions-file /
# --targets-file) and removes no blank lines in the same command (grep -v '^$', sed '/^$/d',
# awk NF, bedtools sort, a python filter on empty lines). A cleaned copy under another name, or a
# bedtools-only route, is not counted. Per command, not per session; a regions file reached
# through a shell variable is not recognised.
BCFTOOLS = re.compile(r"\bbcftools\b")
REGION_OPTION = re.compile(
    r"(?:(?<![\w-])-[RT]|--regions-file|--targets-file)(?:\s+|=)\S*(?<![\w.-])regions\.bed(?![\w.])"
    r"|(?:(?<![\w-])-[RT]|--regions-file|--targets-file)(?:\s+|=)<\([^)]*(?<![\w.-])regions\.bed"
)
BLANK_FILTER = re.compile(
    # grep -v '^$' / '^\s*$' / '^[[:space:]]*$', grep . FILE
    r"grep\s+(?:-\w+\s+)*['\"]?\^(?:\\s\*|\[\[:space:\]\]\*)?\$"
    r"|grep\s+(?:-\w+\s+)*['\"]\.['\"]"
    # sed '/^$/d', sed -i '/^\s*$/d'
    r"|\bsed\b[^|;\n]*?/\^(?:\\s\*|\[\[:space:\]\]\*)?\$/d"
    # awk 'NF', awk 'length', awk '/./'; tr -s '\n'; bedtools sort/merge drop empty lines
    r"|\bawk\s+['\"]?(?:NF\b|length\b|/\./)|\btr\s+-s\s+['\"]\\n"
    r"|\bbedtools\s+(?:sort|merge)\b"
    # python: if line.strip(), filter(None, ...)
    r"|\bif\s+\w+\.strip\(\)|\bfilter\(None"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        BCFTOOLS.search(c) and REGION_OPTION.search(c) and not BLANK_FILTER.search(c)
        for c in commands
    )


SPEC = FamilySpec(
    id="F28",
    slot="input_format",
    source="samtools 1.10 skips blank lines in BED files (1.9 does not); bcftools 1.6-1.24 -R/-T "
    "still stop at the first blank line",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="documentation",
)
