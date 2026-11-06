"""F7 regions_vs_targets: `bcftools view -T` tests only POS, so a gVCF reference block that starts
before a target region and runs into it is silently missed.

Verified on bcftools 1.24 (exit 0, empty stderr): `-R regions.bed` keeps every record that overlaps
a region, using INFO/END for reference blocks, and writes a block once even when it spans two
regions; `-T regions.bed` keeps a record only if POS is in a region (--targets-overlap 0, the
default; `bcftools view --help` prints it next to --regions-overlap 1). bedtools intersect is
END-aware too, so only the -T / -t / --regions-overlap 0 route is trapped.

Twin axis: both twins hold the same record IDs and the same answer. The clean twin's reference
blocks start inside the region they overlap (POS-only tests agree with overlap); in the faulted
twin the same blocks start 8-25 bases earlier, outside every region, so -T drops them. Four blocks
are trap blocks (one spans two regions, as in GEN-02); three more are decoys that touch no region
(two sit exactly one base off a region edge) and are identical in both twins.
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

CONTIGS = {"chr1": (1200, 1500), "chr2": (1000, 1200)}
REGIONS_PER_CONTIG = 4
GAP = 100  # bases between regions: room for the blocks and decoys around each one
LEAD = (8, 25)  # how far before its region a faulted block starts
TRAP_BLOCKS = 4
TWIN_FILES = frozenset({"variants.vcf.gz", "variants.vcf.gz.csi", "README.md"})


@dataclass(frozen=True)
class Region:
    chrom: str
    start0: int  # BED: 0-based start, end exclusive
    end0: int

    @property
    def start1(self) -> int:
        return self.start0 + 1


@dataclass(frozen=True)
class Rec:
    id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    end: int | None = None  # INFO/END of a reference block

    @property
    def last(self) -> int:
        return self.end if self.end is not None else self.pos + len(self.ref) - 1


@dataclass(frozen=True)
class Block:
    id: str
    chrom: str
    clean_pos: int
    faulted_pos: int
    end: int
    ref_bases: tuple[str, str]  # REF base at the clean and at the faulted POS


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    regions: list[Region]
    snvs: list[Rec]
    blocks: list[Block]

    def records(self, twin: str) -> list[Rec]:
        which = 0 if twin == "clean" else 1
        recs = [
            Rec(
                b.id,
                b.chrom,
                (b.clean_pos, b.faulted_pos)[which],
                b.ref_bases[which],
                "<NON_REF>",
                b.end,
            )
            for b in self.blocks
        ]
        order = {c: i for i, c in enumerate(self.contigs)}
        return sorted([*self.snvs, *recs], key=lambda r: (order[r.chrom], r.pos))


def overlaps(rec: Rec, regions: list[Region]) -> bool:
    """POS..last base against the 1-based closed form of each BED interval."""
    return any(rec.chrom == r.chrom and rec.pos <= r.end0 and rec.last >= r.start1 for r in regions)


def starts_in(rec: Rec, regions: list[Region]) -> bool:
    return any(rec.chrom == r.chrom and r.start1 <= rec.pos <= r.end0 for r in regions)


def _regions(rnd: random.Random, contigs: dict[str, int]) -> list[Region]:
    out: list[Region] = []
    for chrom, n in contigs.items():
        taken: list[tuple[int, int]] = []
        while len(taken) < REGIONS_PER_CONTIG:
            start = rnd.randint(60, n - 180)
            end = start + rnd.randint(20, 60)
            if all(end + GAP < s or start > e + GAP for s, e in taken):
                taken.append((start, end))
        out += [Region(chrom, s, e) for s, e in sorted(taken)]
    return out


def _clear(spans: list[tuple[str, int, int]], chrom: str, lo: int, hi: int, pad: int) -> bool:
    return all(c != chrom or hi + pad < a or lo > b + pad for c, a, b in spans)


def _try_blocks(
    rnd: random.Random, contigs: dict[str, int], regions: list[Region]
) -> list[Block] | None:
    pairs = [
        (i, i + 1) for i in range(len(regions) - 1) if regions[i].chrom == regions[i + 1].chrom
    ]
    a, b = min(pairs, key=lambda p: regions[p[1]].start0 - regions[p[0]].end0)
    rest = [i for i in range(len(regions)) if i not in (a, b)]
    narrow = rnd.sample(rest, 3)
    spare = [i for i in rest if i not in narrow]

    def bases() -> tuple[str, str]:
        return rnd.choice("ACGT"), rnd.choice("ACGT")

    def same_base() -> tuple[str, str]:  # decoys do not move between the twins
        base = rnd.choice("ACGT")
        return base, base

    ra, rb = regions[a], regions[b]
    # the wide block: starts at the end of region a (clean) or before it (faulted), ends in b
    blocks = [
        Block(
            "g0",
            ra.chrom,
            ra.end0 - rnd.randint(0, 5),
            ra.start1 - rnd.randint(*LEAD),
            rb.start1 + rnd.randint(2, 8),
            bases(),
        )
    ]
    for j, idx in enumerate(narrow):
        r = regions[idx]
        if j % 2 == 0:  # spans the whole region and sticks out of its end
            clean_pos, end = r.start1 + rnd.randint(0, 5), r.end0 + rnd.randint(0, 40)
        else:  # runs into the region and stops inside it
            clean_pos = r.start1 + rnd.randint(0, 2)
            end = r.start1 + rnd.randint(4, r.end0 - r.start0 - 1)
        faulted_pos = r.start1 - rnd.randint(*LEAD)
        blocks.append(Block(f"g{j + 1}", r.chrom, clean_pos, faulted_pos, end, bases()))
    # decoys, the same in both twins: one base left of a region, one base right of one, one far
    left, right = regions[spare[0]], regions[spare[1]]
    end = left.start0  # the last base is the one just before the region's first
    pos = end - rnd.randint(3, 15)
    blocks.append(Block("g4", left.chrom, pos, pos, end, same_base()))
    pos = right.end0 + 1
    blocks.append(Block("g5", right.chrom, pos, pos, pos + rnd.randint(3, 15), same_base()))
    chrom = rnd.choice(list(contigs))
    lo = rnd.randint(20, contigs[chrom] - 80)
    blocks.append(Block("g6", chrom, lo, lo, lo + rnd.randint(5, 20), same_base()))

    # every block span (faulted form: the longest) clear of the regions, except the trap blocks
    spans = [(b.chrom, b.faulted_pos, b.end) for b in blocks]
    for i, blk in enumerate(blocks):
        others = spans[:i] + spans[i + 1 :]
        if not _clear(others, blk.chrom, blk.faulted_pos, blk.end, 3):
            return None
        touches = any(
            r.chrom == blk.chrom and blk.faulted_pos <= r.end0 and blk.end >= r.start1
            for r in regions
        )
        if touches != (i < TRAP_BLOCKS):
            return None
    return blocks


def _snvs(
    rnd: random.Random, contigs: dict[str, int], regions: list[Region], blocks: list[Block]
) -> list[Rec]:
    wanted: dict[tuple[str, int], None] = {}
    for r in regions:
        # 1-based: the base before the region, its first, its last, the base after it
        for pos in rnd.sample([r.start0, r.start1, r.end0, r.end0 + 1], 2):
            wanted[(r.chrom, pos)] = None
        wanted[(r.chrom, rnd.randint(r.start1 + 1, r.end0 - 1))] = None
    for _ in range(8):
        chrom = rnd.choice(list(contigs))
        wanted[(chrom, rnd.randint(12, contigs[chrom] - 12))] = None
    spans = [(b.chrom, b.faulted_pos, b.end) for b in blocks]
    snvs = []
    for chrom, pos in sorted(wanted):
        if _clear(spans, chrom, pos, pos, 0):
            ref = rnd.choice("ACGT")
            snvs.append(Rec("", chrom, pos, ref, rnd.choice([b for b in "ACGT" if b != ref])))
    order = {c: i for i, c in enumerate(contigs)}
    snvs.sort(key=lambda r: (order[r.chrom], r.pos))
    return [Rec(f"v{i:02d}", r.chrom, r.pos, r.ref, r.alt) for i, r in enumerate(snvs)]


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f07-{seed}")
    for _ in range(500):
        contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
        regions = _regions(rnd, contigs)
        blocks = _try_blocks(rnd, contigs, regions)
        if blocks is not None:
            return Plan(contigs, regions, _snvs(rnd, contigs, regions, blocks), blocks)
    raise RuntimeError(f"no valid F7 plan for seed {seed}")


def truth_answers(plan: Plan, twin: str) -> tuple[list[str], list[str]]:
    """(expected, naive) ID lists, sorted by ID so that both twins give the same expected list."""
    recs = plan.records(twin)
    expected = sorted(r.id for r in recs if overlaps(r, plan.regions))
    naive = sorted(r.id for r in recs if starts_in(r, plan.regions))
    return expected, naive


HEADER = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=END,Number=1,Type=Integer,Description="Last base of the reference block">\n'
    '##ALT=<ID=NON_REF,Description="Reference block: any possible alternate allele">\n'
)


def _vcf_text(plan: Plan, twin: str) -> tuple[str, str]:
    header = HEADER + "".join(f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items())
    body = "".join(
        f"{r.chrom}\t{r.pos}\t{r.id}\t{r.ref}\t{r.alt}\t.\t.\t"
        f"{'.' if r.end is None else f'END={r.end}'}\n"
        for r in plan.records(twin)
    )
    return header, body


def build_f7(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """interval_semantics: records spanning a target region without starting in it."""
    plan = make_plan(seed)
    common.write_vcf_text(work, *_vcf_text(plan, twin), tools)
    (work / "regions.bed").write_text(
        "".join(f"{r.chrom}\t{r.start0}\t{r.end0}\n" for r in plan.regions)
    )
    vcf_line = (
        "variants.vcf.gz (with variants.vcf.gz.csi): SNVs and gVCF reference blocks "
        "(ALT <NON_REF>, INFO/END is the last base the block covers), VCF 4.2, "
        "contigs chr1 and chr2."
        if tier == "explicit"
        else "variants.vcf.gz (with variants.vcf.gz.csi): gVCF-style calls, VCF 4.2, "
        "contigs chr1 and chr2."
    )
    bed = "regions.bed: target regions in BED format."
    if tier == "explicit":
        reach = (
            "some reference blocks start before a region and run into it"
            if twin == "faulted"
            else "every reference block starts inside the region it overlaps"
        )
        bed += (
            " A record overlaps a region if any base from POS to its last base (INFO/END for "
            f"a block) lies in the region; {reach}."
        )
    (work / "README.md").write_text(common.readme([vcf_line, bed]))
    if twin == "clean":
        evidence = ("header declares INFO/END, but every block starts inside its region",)
        if tier == "explicit":
            evidence = ("README: every reference block starts inside the region it overlaps",)
    else:
        evidence = (
            "bcftools view --help: --targets-overlap defaults to 0 (POS in the region), "
            "--regions-overlap to 1 (record overlaps)",
            "README: some reference blocks start before a region and run into it"
            if tier == "explicit"
            else "reference blocks (ALT <NON_REF>, INFO/END) start before the regions they cover",
        )
    prompt = common.prompt_for(
        route,
        "list the IDs of the records in variants.vcf.gz that overlap the target regions in "
        "regions.bed",
        "use `bcftools view -T` with the target regions file to list the IDs of the records in "
        "variants.vcf.gz that overlap the target regions",
        "a JSON list of record IDs",
    )
    expected, naive = truth_answers(plan, twin)
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F7", seed, twin, tier, route)
    return Task(
        task_id, "F7", "interval_semantics", twin, seed, prompt, truth, evidence, tier, route
    )


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _ids(tools: ToolEnv, work: Path, option: str) -> list[str]:
    argv = ["bcftools", "view", "-H", option, "regions.bed", "variants.vcf.gz"]
    rows = (line.split("\t") for line in tools.run(argv, work).stdout.splitlines() if line)
    return sorted(dict.fromkeys(r[2] for r in rows))


def reference_f7(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _ok(_ids(tools, work, "-R"), "bcftools view -R (record overlap, END-aware)")


def naive_f7(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _ok(_ids(tools, work, "-T"), "bcftools view -T (POS in region only)")


# Exposure: a bcftools command ran a start-only selection. -T/-t/--targets(-file) default to
# --targets-overlap 0, and -R/-r with --regions-overlap 0|pos does the same. Naming
# --targets-overlap 1|2|record|variant makes the call safe. Pipeline stages are checked one by one;
# hand-written awk start tests and bundled short options such as -HT are not detected.
START_ONLY = re.compile(
    r"(?:^|\s)(?:-T|-t|--targets-file|--targets)(?:\s+|=)\S"
    r"|--regions-overlap(?:\s+|=)(?:0|pos)\b"
)
OVERLAP_MODE = re.compile(r"--targets-overlap(?:\s+|=)(?:1|2|record|variant)\b")
STAGE_SPLIT = re.compile(r"[|;&\n]+")


def exposed_f7(commands: list[str], work: Path) -> bool:
    for command in commands:
        for stage in STAGE_SPLIT.split(command):
            at = stage.find("bcftools")
            if at >= 0 and START_ONLY.search(stage[at:]) and not OVERLAP_MODE.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F7",
    slot="interval_semantics",
    source="GEN-02",
    build=build_f7,
    reference=reference_f7,
    naive=naive_f7,
    exposed=exposed_f7,
    twin_files=TWIN_FILES,
)
