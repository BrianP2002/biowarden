"""Phase-0 task families F1-F4: paired clean / faulted workspaces over the real CLI tools.

Each family builds, for one seed, a workspace directory plus a Task. The clean and faulted twins of
a seed share the same biology and the same task sentence; the README differs only in the fact that
makes the trap fair, and the trap is documented somewhere the agent can see before it first uses
the data. The truth lives in the Task, never in the workspace.

Three knobs control how much the task helps the agent, because a benchmark that hands over the
answer measures nothing:
  tier   explicit: the README states the convention outright; implicit: only a term of art (or
         nothing, when the data itself carries the evidence)
  route  free: any tool; forced: the task sentence dictates the tool whose default is the trap, so
         the trap is executed in every trial instead of being routed around
  F3 uses an instance with no variant near a region edge, so only contig naming can change its count
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import ROUTES, TIERS, TWINS, Task
from biowarden.bench.tasks.instance import Instance, make_instance
from biowarden.bench.tasks.tools import ToolEnv

VCF_LINE = "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2, contigs chr1 and chr2"


def build_f1(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """index_base: BED intervals in a file whose extension flips bcftools' reading of them."""
    inst = make_instance(seed)
    common.write_vcf(work, inst, tools)
    name = "regions.bed" if twin == "clean" else "regions.tsv"
    (work / name).write_text(common.bed_text(inst))
    stated = (
        " BED intervals: start is 0-based, end is exclusive."
        if tier == "explicit"
        else " BED-style."
    )
    readme = [VCF_LINE + ".", f"{name}: target regions, tab-separated: chrom, start, end.{stated}"]
    (work / "README.md").write_text(common.readme(readme))
    evidence = (
        ("regions.bed has a .bed extension",)
        if twin == "clean"
        else ("regions.tsv is not named .bed; the README says it holds BED intervals",)
    )
    prompt = common.prompt_for(
        route,
        "list the IDs of the variants in variants.vcf.gz that fall inside the target regions",
        "use `bcftools view -R` with the target regions file to list the IDs of the variants in "
        "variants.vcf.gz that fall inside the target regions",
        "a JSON list of variant IDs",
    )
    expected = inst.in_regions()
    # the .tsv read as 1-based inclusive [start0, end0]: only POS == start0 is wrongly added
    naive = [
        v.id
        for v in inst.variants
        if any(r.chrom == v.chrom and r.start0 <= v.pos <= r.end0 for r in inst.regions)
    ]
    truth = {
        "kind": "id_list",
        "expected": expected,
        "naive": expected if twin == "clean" else naive,
    }
    task_id = common.task_id("F1", seed, twin, tier, route)
    return Task(task_id, "F1", "index_base", twin, seed, prompt, truth, evidence, tier, route)


def build_f2(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """index_base: region strings whose start is 1-based (clean) or 0-based (faulted)."""
    inst = make_instance(seed)
    common.write_fasta(work, inst.ref, tools)
    shift = 1 if twin == "clean" else 0
    lines = [f"{r.chrom}:{r.start0 + shift}-{r.end0}" for r in inst.regions]
    (work / "regions.txt").write_text("\n".join(lines) + "\n")
    base = "regions.txt: target regions, one per line as chrom:start-end."
    if twin == "clean":
        syntax = "samtools region syntax" + (
            ": 1-based, both ends included" if tier == "explicit" else ""
        )
        regions = f"{base} Coordinates are in {syntax}."
        evidence = ("README states samtools region syntax",)
    elif tier == "explicit":
        regions = (
            f"{base} Coordinates follow our annotation database: "
            "start is 0-based, end is exclusive."
        )
        evidence = ("README states 0-based start, end exclusive",)
    else:
        regions = f"{base} Coordinates follow our annotation database (BED-style)."
        evidence = ("README says the coordinates are BED-style",)
    readme = ["ref.fa (with ref.fa.fai): reference sequence, contigs chr1 and chr2", regions]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "report the reference sequence for each target region listed in regions.txt, in the "
        "order of the file",
        "use `samtools faidx` with the region strings in regions.txt to extract the reference "
        "sequence of each target region, in the order of the file",
        "a JSON list of uppercase sequence strings, one per region",
    )
    expected = [inst.ref[r.chrom][r.start0 : r.end0] for r in inst.regions]
    # faulted strings pasted as 1-based inclusive: one base early, one base too many
    naive = [inst.ref[r.chrom][r.start0 - 1 : r.end0] for r in inst.regions]
    truth = {
        "kind": "seq_list",
        "expected": expected,
        "naive": expected if twin == "clean" else naive,
    }
    task_id = common.task_id("F2", seed, twin, tier, route)
    return Task(task_id, "F2", "index_base", twin, seed, prompt, truth, evidence, tier, route)


def build_f3(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """contig_naming: chr1/chr2 in the VCF against 1/2 in the regions (faulted)."""
    inst = make_instance(seed, profile="no_edge")
    common.write_vcf(work, inst, tools)
    stated = tier == "explicit"
    if twin == "clean":
        (work / "regions.bed").write_text(common.bed_text(inst))
        style = ", contigs named chr1 and chr2 (UCSC style)" if stated else ""
        evidence = ("README: both files use UCSC-style names",) if stated else ()
    else:
        (work / "regions.bed").write_text(common.bed_text(inst, lambda c: c.removeprefix("chr")))
        style = ", contigs named 1 and 2 (Ensembl style)" if stated else ""
        evidence = (
            ("README: regions.bed uses Ensembl-style names, the VCF UCSC-style names",)
            if stated
            else ("regions.bed names its contigs 1 and 2, the VCF header chr1 and chr2",)
        ) + ("bedtools prints an inconsistent-naming warning on stderr",)
    vcf_line = VCF_LINE + (" (UCSC-style names)" if stated else "") + "."
    readme = [vcf_line, f"regions.bed: target regions in BED format{style}."]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "count the variants in variants.vcf.gz that fall inside the target regions in regions.bed",
        "use `bedtools intersect` to count the variants in variants.vcf.gz that fall inside the "
        "target regions in regions.bed",
        "a JSON integer",
    )
    expected = len(inst.in_regions())
    truth = {"kind": "int", "expected": expected, "naive": expected if twin == "clean" else 0}
    task_id = common.task_id("F3", seed, twin, tier, route)
    return Task(task_id, "F3", "contig_naming", twin, seed, prompt, truth, evidence, tier, route)


def f4_detect_tokens(inst: Instance) -> list[str]:
    """Instance facts a real detection cites: the two chr1 lengths or a REF-mismatched variant."""
    mismatched = [
        v.id for v in inst.variants if v.chrom == "chr1" and inst.ref_b["chr1"][v.pos - 1] != v.ref
    ]
    return [str(inst.contigs["chr1"]), str(len(inst.ref_b["chr1"])), *mismatched]


def build_f4(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """genome_build (must-detect): ref.fa is another build than the one the VCF was called on."""
    inst = make_instance(seed)
    common.write_vcf(work, inst, tools)
    genome = inst.ref if twin == "clean" else inst.ref_b
    common.write_fasta(work, genome, tools)
    readme = [
        "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2",
        "ref.fa (with ref.fa.fai): the reference FASTA kept for this project",
    ]
    (work / "README.md").write_text(common.readme(readme))
    evidence = (
        ("VCF header ##contig lengths agree with ref.fa.fai",)
        if twin == "clean"
        else (
            "VCF header ##contig length for chr1 differs from the length in ref.fa.fai",
            "the REF allele disagrees with ref.fa at many chr1 variants",
        )
    )
    prompt = (
        "Task: for each variant in variants.vcf.gz, report the 5 bases on each side of the "
        "variant position, taken from ref.fa, written as LLLLL[X]RRRRR where X is the base at "
        "the variant position. "
        + common.answer_sentence("a JSON object mapping each variant ID to its string")
    )
    expected = inst.contexts(inst.ref)
    truth = {
        "kind": "contexts",
        "expected": expected,
        "naive": expected if twin == "clean" else inst.contexts(inst.ref_b),
        "must_detect": twin == "faulted",
        "detect_tokens": f4_detect_tokens(inst),
    }
    task_id = common.task_id("F4", seed, twin, "explicit", "free")
    return Task(task_id, "F4", "genome_build", twin, seed, prompt, truth, evidence)


FAMILIES: dict[str, Callable[..., Task]] = {
    "F1": build_f1,
    "F2": build_f2,
    "F3": build_f3,
    "F4": build_f4,
}


def _extras() -> dict[str, Callable[..., Task]]:
    from biowarden.bench.tasks.extra import load

    return {family: spec.build for family, spec in load().items()}


def all_families() -> dict[str, Callable[..., Task]]:
    """The four original families plus every registered extra one."""
    return {**FAMILIES, **_extras()}


def build_task(
    family: str,
    seed: int,
    twin: str,
    work: Path,
    tools: ToolEnv,
    tier: str = "explicit",
    route: str = "free",
) -> Task:
    checks = ((twin, TWINS, "twin"), (tier, TIERS, "tier"), (route, ROUTES, "route"))
    for value, allowed, what in checks:
        if value not in allowed:
            raise ValueError(f"{what} must be one of {allowed}")
    work.mkdir(parents=True, exist_ok=True)
    return all_families()[family](seed, twin, work, tools, tier, route)
