"""F10 indel_bed: the REF allele of a VCF deletion starts with an anchor base that the deletion does
not remove, so the REF span (`bcftools query -f '%CHROM\\t%POS0\\t%END\\n'`, or awk with
length(REF)) is one base too early for the bases the record deletes.

Verified on bcftools 1.24 (exit 0, empty stderr): `%POS0` is POS-1 and `%END` is the last base of
the REF allele, so the pair is the exact BED interval of the REF allele, anchor included.

Design note: the task asks for the interval of the reference bases a record replaces or deletes
(not of its REF allele) because the generic gates need the same answer in both twins. Each trap
site is a deletion of k bases in the faulted twin (POS is the anchor, REF has k+1 bases, ALT is the
anchor alone) and a substitution of the same k bases in the clean twin (an SNV for k = 1, an MNV
otherwise, every ALT base different from REF). A REF-span interval is right for substitutions and
starts one base early for deletions, so the answer is identical in both twins while the naive
answer is wrong on every deletion (6-8 per seed). Both twins also carry plain SNVs and MNVs, so a
REF longer than one base does not by itself mark a fault.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": (700, 900), "chr2": (500, 700)}
SITES_PER_CONTIG = 12
MIN_GAP = 8  # bases between the footprints of neighbouring sites
TWIN_FILES = frozenset({"variants.vcf.gz", "variants.vcf.gz.csi", "README.md"})


@dataclass(frozen=True)
class Site:
    chrom: str
    start0: int  # BED start of the bases the record replaces or deletes
    k: int  # how many bases
    deletion: bool  # a deletion in the faulted twin; clean twin: substitution of the same bases


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    genome: dict[str, str]
    sites: list[Site]

    def interval(self, site: Site) -> str:
        return f"{site.chrom}\t{site.start0}\t{site.start0 + site.k}"

    def record(self, site: Site, twin: str) -> tuple[str, int, str, str]:
        """(chrom, POS, REF, ALT) of the site in the given twin."""
        seq, s, k = self.genome[site.chrom], site.start0, site.k
        if site.deletion and twin == "faulted":
            return site.chrom, s, seq[s - 1 : s + k], seq[s - 1]
        alt = "".join(_other(b, i) for i, b in enumerate(seq[s : s + k]))
        return site.chrom, s + 1, seq[s : s + k], alt


def _other(base: str, salt: int) -> str:
    """A base different from `base`, fixed by the base and its offset (no random state)."""
    others = [b for b in "ACGT" if b != base]
    return others[(salt + "ACGT".index(base)) % 3]


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f10-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    genome = {c: [rnd.choice("ACGT") for _ in range(n)] for c, n in contigs.items()}
    n_sites = SITES_PER_CONTIG * len(contigs)
    kinds = ["del"] * rnd.randint(6, 8) + ["mnv"] * 3
    kinds += ["snv"] * (n_sites - len(kinds))
    rnd.shuffle(kinds)
    sites: list[Site] = []
    for chrom, n in contigs.items():
        start0 = 0
        for _ in range(SITES_PER_CONTIG):
            kind = kinds.pop()
            k = {"snv": 1, "mnv": rnd.randint(2, 3), "del": rnd.choice([1, 2, 2, 3, 4])}[kind]
            start0 += rnd.randint(MIN_GAP + 2, (n - 40) // SITES_PER_CONTIG - 4)
            if kind == "del":
                # left-aligned: the last deleted base differs from the anchor before it
                if genome[chrom][start0 + k - 1] == genome[chrom][start0 - 1]:
                    genome[chrom][start0 + k - 1] = _other(genome[chrom][start0 - 1], 0)
            sites.append(Site(chrom, start0, k, kind == "del"))
            start0 += k
    return Plan(contigs, {c: "".join(s) for c, s in genome.items()}, sites)


def truth_answers(plan: Plan, twin: str) -> tuple[list[str], list[str]]:
    """(expected, naive): the bases each record replaces or deletes, and its REF span."""
    expected = [plan.interval(s) for s in plan.sites]
    naive = []
    for s in plan.sites:
        chrom, pos, ref, _ = plan.record(s, twin)
        naive.append(f"{chrom}\t{pos - 1}\t{pos - 1 + len(ref)}")
    return expected, naive


def _vcf_text(plan: Plan, twin: str) -> tuple[str, str]:
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items()
    )
    body = ""
    for i, site in enumerate(plan.sites):
        chrom, pos, ref, alt = plan.record(site, twin)
        body += f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t.\t.\t.\n"
    return header, body


def build_f10(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """interval_closure: BED interval of the reference bases a record replaces or deletes."""
    plan = make_plan(seed)
    common.write_vcf_text(work, *_vcf_text(plan, twin), tools)
    head = "variants.vcf.gz (with variants.vcf.gz.csi): "
    tail = ", VCF 4.2, contigs chr1 and chr2."
    if tier == "explicit":
        kinds = "SNVs, MNVs and deletions" if twin == "faulted" else "SNVs and MNVs"
        lines = [head + kinds + tail]
        if twin == "faulted":
            lines.append(
                "A deletion is written the VCF way: REF starts with the anchor base just before "
                "the deleted bases, and ALT is that anchor alone."
            )
            evidence = ("README: a deletion's REF starts with an anchor base that ALT keeps",)
        else:
            lines.append("An MNV replaces every base of REF by the base at the same offset in ALT.")
            evidence = ("README: an MNV replaces every base of REF",)
    else:
        lines = [head + "small variants" + tail]
        evidence = (
            ("deletion records have REF = anchor base + deleted bases and ALT = the anchor base",)
            if twin == "faulted"
            else ("no record has an ALT shorter than its REF",)
        )
    (work / "README.md").write_text(common.readme(lines))
    prompt = common.prompt_for(
        route,
        "for each variant record in variants.vcf.gz, in file order, write the BED interval "
        "(0-based start, end exclusive) of the reference bases that the record replaces or "
        "deletes",
        "use `bcftools query -f` and awk to write, for each variant record in variants.vcf.gz, "
        "in file order, the BED interval (0-based start, end exclusive) of the reference bases "
        "that the record replaces or deletes",
        "a JSON list of strings, one per record, each holding the chromosome, start and end "
        "separated by single tab characters",
    )
    expected, naive = truth_answers(plan, twin)
    truth = {"kind": "str_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F10", seed, twin, tier, route)
    return Task(
        task_id, "F10", "interval_closure", twin, seed, prompt, truth, evidence, tier, route
    )


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _records(work: Path, tools: ToolEnv) -> list[list[str]]:
    proc = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work)
    return [line.split("\t") for line in proc.stdout.splitlines() if line]


def reference_f10(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    out = []
    for chrom, pos, _, ref, alt, *_ in _records(work, tools):
        start = int(pos) - 1
        if len(ref) > 1 and len(alt) == 1 and ref[0] == alt:  # anchored deletion: skip the anchor
            start += 1
        out.append(f"{chrom}\t{start}\t{int(pos) - 1 + len(ref)}")
    return _ok(out, "REF span, anchor base excluded for deletions")


def naive_f10(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["bcftools", "query", "-f", "%CHROM\t%POS0\t%END\n", "variants.vcf.gz"], work)
    return _ok(proc.stdout.splitlines(), "bcftools query %POS0 %END (the REF span)")


# Exposure: a command turned VCF records into intervals from POS and REF length (%POS0/%END,
# length(REF), POS-1) without handling the alleles, so the anchor base cannot have been handled.
# Handling means the ALT allele takes part in the computation: a field index into ALT, substr(), or
# two different length() arguments (REF against ALT). Printing %ALT in a -f format string and then
# measuring REF alone is the trap itself and does not count as handling.
INTERVAL_FORMULA = re.compile(r"%POS0|%END|\blength\(|\blen\(|\$2\s*-\s*1|\b[Pp][Oo][Ss]\s*-\s*1")
FORMAT_STRING = re.compile(r"-f\s*(?:'[^']*'|\"[^\"]*\")")
ALT_USED = re.compile(r"\$5\b|\bALT\b|\balt\b|\[4\]|\bsubstr\(")
LENGTH_ARG = re.compile(r"\blen(?:gth)?\(\s*([^)]*?)\s*\)")


def _handles_alt(command: str) -> bool:
    bare = FORMAT_STRING.sub("-f FMT", command)
    return bool(ALT_USED.search(bare) or len(set(LENGTH_ARG.findall(bare))) >= 2)


def exposed_f10(commands: list[str], work: Path) -> bool:
    return any(
        "variants.vcf" in c and INTERVAL_FORMULA.search(c) and not _handles_alt(c) for c in commands
    )


SPEC = FamilySpec(
    id="F10",
    slot="interval_closure",
    source="VCF POS is 1-based and a deletion's REF starts with an anchor base",
    build=build_f10,
    reference=reference_f10,
    naive=naive_f10,
    exposed=exposed_f10,
    twin_files=TWIN_FILES,
)
