"""F18 wrap_truncate (completeness, locus wrapper): the lab's ID-listing wrapper pipes the list
through `head -n 50`, so a long result is cut and the cut looks like a complete answer.

Source SCD-14 (a listing sliced to a cap, with no truncation flag) and GEN-35 (a result list that
simply stops at a default cap). Verified on bcftools 1.24: `bcftools query -r chr2 -f '%ID\\n'
variants.vcf.gz` prints one ID per record on chr2 (56-70 per seed here); the same command piped
through `head -n 50` prints the first 50, exits 0 and writes nothing to stderr (the output is far
below the pipe buffer, so bcftools is never sent SIGPIPE).

Twin axis: the VCF is the same in both twins (97-125 records; chr2 holds 56-70, chr1 40-60). The
clean wrapper prints what `bcftools query` prints; the faulted one pipes it through
`head -n "${LAB_MAXROWS:-50}"` under a comment about terminal readability. The wrapper is plain
text, so `cat` shows the cap; the output (exactly 50 lines, in file order) does not say a cap was
hit. The README names the cap only in the explicit tier. Plain sh; bcftools by name.

Exposure: the shared rule of F17. The wrapper's escape hatch is `LAB_MAXROWS=<n>`; a value of at
least 100 (more than any list here) counts as an override.
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
from biowarden.bench.tasks.extra.f17_wrap_filter import (
    result,
    run_wrapper,
    wrapper_exposed,
    write_script,
)
from biowarden.bench.tasks.tools import ToolEnv

WRAPPER = "lab-ids.sh"
CONTIGS = {"chr1": (1500, 2200), "chr2": (1300, 1800)}
PER_CONTIG = {"chr1": (40, 60), "chr2": (56, 72)}
QUERY = "chr2"
CAP = 50
TWIN_FILES = frozenset({WRAPPER, "README.md"})

HEAD = """#!/bin/sh
# lab-ids.sh: print the ID of every variant record in a VCF/BCF, optionally for one region.
# usage: ./lab-ids.sh VCF [REGION]
set -eu
[ $# -ge 1 ] || { echo "usage: $0 VCF [REGION]" >&2; exit 2; }
if [ -n "${2:-}" ]; then
  set -- -r "$2" "$1"
fi
"""
WRAPPER_CLEAN = HEAD + "bcftools query -f '%ID\\n' \"$@\"\n"
WRAPPER_FAULTED = (
    HEAD
    + "# keep the listing short enough for a terminal\n"
    + f'bcftools query -f \'%ID\\n\' "$@" | head -n "${{LAB_MAXROWS:-{CAP}}}"\n'
)


@dataclass(frozen=True)
class Variant:
    id: str
    chrom: str
    pos: int
    ref: str
    alt: str


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    variants: list[Variant]


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f18-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    variants: list[Variant] = []
    for chrom, n in contigs.items():
        lo, hi = PER_CONTIG[chrom]
        for pos in sorted(rnd.sample(range(10, n - 10), rnd.randint(lo, hi))):
            ref = rnd.choice("ACGT")
            alt = rnd.choice([b for b in "ACGT" if b != ref])
            variants.append(Variant(f"v{len(variants) + 1:03d}", chrom, pos, ref, alt))
    return Plan(contigs, variants)


def truth_answers(plan: Plan) -> tuple[list[str], list[str]]:
    """(every ID on the queried contig, the first CAP of them: what the faulted wrapper prints)."""
    ids = [v.id for v in plan.variants if v.chrom == QUERY]
    return ids, ids[:CAP]


def vcf_text(plan: Plan) -> tuple[str, str]:
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items()
    )
    body = "".join(
        f"{v.chrom}\t{v.pos}\t{v.id}\t{v.ref}\t{v.alt}\t.\t.\t.\n" for v in plan.variants
    )
    return header, body


def build_f18(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """completeness: a result list cut at a cap inside the lab's listing wrapper."""
    plan = make_plan(seed)
    common.write_vcf_text(work, *vcf_text(plan), tools)
    faulted = twin == "faulted"
    write_script(work, WRAPPER, WRAPPER_FAULTED if faulted else WRAPPER_CLEAN)
    line = (
        f"{WRAPPER}: the lab's variant-ID lister around bcftools query "
        "(usage: ./lab-ids.sh VCF [REGION])."
    )
    if tier == "explicit":
        line += (
            f" It pipes the list through head -n {CAP}, so at most {CAP} IDs are printed."
            if faulted
            else " It prints every ID."
        )
    readme = [
        "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2, contigs chr1 and chr2.",
        line,
    ]
    (work / "README.md").write_text(common.readme(readme))
    source = f'{WRAPPER} source: bcftools query ... | head -n "${{LAB_MAXROWS:-{CAP}}}"'
    if faulted:
        evidence = (f"README: {WRAPPER} pipes the list through head -n {CAP}", source)
        evidence = evidence if tier == "explicit" else evidence[1:]
    else:
        evidence = (f"README: {WRAPPER} prints every ID",) if tier == "explicit" else ()
    prompt = common.prompt_for(
        route,
        f"list the IDs of all variants on {QUERY} in variants.vcf.gz "
        f"(the lab usually lists IDs with ./{WRAPPER})",
        f"use the lab's ./{WRAPPER} wrapper to list the IDs of all variants on {QUERY} in "
        "variants.vcf.gz",
        "a JSON list of variant IDs",
    )
    expected, capped = truth_answers(plan)
    truth = {"kind": "id_list", "expected": expected, "naive": capped if faulted else expected}
    task_id = common.task_id("F18", seed, twin, tier, route)
    return Task(task_id, "F18", "completeness", twin, seed, prompt, truth, evidence, tier, route)


def reference_f18(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bcftools", "query", "-r", QUERY, "-f", "%ID\n", "variants.vcf.gz"]
    ids = tools.run(argv, work).stdout.split()
    return result(ids, "bcftools query -r on the contig, called directly (no cap)")


def naive_f18(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = run_wrapper(tools, work, WRAPPER, "variants.vcf.gz", QUERY)
    return result(proc.stdout.split(), f"./{WRAPPER} variants.vcf.gz {QUERY}")


# the wrapper's escape hatch: a LAB_MAXROWS above any list in these files
OVERRIDE_F18 = re.compile(r"\bLAB_MAXROWS=['\"]?\d{3,}")


def exposed_f18(commands: list[str], work: Path) -> bool:
    return wrapper_exposed(commands, WRAPPER, "bcftools", OVERRIDE_F18)


SPEC = FamilySpec(
    id="F18",
    slot="completeness",
    source="SCD-14 (listing cut at a cap with no flag); GEN-35 (default cap on a result list)",
    build=build_f18,
    reference=reference_f18,
    naive=naive_f18,
    exposed=exposed_f18,
    twin_files=TWIN_FILES,
    locus="wrapper",
)
