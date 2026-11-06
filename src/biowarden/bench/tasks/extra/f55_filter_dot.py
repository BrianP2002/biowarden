"""F55 filter_dot (filter_semantics, tool default): `-f PASS` drops the records whose FILTER is ".".

In a VCF the FILTER column is PASS (all filters passed), the name of a failed filter, or "." (no
filter was applied to the record). `bcftools view -f PASS` keeps only PASS: the "." records of a
caller that applies no filters, or of a callset merged from two pipelines, silently vanish. The
same holds for `-i 'FILTER="PASS"'`, `-e 'FILTER!="PASS"'` and `vcftools --keep-filtered PASS`
(there is no value that keeps "." with --keep-filtered); `bcftools view -f .,PASS` and
`vcftools --remove-filtered-all` keep both. Exit status 0, empty stderr.

Task: list the IDs of the variants that no filter flagged. Both twins carry the same sites; the same
10-14 of them carry the name of a failed filter (LowQual, LowDP or StrandBias). The clean twin marks
every other record PASS; in the faulted twin 6-10 of those carry "." instead (records from a caller
that applies no filters). The records are therefore unflagged in both twins, but only PASS survives
the usual `-f PASS`.

Verified here (bcftools 1.24, vcftools 0.1.17; six records a PASS, b LowQual, c PASS, d PASS, e ".",
f PASS): `view -f PASS`, `-i 'FILTER="PASS"'` and `-e 'FILTER!="PASS"'` give a c d f;
`view -f .,PASS` and `vcftools --remove-filtered-all` give a c d e f (LowQual b dropped by both);
`vcftools --keep-filtered PASS` gives a c d f. No stderr message other than the vcftools summary.

  explicit  the README says what each FILTER value means, "." included
  implicit  the README only says the callers' FILTER values are kept as written
  forced    the sentence asks for `bcftools view -f PASS`-style filtering with bcftools
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import (
    FORMAT_GT,
    result,
    segments,
    vcf_text,
    write_vcf_gz,
)
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 2400, "chr2": 1800}
FLAGS = ("LowQual", "LowDP", "StrandBias")
HEADER = FORMAT_GT + "".join(
    f'##FILTER=<ID={name},Description="{name} filter">\n' for name in FLAGS
)


def make_plan(seed: int) -> tuple[list[tuple[str, int, str, str, str]], list[int]]:
    """Sites (contig, pos, ref, alt, clean FILTER value) and the indices that carry "." in the
    faulted twin."""
    rng = random.Random(f"f55-{seed}")
    n_flagged, n_unflagged = rng.randint(10, 14), rng.randint(20, 26)
    kinds = [rng.choice(FLAGS) for _ in range(n_flagged)] + ["PASS"] * n_unflagged
    rng.shuffle(kinds)
    places: set[tuple[str, int]] = set()
    while len(places) < len(kinds):
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    for (chrom, pos), kind in zip(
        sorted(places, key=lambda cp: (order[cp[0]], cp[1])), kinds, strict=True
    ):
        ref = rng.choice("ACGT")
        sites.append((chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref]), kind))
    passing = [i for i, s in enumerate(sites) if s[4] == "PASS"]
    return sites, sorted(rng.sample(passing, rng.randint(6, 10)))


def rows(sites, dots: set[int], rng: random.Random) -> list[str]:
    out = []
    for i, (chrom, pos, ref, alt, filt) in enumerate(sites):
        gts = "\t".join(rng.choice(["0/0", "0/1", "1/1"]) for _ in range(3))
        value = "." if i in dots else filt
        out.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\t{value}\t.\tGT\t{gts}")
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, dots = make_plan(seed)
    rng = random.Random(f"f55-gt-{seed}")
    text = vcf_text(
        CONTIGS,
        ["s1", "s2", "s3"],
        rows(sites, set(dots) if twin == "faulted" else set(), rng),
        HEADER,
    )
    write_vcf_gz(work, text, tools)
    lines = [
        "variants.vcf.gz (with variants.vcf.gz.tbi): SNVs of three samples, contigs chr1 and chr2; "
        "the FILTER column is kept as the callers wrote it."
    ]
    if tier == "explicit":
        lines.append(
            "FILTER values: PASS = all filters passed; a filter name (LowQual, LowDP, StrandBias) "
            "= flagged by that filter; '.' = no filter was applied to the record."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("every FILTER value is PASS or the name of a filter",)
    else:
        evidence = (
            f"{len(dots)} records have FILTER '.' (no filter applied), the others PASS or a name",
            "bcftools view -f PASS drops the records whose FILTER is '.'",
        )
    prompt = common.prompt_for(
        route,
        "list the IDs of the variants in variants.vcf.gz that no filter flagged",
        "use `bcftools view -f` to list the IDs of the variants in variants.vcf.gz that no filter "
        "flagged",
        "a JSON list of variant IDs",
    )
    expected = [f"v{i:02d}" for i, s in enumerate(sites) if s[4] == "PASS"]
    naive = [i for i in expected if int(i[1:]) not in dots] if twin == "faulted" else expected
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F55", seed, twin, tier, route)
    return Task(
        task_id, "F55", "filter_semantics", twin, seed, prompt, truth, evidence, tier, route
    )


# ---- solvers -----------------------------------------------------------------------------------


def bcftools_ids(work: Path, tools: ToolEnv, *options: str) -> list[str]:
    argv = ["bcftools", "view", "-H", *options, "variants.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sorted(line.split("\t")[2] for line in proc.stdout.splitlines() if line)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(bcftools_ids(work, tools, "-f", ".,PASS"), "bcftools view -f .,PASS")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(bcftools_ids(work, tools, "-f", "PASS"), "bcftools view -f PASS")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a command that selects FILTER=PASS records of the variants file without also accepting
# "." (bcftools -f PASS / --apply-filters PASS, FILTER="PASS" tests, vcftools --keep-filtered
# PASS, grep/awk on PASS). Safe: -f .,PASS (either order), a test that names ".", vcftools
# --remove-filtered-all / --remove-filtered NAME, bcftools -e 'FILTER~"Low"'-style name tests.
# Applied per command; an awk test of any column ($7 of the VCF, $2 of a query) counts.
PASS_ONLY = re.compile(
    r"(?:\s-f\s*|--apply-filters[\s=])PASS(?![,\w])"
    r"|FILTER\s*==?\s*[\"']PASS[\"']|FILTER\s*!=\s*[\"']PASS[\"']"
    r"|--keep-filtered[\s=]PASS|\$\d+\s*==\s*[\"']PASS[\"']|\bgrep\b[^|;]*\bPASS\b"
    r"|\bgrep\b[^|;]*-w[^|;]*PASS"
)
ACCEPTS_DOT = re.compile(
    r"PASS,\.|\.,PASS|FILTER\s*[!=]=?\s*[\"']\.[\"']|\$\d+\s*==\s*[\"']\.[\"']|--remove-filtered"
    r"|\|\|\s*\$\d+"
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if PASS_ONLY.search(segment) and not ACCEPTS_DOT.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F55",
    slot="filter_semantics",
    source="VCF 4.2 specification (FILTER: PASS, a filter name, or '.' for 'filters not applied')",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"variants.vcf.gz", "variants.vcf.gz.tbi"}),
    locus="tool_default",
)
