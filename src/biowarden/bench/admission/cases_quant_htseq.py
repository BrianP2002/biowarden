"""Admission cases for htseq.strandedness (family F41) and htseq.min_mapq (family F44).

htseq.strandedness. For one seed both twins of F41 are built (implicit tier): the same reads of the
same genes as a forward-stranded library (clean) and as a dUTP library (even seeds) or an unstranded
one (odd seeds) (faulted). The task counts the reads per gene. Every call is an `htseq-count` of
reads.bam against genes.gtf; the strandedness it asks for (`-s`/`--stranded` yes, no or reverse,
default yes) is the only thing that changes the numbers, so the real tool is run once per value and
twin (`htseq-count -q ...`) and the counts are compared with task.truth (tolerance 5, for the leaked
reads of a stranded library): `yes` is right on the forward twin only, `reverse` on the dUTP twin
only, `no` on both (no gene pair overlaps in F41, so unstranded counting loses nothing). Spelling
variants (`--stranded=yes`, `-q`, `-r pos -t exon -i gene_id`, `-m union`, `--nonunique none`)
share the label of their value; tests/test_rules_quant_htseq.py runs every spelling for real on a
few seeds to check that.

Declarations (key library_strandedness): a right call carries the true library only when its own
strandedness is that library (`-s no` on a dUTP twin is right by accident and is only emitted
undeclared); a wrong call carries its own value (declared but botched) and the true one.

htseq.min_mapq. F44 twins: the same unstranded reads; in the faulted twin 15-30 % of them have MAPQ
0-3. Calls differ in the MAPQ floor (`-a` / `--minaqual`, default 10) and always pass `-s no`:
floor 0 is right on both twins, floors 1, 4 and 10 are wrong on the faulted twin and right on the
clean one. One real run per floor and twin. Declarations (key low_mapq_reads): `keep` is the
call's own value for floor 0, `drop` for any higher floor; the true value is `keep`.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_quant_common import (
    Template,
    cases_for,
    label_of,
    parallel,
)
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f41_htseq_strand import htseq_counts

FILES = ("reads.bam", "genes.gtf")
STRAND_KEY = "library_strandedness"
MAPQ_KEY = "low_mapq_reads"
OWN_STRAND = {"yes": "forward", "no": "unstranded", "reverse": "reverse"}


def _strand(name: str, key: str, *argv: str) -> Template:
    return Template(name, "htseq-count", (*argv, *FILES), key, OWN_STRAND[key])


STRAND_TEMPLATES = (
    _strand("default", "yes"),
    _strand("quiet_default", "yes", "-q"),
    _strand("s_yes", "yes", "-s", "yes"),
    _strand("stranded_eq_yes", "yes", "--stranded=yes", "-q"),
    _strand("s_yes_pos_exon", "yes", "-s", "yes", "-r", "pos", "-t", "exon", "-i", "gene_id"),
    _strand("s_no", "no", "-s", "no"),
    _strand("stranded_no", "no", "--stranded", "no", "-q"),
    _strand("s_no_nonunique", "no", "-s", "no", "--nonunique", "none"),
    _strand("s_reverse", "reverse", "-s", "reverse"),
    _strand("stranded_eq_reverse", "reverse", "--stranded=reverse", "-q"),
    _strand("s_reverse_union", "reverse", "-s", "reverse", "-m", "union"),
)


def true_library(seed: int, twin: str) -> str:
    return "forward" if twin == "clean" else ("reverse" if seed % 2 == 0 else "unstranded")


@case_source("htseq.strandedness")
def strandedness_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F41", seed, twin, work, tools, "implicit", "free")

        def run(value: str, work=work, task=task):
            return value, label_of(task.truth, htseq_counts(work, tools, "-s", value))

        labels = dict(parallel(run, list(OWN_STRAND)))
        out += cases_for(
            STRAND_TEMPLATES,
            labels,
            work,
            key=STRAND_KEY,
            true=true_library(seed, twin),
            family="F41",
            seed=seed,
            twin=twin,
        )
    return out


def _mapq(name: str, floor: int, *opt: str) -> Template:
    return Template(
        name,
        "htseq-count",
        ("-s", "no", *opt, *FILES),
        str(floor),
        "keep" if floor == 0 else "drop",
    )


MAPQ_TEMPLATES = (
    _mapq("default", 10),
    _mapq("a10", 10, "-a", "10"),
    Template(
        "minaqual_eq_10",
        "htseq-count",
        ("--stranded=no", "--minaqual=10", "-q", *FILES),
        "10",
        "drop",
    ),
    _mapq("a0", 0, "-a", "0"),
    _mapq("minaqual_0", 0, "--minaqual", "0"),
    _mapq("a1", 1, "-a", "1"),
    _mapq("a4", 4, "-a", "4"),
)
FLOORS = (10, 0, 1, 4)


@case_source("htseq.min_mapq")
def min_mapq_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F44", seed, twin, work, tools, "implicit", "free")

        def run(floor: int, work=work, task=task):
            counts = htseq_counts(work, tools, "-s", "no", "-a", str(floor))
            return str(floor), label_of(task.truth, counts)

        labels = dict(parallel(run, list(FLOORS)))
        out += cases_for(
            MAPQ_TEMPLATES,
            labels,
            work,
            key=MAPQ_KEY,
            true="keep",
            family="F44",
            seed=seed,
            twin=twin,
        )
    return out


__all__ = ["STRAND_TEMPLATES", "MAPQ_TEMPLATES", "true_library"]
