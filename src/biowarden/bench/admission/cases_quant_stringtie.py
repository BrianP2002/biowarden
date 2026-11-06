"""Admission cases for stringtie.strandedness (a purpose-built workspace, no family).

Per seed two workspaces of the same genes from the quant simulator: a coordinate-sorted BAM of
single-end reads (`reads.bam`) and `genes.gtf`, from a dUTP library (clean) or an unstranded one
(faulted). The task is the number of reads per gene. StringTie reports a coverage per transcript;
with `-e` and one transcript per gene the reads are `coverage * length / 50` (checked against the
simulation in the tests), compared with a tolerance of 1.5. Every call is a `stringtie reads.bam -G
genes.gtf -e ...`; the strand flag is the only thing that changes the numbers, so the real tool is
run once per flag (none, --rf, --fr) and twin: no flag is right on both twins, `--rf` only on the
dUTP twin (it halves the unstranded one), `--fr` on neither. Declarations (library_strandedness):
own value unstranded (no flag), reverse (--rf) or forward (--fr); true value reverse or unstranded.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_quant_common import Template, cases_for, parallel
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, common
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    READ_LEN,
    REVERSE,
    UNSTRANDED,
    counts_by_gene,
    gtf_text,
    make_fragments,
    make_layout,
    sam_genome,
)

COMMON = ("-G", "genes.gtf", "-e", "-A", "abundance.tab", "-o", "out.gtf", "-p", "1")


def build_workspace(seed: int, twin: str, work: Path, tools: ToolEnv) -> dict:
    layout = make_layout(f"st-{seed}", 5)
    library = REVERSE if twin == "clean" else UNSTRANDED
    frags = make_fragments(random.Random(f"st-{seed}-{twin}"), layout, library)
    work.mkdir(parents=True, exist_ok=True)
    (work / "genes.gtf").write_text(gtf_text(layout))
    common.write_bam(work, sam_genome(layout, frags, False), tools, "reads")
    return {
        "kind": "dict_num",
        "expected": counts_by_gene(layout, frags),
        "tol": 1.5,
        "layout": layout,
    }


def stringtie_reads(work: Path, tools: ToolEnv, truth: dict, *flag: str):
    lengths = {g.id: g.length for g in truth["layout"].genes}
    with tempfile.TemporaryDirectory() as scratch:
        out, tab = Path(scratch) / "out.gtf", Path(scratch) / "abundance.tab"
        argv = ["stringtie", "reads.bam", "-G", "genes.gtf", "-e", "-A", str(tab), "-o", str(out)]
        proc = tools.run([*argv, "-p", "1", *flag], work)
        if proc.returncode != 0 or not tab.is_file():
            return None
        rows = [r.split("\t") for r in tab.read_text().splitlines()[1:]]
    return {r[0]: float(r[6]) * lengths[r[0]] / READ_LEN for r in rows}


def _t(name: str, key: str, own: str, *argv: str, first: bool = True) -> Template:
    args = ("reads.bam", *COMMON, *argv) if first else (*argv, "reads.bam", *COMMON)
    return Template(name, "stringtie", args, key, own)


TEMPLATES = (
    _t("default", "none", "unstranded"),
    _t("default_B", "none", "unstranded", "-B"),
    _t("rf", "rf", "reverse", "--rf"),
    _t("rf_first", "rf", "reverse", "--rf", first=False),
    _t("rf_label", "rf", "reverse", "--rf", "-l", "S1"),
    _t("fr", "fr", "forward", "--fr"),
    _t("fr_first", "fr", "forward", "--fr", first=False),
)
FLAGS = {"none": (), "rf": ("--rf",), "fr": ("--fr",)}


@case_source("stringtie.strandedness")
def strandedness_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    works = {twin: scratch / twin for twin in ("clean", "faulted")}
    truths = {twin: build_workspace(seed, twin, works[twin], tools) for twin in works}

    def run(item):
        twin, key = item
        reads = stringtie_reads(works[twin], tools, truths[twin], *FLAGS[key])
        right = reads is not None and answer_matches(truths[twin], reads)
        return (twin, key), "right" if right else "wrong"

    done = dict(parallel(run, [(t, k) for t in works for k in FLAGS]))
    out: list[Case] = []
    for twin in works:
        labels = {k: done[(twin, k)] for k in FLAGS}
        out += cases_for(
            TEMPLATES,
            labels,
            works[twin],
            key="library_strandedness",
            true="reverse" if twin == "clean" else "unstranded",
            family="quant-stringtie",
            seed=seed,
            twin=twin,
        )
    return out
