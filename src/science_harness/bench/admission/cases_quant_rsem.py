"""Admission cases for rsem.strandedness (a purpose-built workspace; family F46 has the same files).

Per seed two workspaces of the same genes from the quant simulator: a paired transcriptome BAM
(`reads_tx.bam`, mates adjacent, no @PG line) of a dUTP library (clean) and of an unstranded one
(faulted), plus an RSEM reference built by `rsem-prepare-reference --gtf genes.gtf genome.fa
rsem_ref`. The task is the number of fragments per gene (RSEM `expected_count`, tolerance 2).
Every call is a `rsem-calculate-expression --alignments --paired-end ...`; `--strandedness` (or
`--forward-prob`) is the only thing that changes the numbers, so the real tool is run once per
value (none, forward, reverse) and twin and the gene-level expected counts are compared with the
truth: `none` is right on both twins, `reverse` on the dUTP twin only (on the unstranded one it
keeps about half of the fragments), `forward` on neither (on the dUTP twin RSEM stops with "No
valid read to estimate the length distribution!", exit 255; a failed run is a wrong call).
Spellings:
`--strandedness=reverse`, `--forward-prob 0` (reverse), `--forward-prob 1` (forward). Declarations
(library_strandedness): the own value is the library the option means; the true value is reverse
(dUTP twin) or unstranded. RSEM costs about 1 s per run, so the tests use 10 seeds.
"""

from __future__ import annotations

import random
from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_quant_common import Template, cases_for, parallel
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, common
from science_harness.bench.tasks.extra.f41_htseq_strand import (
    REVERSE,
    UNSTRANDED,
    counts_by_gene,
    genome_fasta,
    gtf_text,
    make_fragments,
    make_layout,
    sam_transcriptome,
    write_transcriptome_bam,
)

BASE = ("--alignments", "--paired-end", "-p", "1", "--no-bam-output")
ARGS = ("reads_tx.bam", "rsem_ref", "out")


def build_workspace(seed: int, twin: str, work: Path, tools: ToolEnv) -> dict:
    layout = make_layout(f"rs-{seed}", 5)
    library = REVERSE if twin == "clean" else UNSTRANDED
    frags = make_fragments(random.Random(f"rs-{seed}-{twin}"), layout, library)
    work.mkdir(parents=True, exist_ok=True)
    (work / "genome.fa").write_text(genome_fasta(layout))
    (work / "genes.gtf").write_text(gtf_text(layout))
    write_transcriptome_bam(work, sam_transcriptome(layout, frags, True), tools)
    argv = ["rsem-prepare-reference", "--gtf", "genes.gtf", "genome.fa", "rsem_ref"]
    common.check(tools.run(argv, work), argv)
    return {"kind": "dict_num", "expected": counts_by_gene(layout, frags), "tol": 2.0}


def rsem_counts(work: Path, tools: ToolEnv, truth: dict, sample: str, *option: str):
    argv = ["rsem-calculate-expression", *BASE, *option, "reads_tx.bam", "rsem_ref", sample]
    proc = tools.run(argv, work)
    table = work / f"{sample}.genes.results"
    if proc.returncode != 0 or not table.is_file():
        return None
    rows = [r.split("\t") for r in table.read_text().splitlines()[1:]]
    return {r[0]: float(r[4]) for r in rows}


def _t(name: str, key: str, own: str, *option: str) -> Template:
    return Template(name, "rsem-calculate-expression", (*BASE, *option, *ARGS), key, own)


TEMPLATES = (
    _t("none_default", "none", "unstranded"),
    _t("none_explicit", "none", "unstranded", "--strandedness", "none"),
    _t("prob_half", "none", "unstranded", "--forward-prob", "0.5"),
    _t("reverse", "reverse", "reverse", "--strandedness", "reverse"),
    _t("reverse_eq", "reverse", "reverse", "--strandedness=reverse"),
    _t("prob_zero", "reverse", "reverse", "--forward-prob", "0"),
    _t("forward", "forward", "forward", "--strandedness", "forward"),
    _t("prob_one", "forward", "forward", "--forward-prob", "1"),
)
OPTIONS = {
    "none": ("--strandedness", "none"),
    "forward": ("--strandedness", "forward"),
    "reverse": ("--strandedness", "reverse"),
}


@case_source("rsem.strandedness")
def strandedness_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    works = {twin: scratch / twin for twin in ("clean", "faulted")}
    truths = {twin: build_workspace(seed, twin, works[twin], tools) for twin in works}

    def run(item):
        twin, key = item
        counts = rsem_counts(works[twin], tools, truths[twin], f"out_{key}", *OPTIONS[key])
        right = counts is not None and answer_matches(truths[twin], counts)
        return (twin, key), "right" if right else "wrong"

    done = dict(parallel(run, [(t, k) for t in works for k in OPTIONS]))
    out: list[Case] = []
    for twin in works:
        labels = {k: done[(twin, k)] for k in OPTIONS}
        out += cases_for(
            TEMPLATES,
            labels,
            works[twin],
            key="library_strandedness",
            true="reverse" if twin == "clean" else "unstranded",
            family="quant-rsem",
            seed=seed,
            twin=twin,
        )
    return out
