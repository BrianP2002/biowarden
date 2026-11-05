"""Admission cases for kallisto.strand_option (a purpose-built workspace, no family).

For one seed two workspaces of the same genes are built from the quant simulator
(bench.tasks.extra.f41_htseq_strand): paired-end reads of a dUTP library (clean) and of an
unstranded library (faulted); transcripts.fa, genes.gtf, a kallisto index. The task is the number
of fragments per gene (a dict, tolerance 2.5). Every call is a `kallisto quant` of the pair; the
strand option is the only thing that changes the numbers, so the real tool is run once per option
(none, --rf-stranded, --fr-stranded) and twin and the est_counts per gene are compared with the
truth: no option is right on both twins, `--rf-stranded` only on the dUTP twin (on the unstranded
one it keeps about half of the fragments), `--fr-stranded` on neither (on the dUTP twin it exits 1
with nothing pseudoaligned, on the unstranded one it keeps about half). Calls without an option
are not matched by the rule (it never sees them; they are emitted with and without the true
value as well). Declarations (library_strandedness): the own value of
`--rf-stranded` is reverse, of `--fr-stranded` forward, the true value is reverse (dUTP twin) or
unstranded.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_quant_common import Template, cases_for, parallel
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, common
from science_harness.bench.tasks.extra.f41_htseq_strand import (
    REVERSE,
    UNSTRANDED,
    counts_by_gene,
    fastq_text,
    gtf_text,
    make_fragments,
    make_layout,
    transcripts_fasta,
)

FASTQ = ("reads_1.fastq", "reads_2.fastq")
QUANT = ("quant", "-i", "transcripts.idx", "-o", "out")


def build_workspace(seed: int, twin: str, work: Path, tools: ToolEnv) -> dict:
    layout = make_layout(f"kq-{seed}", 5)
    library = REVERSE if twin == "clean" else UNSTRANDED
    frags = make_fragments(random.Random(f"kq-{seed}-{twin}"), layout, library)
    work.mkdir(parents=True, exist_ok=True)
    (work / "transcripts.fa").write_text(transcripts_fasta(layout))
    (work / "genes.gtf").write_text(gtf_text(layout))
    mate1, mate2 = fastq_text(layout, frags, True)
    (work / FASTQ[0]).write_text(mate1)
    (work / FASTQ[1]).write_text(mate2)
    argv = ["kallisto", "index", "-i", "transcripts.idx", "transcripts.fa"]
    common.check(tools.run(argv, work), argv)
    return {
        "kind": "dict_num",
        "expected": counts_by_gene(layout, frags),
        "tol": 2.5,
        "layout": layout,
    }


def kallisto_counts(work: Path, tools: ToolEnv, truth: dict, *option: str):
    """est_counts per gene of a real kallisto quant, or None when it fails."""
    layout = truth["layout"]
    with tempfile.TemporaryDirectory() as scratch:
        argv = ["kallisto", "quant", "-i", "transcripts.idx", "-o", scratch, *option, *FASTQ]
        proc = tools.run(argv, work)
        table = Path(scratch) / "abundance.tsv"
        if proc.returncode != 0 or not table.is_file():
            return None
        rows = [r.split("\t") for r in table.read_text().splitlines()[1:]]
    gene_of = {g.tx: g.id for g in layout.genes}
    return {gene_of[r[0]]: float(r[3]) for r in rows}


def _t(name: str, key: str, own: str | None, *option: str) -> Template:
    return Template(name, "kallisto", (*QUANT[:1], *QUANT[1:], *option, *FASTQ), key, own)


TEMPLATES = (
    _t("default", "none", None),
    _t("rf", "rf", "reverse", "--rf-stranded"),
    _t("rf_t1", "rf", "reverse", "-t", "1", "--rf-stranded"),
    _t("rf_seed", "rf", "reverse", "--rf-stranded", "--seed", "1"),
    _t("fr", "fr", "forward", "--fr-stranded"),
    _t("fr_t1", "fr", "forward", "-t", "1", "--fr-stranded"),
    _t("fr_plaintext", "fr", "forward", "--fr-stranded", "--plaintext"),
)
OPTIONS = {"none": (), "rf": ("--rf-stranded",), "fr": ("--fr-stranded",)}


@case_source("kallisto.strand_option")
def strand_option_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    works = {twin: scratch / twin for twin in ("clean", "faulted")}
    truths = {twin: build_workspace(seed, twin, works[twin], tools) for twin in works}

    def run(item):
        twin, key = item
        counts = kallisto_counts(works[twin], tools, truths[twin], *OPTIONS[key])
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
            family="quant-kallisto",
            seed=seed,
            twin=twin,
        )
    return out
