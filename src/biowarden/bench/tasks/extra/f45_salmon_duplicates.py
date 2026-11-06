"""F45 salmon_duplicates (locus tool_default): `salmon index` drops exact duplicate transcripts.

salmon 2.x collapses transcripts with identical sequences when it builds an index (a recent gene
duplicate, a PAR copy): only the first is indexed and the other has no row in `quant.sf` or
`quant.genes.sf` at all; its reads are credited to the retained copy. `--keepDuplicates` keeps
both, and the reads of the pair are then shared equally. The twins hold the same genes and the same
read counts. In the clean twin the last gene (gene06) is an unrelated sequence. In the faulted twin
its transcript is an exact copy of gene02's, sampled with the same number of fragments.

Verified on salmon 2.8.0 (exit 0) on 377 simulated read pairs: `salmon index -t transcripts.fa
-i idx` prints at INFO level `removed 1 transcripts that were exact sequence duplicates (use
--keepDuplicates to retain them)` and writes the pair to `idx/duplicate_clusters.tsv`;
`salmon quant -i idx -l A -1 .. -2 .. -g genes.gtf` then gives gene02 164 reads (82 + 82) and no
gene06 row. With `--keepDuplicates` gene02 and gene06 get 82.000 each. The quant run itself prints
nothing about it. (salmon never writes an empty stderr: its log is on stderr, so the INFO line is
the only trace.)

  explicit  the README says that gene06 is a recent duplicate of gene02 with the same cDNA
  implicit  the README only describes the files; `grep -v '>' transcripts.fa | sort | uniq -d`
            shows the duplicate
  forced    the task sentence dictates salmon
"""

from __future__ import annotations

import atexit
import hashlib
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    UNSTRANDED,
    counts_by_gene,
    fastq_text,
    gtf_text,
    make_fragments,
    make_layout,
    transcripts_fasta,
)
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"transcripts.fa", "reads_1.fastq", "reads_2.fastq", "README.md"})
_INDEXES: dict[tuple[str, bool], Path] = {}
_SCRATCH: list[Path] = []


def make_plan(seed: int, twin: str):
    duplicate = "same" if twin == "faulted" else "different"
    layout = make_layout(f"f45-{seed}", 5, duplicate=duplicate)
    frags = make_fragments(random.Random(f"f45-{seed}-frags"), layout, UNSTRANDED)
    return layout, frags


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    layout, frags = make_plan(seed, twin)
    (work / "transcripts.fa").write_text(transcripts_fasta(layout))
    (work / "genes.gtf").write_text(gtf_text(layout))
    mate1, mate2 = fastq_text(layout, frags, True)
    (work / "reads_1.fastq").write_text(mate1)
    (work / "reads_2.fastq").write_text(mate2)
    last, src = layout.genes[-1], layout.genes[1]
    lines = [
        "reads_1.fastq, reads_2.fastq: paired-end RNA-seq reads (unstranded library), "
        "one fragment per read pair.",
        "transcripts.fa: the transcript sequences (one per gene, named by transcript ID).",
        "genes.gtf: gene annotation in GTF format; transcript_id links a transcript to its gene.",
    ]
    if tier == "explicit" and twin == "faulted":
        lines[1] += (
            f" {last.tx} ({last.id}) is a recent duplicate of {src.tx} ({src.id}): "
            "the two cDNA sequences are identical."
        )
    elif tier == "explicit":
        lines[1] += " All sequences are distinct."
    (work / "README.md").write_text(common.readme(lines))
    expected = counts_by_gene(layout, frags)
    naive = dict(expected)
    if twin == "faulted":
        naive[src.id] += naive[last.id]
        naive[last.id] = 0
        seen = (
            f"{last.tx} and {src.tx} have identical sequences: salmon index keeps only {src.tx}, "
            f"so {last.id} has no row in quant.genes.sf and {src.id} gets both genes' reads"
        )
    else:
        seen = "every transcript sequence is distinct: salmon index keeps them all"
    evidence = (seen,)
    if tier == "explicit" and twin == "faulted":
        evidence = (f"README says {last.tx} duplicates {src.tx}", *evidence)
    what = "the number of fragments (read pairs) in reads_1.fastq and reads_2.fastq from that gene"
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use salmon to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its fragment count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 1.0}
    task_id = common.task_id("F45", seed, twin, tier, route)
    return Task(task_id, "F45", "completeness", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def _cleanup() -> None:
    for path in _SCRATCH:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup)


def salmon_gene_counts(work: Path, tools: ToolEnv, keep_duplicates: bool) -> dict[str, float]:
    """salmon index + quant with -g genes.gtf; genes without a row get 0 (as an agent reports)."""
    fasta = work / "transcripts.fa"
    key = (hashlib.sha1(fasta.read_bytes()).hexdigest(), keep_duplicates)
    index = _INDEXES.get(key)
    if index is None or not index.is_dir():
        scratch = Path(tempfile.mkdtemp(prefix="f45-index-"))
        _SCRATCH.append(scratch)
        index = scratch / "idx"
        argv = ["salmon", "index", "-t", str(fasta), "-i", str(index), "-p", "1"]
        common.check(
            tools.run([*argv, *(["--keepDuplicates"] if keep_duplicates else [])], work), argv
        )
        _INDEXES[key] = index
    with tempfile.TemporaryDirectory() as scratch_dir:
        out = Path(scratch_dir) / "q"
        argv = ["salmon", "quant", "-i", str(index), "-l", "A", "-p", "1", "-o", str(out)]
        argv += ["-1", "reads_1.fastq", "-2", "reads_2.fastq", "-g", "genes.gtf"]
        common.check(tools.run(argv, work), argv)
        rows = [r.split("\t") for r in (out / "quant.genes.sf").read_text().splitlines()[1:]]
    found = {r[0]: float(r[4]) for r in rows}
    genes = re.findall(r'gene_id "([^"]+)"', (work / "genes.gtf").read_text())
    return {g: round(found.get(g, 0.0), 3) for g in dict.fromkeys(genes)}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    counts = salmon_gene_counts(work, tools, keep_duplicates=True)
    return _result(counts, "salmon index --keepDuplicates, salmon quant -g genes.gtf")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(
        salmon_gene_counts(work, tools, False), "salmon index, salmon quant -g genes.gtf"
    )


def sequences(path: Path) -> list[str]:
    seqs: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            seqs.append("")
        elif seqs:
            seqs[-1] += line.strip().upper()
    return seqs


def has_duplicates(work: Path) -> bool:
    try:
        seqs = sequences(work / "transcripts.fa")
    except OSError:
        return False
    return len(set(seqs)) < len(seqs)


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff `salmon index` runs without --keepDuplicates on a FASTA with duplicates."""
    if not has_duplicates(work):
        return False
    for command in commands:
        for part in re.split(r"[|;&\n]", command):
            if re.search(r"\bsalmon\s+index\b", part) and "--keepDuplicates" not in part:
                return True
    return False


SPEC = FamilySpec(
    id="F45",
    slot="completeness",
    source="salmon 2.x `index --help`: exact-duplicate transcripts are collapsed",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
