"""Admission cases for salmon.duplicate_transcripts and salmon.quant_duplicates (family F45).

Both twins of F45 are built (implicit tier): the same genes and fragment counts, with gene06 an
unrelated transcript (clean) or an exact copy of gene02's (faulted). The task counts the fragments
per gene. salmon costs about 2 s per index build, so the tests use 8 seeds.

salmon.duplicate_transcripts. A call is a `salmon index` (spelling variants: -p 1, long options, the
top-level -q). What changes the numbers is `--keepDuplicates`; one real pipeline (`salmon index`,
`salmon quant -l A -g genes.gtf`, absent genes reported as 0) is run per twin and variant and the
gene counts are compared with task.truth: right on the clean twin whatever the flag, right on the
faulted twin only with `--keepDuplicates`. Declarations (duplicate_transcripts): `keep` is the own
value of a call with the flag, `drop` of one without; the true value is `keep`.

salmon.quant_duplicates (fact rule). The workspace gets real quant outputs: `q_default` (index
without the flag) on both twins and `q_keep` (with it) on the faulted one, each next to its
`cmd_info.json`. The calls read `quant.genes.sf` (awk, python3, perl-free `bash -c` pipelines) and
print `gene count` lines, or only show it (head, wc: inspections, labelled right). Labels come
from running the call and comparing with task.truth: reading q_default of the faulted twin is
wrong (gene06 has no row, gene02 holds both), reading q_keep or any clean table is right, a call
that splits the pooled count after reading duplicate_clusters.tsv is right. Declarations
(duplicate_transcripts): the true value is `keep`.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.admission.cases_quant_common import (
    Template,
    cases_for,
    label_of,
    parallel,
    run_ok,
)
from science_harness.bench.admission.cases_quant_featurecounts import parse_pairs
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.bench.tasks.extra.f45_salmon_duplicates import salmon_gene_counts
from science_harness.engine import Call

INDEX = ("-t", "transcripts.fa", "-i", "idx")


def _t(name: str, key: str, *argv: str, top: tuple[str, ...] = ()) -> Template:
    return Template(
        name, "salmon", (*top, "index", *argv), key, "keep" if key == "keep" else "drop"
    )


TEMPLATES = (
    _t("default", "default", *INDEX),
    _t("p1", "default", "-p", "1", *INDEX),
    _t("long_options", "default", "--transcripts", "transcripts.fa", "--index", "idx"),
    _t("quiet", "default", *INDEX, top=("-q",)),
    _t("keep", "keep", *INDEX, "--keepDuplicates"),
    _t("keep_p1", "keep", "-p", "1", *INDEX, "--keepDuplicates"),
    _t(
        "keep_long", "keep", "--transcripts", "transcripts.fa", "--index", "idx", "--keepDuplicates"
    ),
)


@case_source("salmon.duplicate_transcripts")
def duplicate_transcripts_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    works, tasks = {}, {}
    for twin in ("clean", "faulted"):
        works[twin] = scratch / twin
        tasks[twin] = build_task("F45", seed, twin, works[twin], tools, "implicit", "free")

    def run(item):
        twin, keep = item
        counts = salmon_gene_counts(works[twin], tools, keep)
        return (twin, "keep" if keep else "default"), label_of(tasks[twin].truth, counts)

    done = dict(parallel(run, [(t, k) for t in ("clean", "faulted") for k in (False, True)]))
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        labels = {k: done[(twin, k)] for k in ("default", "keep")}
        out += cases_for(
            TEMPLATES,
            labels,
            works[twin],
            key="duplicate_transcripts",
            true="keep",
            family="F45",
            seed=seed,
            twin=twin,
        )
    return out


# ---- salmon.quant_duplicates -------------------------------------------------------------------

PY_EXTRACT = (
    "for line in open('{file}'):\n"
    "    f = line.rstrip('\\n').split('\\t')\n"
    "    if f[0] != 'Name':\n"
    "        print(f[0], f[4])\n"
)
PY_FILL = (
    "import re\n"
    "genes = dict.fromkeys(re.findall(r'gene_id \"([^\"]+)\"', open('genes.gtf').read()), 0.0)\n"
    "for line in open('{file}'):\n"
    "    f = line.split('\\t')\n"
    "    if f[0] in genes:\n"
    "        genes[f[0]] = float(f[4])\n"
    "for g, n in genes.items():\n"
    "    print(g, n)\n"
)
PY_POOLED = """
import re
q = {}
for line in open('__FILE__'):
    f = line.split('\\t')
    if f[0] != 'Name':
        q[f[0]] = float(f[4])
gtf = open('genes.gtf').read()
tx_gene = {t: g for g, t in re.findall(r'gene_id "([^"]+)"; transcript_id "([^"]+)"', gtf)}
for line in list(open('idx_default/duplicate_clusters.tsv'))[1:]:
    kept, dup = line.split()
    q[tx_gene[kept]] = q[tx_gene[dup]] = q[tx_gene[kept]] / 2
for g, n in q.items():
    print(g, n)
"""

# (name, tool, argv, table, inspection)
READS = (
    ("awk_default", "awk", ("NR>1 {print $1, $5}", "q_default/quant.genes.sf"), "q_default", False),
    (
        "awk_F_default",
        "awk",
        ("-F", "\t", "NR>1 {print $1, $5}", "q_default/quant.genes.sf"),
        "q_default",
        False,
    ),
    (
        "python_default",
        "python3",
        ("-c", PY_EXTRACT.format(file="q_default/quant.genes.sf")),
        "q_default",
        False,
    ),
    (
        "python_fill_default",
        "python3",
        ("-c", PY_FILL.format(file="q_default/quant.genes.sf")),
        "q_default",
        False,
    ),
    (
        "bash_tail_awk",
        "bash",
        ("-c", "tail -n +2 q_default/quant.genes.sf | awk '{print $1, $5}'"),
        "q_default",
        False,
    ),
    (
        "python_pooled",
        "python3",
        ("-c", PY_POOLED.replace("__FILE__", "q_default/quant.genes.sf")),
        "q_default",
        False,
    ),
    ("awk_keep", "awk", ("NR>1 {print $1, $5}", "q_keep/quant.genes.sf"), "q_keep", False),
    (
        "python_keep",
        "python3",
        ("-c", PY_EXTRACT.format(file="q_keep/quant.genes.sf")),
        "q_keep",
        False,
    ),
    ("head", "head", ("-3", "q_default/quant.genes.sf"), "q_default", True),
    ("wc", "wc", ("-l", "q_default/quant.genes.sf"), "q_default", True),
)


@case_source("salmon.quant_duplicates")
def quant_duplicates_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    works, tasks = {}, {}
    for twin in ("clean", "faulted"):
        works[twin] = scratch / twin
        tasks[twin] = build_task("F45", seed, twin, works[twin], tools, "implicit", "free")

    def prepare(item):
        twin, name, keep = item
        work = works[twin]
        idx = f"idx_{name[2:]}"
        tools.run(
            [
                "salmon",
                "index",
                "-t",
                "transcripts.fa",
                "-i",
                idx,
                "-p",
                "1",
                *(["--keepDuplicates"] if keep else []),
            ],
            work,
        )
        argv = [
            "salmon",
            "quant",
            "-i",
            idx,
            "-l",
            "A",
            "-1",
            "reads_1.fastq",
            "-2",
            "reads_2.fastq",
        ]
        tools.run([*argv, "-o", name, "-p", "1", "-g", "genes.gtf"], work)

    parallel(
        prepare,
        [
            ("clean", "q_default", False),
            ("faulted", "q_default", False),
            ("faulted", "q_keep", True),
        ],
    )
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = works[twin]
        for name, tool, argv, table, inspection in READS:
            if not (work / table).is_dir():
                continue
            proc = run_ok(tools, [tool, *argv], work)
            label: Label = "wrong"
            if proc is not None and proc.returncode == 0:
                answer = parse_pairs(proc.stdout)
                if inspection or (answer is not None and answer_matches(tasks[twin].truth, answer)):
                    label = "right"
            out.append(Case(Call(tool, argv, work), label, False, "F45", seed, twin, name))
            for value in ("keep",) if label == "right" else ("keep", "drop"):
                declared = Call(tool, argv, work, {"duplicate_transcripts": value})
                out.append(Case(declared, label, True, "F45", seed, twin, f"{name}+{value}"))
    return out
