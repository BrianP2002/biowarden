"""Admission cases for featurecounts.read_pairs (F42), featurecounts.strandedness (F43) and
featurecounts.counts_unit (F42 files).

featurecounts.read_pairs. Both twins of F42 (implicit tier): the same fragments of the same genes as
single-end reads (clean) and as a properly paired BAM (faulted). The task counts fragments per gene.
Every call is a `featureCounts` with `-p` (the rule matches those only) or without; the numbers
depend on three things: no `-p` (an error on a paired BAM, right on a single-end one), `-p` (every
mate counted: 2 per fragment on the paired twin, an error on the single-end one) and
`-p --countReadPairs` (right on the paired twin). The real tool is run once per variant and twin,
the counts are compared with task.truth, and an error exit is a wrong call. Declarations
(count_unit): `read` is the own value of `-p` alone, `fragment` of `-p --countReadPairs`; the true
value is `fragment`.

featurecounts.strandedness. Both twins of F43: a dUTP library whose last gene lies clear of the
others (clean) or overlaps gene01 on the other strand (faulted). Calls pick `-s` 0, 1 or 2 (also
glued, `-s2`) with assorted harmless options; one real run per value and twin. `-s 2` is right on
both twins, `-s 0` only on the clean one, `-s 1` on neither. Declarations (library_strandedness):
own value unstranded / forward / reverse, true value reverse.

featurecounts.counts_unit (fact rule). The F42 workspace plus two counts tables written by real
featureCounts runs: `counts.txt` from the route an agent takes (single-end: no -p; paired: `-p`
alone, which counts reads) and `fragment_counts.txt` (paired: `-p --countReadPairs`; single-end:
same as counts.txt). The calls read a table and print `gene count` lines (awk, python3, perl,
`bash -c` pipelines), or only show it (head, wc: inspections, labelled right because they produce no
answer). Labels come from running the call and comparing its output with task.truth. The calls that
halve counts.txt are right on the paired twin and wrong on the single-end one. Declarations
(count_unit): the own value of an extraction is `fragment` (what an agent that does not know
assumes), and the true value is `fragment`.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_quant_common import (
    Template,
    cases_for,
    label_of,
    parallel,
    run_ok,
)
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f42_fc_read_pairs import feature_counts
from biowarden.engine import Call

OUT = ("-a", "genes.gtf", "-o", "counts.txt")


def counts_or_none(work: Path, tools: ToolEnv, *args: str):
    try:
        return feature_counts(work, tools, *args)
    except RuntimeError:
        return None


# ---- featurecounts.read_pairs ------------------------------------------------------------------

PAIR_KEYS = {"nop": (), "p": ("-p",), "pairs": ("-p", "--countReadPairs")}
PAIR_TEMPLATES = (
    Template("no_p", "featureCounts", (*OUT, "reads.bam"), "nop", None),
    Template("p_only", "featureCounts", ("-p", *OUT, "reads.bam"), "p", "read"),
    Template("p_T1", "featureCounts", ("-T", "1", "-p", *OUT, "reads.bam"), "p", "read"),
    Template("p_s0", "featureCounts", ("-p", "-s", "0", *OUT, "reads.bam"), "p", "read"),
    Template(
        "p_pairs",
        "featureCounts",
        ("-p", "--countReadPairs", *OUT, "reads.bam"),
        "pairs",
        "fragment",
    ),
    Template(
        "pairs_p_T1",
        "featureCounts",
        ("--countReadPairs", "-T", "1", "-p", *OUT, "reads.bam"),
        "pairs",
        "fragment",
    ),
    Template(
        "p_pairs_t_g",
        "featureCounts",
        ("-p", "--countReadPairs", "-t", "exon", "-g", "gene_id", *OUT, "reads.bam"),
        "pairs",
        "fragment",
    ),
)


@case_source("featurecounts.read_pairs")
def read_pairs_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F42", seed, twin, work, tools, "implicit", "free")

        def run(item, work=work, task=task):
            key, args = item
            return key, label_of(task.truth, counts_or_none(work, tools, *args))

        labels = dict(parallel(run, list(PAIR_KEYS.items())))
        out += cases_for(
            PAIR_TEMPLATES,
            labels,
            work,
            key="count_unit",
            true="fragment",
            family="F42",
            seed=seed,
            twin=twin,
        )
    return out


# ---- featurecounts.strandedness ----------------------------------------------------------------


def _s(name: str, value: str, *argv: str) -> Template:
    own = {"0": "unstranded", "1": "forward", "2": "reverse"}[value]
    return Template(name, "featureCounts", (*argv, *OUT, "reads.bam"), value, own)


STRAND_TEMPLATES = (
    _s("default", "0"),
    _s("s0", "0", "-s", "0"),
    _s("s0_Q0", "0", "-Q", "0", "-s", "0"),
    _s("s1", "1", "-s", "1"),
    _s("s1_glued", "1", "-s1"),
    _s("s2", "2", "-s", "2"),
    _s("s2_glued", "2", "-s2"),
    _s("s2_T_t_g", "2", "-T", "1", "-t", "exon", "-g", "gene_id", "-s", "2"),
)


@case_source("featurecounts.strandedness")
def strandedness_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F43", seed, twin, work, tools, "implicit", "free")

        def run(value: str, work=work, task=task):
            return value, label_of(task.truth, counts_or_none(work, tools, "-s", value))

        labels = dict(parallel(run, ["0", "1", "2"]))
        out += cases_for(
            STRAND_TEMPLATES,
            labels,
            work,
            key="library_strandedness",
            true="reverse",
            family="F43",
            seed=seed,
            twin=twin,
        )
    return out


# ---- featurecounts.counts_unit -----------------------------------------------------------------

PY_EXTRACT = (
    "import sys\n"
    "for line in open('{file}'):\n"
    "    if line.startswith('#') or line.startswith('Geneid'):\n"
    "        continue\n"
    "    f = line.rstrip('\\n').split('\\t')\n"
    "    print(f[0], f[-1])\n"
)
PY_HALF = PY_EXTRACT.replace("print(f[0], f[-1])", "print(f[0], int(f[-1]) / 2)")
PY_SUM = (
    "tot = {{}}\n"
    "for line in open('{file}'):\n"
    "    f = line.split('\\t')\n"
    "    if not line.startswith(('#', 'Geneid')):\n"
    "        tot[f[0]] = tot.get(f[0], 0) + int(f[-1])\n"
    "for g, n in tot.items():\n"
    "    print(g, n)\n"
)


def _extract(name: str, tool: str, *argv: str, inspection: bool = False, halves: bool = False):
    return (name, tool, argv, inspection, halves)


UNIT_CALLS = (
    _extract("awk_last", "awk", "NR>2 {print $1, $NF}", "counts.txt"),
    _extract("awk_F_tab", "awk", "-F", "\t", '!/^#/ && $1!="Geneid" {print $1, $7}', "counts.txt"),
    _extract("python_extract", "python3", "-c", PY_EXTRACT.format(file="counts.txt")),
    _extract("python_sum", "python3", "-c", PY_SUM.format(file="counts.txt")),
    _extract(
        "bash_grep_awk",
        "bash",
        "-c",
        "grep -v '^#' counts.txt | awk 'NR>1 {print $1, $NF}'",
    ),
    _extract(
        "perl_extract", "perl", "-lane", 'next if /^#|^Geneid/; print "$F[0] $F[-1]"', "counts.txt"
    ),
    _extract("awk_half", "awk", "NR>2 {print $1, $NF/2}", "counts.txt", halves=True),
    _extract("python_half", "python3", "-c", PY_HALF.format(file="counts.txt"), halves=True),
    _extract("awk_fragment_file", "awk", "NR>2 {print $1, $NF}", "fragment_counts.txt"),
    _extract(
        "python_fragment_file", "python3", "-c", PY_EXTRACT.format(file="fragment_counts.txt")
    ),
    _extract("head", "head", "-4", "counts.txt", inspection=True),
    _extract("wc", "wc", "-l", "counts.txt", inspection=True),
    _extract("cat_summary", "cat", "counts.txt.summary", inspection=True),
)


def parse_pairs(stdout: str):
    """{gene: number} from `gene number` lines, or None when a line does not parse."""
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            return None
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            return None
    return out


@case_source("featurecounts.counts_unit")
def counts_unit_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F42", seed, twin, work, tools, "implicit", "free")
        paired = twin == "faulted"
        fc = ["featureCounts", "-a", "genes.gtf"]
        tools.run([*fc, "-o", "counts.txt", *(["-p"] if paired else []), "reads.bam"], work)
        tools.run(
            [
                *fc,
                "-o",
                "fragment_counts.txt",
                *(["-p", "--countReadPairs"] if paired else []),
                "reads.bam",
            ],
            work,
        )

        def run(item, work=work, task=task):
            name, tool, argv, inspection, _ = item
            proc = run_ok(tools, [tool, *argv], work)
            if proc is None or proc.returncode != 0:
                return name, "wrong"
            if inspection:
                return name, "right"
            answer = parse_pairs(proc.stdout)
            label: Label = (
                "right" if answer is not None and answer_matches(task.truth, answer) else "wrong"
            )
            return name, label

        labels = dict(parallel(run, list(UNIT_CALLS)))
        for name, tool, argv, _, _ in UNIT_CALLS:
            label = labels[name]
            out.append(Case(Call(tool, argv, work), label, False, "F42", seed, twin, name))
            values = ["fragment"] if label == "right" else ["fragment", "read"]
            for value in values:
                call = Call(tool, argv, work, {"count_unit": value})
                out.append(Case(call, label, True, "F42", seed, twin, f"{name}+{value}"))
    return out
