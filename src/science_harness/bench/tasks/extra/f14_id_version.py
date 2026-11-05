"""F14 id_version: Ensembl gene IDs with a version suffix on one side of a join.

Source GEN-16: pipelines that key on Ensembl IDs match them as exact strings, so
ENSG00000000123.7 never meets ENSG00000000123 and the join drops rows without an error.
The twins hold the same genes; genes.tsv carries unversioned IDs like expr.tsv (clean) or
versioned IDs (faulted). Task: how many genes listed in genes.tsv have a row in expr.tsv.
The exact-string answer on the faulted twin is 0.

  explicit  the README says whether genes.tsv IDs carry version suffixes
  implicit  the README only says "Ensembl gene IDs"; the suffixes in the data are the evidence
  forced    the task sentence dictates `join`, which matches exact strings
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

VERSION = re.compile(r"\.\d+$")
BIOTYPES = ("protein_coding", "lncRNA", "pseudogene", "miRNA")
SAMPLES = ("S1", "S2", "S3")


def _gene_ids(rng: random.Random, n: int) -> list[str]:
    numbers = sorted(rng.sample(range(1000, 280000), n))
    return [f"ENSG{k:011d}" for k in numbers]


def _symbol(rng: random.Random) -> str:
    return "".join(rng.choices("ABCDEFGHKLMNPRSTWXYZ", k=rng.randint(3, 4))) + str(
        rng.randint(1, 9)
    )


def _read_ids(path: Path) -> list[str]:
    """First column of a TSV with a header line."""
    return [line.split("\t")[0] for line in path.read_text().splitlines()[1:]]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f14-{seed}")
    n_genes = rng.randint(60, 90)
    ids = _gene_ids(rng, n_genes + rng.randint(10, 20))
    rng.shuffle(ids)
    listed, extra = sorted(ids[:n_genes]), ids[n_genes:]
    in_expr = set(rng.sample(listed, round(n_genes * rng.uniform(0.55, 0.75))))
    versions = {g: rng.randint(1, 15) for g in listed}
    rows = [
        f"{g}{'' if twin == 'clean' else '.' + str(versions[g])}\t{_symbol(rng)}\t"
        f"{rng.choices(BIOTYPES, weights=(6, 3, 2, 1))[0]}\n"
        for g in listed
    ]
    (work / "genes.tsv").write_text("ensembl_gene_id\tsymbol\tbiotype\n" + "".join(rows))
    expr_ids = sorted(in_expr | set(extra))
    counts = (
        f"{g}\t" + "\t".join(str(int(rng.lognormvariate(4.5, 1.5))) for _ in SAMPLES) + "\n"
        for g in expr_ids
    )
    (work / "expr.tsv").write_text("gene\t" + "\t".join(SAMPLES) + "\n" + "".join(counts))
    stated = tier == "explicit"
    if twin == "clean":
        suffix = " without version suffixes (ENSG00000000123)" if stated else ""
        evidence = ("README: both files use IDs without version suffixes",) if stated else ()
    else:
        suffix = " with version suffixes (ENSG00000000123.7)" if stated else ""
        seen = "genes.tsv IDs end in .N version suffixes; expr.tsv IDs have none"
        evidence = ("README: genes.tsv IDs carry version suffixes", seen) if stated else (seen,)
    readme = [
        "genes.tsv: gene panel, tab-separated: ensembl_gene_id, symbol, biotype. "
        f"The IDs are Ensembl gene IDs{suffix}.",
        "expr.tsv: expression counts, tab-separated: gene, then one column per sample. "
        "The IDs are Ensembl gene IDs" + (" without version suffixes." if stated else "."),
    ]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "count the genes listed in genes.tsv that have a row in expr.tsv",
        "use `join` on the gene ID columns to count the genes listed in genes.tsv that have a "
        "row in expr.tsv",
        "a JSON integer",
    )
    expected = len(in_expr)
    truth = {"kind": "int", "expected": expected, "naive": expected if twin == "clean" else 0}
    task_id = common.task_id("F14", seed, twin, tier, route)
    return Task(task_id, "F14", "id_namespace", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    listed = _read_ids(work / "genes.tsv")
    expr = {VERSION.sub("", g) for g in _read_ids(work / "expr.tsv")}
    flags = ["genes.tsv IDs carry version suffixes: stripped before matching"]
    versioned = any(VERSION.search(g) for g in listed)
    count = len({VERSION.sub("", g) for g in listed} & expr)
    return _result(count, "join on IDs with the version suffix removed", flags if versioned else [])


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    listed, expr = _read_ids(work / "genes.tsv"), set(_read_ids(work / "expr.tsv"))
    return _result(len(set(listed) & expr), "exact string match of the gene IDs")


FILES = re.compile(r"\bgenes\.tsv\b[\s\S]*\bexpr\.tsv\b|\bexpr\.tsv\b[\s\S]*\bgenes\.tsv\b")
COMPUTE = re.compile(r"\b(?:join|comm|grep|awk|gawk|python3?|perl|sort|uniq|diff|paste)\b")
# anything that removes, ignores or prefix-matches the suffix on the ID
HANDLES_VERSION = re.compile(
    "|".join(
        [
            r"\bg?sub\(|\bsubstr\(",  # awk
            r"split\([^)]*['\"]\\?\.['\"]|\.rsplit\(|\.partition\(",  # split on the dot
            r"\bre\.(?:sub|findall|match|search|compile)",  # python regex
            r"\bsed\b[^\n|;]*\bs\W",  # sed substitution
            r"\\\.|\[\^?\.\]",  # an escaped or bracketed dot in a pattern
            r"\bcut\b[^\n|;]*(?:-d\s*['\"]?\.|-c\s*\d)",  # cut on the dot or by characters
            r"-F\s*['\"]?\.|FS\s*=\s*['\"]?[.\[]",  # awk field separator with a dot
            r"\$\{\w+%|\[:15\]|\[:-\d+\]|ENSG\[0-9\]",  # bash %, slices, ID regex
        ]
    )
)
# a grep -f whose search target (last operand) is genes.tsv takes its patterns from elsewhere,
# e.g. expr.tsv: unversioned patterns match versioned lines as substrings
GREP_ON_GENES = re.compile(
    r"\bgrep\b[^|;\n]*\s-\w*f\b(?:<\([^)]*\)|[^|;\n(])*\bgenes\.tsv['\"]?\s*(?:$|[|;&>])",
    re.MULTILINE,
)


# a command that puts the two ID lists side by side: join, comm, grep -f, awk's two-file idiom, a
# set intersection (python, pandas); str.join (".join(") is not one
COMBINES = re.compile(
    r"(?<![\w.])join\b(?!\()|\bcomm\b|\bgrep\b[^|;\n]*\s-\w*f\b|\bNR\s*==\s*FNR|\bFNR\s*==\s*NR"
    r"|\s&\s|\.intersection\(|\.isin\(|\bmerge\("
)
# the answer file may name the method ("comm", "join") without running it
ANSWER_JSON = re.compile(r"\{[^{}]*[\"']answer[\"'][^{}]*\}")


def handles(command: str) -> bool:
    return bool(HANDLES_VERSION.search(command) or GREP_ON_GENES.search(command))


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command joins the raw IDs of the two files, or the session never handles them.

    Handling means a regex/split/cut/sed/substr on the ID, or grep -f searching genes.tsv
    (patterns from expr.tsv). A raw join is judged per command, as in F13 and F15: a model that
    joins the raw IDs first and strips the suffix later still executed the trapped route (exposure
    is "ran it", not "kept the answer"), and a `sed` or a dot check in some other command does not
    excuse it. It is a command that names both files, combines them (join, comm, grep -f, a set
    intersection) and does not handle the suffix itself.

    Without such a command the commands are searched as one text, as before, so a script written
    by one command and run by another counts: both files are used in a computation and nothing in
    the session handles the suffix. A command that only prints the files (head, cat) is not a
    computation. Limitations: a join of lists that earlier commands cut out of the files is only
    seen when nothing in the session handles the suffix, and other ways to ignore the suffix (e.g.
    str.replace) are not seen.
    """
    for command in commands:
        bare = ANSWER_JSON.sub("", command)
        if FILES.search(bare) and COMBINES.search(bare) and not handles(bare):
            return True
    text = "\n".join(commands)
    if not FILES.search(text) or not COMPUTE.search(text):
        return False
    return not handles(text)


SPEC = FamilySpec(
    id="F14",
    slot="id_namespace",
    source="GEN-16",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"genes.tsv", "README.md"}),
)
