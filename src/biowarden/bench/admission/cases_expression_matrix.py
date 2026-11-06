"""Admission cases for matrix.log_scale_state (family F15, tier implicit, both twins).

Per seed two workspaces are built with the family builder: the clean twin (expr.tsv holds raw
integer counts) and the faulted twin (expr.tsv holds log2(count + 1) rounded to 4 decimals). The
task is "count the genes whose mean raw count is above 100"; every call below is a realistic way an
agent computes that, written in awk, python3 and perl, including spellings that differ in the
file name (./expr.tsv, a gzip copy), the arithmetic (2^x-1, exp(x*log(2))-1, a sum compared with
600) and the output (print n or print n+0).

LABELS come from running the call in the workspace with the real interpreter: the first integer on
stdout is compared with task.truth by `answer_matches`; empty or non-integer output is wrong.
The rule is never consulted. Each call is then paired with declarations:

  right call   clean twin: the file values; faulted twin: a back-transform
               -> right_declared with the declaration that matches the data (raw_counts on
                  clean, log_transformed on faulted)
               -> right_undeclared on the CLEAN twin only (see below)
  wrong call   clean twin: a back-transform of raw counts (2^count); faulted twin: the file values
               -> wrong_undeclared, and wrong_declared with two of the three declarations drawn per
                  call (raw_counts, log_transformed, normalized). Some of those the rule cannot
                  know to be wrong (log_transformed declared, then no back-transform): they stay in
                  the cases as misses.

NOT generated: a correct back-transform on the faulted twin WITHOUT a declaration. The rule's
contract is that data in a risky state need the declaration whatever the call computes (a fact
rule fires on the data, not on the arithmetic), so such a call fails by design; counting it as
friction on correct data would misdescribe the contract. tests/test_rules_expression_matrix.py
asserts this behaviour explicitly instead.
"""

from __future__ import annotations

import gzip
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import build_task
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.tools import ToolEnv
from biowarden.engine import Call

RULE = "matrix.log_scale_state"
KEY = "matrix_state"
DECLARATIONS = ("raw_counts", "log_transformed", "normalized")
TRUE_STATE = {"clean": "raw_counts", "faulted": "log_transformed"}

AWK_MEAN = "NR>1{s=0;for(i=2;i<=NF;i++)s+=%s; if(s/(NF-1)>100)n++} END{print %s}"
AWK_SUM = "NR>1{s=0;for(i=2;i<=NF;i++)s+=%s} s>600{n++} END{print n}"
AWK_ARRAY = "NR>1{for(i=2;i<=NF;i++)t[$1]+=%s} END{for(g in t) if(t[g]/6>100)n++; print n+0}"
PY_CSV = (
    "import csv; rows=list(csv.reader(open('{f}'),delimiter='\\t'))[1:]; "
    "print(sum(sum({v} for x in r[1:])/len(r[1:])>100 for r in rows))"
)
PY_GZ = (
    "import gzip; rows=[l.split() for l in gzip.open('expr.tsv.gz','rt')][1:]; "
    "print(sum(sum({v} for x in r[1:])/len(r[1:])>100 for r in rows))"
)
PERL = "next if $.==1; $s=0; $s += %s for 1..$#F; $n++ if $s/$#F>100; END{print $n+0}"

# name -> (kind, tool, argv); kind is only a description, the label comes from running the call
TEMPLATES: dict[str, tuple[str, str, list[str]]] = {
    "awk_mean_values": ("values", "awk", ["-F\\t", AWK_MEAN % ("$i", "n"), "expr.tsv"]),
    "awk_mean_values_plus0": ("values", "awk", ["-F\\t", AWK_MEAN % ("$i", "n+0"), "expr.tsv"]),
    "awk_mean_values_dot": ("values", "awk", ["-F\\t", AWK_MEAN % ("$i", "n"), "./expr.tsv"]),
    "awk_sum_values": ("values", "awk", ["-F\\t", AWK_SUM % "$i", "expr.tsv"]),
    "awk_array_values": ("values", "awk", ["-F\\t", AWK_ARRAY % "$i", "expr.tsv"]),
    "awk_mean_pow": ("backtransform", "awk", ["-F\\t", AWK_MEAN % ("2^$i-1", "n"), "expr.tsv"]),
    "awk_mean_pow_plus0": (
        "backtransform",
        "awk",
        ["-F\\t", AWK_MEAN % ("2^$i-1", "n+0"), "expr.tsv"],
    ),
    "awk_mean_exp": (
        "backtransform",
        "awk",
        ["-F\\t", AWK_MEAN % ("exp($i*log(2))-1", "n"), "expr.tsv"],
    ),
    "awk_sum_pow": ("backtransform", "awk", ["-F\\t", AWK_SUM % "(2^$i-1)", "expr.tsv"]),
    "awk_array_pow": ("backtransform", "awk", ["-F\\t", AWK_ARRAY % "(2^$i-1)", "expr.tsv"]),
    "py_csv_values": ("values", "python3", ["-c", PY_CSV.format(f="expr.tsv", v="float(x)")]),
    "py_csv_pow": (
        "backtransform",
        "python3",
        ["-c", PY_CSV.format(f="expr.tsv", v="2**float(x)-1")],
    ),
    "py_gz_values": ("values", "python3", ["-c", PY_GZ.format(v="float(x)")]),
    "py_gz_pow": ("backtransform", "python3", ["-c", PY_GZ.format(v="2**float(x)-1")]),
    "perl_values": ("values", "perl", ["-F\\t", "-lane", PERL % "$F[$_]", "expr.tsv"]),
    "perl_pow": ("backtransform", "perl", ["-F\\t", "-lane", PERL % "2**$F[$_]-1", "expr.tsv"]),
}
# always included, so every seed has at least one correct call per twin
CORE = ("awk_mean_values", "awk_mean_pow", "py_csv_values", "py_csv_pow")


def _answer(stdout: str) -> int | None:
    match = re.match(r"\s*(-?\d+)\s*$", stdout)
    return int(match.group(1)) if match else None


def _run(tools: ToolEnv, call: Call) -> str:
    argv = [sys.executable if call.tool == "python3" else call.tool, *call.argv]
    return tools.run(argv, call.cwd).stdout


def _build(seed: int, twin: str, scratch: Path, tools: ToolEnv) -> tuple[Path, Task]:
    work = scratch / twin
    task = build_task("F15", seed, twin, work, tools, "implicit", "free")
    text = (work / "expr.tsv").read_bytes()
    (work / "expr.tsv.gz").write_bytes(gzip.compress(text, mtime=0))
    return work, task


def _case(call: Call, label: Label, declared: str | None, seed: int, twin: str, note: str) -> Case:
    final = Call(call.tool, call.argv, call.cwd, {} if declared is None else {KEY: declared})
    return Case(final, label, declared is not None, "F15", seed, twin, note)


@case_source(RULE)
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    rng = random.Random(f"f15-cases-{seed}")
    optional = [name for name in TEMPLATES if name not in CORE]
    chosen = [*CORE, *rng.sample(optional, 4)]
    built = {twin: _build(seed, twin, scratch, tools) for twin in ("clean", "faulted")}
    jobs = [
        (twin, name, Call(TEMPLATES[name][1], tuple(TEMPLATES[name][2]), built[twin][0]))
        for twin in built
        for name in chosen
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        outputs = list(pool.map(lambda job: _run(tools, job[2]), jobs))
    out: list[Case] = []
    for (twin, name, call), stdout in zip(jobs, outputs, strict=True):
        truth = built[twin][1].truth
        label: Label = "right" if answer_matches(truth, _answer(stdout)) else "wrong"
        note = f"{name} on {twin}: stdout {stdout.strip()[:12]!r}"
        if label == "right":
            out.append(_case(call, label, TRUE_STATE[twin], seed, twin, note))
            if twin == "clean":
                out.append(_case(call, label, None, seed, twin, note))
        else:
            out.append(_case(call, label, None, seed, twin, note))
            for declared in rng.sample(DECLARATIONS, 2):
                out.append(_case(call, label, declared, seed, twin, note))
    return out
