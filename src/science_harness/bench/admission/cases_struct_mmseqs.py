"""Admission cases for mmseqs.cluster_coverage (family F94: clusters at 90% identity of a set with
partial sequences).

Workspaces are the benchmark's own (implicit tier): the clean twin holds full-length proteins only,
the faulted twin the same proteins plus fragments. The label of a call is the number of clusters it
returns (a real `mmseqs easy-cluster seqs.fasta clu tmp --min-seq-id 0.9 ... --threads 1 -v 1`
run in a scratch directory) compared with `task.truth`. Because one run costs about 2.5 s, a
distinct effective option set is run once per workspace; the admission test uses 8 seeds.

  default coverage (wrong on the faulted twin) and `--cov-mode 1` (right). A lowered `-c` under
  the default mode is not generated: whether it also gets the number of clusters right depends on
  which sequence mmseqs picks as representative (`-c 0.0` came out wrong on 25 of 30 generated
  sets, `-c 0.3` on 2 of 2 tried), so the label would be luck. `--cov-mode 2` is checked in the
  unit tests only (a run costs seconds).
Declarations: with_parent for right calls; with_parent and separate for the others (the task asks
for fragments with their parent).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.bench.tasks.extra.f94_mmseqs_coverage import cluster_count
from science_harness.engine import Call

KEY = "fragment_policy"
BASE = ("easy-cluster", "seqs.fasta", "clu", "tmp", "--min-seq-id", "0.9")
# name -> (extra options, convention the call implements)
VARIANTS = {
    "default": ((), "separate"),
    "cov1": (("--cov-mode", "1"), "with_parent"),
}


def run_variant(tools: ToolEnv, text: str, name: str) -> int:
    return cluster_count(tools, text, *VARIANTS[name][0])


@case_source("mmseqs.cluster_coverage")
def cluster_coverage_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F94", seed, twin, work, tools, "implicit", "free")
        text = (work / "seqs.fasta").read_text()
        names = tuple(VARIANTS)
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = pool.map(run_variant, [tools] * len(names), [text] * len(names), names)
            counts = dict(zip(names, runs, strict=True))
        for name, (extra, uses) in VARIANTS.items():
            argv = (*BASE, *extra)
            label: Label = "right" if answer_matches(task.truth, counts[name]) else "wrong"
            out.append(Case(Call("mmseqs", argv, work), label, False, "F94", seed, twin, name))
            decls = [{KEY: "with_parent"}]
            if label == "wrong" and uses != "with_parent":
                decls.append({KEY: uses})
            for decl in decls:
                out.append(
                    Case(Call("mmseqs", argv, work, decl), label, True, "F94", seed, twin, name)
                )
    return out
