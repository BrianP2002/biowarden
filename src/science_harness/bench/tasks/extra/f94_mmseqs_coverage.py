"""F94 mmseqs_coverage: `mmseqs easy-cluster` demands 80% coverage of both sequences by default.

Source: the MMseqs2 defaults (`-c 0.8 --cov-mode 0`, `--min-seq-id 0.0`; easy-search uses `-c 0`).
Verified here with MMseqs2 18.8cc5c on 41 sequences (12 families of 2-4 members at 98% identity and
10 fragments, each an exact piece of 35-60% of a member):

  mmseqs easy-cluster seqs.fa clu tmp --min-seq-id 0.9                  20 clusters
  mmseqs easy-cluster seqs.fa clu tmp --min-seq-id 0.9 -c 0.0            12 clusters
  mmseqs easy-cluster seqs.fa clu tmp --min-seq-id 0.9 --cov-mode 1      12 clusters

and on the generated files of seeds 0-29 (10-14 families, 7-15 fragments): the default call gives
families + fragments on all 30 faulted twins and families on all 30 clean ones; `--cov-mode 1`
(coverage of the target, the member) gives the number of families on all of them; lowering `-c`
alone with the default `--cov-mode 0` still leaves fragments out: `-c 0.0` on 25 of the 30 faulted
twins, and `-c 0.1`, `0.2`, `0.3` on the two seeds tried (seed 0: 12 clusters for 10 families), so
only the coverage MODE fixes it (with `-c 0.3` a 37-residue fragment becomes the representative of
the four full-length members and a second fragment from another region of the same protein cannot
join it). Exit 0;
nothing on stderr (-v 1). A fragment of a protein has 100% identity over its own
length but covers only 35-60% of the full-length sequence, so with the default coverage it stays
out of its parent's cluster and becomes a cluster of its own. `--min-seq-id 0.9` alone therefore
does not mean "90% identity clusters" for a set that holds partial sequences.

Twins: the same families. Clean: only full-length members. Faulted: the same members plus 7-15
fragments (disjoint pieces, so two fragments never cluster with each other). The recorded answer
is the number of families; the default call returns families + fragments on the faulted twin.
The ids are neutral (prot01, prot02, ... in file order, numbered after the shuffle, contiguous in
each twin): nothing in an id names a family or a member, and only the header text "partial
sequence" marks a fragment. (The first version put the family number and a member or fragment
index in every id, so counting the distinct prefixes gave the answer.)

  explicit  the README states the clustering rule (identity only, fragments belong to their parent)
  implicit  the README only says that some entries are partial sequences
  forced    the task sentence dictates mmseqs easy-cluster
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f89_chain_ids import (
    AA_LETTERS,
    TOOLS_NOTE,
    random_sequence,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"seqs.fasta", "README.md"})


@functools.lru_cache(maxsize=64)
def make_records(seed: int) -> tuple[tuple[tuple[str, str, bool], ...], int, int]:
    """((name, sequence, is_fragment), ...) in file order, number of families, fragments.

    The names are neutral and numbered after the shuffle (the clean twin renumbers, see `fasta`).
    """
    rng = random.Random(f"f94-{seed}")
    n_fam = rng.randint(10, 14)
    entries: list[tuple[str, bool]] = []
    frag_families = set(rng.sample(range(n_fam), rng.randint(6, 9)))
    n_frag = 0
    for f in range(n_fam):
        length = rng.randint(90, 140)
        root = random_sequence(rng, length)
        members = []
        for _ in range(rng.randint(2, 4)):
            s = list(root)
            for _ in range(max(1, int(0.02 * length))):
                s[rng.randrange(length)] = rng.choice(AA_LETTERS)
            members.append("".join(s))
            entries.append((members[-1], False))
        if f in frag_families:
            source = rng.choice(members)
            size = int(length * rng.uniform(0.35, 0.45))
            for k, start in enumerate((rng.randint(0, 5), length // 2 + rng.randint(0, 5))):
                if k == 1 and rng.random() < 0.5:
                    continue
                entries.append((source[start : start + size], True))
                n_frag += 1
    rng.shuffle(entries)
    recs = tuple((f"prot{i + 1:02d}", seq, frag) for i, (seq, frag) in enumerate(entries))
    return recs, n_fam, n_frag


def fasta(recs, twin: str) -> str:
    """The FASTA text; the clean twin drops the fragments and numbers the rest from 1."""
    kept = [(seq, frag) for _, seq, frag in recs if not (frag and twin == "clean")]
    return "".join(
        f">prot{i + 1:02d}{' partial sequence' if frag else ''}\n{seq}\n"
        for i, (seq, frag) in enumerate(kept)
    )


@functools.lru_cache(maxsize=512)
def cluster_count(tools: ToolEnv, text: str, *extra: str) -> int:
    """Number of clusters of `mmseqs easy-cluster ... --min-seq-id 0.9 <extra>` (own temp dir)."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        (cwd / "seqs.fasta").write_text(text)
        argv = ["mmseqs", "easy-cluster", "seqs.fasta", "clu", "tmp", "--min-seq-id", "0.9"]
        proc = tools.run([*argv, *extra, "--threads", "1", "-v", "1"], cwd)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        rows = (cwd / "clu_cluster.tsv").read_text().splitlines()
        return len({r.split("\t")[0] for r in rows})


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    recs, n_fam, n_frag = make_records(seed)
    text = fasta(recs, twin)
    (work / "seqs.fasta").write_text(text)
    n_seq = text.count(">")
    lines = [f"seqs.fasta: {n_seq} protein sequences (FASTA)."]
    if tier == "explicit":
        lines.append(
            "Some entries are partial sequences (fragments, marked 'partial') cut from a "
            "full-length protein in the file. Clusters are defined by sequence identity alone: "
            "a fragment belongs to the cluster of the protein it was cut from."
        )
    elif twin == "faulted":
        lines.append("Some entries are partial sequences, marked 'partial'.")
    if twin == "clean":
        evidence = ("every entry is a full-length protein; no entry is marked partial",)
    else:
        evidence = (f"{n_frag} entries are marked 'partial', each a piece of 35-45% of another",)
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    prompt = common.prompt_for(
        route,
        "cluster the sequences in seqs.fasta at 90% sequence identity and report the number of "
        "clusters (a partial sequence belongs to the cluster of the protein it was cut from)",
        "use `mmseqs easy-cluster` to cluster the sequences in seqs.fasta at 90% sequence "
        "identity and report the number of clusters (a partial sequence belongs to the cluster "
        "of the protein it was cut from)",
        "a JSON integer",
    )
    truth = {
        "kind": "int",
        "expected": n_fam,
        "naive": n_fam if twin == "clean" else n_fam + n_frag,
    }
    task_id = common.task_id("F94", seed, twin, tier, route)
    return Task(task_id, "F94", "coverage", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = cluster_count(tools, (work / "seqs.fasta").read_text(), "--cov-mode", "1")
    return _result(n, "mmseqs easy-cluster --min-seq-id 0.9 --cov-mode 1 (member coverage)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = cluster_count(tools, (work / "seqs.fasta").read_text())
    return _result(n, "mmseqs easy-cluster --min-seq-id 0.9 (default coverage)")


# Exposure: a mmseqs clustering call on the file without `--cov-mode 1` (coverage of the target,
# the member) or `--cov-mode 2`. A lower `-c` under the default mode does not fix it (see above).
# Judged per command. Other clusterers (cd-hit, vsearch) are not exposures.
CLUSTER = re.compile(r"\bmmseqs\s+(?:easy-cluster|cluster|easy-linclust|linclust)\b")
IDENTITY_ONLY = re.compile(r"--cov-mode[ =][12]\b")


def exposed(commands: list[str], work: Path) -> bool:
    return any(CLUSTER.search(c) and not IDENTITY_ONLY.search(c) for c in commands)


SPEC = FamilySpec(
    id="F94",
    slot="coverage",
    source="MMseqs2 defaults (easy-cluster -c 0.8 --cov-mode 0)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
