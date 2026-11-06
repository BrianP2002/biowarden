"""F96 modified_residues: sequence extraction turns selenomethionine (MSE) into X or drops it.

Source R2-STR-03 / R2-STR-28 (ProteinMPNN and LigandMPNN turn residues they do not know into gaps
or drop them) and the readers verified here (pdb-tools 2.7.0, gemmi 0.7.5, Biopython 1.88, ProDy
2.6.1) on a 25-residue chain with MSE at residues 6 and 15 as HETATM records, a SEP at 15, a gap
(20-22 missing) and one water:

  pdb_tofasta model.pdb                          MKTAYXAKQRQISFXKSHFAGLDEX  (X for MSE, SEP, HOH)
  gemmi.one_letter_code(residue names)           the same string
  Bio.SeqUtils.seq1 on the residue names         the same string
  Bio.PDB.PPBuilder().build_peptides             'MKTAY', 'AKQRQISF', 'KSHF', 'AGLDE': the MSE and
                                                 SEP residues are missing from every peptide
  ProDy chain.getSequence()                      MKTAYMAKQRQISFSKSHFAGLDE   (parent letter: M, S)

All exit 0 with empty stderr (ProDy prints its "@> parsed" line). Selenomethionine labelling is
routine in crystallography, so the HETATM MSE case is common. pdb_tofasta also appends one X per
water in the chain, so the waterless files here keep that second quirk out of the clean twin.

Twins: the same protein. Clean: methionines are ordinary ATOM MET records. Faulted: two or three of
them are MSE HETATM records (the same backbone and CB atoms). The answer is the one-letter
sequence with MSE written as M.

  explicit  the README says selenomethionine counts as M
  implicit  the prompt only says a modified residue is written as its parent residue
  forced    the task sentence dictates pdb_tofasta
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f89_chain_ids import (
    AA_LETTERS,
    ONE,
    TOOLS_NOTE,
    Chain,
    make_residues,
    pdb_text,
    random_fold,
)
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.pdb", "README.md"})
LETTERS = AA_LETTERS.replace("M", "")


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f96-{seed}")
    n = rng.randint(28, 40)
    seq = [rng.choice(LETTERS) for _ in range(n)]
    met = rng.sample(range(2, n - 2), rng.randint(3, 4))
    for i in met:
        seq[i] = "M"
    modified = met[: rng.randint(2, 3)]  # which methionines are MSE in the faulted twin
    return {"n": n, "seq": "".join(seq), "modified": modified, "fold": random_fold(rng, n)}


def pdb_file(plan: dict[str, Any], twin: str) -> str:
    residues = make_residues(random.Random(0), plan["seq"], 1, fold=plan["fold"])
    if twin == "faulted":
        for i in plan["modified"]:
            residues[i].name = "MSE"
            residues[i].het = True
    return pdb_text([Chain("A", residues)])


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "model.pdb").write_text(pdb_file(plan, twin))
    line = "model.pdb: one protein chain (A), PDB format."
    if tier == "explicit":
        line += " Selenomethionine (MSE) counts as methionine (M)."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = ("every residue is an ATOM record of a standard amino acid",)
    else:
        where = ", ".join(str(i + 1) for i in plan["modified"])
        evidence = (f"residues {where} are MSE (selenomethionine) HETATM records",)
    prompt = common.prompt_for(
        route,
        "report the amino-acid sequence of chain A of model.pdb in one-letter code (a modified "
        "residue is written as the standard residue it derives from)",
        "use `pdb_tofasta` to get the amino-acid sequence of chain A of model.pdb in one-letter "
        "code (a modified residue is written as the standard residue it derives from)",
        "a JSON list holding that one sequence string",
    )
    n_mod = len(plan["modified"])
    naive_seq = list(plan["seq"])
    for i in plan["modified"]:
        naive_seq[i] = "X"
    truth = {
        "kind": "seq_list",
        "expected": [plan["seq"]],
        "naive": [plan["seq"] if twin == "clean" or not n_mod else "".join(naive_seq)],
    }
    task_id = common.task_id("F96", seed, twin, tier, route)
    return Task(
        task_id, "F96", "residue_alphabet", twin, seed, prompt, truth, evidence, tier, route
    )


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


PARENT = {"MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y"}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    letters = []
    for line in (work / "model.pdb").read_text().splitlines():
        if line[:6] in ("ATOM  ", "HETATM") and line[12:16].strip() == "CA":
            name = line[17:20]
            letters.append(ONE.get(name) or PARENT.get(name, "X"))
    return _result(["".join(letters)], "CA records in order, modified residues mapped to parents")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["pdb_tofasta", "model.pdb"], work)
    seq = "".join(line for line in proc.stdout.splitlines() if not line.startswith(">"))
    return _result([seq], "pdb_tofasta model.pdb")


# Exposure: a sequence extractor that writes X for (or drops) modified residues, with no handling
# of MSE / HETATM / parent residues in the same command. ProDy's getSequence (parent letters),
# a table that lists MSE and anything naming HETATM / MSE / modified residues count as handled. A
# table of the twenty residues written in the command (`{'ALA': 'A', ...}`) or a lookup that
# falls back to X (`d.get(name, 'X')`) is an extractor of its own and counts as one.
EXTRACT = re.compile(
    r"\bpdb_tofasta\b|one_letter_code|\bseq1\(|three_to_one|three_to_index|PPBuilder|CaPPBuilder"
    r"|get_sequence|\bSeqIO\b.*pdb-seqres|\bpdb2fasta\b"
    r"""|['"]ALA['"]\s*:\s*['"]A['"]|\.get\((?:[^()]|\([^()]*\))*,\s*['"]X['"]\s*\)"""
)
HANDLED = re.compile(
    r"(?i)\bMSE\b|HETATM|modified|parent|getSequence|\bhet\b|standard=False|aa_only"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any("model.pdb" in c and EXTRACT.search(c) and not HANDLED.search(c) for c in commands)


SPEC = FamilySpec(
    id="F96",
    slot="residue_alphabet",
    source="R2-STR-03",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
