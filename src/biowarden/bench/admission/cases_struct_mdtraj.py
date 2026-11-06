"""Admission cases for mdtraj.resid_index (family F92: CA-CA distance between two PDB residues).

Workspaces are the benchmark's own (implicit tier): the clean twin numbers the chain from 0 (so
mdtraj's residue index equals the number), the faulted twin from 1. Every call below is a
`python-struct -c` snippet (or an awk call) run with the real libraries; the printed distance (in
angstroms) is compared with `task.truth` (tolerance 0.05); labels never come from the rule.

  mdtraj `resid p` / `resid q` (the index: wrong on the faulted twin), mdtraj `resSeq` (right),
  mdtraj through a list comprehension on `residue.resSeq` (right), mdtraj `topology.residue(p)`
  (index again: wrong on the faulted twin, not seen by the rule), mdtraj `resSeq` with the
  nanometre result left unconverted (wrong on both twins, units), gemmi by residue number, Biopython
  by residue id, ProDy `resnum`, awk over the CA records (right), and a printed header of the
  topology (inspection).
Declarations: pdb_number for right calls; pdb_number and residue_index for the `resid` calls.
"""

from __future__ import annotations

import re

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_struct_common import (
    Tpl,
    build_cases,
    last_float,
    py,
)
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.base import Task

LOAD = "import mdtraj as md; t = md.load_pdb('protein.pdb'); top = t.topology; "


def templates(task: Task) -> list[Tpl]:
    p, q = re.search(r"residues (\d+) and (\d+)", task.prompt).groups()

    def dist(select: str, scale: str = " * 10") -> str:
        return (
            f"{LOAD}a = top.select({select % p!r})[0]; b = top.select({select % q!r})[0]; "
            f"print(float(md.compute_distances(t, [[a, b]])[0, 0]){scale})"
        )

    return [
        py("mdtraj_resid", dist("resid %s and name CA"), "residue_index", last_float),
        py(
            "mdtraj_resid_calpha",
            dist("resid %s and name CA", " * 10.0"),
            "residue_index",
            last_float,
        ),
        py("mdtraj_resSeq", dist("resSeq %s and name CA"), "pdb_number", last_float),
        py(
            "mdtraj_resSeq_comprehension",
            f"{LOAD}ix = lambda n: [a.index for a in top.atoms if a.residue.resSeq == n "
            f"and a.name == 'CA'][0]; "
            f"print(float(md.compute_distances(t, [[ix({p}), ix({q})]])[0, 0]) * 10)",
            "pdb_number",
            last_float,
        ),
        py(
            "mdtraj_residue_index_lookup",
            f"{LOAD}ix = lambda n: [a.index for a in top.residue(n).atoms if a.name == 'CA'][0]; "
            f"print(float(md.compute_distances(t, [[ix({p}), ix({q})]])[0, 0]) * 10)",
            "residue_index",
            last_float,
        ),
        py("mdtraj_resSeq_nm", dist("resSeq %s and name CA", ""), "pdb_number", last_float),
        py(
            "gemmi_by_number",
            "import gemmi; st = gemmi.read_structure('protein.pdb'); ch = st[0]['A']; "
            "ca = {r.seqid.num: r['CA'][0].pos for r in ch}; "
            f"print(ca[{p}].dist(ca[{q}]))",
            "pdb_number",
            last_float,
        ),
        py(
            "biopython_by_id",
            "from Bio.PDB import PDBParser; st = PDBParser(QUIET=True).get_structure('x', "
            f"'protein.pdb'); ch = st[0]['A']; print(ch[{p}]['CA'] - ch[{q}]['CA'])",
            "pdb_number",
            last_float,
        ),
        py(
            "prody_resnum",
            "import prody, numpy; ag = prody.parsePDB('protein.pdb'); "
            f"a = ag.select('resnum {p} and name CA').getCoords()[0]; "
            f"b = ag.select('resnum {q} and name CA').getCoords()[0]; "
            "print(float(numpy.linalg.norm(a - b)))",
            "pdb_number",
            last_float,
        ),
        py(
            "mdtraj_topology_summary",
            f"{LOAD}print(top.n_residues, [r.resSeq for r in list(top.residues)[:3]], "
            "top.n_chains)",
            None,
            None,
        ),
    ]


@case_source("mdtraj.resid_index")
def resid_index_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F92",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=templates,
        key="residue_numbering",
        needed=lambda twin, task: "pdb_number",
        values=("pdb_number", "residue_index"),
    )
