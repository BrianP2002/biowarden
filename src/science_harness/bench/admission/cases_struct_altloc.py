"""Admission cases for struct.altloc_policy (family F90: CA-CA distance of two residues whose atoms
all carry alternate locations).

Workspaces are the benchmark's own (implicit tier): clean twin, altloc A is the major conformer;
faulted twin, B is. Every call below is a `python-struct -c` snippet run with the real libraries;
its printed distance (angstroms) is compared with `task.truth` (tolerance 0.05); the labels never
come from the rule.

  ProDy parsePDB default (first conformer: wrong on the faulted twin), ProDy with altloc 'A' and 'B'
  loaded and chosen by occupancy, gemmi after remove_alternative_conformations (first conformer),
  gemmi taking the atom of largest occupancy, gemmi taking the first CA of the residue (first
  conformer), Biopython PDBParser (highest occupancy), plain Python over the ATOM lines taking the
  first matching line (wrong) or the largest occupancy (right), and a printed list of the altloc
  records (inspection).
Declarations: highest_occupancy for right calls; highest_occupancy and first_listed for the calls
that keep the first conformer.
"""

from __future__ import annotations

import re

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_struct_common import (
    Tpl,
    build_cases,
    last_float,
    py,
)
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task

DIST = "print(float(((a - b) ** 2).sum() ** 0.5))"


def templates(task: Task) -> list[Tpl]:
    p, q = re.search(r"residues (\d+) and (\d+)", task.prompt).groups()
    return [
        py(
            "prody_default",
            "import prody; ag = prody.parsePDB('model.pdb'); "
            f"a = ag.select('resnum {p} name CA').getCoords()[0]; "
            f"b = ag.select('resnum {q} name CA').getCoords()[0]; {DIST}",
            "first_listed",
            last_float,
        ),
        py(
            "prody_calpha_flag",
            "import prody; ag = prody.parsePDB('model.pdb', subset='calpha'); "
            f"a = ag.select('resnum {p}').getCoords()[0]; "
            f"b = ag.select('resnum {q}').getCoords()[0]; {DIST}",
            "first_listed",
            last_float,
        ),
        py(
            "prody_altloc_by_occupancy",
            "import prody, numpy as np; "
            "A = prody.parsePDB('model.pdb', altloc='A'); "
            "B = prody.parsePDB('model.pdb', altloc='B'); "
            "pick = lambda n: (A if A.select(f'resnum {n} name CA').getOccupancies()[0] "
            ">= B.select(f'resnum {n} name CA').getOccupancies()[0] else B)"
            ".select(f'resnum {n} name CA').getCoords()[0]; "
            f"a, b = pick({p}), pick({q}); {DIST}",
            "highest_occupancy",
            last_float,
        ),
        py(
            "gemmi_remove_alternative_conformations",
            "import gemmi, numpy as np; st = gemmi.read_structure('model.pdb'); "
            "st.remove_alternative_conformations(); ch = st[0]['A']; "
            "pos = lambda n: [r['CA'][0].pos for r in ch if r.seqid.num == n][0]; "
            f"d = pos({p}).dist(pos({q})); print(d)",
            "first_listed",
            last_float,
        ),
        py(
            "gemmi_largest_occupancy",
            "import gemmi; st = gemmi.read_structure('model.pdb'); ch = st[0]['A']; "
            "best = lambda n: max((a for r in ch if r.seqid.num == n for a in r "
            "if a.name == 'CA'), key=lambda a: a.occ).pos; "
            f"print(best({p}).dist(best({q})))",
            "highest_occupancy",
            last_float,
        ),
        py(
            "gemmi_first_ca",
            "import gemmi; st = gemmi.read_structure('model.pdb'); ch = st[0]['A']; "
            "first = lambda n: [a for r in ch if r.seqid.num == n for a in r "
            "if a.name == 'CA'][0].pos; "
            f"print(first({p}).dist(first({q})))",
            "first_listed",
            last_float,
        ),
        py(
            "biopython_default",
            "from Bio.PDB import PDBParser; "
            "ch = PDBParser(QUIET=True).get_structure('x', 'model.pdb')[0]['A']; "
            f"print(ch[{p}]['CA'] - ch[{q}]['CA'])",
            "highest_occupancy",
            last_float,
        ),
        py(
            "text_first_line",
            "rows = [l for l in open('model.pdb') if l.startswith('ATOM') and l[12:16] == ' CA ']; "
            "xyz = lambda n: [tuple(float(l[c:c + 8]) for c in (30, 38, 46)) for l in rows "
            "if int(l[22:26]) == n][0]; "
            f"a, b = xyz({p}), xyz({q}); "
            "print(sum((u - v) ** 2 for u, v in zip(a, b)) ** 0.5)",
            "first_listed",
            last_float,
        ),
        py(
            "text_largest_occupancy",
            "rows = [l for l in open('model.pdb') if l.startswith('ATOM') and l[12:16] == ' CA ']; "
            "xyz = lambda n: max(((float(l[54:60]), "
            "tuple(float(l[c:c + 8]) for c in (30, 38, 46))) "
            "for l in rows if int(l[22:26]) == n))[1]; "
            f"a, b = xyz({p}), xyz({q}); "
            "print(sum((u - v) ** 2 for u, v in zip(a, b)) ** 0.5)",
            "highest_occupancy",
            last_float,
        ),
        py(
            "list_altloc_records",
            "print([l[:27] for l in open('model.pdb') if l[16] in 'AB'][:6])",
            None,
            None,
        ),
        py(
            "count_atoms_and_conformers",
            "import gemmi; st = gemmi.read_structure('model.pdb'); "
            "print(sum(1 for r in st[0]['A'] for a in r), len(st[0]['A']))",
            None,
            None,
        ),
    ]


@case_source("struct.altloc_policy")
def altloc_policy_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F90",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=templates,
        key="altloc_policy",
        needed=lambda twin, task: "highest_occupancy",
        values=("highest_occupancy", "first_listed", "all_conformers"),
    )
