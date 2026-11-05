"""Admission cases for prody.chain_scheme (family F89) and prody.range_end (family F91).

Workspaces are the benchmark's own (implicit tier). Every call below is a `python-struct -c`
snippet run with the real libraries (ProDy 2.6.1, gemmi, Biopython, MDAnalysis) in the workspace;
its output is turned into the task's answer and compared with `task.truth`
(bench.score.answer_matches); the labels never come from the rules.

prody.chain_scheme / F89 (residues per chain, keyed by the author chain ID of the PDB entry):
  ProDy parseMMCIF / parsePDB('x.cif') default (label IDs: wrong on the faulted twin), ProDy with
  unite_chains=True, ProDy keyed by segment name (ProDy keeps auth_asym_id there), gemmi,
  Biopython MMCIFParser, a raw mmCIF dictionary grouped by label_asym_id (wrong on the faulted
  twin) or by auth_asym_id, the head of the file (inspection).
  Declarations: auth for right calls; auth and label for the label-based ones.

prody.range_end / F91 (number of cysteines in residues a..b, both ends included):
  ProDy 'resnum a:b' (also written 'resid a:b' and 'chain A and resnum a:b'; the end is left out,
  wrong on the faulted twin where the last residue is a cysteine), ProDy 'resnum a to b', a ProDy
  loop over residues, MDAnalysis 'resid a:b' (inclusive), gemmi and Biopython loops with <= (right)
  and with < (wrong, by hand), an awk count, and a printed list of residue names (inspection).
  A call that writes 'resnum a:b+1' by hand is right but is flagged unless it declares
  range_end=exclusive; it is deliberately left out (documented limit of the rule).
"""

from __future__ import annotations

import re

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_struct_common import (
    Tpl,
    build_cases,
    last_int,
    last_json,
    py,
)
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task

JSON = "import json; "
F89_TEMPLATES = [
    py(
        "prody_parseMMCIF",
        "import prody, json; ag = prody.parseMMCIF('complex.cif'); "
        "print(json.dumps({c.getChid(): c.numResidues() for c in ag.getHierView().iterChains()}))",
        "label",
        last_json,
    ),
    py(
        "prody_parsePDB_cif",
        "import prody, json; ag = prody.parsePDB('complex.cif'); "
        "print(json.dumps({c.getChid(): c.numResidues() for c in ag.getHierView().iterChains()}))",
        "label",
        last_json,
    ),
    py(
        "prody_parseMMCIF_selection",
        "import prody, json; ag = prody.parseMMCIF('complex.cif'); "
        "ids = sorted(set(ag.getChids())); "
        "print(json.dumps({i: ag.select('chain ' + i + ' and name CA').numAtoms() for i in ids}))",
        "label",
        last_json,
    ),
    py(
        "prody_unite_chains",
        "import prody, json; ag = prody.parseMMCIF('complex.cif', unite_chains=True); "
        "print(json.dumps({c.getChid(): c.numResidues() for c in ag.getHierView().iterChains()}))",
        "auth",
        last_json,
    ),
    py(
        "prody_segnames",
        "import prody, json; ag = prody.parseMMCIF('complex.cif'); "
        "print(json.dumps({s: ag.select('segment ' + s + ' and name CA').numAtoms() "
        "for s in sorted(set(ag.getSegnames()))}))",
        "auth",
        last_json,
    ),
    py(
        "gemmi_read_structure",
        "import gemmi, json; st = gemmi.read_structure('complex.cif'); "
        "print(json.dumps({ch.name: len(ch) for ch in st[0]}))",
        "auth",
        last_json,
    ),
    py(
        "biopython_mmcif",
        "from Bio.PDB import MMCIFParser; import json; "
        "st = MMCIFParser(QUIET=True).get_structure('x', 'complex.cif'); "
        "print(json.dumps({c.id: len(c) for c in st[0]}))",
        "auth",
        last_json,
    ),
    py(
        "mmcif_dict_label",
        "from Bio.PDB.MMCIF2Dict import MMCIF2Dict; import json; d = MMCIF2Dict('complex.cif'); "
        "pairs = set(zip(d['_atom_site.label_asym_id'], d['_atom_site.label_seq_id'])); "
        "print(json.dumps({a: sum(1 for p in pairs if p[0] == a) "
        "for a in sorted({p[0] for p in pairs})}))",
        "label",
        last_json,
    ),
    py(
        "mmcif_dict_auth",
        "from Bio.PDB.MMCIF2Dict import MMCIF2Dict; import json; d = MMCIF2Dict('complex.cif'); "
        "pairs = set(zip(d['_atom_site.auth_asym_id'], d['_atom_site.auth_seq_id'])); "
        "print(json.dumps({a: sum(1 for p in pairs if p[0] == a) "
        "for a in sorted({p[0] for p in pairs})}))",
        "auth",
        last_json,
    ),
    py("head_cif", "print(''.join(open('complex.cif').readlines()[:12]))", None, None),
]


def f89_templates(task: Task) -> list[Tpl]:
    return F89_TEMPLATES


@case_source("prody.chain_scheme")
def chain_scheme_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F89",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=f89_templates,
        key="chain_id_scheme",
        needed=lambda twin, task: "auth",
        values=("auth", "label"),
    )


def f91_templates(task: Task) -> list[Tpl]:
    a, b = re.search(r"residues (\d+) to (\d+)", task.prompt).groups()
    load = "import prody; ag = prody.parsePDB('protein.pdb'); "
    count = "print(0 if sel is None else sel.numAtoms())"
    return [
        py(
            "prody_resnum_colon",
            f"{load}sel = ag.select('resnum {a}:{b} and name CA and resname CYS'); {count}",
            "exclusive",
            last_int,
        ),
        py(
            "prody_resid_colon",
            f"{load}sel = ag.select('resid {a}:{b} and calpha and resname CYS'); {count}",
            "exclusive",
            last_int,
        ),
        py(
            "prody_chain_resnum_colon",
            f"{load}sel = ag.select('chain A and resnum {a}:{b} and resname CYS and name CA'); "
            f"{count}",
            "exclusive",
            last_int,
        ),
        py(
            "prody_resnum_to",
            f"{load}sel = ag.select('resnum {a} to {b} and name CA and resname CYS'); {count}",
            "inclusive",
            last_int,
        ),
        py(
            "prody_residue_loop",
            f"{load}print(sum(1 for r in ag.getHierView().iterResidues() "
            f"if {a} <= r.getResnum() <= {b} and r.getResname() == 'CYS'))",
            "inclusive",
            last_int,
        ),
        py(
            "mdanalysis_resid_colon",
            "import warnings; warnings.simplefilter('ignore'); import MDAnalysis as mda; "
            "u = mda.Universe('protein.pdb'); "
            f"print(u.select_atoms('resid {a}:{b} and resname CYS and name CA').n_residues)",
            "inclusive",
            last_int,
        ),
        py(
            "gemmi_loop_inclusive",
            "import gemmi; st = gemmi.read_structure('protein.pdb'); "
            f"print(sum(1 for r in st[0]['A'] if {a} <= r.seqid.num <= {b} and r.name == 'CYS'))",
            "inclusive",
            last_int,
        ),
        py(
            "gemmi_loop_exclusive_end",
            "import gemmi; st = gemmi.read_structure('protein.pdb'); "
            f"print(sum(1 for r in st[0]['A'] if {a} <= r.seqid.num < {b} and r.name == 'CYS'))",
            "exclusive",
            last_int,
        ),
        py(
            "biopython_loop",
            "from Bio.PDB import PDBParser; st = PDBParser(QUIET=True).get_structure('x', "
            f"'protein.pdb'); print(sum(1 for r in st[0]['A'] if {a} <= r.id[1] <= {b} "
            "and r.get_resname() == 'CYS'))",
            "inclusive",
            last_int,
        ),
        Tpl(
            "awk_count",
            "awk",
            (
                f'$1=="ATOM" && $3=="CA" && $4=="CYS" && $6>={a} && $6<={b} '
                "{n++} END{print n+0}",
                "protein.pdb",
            ),
            "inclusive",
            last_int,
        ),
        py(
            "prody_residue_names",
            f"{load}print(' '.join(r.getResname() for r in ag.getHierView().iterResidues() "
            f"if {a} <= r.getResnum() <= {b}))",
            None,
            None,
        ),
    ]


@case_source("prody.range_end")
def range_end_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F91",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=f91_templates,
        key="range_end",
        needed=lambda twin, task: "inclusive",
        values=("inclusive", "exclusive"),
    )
