"""Admission cases for pdbtools.selaltloc_digits (family F95) and pdbtools.modified_residues (F96).

Workspaces are the benchmark's own (implicit tier). Every call is run with the real tools and its
output is compared with `task.truth` (bench.score.answer_matches); labels never come from a rule.

pdbtools.selaltloc_digits / F95 (atoms left after keeping one conformer of every atom; the faulted
twin numbers the chain so that the fully split residue is residue 9, the clean twin from 4): the
`pdb_selaltloc` call is the rule's call, the label is the atom count of the real pipeline.
  pdb_selaltloc model.pdb | grep -c '^ATOM'; the same with -A and -B (select by altloc), with a
  `./` path, and as the second stage of `cat model.pdb | pdb_selaltloc`. The faulted twin loses a
  whole residue on all of them. Declaration: verified (right calls and wrong calls alike; a
  declaration is a claim about the caller's next step).

pdbtools.modified_residues / F96 (sequence of a chain with selenomethionine, MSE written as M):
  pdb_tofasta (X), gemmi one_letter_code on residue names (X), Biopython seq1 on names (X),
  Biopython PPBuilder joined (MSE missing), ProDy getSequence (parent letter), a gemmi table
  lookup mapped to the parent letter, Biopython's extended three-to-one table, a hand-written
  dictionary with MSE, an awk mapping from a 20-residue table (X), and a grep of the HETATM
  records (inspection).
  Declaration: parent_letter for right calls; parent_letter and the call's own behaviour for wrong
  ones.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_struct_common import (
    Tpl,
    build_cases,
    last_json,
    py,
)
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call

SEL_KEY = "selaltloc_residues"
SELALTLOC_CALLS = {
    "plain": ("pdb_selaltloc", ("model.pdb",), ()),
    "altloc_A": ("pdb_selaltloc", ("-A", "model.pdb"), ()),
    "altloc_B": ("pdb_selaltloc", ("-B", "model.pdb"), ()),
    "dot_slash": ("pdb_selaltloc", ("./model.pdb",), ()),
    "stdin": ("pdb_selaltloc", (), ("cat", "model.pdb")),
}


def pipeline(argv: tuple[str, ...], context: tuple[str, ...]) -> str:
    head = f"{' '.join(context)} | " if context else ""
    return f"{head}pdb_selaltloc {' '.join(argv)} | grep -c '^ATOM'"


@case_source("pdbtools.selaltloc_digits")
def selaltloc_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F95", seed, twin, work, tools, "implicit", "free")
        for name, (tool, argv, context) in SELALTLOC_CALLS.items():
            proc = tools.run(["bash", "-c", pipeline(argv, context)], work)
            ok = proc.returncode == 0 and answer_matches(task.truth, proc.stdout.strip())
            label: Label = "right" if ok else "wrong"
            call = Call(tool, argv, work, context=context)
            out.append(Case(call, label, False, "F95", seed, twin, name))
            declared = Call(tool, argv, work, {SEL_KEY: "verified"}, context)
            out.append(Case(declared, label, True, "F95", seed, twin, name))
    return out


# -- F96 ---------------------------------------------------------------------------------------

NAMES = (
    "import gemmi, json; st = gemmi.read_structure('model.pdb'); "
    "names = [r.name for r in st[0]['A']]; "
)
F96_TEMPLATES = [
    Tpl(
        "pdb_tofasta",
        "pdb_tofasta",
        ("model.pdb",),
        "X",
        lambda out: ["".join(line for line in out.splitlines() if not line.startswith(">"))],
    ),
    py(
        "gemmi_one_letter_code",
        NAMES + "print(json.dumps([gemmi.one_letter_code(names)]))",
        "X",
        last_json,
    ),
    py(
        "biopython_seq1",
        "from Bio.SeqUtils import seq1; import json; from Bio.PDB import PDBParser; "
        "ch = PDBParser(QUIET=True).get_structure('x', 'model.pdb')[0]['A']; "
        "print(json.dumps([seq1(''.join(r.get_resname() for r in ch))]))",
        "X",
        last_json,
    ),
    py(
        "biopython_ppbuilder",
        "from Bio.PDB import PDBParser, PPBuilder; import json; "
        "st = PDBParser(QUIET=True).get_structure('x', 'model.pdb'); "
        "peps = PPBuilder().build_peptides(st[0]); "
        "print(json.dumps([''.join(str(p.get_sequence()) for p in peps)]))",
        "omitted",
        last_json,
    ),
    py(
        "prody_getSequence",
        "import prody, json; ag = prody.parsePDB('model.pdb'); "
        "print(json.dumps([ag.select('calpha').getSequence()]))",
        "parent_letter",
        last_json,
    ),
    py(
        "gemmi_tabulated_parent",
        NAMES + "print(json.dumps([''.join(gemmi.find_tabulated_residue(n).one_letter_code.upper() "
        "for n in names)]))",
        "parent_letter",
        last_json,
    ),
    py(
        "biopython_extended_table",
        "from Bio.Data.IUPACData import protein_letters_3to1_extended as t; import json; "
        "from Bio.PDB import PDBParser; "
        "ch = PDBParser(QUIET=True).get_structure('x', 'model.pdb')[0]['A']; "
        "print(json.dumps([''.join(t[r.get_resname().capitalize()] for r in ch)]))",
        "parent_letter",
        last_json,
    ),
    py(
        "python_dictionary_with_mse",
        "import json; m = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',"
        "'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',"
        "'THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}; "
        "names = [l[17:20] for l in open('model.pdb') if l[12:16] == ' CA ']; "
        "print(json.dumps([''.join(m[n] for n in names)]))",
        "parent_letter",
        last_json,
    ),
    Tpl(
        "awk_twenty_residue_table",
        "awk",
        (
            'BEGIN{s="ALAARGASNASPCYSGLNGLUGLYHISILELEULYSMETPHEPROSERTHRTRPTYRVAL";'
            'o="ARNDCQEGHILKMFPSTWYV"} $1~/^(ATOM|HETATM)$/ && $3=="CA"{i=index(s,$4); '
            'q=q (i? substr(o,(i+2)/3,1):"X")} END{print "[\\""q"\\"]"}',
            "model.pdb",
        ),
        "X",
        last_json,
    ),
    Tpl("grep_hetatm", "grep", ("HETATM", "model.pdb"), None, None),
]


def f96_templates(task) -> list[Tpl]:
    return F96_TEMPLATES


@case_source("pdbtools.modified_residues")
def modified_residues_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F96",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=f96_templates,
        key="modified_residues",
        needed=lambda twin, task: "parent_letter",
        values=("parent_letter", "X", "omitted"),
    )
