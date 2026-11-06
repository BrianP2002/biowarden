"""Admission cases for struct.bfactor_scale (family F97: residues with pLDDT above 70).

Workspaces are the benchmark's own (implicit tier): the clean twin writes pLDDT on a 0-100 scale in
the B-factor column, the faulted twin the same values divided by 100. Every call is run with the
real tools (awk, python-struct snippets with ProDy, gemmi, Biopython); the printed count is compared
with `task.truth`; labels never come from the rule.

  a cutoff of 70 on the raw column (awk, plain Python, ProDy `beta > 70`, gemmi, Biopython): right
  on the clean twin, 0 on the faulted twin; a cutoff of 0.7 on the raw column (the other way
  round); a count that rescales first (`* 100` in awk, `max(...) <= 1` auto-detection in Python
  and in ProDy: right on both); a printed minimum and maximum (inspection).
Declarations: the scale of the twin for right calls; the scale of the twin and the scale the call's
cutoff assumes for wrong ones.
"""

from __future__ import annotations

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_struct_common import Tpl, build_cases, last_int, py
from biowarden.bench.tasks import ToolEnv

CA = '$1=="ATOM" && $3=="CA"'
LOAD = "import prody; ag = prody.parsePDB('predicted.pdb'); ca = ag.select('calpha'); "
GEMMI = "import gemmi; st = gemmi.read_structure('predicted.pdb'); "
RAW = "float(l[60:66])"
ROWS = "for l in open('predicted.pdb') if l.startswith('ATOM') and l[12:16] == ' CA '"
BIO = (
    "from Bio.PDB import PDBParser; "
    "st = PDBParser(QUIET=True).get_structure('x', 'predicted.pdb'); "
)


def awk(name: str, test: str, uses: str | None) -> Tpl:
    program = f"{CA} && {test} {{n++}} END{{print n+0}}"
    return Tpl(name, "awk", (program, "predicted.pdb"), uses, last_int)


TEMPLATES = [
    awk("awk_raw_70", "substr($0,61,6)+0>70", "0_100"),
    awk("awk_raw_ge_70", "substr($0,61,6)+0>=70", "0_100"),
    awk("awk_raw_0_7", "substr($0,61,6)+0>0.7", "0_1"),
    awk("awk_rescaled", "substr($0,61,6)*100>70", "0_1"),
    py("python_raw_70", f"print(sum({RAW} > 70 {ROWS}))", "0_100", last_int),
    py("python_raw_0_7", f"print(sum({RAW} > 0.7 {ROWS}))", "0_1", last_int),
    py(
        "python_autodetect",
        f"b = [{RAW} {ROWS}]; s = 100 if max(b) <= 1 else 1; print(sum(x * s > 70 for x in b))",
        None,
        last_int,
    ),
    py(
        "prody_beta_70",
        LOAD + "s = ag.select('calpha and beta > 70'); print(0 if s is None else s.numAtoms())",
        "0_100",
        last_int,
    ),
    py("prody_betas_70", LOAD + "print(int((ca.getBetas() > 70).sum()))", "0_100", last_int),
    py(
        "prody_autodetect",
        LOAD + "b = ca.getBetas(); b = b * 100 if b.max() <= 1 else b; print(int((b > 70).sum()))",
        None,
        last_int,
    ),
    py(
        "gemmi_b_iso_70",
        GEMMI + "print(sum(r['CA'][0].b_iso > 70 for r in st[0]['A']))",
        "0_100",
        last_int,
    ),
    py(
        "gemmi_b_iso_0_7",
        GEMMI + "print(sum(r['CA'][0].b_iso > 0.7 for r in st[0]['A']))",
        "0_1",
        last_int,
    ),
    py(
        "biopython_bfactor_70",
        BIO + "print(sum(r['CA'].get_bfactor() > 70 for r in st[0]['A']))",
        "0_100",
        last_int,
    ),
    py(
        "biopython_scale_by_max",
        BIO + "b = [r['CA'].get_bfactor() for r in st[0]['A']]; k = 100 if max(b) <= 1 else 1; "
        "print(sum(x * k > 70 for x in b))",
        None,
        last_int,
    ),
    Tpl(
        "awk_min_max",
        "awk",
        (
            f'{CA} {{b=substr($0,61,6)+0; if(min==""||b<min)min=b; if(b>max)max=b}} '
            "END{print min, max}",
            "predicted.pdb",
        ),
        None,
        None,
    ),
]


def templates(task) -> list[Tpl]:
    return TEMPLATES


@case_source("struct.bfactor_scale")
def bfactor_scale_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F97",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=templates,
        key="bfactor_scale",
        needed=lambda twin, task: "0_100" if twin == "clean" else "0_1",
        values=("0_1", "0_100"),
    )
