"""F89 chain_ids: ProDy reads the chain IDs of an mmCIF from label_asym_id, not from auth_asym_id.

Source R2-STR-07 (ProDy issue 1985, LigandMPNN pull 81). An mmCIF atom_site holds two chain names:
`label_asym_id` (assigned by the archive, one per entity instance) and `auth_asym_id` (the author's
chain ID, the one in the paper and the PDB file). gemmi and Biopython (`MMCIFParser`) report the
author ID; `prody.parseMMCIF` (and `parsePDB` on a .cif, which falls back to it with a warning)
reports `label_asym_id` unless `unite_chains=True`.

Verified here (ProDy 2.6.1, gemmi 0.7.5, Biopython 1.88) on a two-chain mmCIF whose author chains
are E and I and whose label chains are A and B: gemmi chain names E, I; Biopython E, I;
`prody.parseMMCIF(f)` chain IDs A, B (exit 0, stderr only the "@> 79 atoms ... parsed" line);
`parseMMCIF(f, unite_chains=True)` E, I.

Twins: the same three polymer chains (author A, B, C with different lengths). Clean: label IDs equal
the author IDs. Faulted: the archive labels follow another order (author A is label B, author B is
label C, author C is label A), which is what an entry looks like when the depositor's chain letters
do not follow the entity order. The ProDy default then names every chain after another chain's
residue count.

This module also holds the synthetic-structure helpers the other `struct` families share (NeRF
backbones, PDB and mmCIF writers); keep them stdlib-only.

  explicit  the README says the chain IDs are the author IDs (auth_asym_id)
  implicit  the README says the chains are named as in the PDB entry, with the chain letters the
            authors use in the paper (it names neither auth_asym_id nor label_asym_id)
  forced    the task sentence dictates ProDy
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

# ---------------------------------------------------------------------------------------------
# shared helpers for the struct families (F89-F100): tiny synthetic proteins
# ---------------------------------------------------------------------------------------------

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip
ONE = {v: k for k, v in AA3.items()}
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
LETTERS_NO_G = AA_LETTERS.replace("G", "")

Vec = tuple[float, float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a: Vec) -> Vec:
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / n, a[1] / n, a[2] / n)


def _place(a: Vec, b: Vec, c: Vec, length: float, angle: float, torsion: float) -> Vec:
    """Natural-extension reference frame: the atom bonded to c at `length`, `angle`, `torsion`."""
    bc = _unit(_sub(c, b))
    n = _unit(_cross(_sub(b, a), bc))
    m = _cross(n, bc)
    x, y, z = (
        -length * math.cos(angle),
        length * math.sin(angle) * math.cos(torsion),
        (length * math.sin(angle) * math.sin(torsion)),
    )
    return (
        c[0] + x * bc[0] + y * m[0] + z * n[0],
        c[1] + x * bc[1] + y * m[1] + z * n[1],
        c[2] + x * bc[2] + y * m[2] + z * n[2],
    )


def random_fold(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
    """Phi/psi angles (degrees) of a protein-like chain: helices, strands and short loops."""
    phis: list[float] = []
    psis: list[float] = []
    while len(phis) < n:
        kind = rng.choice("hhel")
        for _ in range(rng.randint(5, 10) if kind != "l" else rng.randint(2, 4)):
            if kind == "h":
                phis.append(-57 + rng.gauss(0, 5))
                psis.append(-47 + rng.gauss(0, 5))
            elif kind == "e":
                phis.append(-120 + rng.gauss(0, 8))
                psis.append(130 + rng.gauss(0, 8))
            else:
                phis.append(rng.choice([-80, -65, 60, -100]) + rng.gauss(0, 12))
                psis.append(rng.choice([140, 0, 30, -30, 120]) + rng.gauss(0, 12))
    return phis[:n], psis[:n]


def backbone(phis: list[float], psis: list[float]) -> list[dict[str, Vec]]:
    """N, CA, C, O (and CB) coordinates of each residue from its phi/psi angles."""
    rad = math.radians
    n0: Vec = (0.0, 0.0, 0.0)
    ca0: Vec = (1.458, 0.0, 0.0)
    c0: Vec = (ca0[0] + 1.525 * math.cos(rad(69.0)), 1.525 * math.sin(rad(69.0)), 0.0)
    out: list[dict[str, Vec]] = []
    prev: tuple[Vec, Vec, Vec] = (n0, ca0, c0)
    for i in range(len(phis)):
        if i == 0:
            n_i, ca_i, c_i = prev
        else:
            n_i = _place(prev[0], prev[1], prev[2], 1.329, rad(116.2), rad(psis[i - 1]))
            ca_i = _place(prev[1], prev[2], n_i, 1.458, rad(121.7), math.pi)
            c_i = _place(prev[2], n_i, ca_i, 1.525, rad(111.0), rad(phis[i]))
        o_i = _place(n_i, ca_i, c_i, 1.231, rad(120.8), rad(psis[i]) + math.pi)
        b, c = _sub(ca_i, n_i), _sub(c_i, ca_i)
        a = _cross(b, c)
        cb = tuple(
            -0.58273431 * a[k] + 0.56802827 * b[k] - 0.54067466 * c[k] + ca_i[k] for k in range(3)
        )
        out.append({"N": n_i, "CA": ca_i, "C": c_i, "O": o_i, "CB": cb})  # type: ignore[dict-item]
        prev = (n_i, ca_i, c_i)
    return out


@dataclass
class Atom:
    name: str
    xyz: Vec
    element: str = ""
    occ: float = 1.0
    bfac: float = 20.0
    altloc: str = ""

    @property
    def elem(self) -> str:
        return self.element or self.name.strip()[0]


@dataclass
class Residue:
    name: str  # three-letter code
    num: int
    atoms: list[Atom]
    icode: str = ""
    het: bool = False  # HETATM record (modified residues, waters, ligands)


@dataclass
class Chain:
    auth: str  # author chain ID (PDB column 22, auth_asym_id)
    residues: list[Residue]
    label: str = ""  # label_asym_id (mmCIF); defaults to the author ID
    entity: int = 1
    polymer: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def make_residues(
    rng: random.Random,
    seq: str,
    start: int = 1,
    shift: Vec = (0.0, 0.0, 0.0),
    with_cb: bool = True,
    fold: tuple[list[float], list[float]] | None = None,
) -> list[Residue]:
    """A synthetic chain: residues numbered from `start`, backbone from a random fold."""
    phis, psis = fold or random_fold(rng, len(seq))
    out: list[Residue] = []
    for i, (letter, xyz) in enumerate(zip(seq, backbone(phis, psis), strict=True)):
        atoms = [
            Atom(n, (xyz[n][0] + shift[0], xyz[n][1] + shift[1], xyz[n][2] + shift[2]))
            for n in ("N", "CA", "C", "O", *(("CB",) if with_cb and letter != "G" else ()))
        ]
        out.append(Residue(AA3[letter], start + i, atoms))
    return out


def random_sequence(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA_LETTERS) for _ in range(n))


def pdb_atom_line(
    serial: int, atom: Atom, res: Residue, chain: str, het: bool | None = None
) -> str:
    name = f" {atom.name:<3}" if len(atom.name) < 4 else atom.name
    record = "HETATM" if (res.het if het is None else het) else "ATOM  "
    x, y, z = atom.xyz
    return (
        f"{record}{serial:5d} {name}{atom.altloc or ' '}{res.name:>3} {chain}{res.num:4d}"
        f"{res.icode or ' '}   {x:8.3f}{y:8.3f}{z:8.3f}{atom.occ:6.2f}{atom.bfac:6.2f}"
        f"          {atom.elem:>2}"
    )


def pdb_text(chains: list[Chain], remarks: list[str] | None = None) -> str:
    """PDB-format text (ATOM/HETATM, a TER after every polymer chain, END)."""
    lines = [f"REMARK {r}" for r in (remarks or [])]
    serial = 1
    for chain in chains:
        for res in chain.residues:
            for atom in res.atoms:
                lines.append(pdb_atom_line(serial, atom, res, chain.auth))
                serial += 1
        if chain.polymer:
            lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def mmcif_text(chains: list[Chain], name: str = "model", poly_scheme: bool = False) -> str:
    """A small mmCIF: entity tables plus an `_atom_site` loop with label_* and auth_* columns.

    With `poly_scheme` it also writes `_entity_poly_seq`, `_struct_asym` and the
    `_pdbx_poly_seq_scheme` residue table that archive files carry (mkdssp reads its residues
    from the latter).
    """
    ents: dict[int, Chain] = {}
    for c in chains:
        ents.setdefault(c.entity, c)
    out = [f"data_{name}", f"_entry.id {name}", "loop_", "_entity.id", "_entity.type"]
    out += [f"{e} {'polymer' if c.polymer else 'non-polymer'}" for e, c in ents.items()]
    out += ["loop_", "_entity_poly.entity_id", "_entity_poly.type"]
    out += [
        "_entity_poly.pdbx_seq_one_letter_code_can",
        "_entity_poly.pdbx_strand_id",
    ]
    for e, c in ents.items():
        if c.polymer:
            letters = "".join(ONE.get(r.name, "X") for r in c.residues)
            out.append(f"{e} polypeptide(L) {letters} {c.auth}")
    if poly_scheme:
        out += ["loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num"]
        out += ["_entity_poly_seq.mon_id", "_entity_poly_seq.hetero"]
        for e, c in ents.items():
            out += [f"{e} {i + 1} {r.name} n" for i, r in enumerate(c.residues) if c.polymer]
        out += ["loop_", "_struct_asym.id", "_struct_asym.entity_id"]
        out += [f"{c.label or c.auth} {c.entity}" for c in chains]
        out += ["loop_"] + [f"_pdbx_poly_seq_scheme.{t}" for t in SCHEME_TAGS]
        for c in chains:
            for i, r in enumerate(c.residues):
                if c.polymer:
                    ins = r.icode or "."
                    out.append(
                        f"{c.label or c.auth} {c.entity} {i + 1} {r.name} {i + 1} {r.num} {r.num} "
                        f"{r.name} {r.name} {c.auth} {ins} n"
                    )
    out += ["loop_"]
    out += [f"_atom_site.{t}" for t in ATOM_SITE_TAGS]
    serial = 1
    for chain in chains:
        label = chain.label or chain.auth
        for i, res in enumerate(chain.residues):
            for a in res.atoms:
                x, y, z = a.xyz
                out.append(
                    f"{'HETATM' if res.het else 'ATOM'} {serial} {a.elem} {a.name} "
                    f"{a.altloc or '.'} {res.name} {label} {chain.entity} "
                    f"{i + 1 if chain.polymer else '.'} {res.icode or '?'} "
                    f"{x:.3f} {y:.3f} {z:.3f} {a.occ:.2f} {a.bfac:.2f} "
                    f"{res.num} {res.name} {chain.auth} {a.name} 1"
                )
                serial += 1
    return "\n".join(out) + "\n"


SCHEME_TAGS = (
    "asym_id entity_id seq_id mon_id ndb_seq_num pdb_seq_num auth_seq_num pdb_mon_id auth_mon_id "
    "pdb_strand_id pdb_ins_code hetero"
).split()
ATOM_SITE_TAGS = (
    "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id "
    "label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy "
    "B_iso_or_equiv auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num"
).split()


def one_letter(res: Residue) -> str:
    return ONE.get(res.name, "X")


TOOLS_NOTE = (
    "Tools: `python-struct` is Python with gemmi, Biopython, ProDy, MDAnalysis, mdtraj and RDKit; "
    "TMalign, mmseqs, foldseek, mkdssp and pdb-tools (pdb_*) are on the PATH."
)


def python_struct(tools: ToolEnv, work: Path, code: str):
    """Run `code` with the named interpreter (user site off, so the conda libraries are used)."""
    return tools.run(["python-struct", "-s", "-c", code], work)


# ---------------------------------------------------------------------------------------------
# F89
# ---------------------------------------------------------------------------------------------

AUTH_IDS = "ABC"
TWIN_FILES = frozenset({"complex.cif", "README.md"})


def _plan(seed: int) -> list[tuple[str, int]]:
    """Author chain IDs with distinct residue counts."""
    rng = random.Random(f"f89-{seed}")
    return list(zip(AUTH_IDS, rng.sample(range(16, 38), 3), strict=True))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f89-struct-{seed}")
    plan = _plan(seed)
    # faulted: the archive's label order is B, C, A for the author chains A, B, C
    label_of = {"A": "A", "B": "B", "C": "C"}
    order = list(AUTH_IDS)
    if twin == "faulted":
        label_of = {"A": "B", "B": "C", "C": "A"}
        order = ["C", "A", "B"]  # file order follows label_asym_id: A (= author C), B, C
    chains: list[Chain] = []
    by_auth = {}
    for k, (auth, n) in enumerate(plan):
        res = make_residues(
            rng, random_sequence(rng, n), start=rng.randint(1, 30), shift=(24.0 * k, 0.0, 0.0)
        )
        by_auth[auth] = (res, n)
    for auth in order:
        res, _ = by_auth[auth]
        label = label_of[auth]
        chains.append(Chain(auth, res, label=label, entity=ord(label) - 64))
    (work / "complex.cif").write_text(mmcif_text(chains, "complex"))
    line = "complex.cif: protein complex, mmCIF from the PDB, three protein chains (no ligands)."
    if tier == "explicit":
        line += " Chains are named with their author chain IDs (auth_asym_id), as in the paper."
    else:
        line += (
            " Chains are named as in the PDB entry, with the chain letters the authors use in "
            "the paper."
        )
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = (
            ("README: chain IDs are the author IDs (auth_asym_id)", "label_asym_id = auth_asym_id")
            if tier == "explicit"
            else ("label_asym_id equals auth_asym_id for every chain",)
        )
    else:
        shown = "label_asym_id differs from auth_asym_id (author A is label B, B is C, C is A)"
        evidence = (
            ("README: chain IDs are the author IDs (auth_asym_id)", shown)
            if tier == "explicit"
            else (shown,)
        )
    prompt = common.prompt_for(
        route,
        "report the number of residues in each chain of complex.cif, using the chain IDs of the "
        "PDB entry",
        "use ProDy (`prody.parseMMCIF`) to report the number of residues in each chain of "
        "complex.cif, using the chain IDs of the PDB entry",
        "a JSON object mapping each chain ID to its number of residues",
    )
    expected = {auth: n for auth, n in plan}
    naive = {label_of[auth]: n for auth, n in plan}
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.0}
    task_id = common.task_id("F89", seed, twin, tier, route)
    return Task(task_id, "F89", "chain_id_scheme", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    import gemmi

    st = gemmi.read_structure(str(work / "complex.cif"))
    counts = {ch.name: len(ch) for ch in st[0]}
    flags = ["label_asym_id differs from auth_asym_id: author IDs used"]
    differs = any(sub.subchain_id() != ch.name for ch in st[0] for sub in ch.subchains())
    return _result(counts, "gemmi (author chain names)", flags if differs else [])


PRODY_COUNT = (
    "import prody, json; ag = prody.parseMMCIF('complex.cif'); "
    "print(json.dumps({c.getChid(): c.numResidues() for c in ag.getHierView().iterChains()}))"
)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    import json

    proc = python_struct(tools, work, PRODY_COUNT)
    counts = json.loads(proc.stdout.strip().splitlines()[-1])
    return _result(counts, "prody.parseMMCIF default chain IDs")


PRODY_CIF = re.compile(
    r"parse(?:MMCIF|CIF)\b|parsePDB\([^)]*\.cif|MMCIF2Dict|label_asym_id|\.subchain"
)
SAFE = re.compile(r"unite_chains\s*=\s*True|auth_asym_id|gemmi|MMCIFParser")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command takes chain IDs of complex.cif from ProDy's default (label ids) or
    from a label_asym_id column, without unite_chains=True / auth_asym_id / gemmi / Biopython in
    the same command (judged per command)."""
    return any("complex.cif" in c and PRODY_CIF.search(c) and not SAFE.search(c) for c in commands)


SPEC = FamilySpec(
    id="F89",
    slot="chain_id_scheme",
    source="R2-STR-07",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
