"""Admission cases for structure.chain_order (incident T01, slot chain_order, backend mockfold).

There is no benchmark family for the mock predictor, so each seed builds its own workspace from the
mock fixtures: a few JSON request files (entities with tFold role codes B, A, D, G, M, N, P; the
sequences are synthetic variants of the ones in tests/helpers.py ROLES, with point mutations and
different lengths per seed) and `mockfold predict` calls over them. The calls vary the backend
(tfoldlike, boltzlike), the entity subset (alpha/beta or gamma/delta, a request without beta2m under
`--fault silent_missing_n`), the order of the entities (tool order, swapped TCR chains, reversed, a
random permutation, pMHC first, M and N swapped), the spelling of the options (`--opt v`,
`--opt=v`, abbreviations such as `--back`/`--req`/`--out`, any option order, `./req.json`, an
absolute path), a repeated `--request` (argparse keeps the last one) and four requests the tool
refuses itself (no beta2m, an unknown role code, both TCR pairs, a missing request file).

LABELS come from running mockfold on every call (the same `python -I -m ...mockfold` module the
console script wraps, in the workspace) and looking at the PDB it wrote; the rule is never
consulted. The label depends on how the caller pairs output chains with request entities, which is
what the declaration states:
  undeclared / by_request_position  chain k must hold request entity k (the oracle
      `chain_sequence_roundtrip`, used here only as a labelling tool: chain ID and sequence);
  by_role  the chain whose ID is the entity's role code must hold that entity's sequence, for every
      entity and with no other chain (checked directly on the PDB).
A call the tool refuses (exit 2, nothing written) is labelled right under every declaration: it
produces no result that could be silently wrong. Each call is emitted without a declaration and with
chain_pairing=by_request_position; tfoldlike calls also with chain_pairing=by_role (the value is
meaningless for boltzlike, whose chains are named A, B, C, ...). So a permuted request is wrong
when undeclared or declared by_request_position and right when declared by_role: the declared
convention is what makes the pairing correct, and whether the caller then really pairs by role is
beyond what any rule can see.

NOT generated: `--fault dg_overwrites_ba` (it drops a chain pair, a chain-count problem for another
slot), and `python -m ...mockfold` spellings (the rule applies to the `mockfold` console script).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.tasks.tools import ToolEnv
from biowarden.contracts import Entity, Request, load_packaged
from biowarden.engine import Call
from biowarden.oracles.invariants import OracleContext, chain_sequence_roundtrip
from biowarden.oracles.structure import OutputError, parse_output

RULE = "structure.chain_order"
KEY = "chain_pairing"
BY_POSITION, BY_ROLE = "by_request_position", "by_role"
MODULE = "biowarden.bench.fixtures.mockfold"
FAMILY = "T01"

# role code -> (label, template sequence); public sequences, see tests/helpers.py
ROLES = {
    "B": ("tcr_beta", "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEY"),
    "A": ("tcr_alpha", "KEVEQNSGPLSVPEGAIASLNCTYSDRGSQS"),
    "D": ("tcr_delta", "AQSVTQPDARVTVSEGASLQLRCKYSYSATP"),
    "G": ("tcr_gamma", "QKVQQSPESLIVPEGGTATLTCNYSDSHG"),
    "M": ("mhc", "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSD"),
    "N": ("b2m", "IQRTPKIQVYSRHPAENGKSNFLNCYVSGF"),
    "P": ("peptide", "GILGFVFTL"),
}
AMINO = "ACDEFGHIKLMNPQRSTVWY"
# (backend option, request option, out-dir option, fault option): spellings argparse accepts
SPELLINGS = (
    ("--backend", "--request", "--out-dir", "--fault"),
    ("--back", "--req", "--out", "--fau"),
    ("--b", "--r", "--o", "--f"),
)


def vary(rng: random.Random, base: str) -> str:
    """A synthetic sequence close to the template: point mutations, a shorter or longer end."""
    seq = list(base)
    for _ in range(rng.randint(0, 3)):
        i = rng.randrange(len(seq))
        seq[i] = rng.choice(AMINO.replace(seq[i], ""))
    if rng.random() < 0.3 and len(seq) > 9:
        del seq[len(seq) - rng.randint(1, 3) :]
    if rng.random() < 0.3:
        seq += rng.choices(AMINO, k=rng.randint(1, 4))
    return "".join(seq)


@dataclass
class Scenario:
    name: str
    backend: str
    requests: list[str]  # file names given to --request, in call order (the last one counts)
    out_dir: str
    rows: list[dict[str, str]]  # the entities of the last request file
    faults: tuple[str, ...] = ()
    loud: bool = False  # mockfold refuses the request itself (exit 2, nothing written)
    pairings: tuple[str | None, ...] = (None, BY_POSITION)


class Builder:
    """Writes the request files of one seed and collects its scenarios."""

    def __init__(self, rng: random.Random, work: Path) -> None:
        self.rng, self.work = rng, work
        self.seqs = {role: vary(rng, template) for role, (_, template) in ROLES.items()}
        self.scenarios: list[Scenario] = []

    def rows(self, roles: str, bad_role: str | None = None, roleless: bool = False):
        rows = []
        for i, role in enumerate(roles):
            row = {"role": bad_role if (bad_role is not None and i == 1) else role}
            if self.rng.random() < 0.7:
                row["label"] = ROLES[role][0]
            row["sequence"] = self.seqs[role]
            if roleless:
                del row["role"]
            if self.rng.random() < 0.3:  # key order in the file does not matter to the tool
                row = dict(reversed(list(row.items())))
            rows.append(row)
        return rows

    def write(self, name: str, rows: list[dict[str, str]]) -> str:
        body: dict[str, object] = {"entities": rows}
        if self.rng.random() < 0.3:
            body = {"name": f"complex-{self.rng.randrange(1000)}", **body}
        text = json.dumps(body, indent=self.rng.choice((None, 1)))
        (self.work / name).write_text(text + "\n")
        return name

    def add(
        self,
        name: str,
        backend: str,
        roles: str,
        *,
        faults: tuple[str, ...] = (),
        before: tuple[str, ...] = (),
        loud: bool = False,
        bad_role: str | None = None,
        roleless: bool = False,
        write: bool = True,
    ) -> str:
        """Add a scenario; `before` are request files named earlier in the same call."""
        rows = self.rows(roles, bad_role, roleless)
        file = f"req_{name}.json"
        if write:
            self.write(file, rows)
        pairings = (None, BY_POSITION, BY_ROLE) if backend == "tfoldlike" else (None, BY_POSITION)
        self.scenarios.append(
            Scenario(name, backend, [*before, file], f"out_{name}", rows, faults, loud, pairings)
        )
        return file

    def argv(self, scenario: Scenario) -> tuple[str, ...]:
        names = self.rng.choice(SPELLINGS)
        attached = self.rng.random() < 0.3
        form = self.rng.choice(("name", "dot", "abs"))

        def path(file: str) -> str:
            return {"name": file, "dot": f"./{file}", "abs": str(self.work / file)}[form]

        def option(opt: str, value: str) -> list[str]:
            return [f"{opt}={value}"] if attached else [opt, value]

        parts = {
            "backend": option(names[0], scenario.backend),
            "request": [t for f in scenario.requests for t in option(names[1], path(f))],
            "out": option(names[2], scenario.out_dir),
        }
        argv = ["predict"]
        for key in self.rng.sample(sorted(parts), 3):
            argv += parts[key]
        for fault in scenario.faults:
            argv += option(names[3], fault)
        return tuple(argv)


def build_scenarios(rng: random.Random, work: Path) -> Builder:
    b = Builder(rng, work)
    tcr = "DG" if rng.random() < 0.4 else "BA"
    tool = tcr + "MNP"  # the order tfoldlike assembles in
    shuffled = "".join(rng.sample(tool, len(tool)))
    while shuffled == tool:
        shuffled = "".join(rng.sample(tool, len(tool)))
    wrong = {
        "swap_tcr": tcr[::-1] + "MNP",
        "reversed": tool[::-1],
        "shuffled": shuffled,
        "pmhc_first": rng.choice(("MNP", "PMN", "NMP")) + tcr,
        "swap_mn": tcr + "NMP",
    }
    # tfoldlike: the request already in the order the backend assembles, and the permuted ones
    b.add("tfold_in_order", "tfoldlike", tool)
    for name, roles in wrong.items():
        b.add(f"tfold_{name}", "tfoldlike", roles)
    if rng.random() < 0.5:  # a request without beta2m, which only the silent_missing_n fault takes
        b.add("tfold_no_n_in_order", "tfoldlike", tcr + "MP", faults=("silent_missing_n",))
        b.add("tfold_no_n_swapped", "tfoldlike", "MP" + tcr[::-1], faults=("silent_missing_n",))
    # a repeated --request: argparse keeps the last one, the first is a decoy
    decoy_ok = b.write("decoy_ok.json", b.rows(tool))
    decoy_bad = b.write("decoy_bad.json", b.rows(wrong["shuffled"]))
    b.add("tfold_last_request_bad", "tfoldlike", wrong["swap_tcr"], before=(decoy_ok,))
    b.add("tfold_last_request_ok", "tfoldlike", tool, before=(decoy_bad,))
    # boltzlike keeps the request order whatever it is, and ignores the roles
    b.add("boltz_in_order", "boltzlike", tool)
    b.add("boltz_shuffled", "boltzlike", shuffled)
    b.add("boltz_swap_tcr", "boltzlike", wrong["swap_tcr"])
    b.add("boltz_no_roles", "boltzlike", "".join(rng.sample("MNP", 3)), roleless=True)
    # requests the tool refuses itself
    b.add("tfold_no_n", "tfoldlike", "".join(rng.sample(tcr + "MP", 4)), loud=True)
    b.add(
        "tfold_bad_role",
        "tfoldlike",
        wrong["swap_tcr"],
        bad_role=rng.choice(("X", "b", "")),
        loud=True,
    )
    b.add("tfold_both_pairs", "tfoldlike", "".join(rng.sample("BADGMNP", 7)), loud=True)
    b.add("tfold_missing_file", "tfoldlike", tool, loud=True, write=False)
    return b


def _run(work: Path, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-I", "-m", MODULE, *argv]
    return subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)


def _request(rows: list[dict[str, str]]) -> Request:
    return Request(
        entities=[
            Entity(
                role=row.get("role", ""),
                label=row.get("label") or f"e{i}",
                sequence=row["sequence"],
            )
            for i, row in enumerate(rows)
        ]
    )


def positional_pairing_holds(backend: str, rows: list[dict[str, str]], out: Path) -> bool:
    """Does output chain k hold request entity k (chain ID and sequence)? Via the v0 oracle."""
    try:
        parsed = parse_output(out / "complex.pdb", out / "confidence.json")
    except OutputError:
        return False
    contract = load_packaged(f"mockfold_{backend}")
    findings = chain_sequence_roundtrip(OracleContext(contract, _request(rows), parsed))
    return not findings


def role_pairing_holds(rows: list[dict[str, str]], out: Path) -> bool:
    """Does the chain named by each entity's role hold that entity, with no other chain?"""
    try:
        parsed = parse_output(out / "complex.pdb", out / "confidence.json")
    except OutputError:
        return False
    by_id = {chain.chain_id: chain.sequence for chain in parsed.chains}
    return len(parsed.chains) == len(rows) and all(
        by_id.get(row.get("role", "")) == row["sequence"] for row in rows
    )


def label_of(scenario: Scenario, proc, work: Path, pairing: str | None) -> Label:
    if scenario.loud:
        return "right"  # a loud failure produces nothing that could be silently wrong
    out = work / scenario.out_dir
    if pairing == BY_ROLE:
        holds = role_pairing_holds(scenario.rows, out)
    else:
        holds = positional_pairing_holds(scenario.backend, scenario.rows, out)
    return "right" if holds else "wrong"


@case_source(RULE)
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "work"
    work.mkdir(parents=True, exist_ok=True)
    builder = build_scenarios(random.Random(f"{RULE}:{seed}"), work)
    argvs = [builder.argv(s) for s in builder.scenarios]
    with ThreadPoolExecutor(max_workers=16) as pool:
        procs = list(pool.map(lambda argv: _run(work, argv), argvs))
    out: list[Case] = []
    for scenario, argv, proc in zip(builder.scenarios, argvs, procs, strict=True):
        if (proc.returncode != 0) != scenario.loud:  # the scenario does not do what it claims
            raise RuntimeError(f"{scenario.name}: exit {proc.returncode} {proc.stderr.strip()}")
        for pairing in scenario.pairings:
            label = label_of(scenario, proc, work, pairing)
            declared = {} if pairing is None else {KEY: pairing}
            out.append(
                Case(
                    Call("mockfold", argv, work, declared),
                    label,
                    pairing is not None,
                    FAMILY,
                    seed,
                    scenario.backend,
                    f"{scenario.name}:{pairing or 'undeclared'}",
                )
            )
    return out
