"""Rule structure.chain_order (slot chain_order, incident T01): mockfold's chain assembly order.

The trap (docs/research/toolset.md T01; tFold-TCR `_build_inputs`). Structure predictors differ in
how they order and name the chains of a complex. `mockfold predict --backend tfoldlike` assembles
B:A + M:N + P (D:G + M:N + P for gamma/delta) in that fixed order whatever the order of the request
and names the output chains by role code; `--backend boltzlike` keeps the request order and names
the chains A, B, C, ... Downstream code (scorers, ipSAE, DockQ, the agent's own loop) pairs output
chain k with request entity k, as it does for every other predictor. Hand tfoldlike the entities in
another order and it exits 0 with a valid PDB whose chains are silently mispaired.

Verified here (mockfold 0.1.0, `mockfold predict ...` run in a temp dir, request entities with the
roles of tests/helpers.py ROLES):
  --backend tfoldlike, request roles A,B,M,N,P -> exit 0, stderr empty, output chains B,A,M,N,P
      (chain 0 holds entity 1's sequence); request P,N,M,G,D -> chains D,G,M,N,P.
  --backend boltzlike, the same requests        -> chains A,B,C,D,E (request order kept); roles may
      be absent from the entities.
  Everything that is not a silent reorder is loud (exit 2, nothing written; a JSON list instead of
  an object even ends in a traceback and exit 1): a role outside
  B,A,D,G,M,N,P ("unknown role code 'b'", the codes are case-sensitive, an empty one too), a
  duplicate role, B without A or D without G, B,A together with D,G ("alpha/beta and gamma/delta
  chains cannot be combined"), M or P missing, N missing ("role N (beta2m) is missing"), a request
  file that is missing, not JSON or without an 'entities' list, an unknown or absent --backend.
  `--fault silent_missing_n` lets a request without N through (chains B,A,M,P); `--fault
  dg_overwrites_ba` lets a request holding both pairs through and writes only D,G,M,N,P.
  argparse accepts any unambiguous prefix of a long option: `--back tfoldlike --req r.json --out o`
  and `--b=tfoldlike --r=r.json --o o` behave like the full spellings; a repeated option keeps the
  LAST value (`--backend boltzlike --backend tfoldlike` ran tfoldlike; `--request five.json
  --request two.json` on boltzlike wrote chains A,B, the two entities of the last file).

What the rule does. It applies to `mockfold predict`. The backend and the request file are read
from the call (`--backend`, `--request`, `--fault`: `--opt v`, `--opt=v`, every abbreviation); the
order the backend assembles comes from the v0 contract data (contracts/data/mockfold_<backend>.yaml,
loaded with the plain contracts loader, not through validators/precall.py): `chain_order.order`,
`required`, `exactly_one_of`. A request that follows it is exactly one whose roles, as listed, equal
the roles sorted by that order (the v0 `order_mismatch` test).
  no --backend, no --request, -h/--help (mockfold stops or prints help itself)   -> pass
  a --backend without a v0 contract (mockfold itself refuses it today)           -> unknown
  backend keeps the request order (boltzlike: tool_policy input_order)  -> pass, file not read
  declared chain_pairing=by_role (the caller pairs by role code: safe)  -> pass, file not read
  request file missing, unreadable or without an entities list          -> unknown (it may be
      written by an earlier command of the same line; the rule cannot judge it)
  roles outside the backend, duplicated, a pair or a required role missing, both pairs (the tool
      stops with an error itself; not a silent failure)                  -> pass
  both pairs under --fault dg_overwrites_ba (the tool drops B,A: a record-count problem, the
      order says nothing about it)                                       -> unknown
  roles already listed in the backend's order                           -> pass
  roles listed in another order, no declaration                         -> fail (order_mismatch)
  roles in another order, declared chain_pairing=by_request_position     -> fail (declared_mismatch)
The fix text comes first and is the reordered entity list (as the v0 `proposed_fix`), then the
by_role declaration for callers that pair by role code. The call is never rewritten. Declaring
by_role is a claim about the caller's code, which the rule cannot verify.

Not seen: a request produced by the same command line before the call (unknown), a path with shell
variables or globs (unknown), `python -m biowarden.bench.fixtures.mockfold predict ...` (the
tool name is then `python`; the rule applies to the `mockfold` console script only), and any
request that is not a JSON file. The rule models the fault-free tool except for the two documented
faults above.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from functools import lru_cache

from biowarden.contracts.loader import PACKAGED, load_packaged
from biowarden.contracts.schema import ChainOrderSlot
from biowarden.engine.argv import has_option, option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed

BY_POSITION, BY_ROLE = "by_request_position", "by_role"
FAULT_DROPS_PAIR = "dg_overwrites_ba"  # a request with B,A and D,G is assembled from D:G alone
FAULT_ALLOWS_NO_N = "silent_missing_n"  # a request without N builds the complex without it


def spellings(option: str) -> list[str]:
    """Every spelling argparse accepts for a long option: the prefixes with a letter after `--`.

    The options of `mockfold predict` (--backend, --request, --out-dir, --fault, --help) start with
    different letters, so every prefix is unambiguous.
    """
    return [option[:n] for n in range(3, len(option) + 1)]


def last_value(argv: tuple[str, ...], option: str) -> str | None:
    """argparse keeps the last value of a repeated option."""
    values = option_values(argv, *spellings(option))
    return values[-1] if values else None


@lru_cache(maxsize=16)
def backend_slot(backend: str) -> ChainOrderSlot | None:
    """The chain_order slot of the v0 contract for this backend, None if there is no contract."""
    name = f"mockfold_{backend}"
    return load_packaged(name).slots.chain_order if name in PACKAGED else None


class Unreadable(Exception):
    """The request file cannot be turned into a list of entities."""


@dataclass(frozen=True)
class Entry:
    key: object  # the role code as written (any JSON value)
    label: str


def read_entries(ctx: Ctx, path: str, key: str) -> list[Entry]:
    """The entities of the request file, in file order."""
    try:
        text = "\n".join(ctx.lines(path))
    except OSError as exc:
        raise Unreadable(exc.strerror or type(exc).__name__) from exc
    except (EOFError, zlib.error) as exc:
        raise Unreadable(f"corrupt gzip: {exc}") from exc
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise Unreadable(f"not JSON: {exc}") from exc
    entities = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(entities, list) or not entities:
        raise Unreadable("no non-empty 'entities' list")
    entries = []
    for item in entities:
        row = item if isinstance(item, dict) else {}
        label = row.get("label")
        entries.append(Entry(row.get(key), label if isinstance(label, str) else ""))
    return entries


def structural_problem(slot: ChainOrderSlot, keys: list[object], faults: set[str]) -> str | None:
    """Why the tool itself refuses these role codes (mirrors v0 `_check_declared_by_tool`)."""
    if any(not isinstance(k, str) or k not in slot.order for k in keys):
        return f"a {slot.key} code outside {','.join(slot.order)}"
    present = [str(k) for k in keys]
    if len(set(present)) != len(present):
        return f"a repeated {slot.key} code"
    required = [k for k in slot.required if not (k == "N" and FAULT_ALLOWS_NO_N in faults)]
    if missing := [k for k in required if k not in present]:
        return f"required {slot.key} code(s) {','.join(missing)} missing"
    groups = [g for g in slot.exactly_one_of if any(k in present for k in g)]
    if any(not all(k in present for k in g) for g in groups):
        return "a pair given incomplete"
    if len(groups) != 1 and not (len(groups) == 2 and FAULT_DROPS_PAIR in faults):
        return f"{len(groups)} of the groups {slot.exactly_one_of} present"
    return None


def _described(roles: list[str], labels: dict[str, str]) -> str:
    return ", ".join(f"{r} ({labels[r]})" if labels.get(r) else r for r in roles)


@checker("structure_chain_order")
def chain_order(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    argv, key = call.argv, rule.declare.key
    if has_option(argv, "-h", *spellings("--help")):
        return passed("help is printed, nothing is predicted")
    backend = last_value(argv, "--backend")
    if backend is None:
        return passed("no --backend: mockfold stops with a usage error")
    slot = backend_slot(backend)
    if slot is None:
        return CheckResult(
            "unknown", "backend_unknown", f"no chain-order contract for --backend {backend}"
        )
    if slot.tool_policy == "input_order":
        return passed(f"{backend} keeps the request order")
    assert slot.key is not None
    declared = call.declared.get(key)
    if declared == BY_ROLE:
        return passed(f"{key}={BY_ROLE}: chains are paired by role code")
    path = last_value(argv, "--request")
    if path is None:
        return passed("no --request: mockfold stops with a usage error")
    by_role = f"re-run with --declare {key}={BY_ROLE} if your code pairs chains by role code"
    try:
        entries = read_entries(ctx, path, slot.key)
    except Unreadable as exc:
        return CheckResult(
            "unknown",
            "request_unreadable",
            f"cannot read the entities of {path} ({exc}), so their order is unverified",
            fix=(
                f"list its entities in the order {','.join(slot.order)} (the roles you use), or",
                by_role,
            ),
        )
    faults = set(option_values(argv, *spellings("--fault")))
    if problem := structural_problem(slot, [e.key for e in entries], faults):
        return passed(f"{backend} refuses {path} itself ({problem})")
    listed = [str(e.key) for e in entries]
    both_pairs = sum(any(k in listed for k in g) for g in slot.exactly_one_of) == 2
    if both_pairs and FAULT_DROPS_PAIR in faults:
        return CheckResult(
            "unknown",
            "pair_dropped",
            f"--fault {FAULT_DROPS_PAIR} makes {backend} drop one pair of {path}; that is a "
            "chain-count problem, the chain order does not decide it",
        )
    assembled = sorted(listed, key=slot.order.index)
    if listed == assembled:
        return passed(f"{path} already lists the roles in the {backend} order")
    labels = {str(e.key): e.label for e in entries}
    moved = sum(a != b for a, b in zip(listed, assembled, strict=True))
    fact = (
        f"{backend} assembles chains in the fixed {slot.key} order {','.join(assembled)} "
        f"whatever the request order, while {path} lists {','.join(listed)}, so output chain k is "
        f"not request entity k at {moved} of {len(listed)} positions and code that pairs them by "
        "position mispairs those entities"
    )
    fix = (f"reorder the entities of {path} to {_described(assembled, labels)}; or", by_role)
    if declared == BY_POSITION:
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}={BY_POSITION}, but {fact}",
            expected=assembled,
            got=listed,
            fix=fix,
        )
    return CheckResult("fail", "order_mismatch", fact, expected=assembled, got=listed, fix=fix)
