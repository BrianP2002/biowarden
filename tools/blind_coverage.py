"""Replay silent-error cases written without sight of the rules, and count what the layer catches.

    python tools/blind_coverage.py verify CASES --out FILE                  # ground truth, no layer
    python tools/blind_coverage.py replay CASES --verified FILE --out FILE  # again, under the layer
    python tools/blind_coverage.py accept REVIEWS --out FILE                # their decisions
    python tools/blind_coverage.py summary FILE [--accepted FILE]           # the tables

A case (`CASES/<id>.json`) is a pair of commands on the same tiny input: `faulted`, which a hurried
analyst or an agent would plausibly write and which gives a wrong result without any sign of it,
and `clean`, which gives the right one. `verify` runs both for real, twice, in a fresh scratch
directory each time (the files of the case, then its setup lines, then the command, then the
`observe` line that prints the decisive result) and keeps the cases where both exit 0 and the two
observations differ and repeat. `replay` runs the verified cases again with the command under
`biowarden` in its default mode, where every rule blocks, and records what the layer did.

A block is either a finding (a rule read the call or its files and found the error it knows) or a
request (the call lacks a `--declare` the rule asks for on every call in its scope). For each case
the layer's answer to the faulted command is one of

    caught       a finding, and the clean command has none
    both         a finding, and the clean command has one too: the rule does not tell them apart
    declaration  only requests: the call is held up without being told what is wrong
    warned       only warned
    missed       nothing

A finding on the clean command is a false block; a request on it is noted separately. Whether a
finding names the error of the case is a judgement of its own; `summary` takes it from a file when
there is one. Every run happens under `bwrap` (the file system read-only but the scratch
directory, no network) unless `--no-sandbox` is given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ID = re.compile(r"^[a-z]+-\d{2,3}$")
NAME = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$")
FIELDS = "id domain tool trap_class intent silent_error files setup faulted clean observe".split()
MAX_COMMAND = 2000
MAX_FILE = 50_000
BLOCKED = 97  # the exit status of a blocked call
ASKS = ("declaration_missing", "declaration_invalid")  # codes of a request, not a finding
LOADER_NOISE = re.compile(r"no version information available")


def load_scan() -> ModuleType:
    """tools/real_code_scan.py, which has the sandboxed runner both scripts use."""
    spec = importlib.util.spec_from_file_location("real_code_scan", HERE / "real_code_scan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["real_code_scan"] = module
    spec.loader.exec_module(module)
    return module


# ---- the cases --------------------------------------------------------------------------------


def check_case(case: dict, stem: str) -> list[str]:
    """What is wrong with a case file; empty when it can be run."""
    problems = [f"missing field {f}" for f in FIELDS if f not in case]
    if problems:
        return problems
    if not ID.match(case["id"]) or case["id"] != stem:
        problems.append("id is not the file name")
    for field in ("domain", "tool", "trap_class", "intent", "silent_error", "observe"):
        if not isinstance(case[field], str) or not case[field].strip():
            problems.append(f"{field} is empty")
    for field in ("faulted", "clean"):
        text = case[field]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_COMMAND:
            problems.append(f"{field} is not one command of up to {MAX_COMMAND} characters")
    if case["faulted"] == case["clean"]:
        problems.append("faulted and clean are the same command")
    files = case["files"]
    if not isinstance(files, dict) or len(files) > 20:
        problems.append("files is not a dict of up to 20 entries")
    else:
        for name, text in files.items():
            if not NAME.match(name) or ".." in name:
                problems.append(f"file name {name!r} is not a plain relative name")
            if not isinstance(text, str) or len(text) > MAX_FILE:
                problems.append(f"file {name!r} is not text of up to {MAX_FILE} characters")
    setup = case["setup"]
    if not isinstance(setup, list) or not all(isinstance(s, str) for s in setup) or len(setup) > 20:
        problems.append("setup is not a list of up to 20 shell lines")
    return problems


def load_cases(directory: Path) -> tuple[list[dict], dict[str, list[str]]]:
    cases, rejected = [], {}
    for path in sorted(directory.glob("*.json")):
        try:
            case = json.loads(path.read_text())
        except ValueError as exc:
            rejected[path.stem] = [f"not JSON: {exc}"]
            continue
        problems = check_case(case, path.stem) if isinstance(case, dict) else ["not an object"]
        if problems:
            rejected[path.stem] = problems
        else:
            cases.append(case)
    return cases, rejected


# ---- one variant ------------------------------------------------------------------------------


class Variant:
    """Runs one command of a case in a fresh scratch directory, alone or under the guard."""

    def __init__(self, scan: ModuleType, env: dict[str, str], bwrap: str | None) -> None:
        self.scan, self.env, self.bwrap = scan, env, bwrap

    def run(self, case: dict, which: str, guarded: bool) -> dict:
        with tempfile.TemporaryDirectory(prefix="biowarden-case-") as scratch:
            work = Path(scratch) / "ws"
            work.mkdir()
            for name, text in case["files"].items():
                target = work / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)

            def go(command: str, guard: bool = False):
                return self.scan.run_command(
                    ["bash", "-o", "pipefail", "-c", command],
                    workdir=work,
                    guarded=guard,
                    env=self.env,
                    bwrap=self.bwrap,
                    extra_env={},
                )

            setup = go("set -e\n" + "\n".join(case["setup"])) if case["setup"] else None
            if setup is not None and setup.code != 0:
                return {"setup_failed": True, "stderr": tail(setup.stderr)}
            main = go(case[which], guard=guarded)
            seen = go(case["observe"])
        stderr = "\n".join(
            line for line in main.stderr.splitlines() if not LOADER_NOISE.search(line)
        )
        return {
            "exit": main.code,
            "stderr": tail(stderr),
            "observed": seen.stdout.strip(),
            "observe_exit": seen.code,
            "seconds": round(main.seconds, 2),
            "ledger": main.ledger,
        }


def tail(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


# ---- verify -----------------------------------------------------------------------------------


def verify_case(runner: Variant, case: dict) -> dict:
    """Run both commands twice without the layer and say whether the case is a silent error."""
    runs = {w: [runner.run(case, w, False) for _ in range(2)] for w in ("faulted", "clean")}
    entry: dict = {"id": case["id"], "domain": case["domain"], "tool": case["tool"]}
    reasons = []
    for which, pair in runs.items():
        if any(r.get("setup_failed") for r in pair):
            reasons.append(f"setup failed ({which})")
        elif any(r["exit"] != 0 for r in pair):
            reasons.append(f"{which} does not exit 0")
        elif pair[0]["observed"] != pair[1]["observed"]:
            reasons.append(f"{which}: observe differs between two runs")
        elif not pair[0]["observed"]:
            reasons.append(f"{which}: observe prints nothing")
    if not reasons and runs["faulted"][0]["observed"] == runs["clean"][0]["observed"]:
        reasons.append("faulted and clean give the same observation")
    entry["verified"] = not reasons
    entry["reasons"] = reasons
    if not any(r.get("setup_failed") for pair in runs.values() for r in pair):
        entry["observed"] = {w: pair[0]["observed"][:2000] for w, pair in runs.items()}
        entry["stderr"] = {w: pair[0]["stderr"] for w, pair in runs.items()}
        entry["tool_speaks"] = bool(runs["faulted"][0]["stderr"].strip())
    return entry


# ---- replay -----------------------------------------------------------------------------------


def decide(run: dict) -> dict:
    """What the layer did with one guarded run, from its ledger and exit status."""
    rows = run.get("ledger", [])
    actions = {row["action"] for row in rows}
    fired = sorted(
        {
            (o["rule"], o["code"], o["effective"])
            for row in rows
            for o in row["outcomes"]
            if o["effective"] != "none"
        }
    )
    blocking = [f for f in fired if f[2] == "block"]
    if "block" in actions or run.get("exit") == BLOCKED:
        verdict = "block"
    elif "warn" in actions:
        verdict = "warn"
    else:
        verdict = "allow"
    return {
        "decision": verdict,
        "findings": [list(f) for f in blocking if f[1] not in ASKS],
        "requests": [list(f) for f in blocking if f[1] in ASKS],
        "warnings": [list(f) for f in fired if f[2] == "warn"],
        "tools_seen": sorted({row["tool"] for row in rows}),
        "calls_checked": len(rows),
        "message": run["stderr"] if verdict == "block" else "",
        "exit": run.get("exit"),
        "observed": run.get("observed", ""),
    }


def replay_case(runner: Variant, case: dict, truth: dict) -> dict:
    entry = {"id": case["id"], "domain": case["domain"], "tool": case["tool"]}
    for which in ("faulted", "clean"):
        guarded = decide(runner.run(case, which, True))
        guarded["same_result_as_unguarded"] = (
            guarded["decision"] != "block"
            and guarded["observed"][:2000] == truth["observed"][which]
        )
        entry[which] = guarded
    faulted, clean = entry["faulted"], entry["clean"]
    if faulted["findings"]:
        entry["outcome"] = "both" if clean["findings"] else "caught"
    elif faulted["decision"] == "block":
        entry["outcome"] = "declaration"
    else:
        entry["outcome"] = "warned" if faulted["decision"] == "warn" else "missed"
    entry["false_block"] = bool(clean["findings"])
    entry["clean_asked"] = bool(clean["requests"]) and not clean["findings"]
    return entry


# ---- the reviewers ----------------------------------------------------------------------------


def accepted_ids(reviews: Sequence[dict]) -> list[str]:
    """The cases every reviewer question was answered yes for, one per group of duplicates.

    `reviews` is a list of review records ({id, faulted_wrong, clean_right, plausible, silent,
    duplicate_of, verdict}). A case whose verdict is accept needs the four answers to be true; of
    accepted cases that name each other as duplicates only the first id is kept."""
    ok = {
        r["id"]: r
        for r in reviews
        if r["verdict"] == "accept"
        and all(r[k] for k in ("faulted_wrong", "clean_right", "plausible", "silent"))
    }
    parent = {i: i for i in ok}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, r in ok.items():
        other = r.get("duplicate_of") or ""
        if other in ok:
            parent[find(max(i, other))] = find(min(i, other))
    groups: dict[str, list[str]] = {}
    for i in ok:
        groups.setdefault(find(i), []).append(i)
    return sorted(min(members) for members in groups.values())


# ---- the tables -------------------------------------------------------------------------------

OUTCOMES = ("caught", "both", "declaration", "warned", "missed")


def wilson_text(k: int, n: int) -> str:
    from biowarden.bench.score import wilson

    low, high = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * low:.0f}-{100 * high:.0f}%)" if n else "0/0"


def summary(
    replay: dict,
    accepted: set[str] | None = None,
    on_target: dict | None = None,
    families: dict | None = None,
    clean: dict | None = None,
) -> str:
    """The outcome table; `families` maps a case id to the benchmark family whose trap it repeats
    (empty for a new trap), which adds the two rows that tell the two kinds apart."""
    rows = [r for r in replay["cases"] if accepted is None or r["id"] in accepted]
    head = "| Domain | Cases | Caught | Both twins | Declaration | Warned | Missed"
    head += " | False blocks | Clean asked |"
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    domains = sorted({r["domain"] for r in rows})
    groups = [(d, [r for r in rows if r["domain"] == d]) for d in domains]
    seen = [r for r in rows if r["faulted"]["calls_checked"] > 0]
    groups += [("all", rows), ("call seen by a shim", seen)]
    if families is not None:
        again = [r for r in rows if families.get(r["id"])]
        groups += [
            ("repeats a family trap", again),
            ("new trap", [r for r in rows if r not in again]),
        ]
    for name, group in groups:
        count = Counter(r["outcome"] for r in group)
        n = len(group)
        cells = [wilson_text(count[o], n) if o == "caught" else str(count[o]) for o in OUTCOMES]
        false = sum(r["false_block"] for r in group)
        asked = sum(r["clean_asked"] for r in group)
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + f" | {false} | {asked} |")
    if on_target:
        caught = [r for r in rows if r["outcome"] == "caught"]
        hits = sum(1 for r in caught if on_target.get(r["id"], {}).get("on_target"))
        lines.append(f"\nCaught for the reason of the case: {hits} of {len(caught)}.")
    if clean:
        judged = Counter(clean.get(r["id"], "n/a") for r in rows if r["false_block"])
        total = sum(judged.values())
        parts = ", ".join(
            f"{judged[k]} {k.replace('_', ' ')}" for k in ("valid", "false_positive", "unclear")
        )
        lines.append(f"\nFindings on the clean command, judged ({total}): {parts}.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind_coverage", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "replay"):
        step = sub.add_parser(name)
        step.add_argument("cases", type=Path)
        step.add_argument("--out", type=Path, required=True)
        step.add_argument("--tools", type=Path, help="tools.json of the benchmark (else this PATH)")
        step.add_argument("--no-sandbox", action="store_true")
        step.add_argument("--only", action="append", default=[], help="run only this case id")
    sub.choices["replay"].add_argument("--verified", type=Path, required=True)
    accept = sub.add_parser("accept")
    accept.add_argument("reviews", type=Path, help="JSON list of review records")
    accept.add_argument("--out", type=Path, required=True)
    table = sub.add_parser("summary")
    table.add_argument("replay", type=Path)
    table.add_argument("--accepted", type=Path, help="JSON list of the case ids to count")
    table.add_argument("--on-target", type=Path, help="JSON {id: {on_target: bool}}")
    table.add_argument("--families", type=Path, help="JSON {id: family id, or empty}")
    table.add_argument("--labels", type=Path, help="JSON {id: {family, on_target, clean_finding}}")
    args = parser.parse_args(argv)
    if args.command == "accept":
        ids = accepted_ids(json.loads(args.reviews.read_text()))
        args.out.write_text(json.dumps(ids, indent=1) + "\n")
        print(f"{len(ids)} cases accepted")
        return 0
    if args.command == "summary":
        replay = json.loads(args.replay.read_text())
        accepted = set(json.loads(args.accepted.read_text())) if args.accepted else None
        on_target = json.loads(args.on_target.read_text()) if args.on_target else None
        families = json.loads(args.families.read_text()) if args.families else None
        clean = None
        if args.labels:
            labels = json.loads(args.labels.read_text())
            families = {i: item["family"] for i, item in labels.items()}
            on_target = {i: {"on_target": item["on_target"] == "yes"} for i, item in labels.items()}
            clean = {i: item["clean_finding"] for i, item in labels.items()}
        print(summary(replay, accepted, on_target, families, clean))
        return 0
    scan = load_scan()
    bwrap = None if args.no_sandbox else scan.find_bwrap()
    if bwrap is None and not args.no_sandbox:
        print("no working bwrap; give --no-sandbox to run the cases without one", file=sys.stderr)
        return 2
    runner = Variant(scan, scan.tool_environment(args.tools), bwrap)
    cases, rejected = load_cases(args.cases)
    cases = [c for c in cases if not args.only or c["id"] in args.only]
    if args.command == "verify":
        results = [verify_case(runner, c) for c in cases]
        out = {"schema": 1, "sandbox": bool(bwrap), "rejected": rejected, "cases": results}
    else:
        from biowarden.engine.spec import packaged_specs
        from biowarden.engine.tools import watched_tools

        truth = {c["id"]: c for c in json.loads(args.verified.read_text())["cases"]}
        todo = [c for c in cases if truth.get(c["id"], {}).get("verified")]
        results = [replay_case(runner, c, truth[c["id"]]) for c in todo]
        out = {
            "schema": 1,
            "sandbox": bool(bwrap),
            "watched": sorted(watched_tools(packaged_specs())),
            "cases": results,
        }
    for entry in results:
        print(
            entry["id"],
            entry.get("outcome") or ("verified" if entry["verified"] else entry["reasons"]),
        )
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
