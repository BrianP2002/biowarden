"""Replay finished benchmark trials through the engine: what would the rules have blocked?

    python -I -m science_harness.bench.admission.replay --run runs/minimal-2 [--run DIR ...]

For every trial under DIR/trials/<model>/<trial>/ (score.json, task.json, transcript.json, work/)
the bash commands the agent issued are read with `commands_of`, split into simple commands with
`engine.shellsplit`, turned into engine `Call`s whose directory is the trial's `work` directory
(the sandbox mount `/work` is mapped onto it, `cd` is tracked), and evaluated by an
`Engine(mode="block")` over the packaged specs with the real tools of `ToolEnv.discover`.
The table gives, per family and twin, how many trials have at least one command that would have
been blocked, split by the trial's outcome (pass, silent_wrong, any other).

How to read it (caveats, they matter):
  * The calls carry no declaration, because the agents of these runs never knew the convention.
    The replay therefore measures what the FIRST call of each kind would have triggered, not the
    full interaction: after a block the agent would have reacted (declared, or changed course) and
    every later command would have been different. "Blocked" is an upper bound on the share of
    trials a block-mode engine would have interrupted, and a lower bound on the share of faulted
    trials in which the trap was reachable.
  * The workspace is the FINAL one. Files the agent created (scripts, result.json, intermediate
    tables) exist, files it deleted or rewrote are gone or changed. A rule that reads data sees
    the data as the agent left it.
  * Fact rules see what the engine sees: the argv. `shellsplit` appends, to each call, the target
    of a `<` redirect and the body of a here-document (code piped into `python3 -`), because a
    front end that sees the whole command line would; `--strict` turns that off. A file named
    only inside a script written earlier (`python3 analyze.py`) or reaching a tool through a pipe
    (`cat f | awk ...`) is not seen, exactly as the engine would not see it.
  * Commands whose directory cannot be known (`cd $VAR`) are evaluated in the work directory.
  * A rule whose checker crashes (a missing tool, an unreadable file) is counted as `crashed`,
    never as a block: a fail-closed crash is a defect to fix, not a measured block.
  * Specs that do not load, or name a checker that does not exist yet, are skipped and listed.

Each record also names every rule that did not pass (`fired`: "rule:code:effective", warn-only and
unknown verdicts included, crashes apart) and carries the trial's seed, arm, cell, `exposed` and
`signature`, so that `bench.stage2` can ask which of a family's OWN rules would have blocked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from science_harness.bench.runner.exposure import commands_of
from science_harness.bench.tasks.tools import ToolEnv
from science_harness.engine import Call, Engine, Spec
from science_harness.engine.call import DeclarationError
from science_harness.engine.shellsplit import SimpleCommand, split_command
from science_harness.engine.spec import SPEC_DIR, load_spec, packaged_specs
from science_harness.engine.tools import call_rule_tools

SANDBOX_ROOT = "/work"  # where a trial's work directory appears inside the sandbox
OUTCOME_GROUPS = ("pass", "silent_wrong", "other")
CRASH_CODES = ("checker_crashed", "layer_error")  # a fault of the rule or the layer, not a verdict
_SANDBOX_PATH = re.compile(r"(?<![\w.~-])" + re.escape(SANDBOX_ROOT) + r"(?![\w.-])")


# -- specs ------------------------------------------------------------------------------------


def usable_specs(names: Sequence[str] = ()) -> tuple[list[Spec], list[str]]:
    """The packaged specs that load and whose checkers exist, plus what was skipped and why.

    `names` limits the set to specs with those names (file stem or spec name).
    """
    skipped: list[str] = []
    try:
        specs = packaged_specs()
    except Exception:  # one broken file must not hide the others
        specs = []
        for path in sorted(SPEC_DIR.glob("*.yaml")):
            try:
                specs.append(load_spec(path))
            except Exception as exc:
                skipped.append(f"{path.name}: {type(exc).__name__}")
    if names:
        specs = [s for s in specs if s.name in names]
    usable: list[Spec] = []
    for spec in specs:
        try:
            Engine([spec], mode="block")
        except Exception as exc:  # unknown checker, a rule module that fails to import
            skipped.append(f"{spec.name}: {type(exc).__name__}: {exc}")
        else:
            usable.append(spec)
    return usable, skipped


# -- one trial --------------------------------------------------------------------------------


@dataclass
class TrialReplay:
    run: str
    trial: str
    model: str
    family: str
    twin: str
    outcome: str
    n_commands: int = 0  # bash tool calls the agent made
    n_calls: int = 0  # simple commands handed to the engine
    blocked: bool = False
    warned: bool = False  # a warning-only verdict and no block
    crashed: bool = False
    first_block: int | None = None  # index of the bash call holding the first blocked command
    blocking_rules: tuple[str, ...] = ()
    unparsed: int = 0
    problem: str = ""  # the trial could not be read completely
    codes: tuple[str, ...] = field(default=())
    seed: int | None = None
    arm: str = ""
    contract: str = ""
    tier: str = ""
    route: str = ""
    exposed: bool | None = None  # as the trial's score.json says
    signature: bool | None = None
    fired: tuple[str, ...] = ()  # "rule:code:effective" of every outcome that was not a pass
    n_unknown: int = 0  # outcomes with an unknown verdict (they only ever warn)
    crashed_rules: tuple[str, ...] = ()  # rules whose check crashed or could not be run


def outcome_group(outcome: str) -> str:
    return outcome if outcome in ("pass", "silent_wrong") else "other"


def map_sandbox(text: str, work: Path) -> str:
    """Rewrite the sandbox path /work (not /workflow, ./work) to the trial's work directory."""
    return _SANDBOX_PATH.sub(lambda _: str(work), text)


def directory_of(command: SimpleCommand, work: Path) -> Path:
    cwd = command.cwd
    if cwd is None or cwd.startswith("~") or not cwd.startswith("/"):
        return work
    return Path(map_sandbox(cwd, work))


def call_of(command: SimpleCommand, work: Path, strict: bool = False) -> Call:
    """The engine call for one simple command, as a front end that sees the command line would."""
    argv = list(command.argv)
    if not strict:
        argv += [r.target for r in command.redirects if r.op in ("<", "<>") and r.target]
        if command.heredoc:
            argv.append(command.heredoc)
    argv = [map_sandbox(token, work) for token in argv]
    env = {k: v for k, v in command.env}
    return Call.from_argv(command.tool, argv, directory_of(command, work), env=env)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def replay_trial(
    trial_dir: Path, engine: Engine, run: str = "", strict: bool = False
) -> TrialReplay | None:
    """Replay one trial directory; None when it has no score.json (not a finished trial)."""
    try:
        score = _read_json(trial_dir / "score.json")
    except (OSError, ValueError):
        return None
    task: dict[str, Any] = {}
    try:
        task = _read_json(trial_dir / "task.json")
    except (OSError, ValueError):
        pass
    seed = score.get("seed", task.get("seed"))
    record = TrialReplay(
        run=run,
        trial=str(score.get("trial") or f"{trial_dir.parent.name}/{trial_dir.name}"),
        model=str(score.get("model") or trial_dir.parent.name),
        family=str(score.get("family") or task.get("family") or "?"),
        twin=str(score.get("twin") or task.get("twin") or "?"),
        outcome=str(score.get("outcome") or "?"),
        seed=int(seed)
        if isinstance(seed, (int, str)) and str(seed).lstrip("-").isdigit()
        else None,
        arm=str(score.get("arm") or ""),
        contract=str(score.get("contract") or ""),
        tier=str(score.get("tier") or task.get("tier") or ""),
        route=str(score.get("route") or task.get("route") or ""),
        exposed=score.get("exposed"),
        signature=score.get("signature"),
    )
    try:
        commands = commands_of(_read_json(trial_dir / "transcript.json"))
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        record.problem = f"transcript: {type(exc).__name__}"
        return record
    work = (trial_dir / "work").absolute()  # a relative path would be resolved twice by Ctx
    record.n_commands = len(commands)
    blocking: list[str] = []
    fired: set[str] = set()
    crashed_rules: set[str] = set()
    codes: Counter[str] = Counter()
    for index, text in enumerate(commands):
        split = split_command(text, SANDBOX_ROOT)
        record.unparsed += len(split.unparsed)
        context: list[str] = []  # words of earlier commands in this string: the pipeline front end
        for command in split.commands:
            try:
                call = call_of(command, work, strict)
            except DeclarationError:
                record.problem = "malformed --declare marker"
                continue
            if not strict:
                context.extend(call.argv)
                call = Call(
                    call.tool,
                    call.argv,
                    call.cwd,
                    call.declared,
                    tuple(context[: -len(call.argv) or None]),
                )
            record.n_calls += 1
            try:
                decision = engine.evaluate(call)
            except Exception as exc:  # a rule or spec defect must not stop the replay
                record.crashed = True
                record.problem = f"engine: {type(exc).__name__}"
                codes["engine_error"] += 1
                continue
            for outcome in decision.outcomes:
                code = outcome.result.code
                if code in CRASH_CODES:
                    record.crashed = True
                    crashed_rules.add(outcome.rule.id)
                    codes[code] += 1
                    continue
                if outcome.effective == "none":
                    continue  # a pass
                fired.add(f"{outcome.rule.id}:{code}:{outcome.effective}")
                record.n_unknown += outcome.result.verdict == "unknown"
                if outcome.effective == "block":
                    blocking.append(outcome.rule.id)
                    codes[code] += 1
                    if record.first_block is None:
                        record.first_block = index
                elif outcome.effective == "warn":
                    record.warned = True
    record.blocked = bool(blocking)
    record.warned = record.warned and not record.blocked
    record.blocking_rules = tuple(sorted(set(blocking)))
    record.codes = tuple(sorted(codes))
    record.fired = tuple(sorted(fired))
    record.crashed_rules = tuple(sorted(crashed_rules))
    return record


def make_engine(specs: Sequence[Spec], tools: ToolEnv | None) -> Engine:
    return Engine(
        specs,
        mode="block",
        real_bins=tools.real_bins(call_rule_tools(specs)) if tools else {},
        real_env=tools.process_env() if tools else None,
        retry_cap=10**9,  # the retry cap is about repeated identical calls, not a replay concern
    )


def replay_runs(
    run_dirs: Iterable[Path],
    specs: Sequence[Spec],
    tools: ToolEnv | None,
    strict: bool = False,
) -> list[TrialReplay]:
    engine = make_engine(specs, tools)
    records: list[TrialReplay] = []
    for run_dir in run_dirs:
        for score in sorted(Path(run_dir).glob("trials/*/*/score.json")):
            record = replay_trial(score.parent, engine, Path(run_dir).name, strict)
            if record is not None:
                records.append(record)
    return records


# -- summary ----------------------------------------------------------------------------------


@dataclass
class Share:
    n: int = 0
    blocked: int = 0
    warned: int = 0
    crashed: int = 0

    def add(self, record: TrialReplay) -> None:
        self.n += 1
        self.blocked += record.blocked
        self.warned += record.warned
        self.crashed += record.crashed

    @property
    def rate(self) -> float | None:
        return self.blocked / self.n if self.n else None


def family_order(family: str) -> tuple[int, str]:
    digits = re.sub(r"\D", "", family)
    return (int(digits) if digits else 10**6, family)


def summarise(records: Iterable[TrialReplay]) -> dict[tuple[str, str], dict[str, Share]]:
    """(family, twin) -> outcome group -> Share."""
    table: dict[tuple[str, str], dict[str, Share]] = defaultdict(
        lambda: {g: Share() for g in OUTCOME_GROUPS}
    )
    for record in records:
        table[(record.family, record.twin)][outcome_group(record.outcome)].add(record)
    return dict(sorted(table.items(), key=lambda kv: (family_order(kv[0][0]), kv[0][1])))


def _cell(share: Share) -> str:
    if not share.n:
        return "-"
    return f"{share.blocked}/{share.n} {100 * share.blocked / share.n:.0f}%"


def render(
    summary: dict[tuple[str, str], dict[str, Share]],
    records: Sequence[TrialReplay] = (),
    skipped: Sequence[str] = (),
) -> str:
    head = (
        f"{'family':<8}{'twin':<9}{'pass blocked':>16}"
        f"{'silent_wrong blocked':>24}{'other blocked':>18}"
    )
    lines = [head, "-" * len(head)]
    total = {g: Share() for g in OUTCOME_GROUPS}
    for (family, twin), groups in summary.items():
        lines.append(
            f"{family:<8}{twin:<9}{_cell(groups['pass']):>16}"
            f"{_cell(groups['silent_wrong']):>24}{_cell(groups['other']):>18}"
        )
        for g in OUTCOME_GROUPS:
            for field_name in ("n", "blocked", "warned", "crashed"):
                setattr(
                    total[g],
                    field_name,
                    getattr(total[g], field_name) + getattr(groups[g], field_name),
                )
    lines.append("-" * len(head))
    lines.append(
        f"{'all':<8}{'':<9}{_cell(total['pass']):>16}{_cell(total['silent_wrong']):>24}"
        f"{_cell(total['other']):>18}"
    )
    by_rule: Counter[str] = Counter()
    for record in records:
        by_rule.update(record.blocking_rules)
    lines.append("")
    lines.append(
        f"trials {len(records)}; commands {sum(r.n_commands for r in records)}; "
        f"unparsed fragments {sum(r.unparsed for r in records)}; "
        f"trials with a crashed checker {sum(r.crashed for r in records)}; "
        f"trials unreadable {sum(bool(r.problem) for r in records)}"
    )
    if by_rule:
        lines.append(
            "trials blocked by rule: " + ", ".join(f"{k} {v}" for k, v in sorted(by_rule.items()))
        )
    for line in skipped:
        lines.append(f"skipped spec: {line}")
    lines.append(
        "blocked = at least one command would have been refused by an engine in block mode; "
        "the calls carry no declaration and the workspace is the final one."
    )
    return "\n".join(lines)


# -- command line -----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -I -m science_harness.bench.admission.replay",
        description="Replay finished trials through the rule engine (block mode).",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        metavar="DIR",
        help="a run directory such as runs/minimal-2 (repeatable)",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="NAME",
        help="only this packaged spec (repeatable); default all that load",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="argv only: do not append < targets and here-document bodies",
    )
    parser.add_argument(
        "--json", type=Path, metavar="FILE", help="also write the per-trial records"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="where ToolEnv.discover looks for .local/tools.json",
    )
    args = parser.parse_args(argv)

    missing = [str(d) for d in args.run if not (d / "trials").is_dir()]
    if missing:
        print(f"no trials directory in: {', '.join(missing)}", file=sys.stderr)
        return 2
    tools = ToolEnv.discover(args.root)
    if tools is None:
        print(
            "warning: samtools, bcftools or bedtools not found; rules that run them will crash "
            "and be counted as crashed",
            file=sys.stderr,
        )
    specs, skipped = usable_specs(args.spec)
    if args.spec and len(specs) < len(set(args.spec)):
        known = {s.name for s in specs}
        print(
            f"no usable packaged spec named: {', '.join(sorted(set(args.spec) - known))}",
            file=sys.stderr,
        )
        return 2
    records = replay_runs(args.run, specs, tools, args.strict)
    print(
        f"specs: {', '.join(s.name for s in specs) or '(none)'}; runs: "
        f"{', '.join(str(d) for d in args.run)}" + ("; strict argv" if args.strict else "")
    )
    print(render(summarise(records), records, skipped))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in records], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
