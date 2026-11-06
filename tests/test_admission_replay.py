"""Replay of finished trials through the engine, on synthetic trial directories."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from biowarden.bench.admission import replay
from biowarden.bench.admission.replay import (
    TrialReplay,
    map_sandbox,
    replay_runs,
    summarise,
    usable_specs,
)
from biowarden.engine import CheckResult, Spec, checker
from biowarden.engine import spec as spec_module
from biowarden.engine.spec import load_packaged

SPEC = load_packaged("expression_matrix")
RULE = "matrix.log_scale_state"
AWK = "awk -F'\\t' 'NR>1{s=0;for(i=2;i<=NF;i++)s+=$i; if(s/(NF-1)>100)n++} END{print n+0}' expr.tsv"


def matrix_text(log: bool) -> str:
    rng = random.Random(5)
    rows = ["gene\tS1\tS2\tS3\tS4"]
    for i in range(30):
        counts = [int(rng.lognormvariate(4.5, 1.5)) for _ in range(4)]
        counts[0] = 90000 if i == 0 else counts[0]
        cells = [f"{math.log2(c + 1):.4f}" if log else str(c) for c in counts]
        rows.append("\t".join([f"ENSG{i:011d}", *cells]))
    return "\n".join(rows) + "\n"


def bash_message(index: int, command: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"t{index}",
                    "name": "bash",
                    "input": {"command": command},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{index}", "content": "ok"}],
        },
    ]


def make_trial(
    run: Path,
    name: str,
    family: str,
    twin: str,
    outcome: str,
    commands: list[str],
    files: dict[str, str] | None = None,
    model: str = "vendor__model-a",
    transcript: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    trial = run / "trials" / model / name
    work = trial / "work"
    work.mkdir(parents=True)
    for rel, text in (
        files if files is not None else {"expr.tsv": matrix_text(twin == "faulted")}
    ).items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text)
    score = {
        "trial": f"{model}/{name}",
        "model": model,
        "family": family,
        "twin": twin,
        "outcome": outcome,
        **(extra or {}),
    }
    (trial / "score.json").write_text(json.dumps(score))
    (trial / "task.json").write_text(json.dumps({"family": family, "twin": twin}))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Task: count genes."}]
    for i, command in enumerate(commands):
        messages += bash_message(i, command)
    (trial / "transcript.json").write_text(
        json.dumps(messages if transcript is None else transcript)
    )
    return trial


def replay_one(
    run: Path, specs: list[Spec] | None = None, strict: bool = False
) -> list[TrialReplay]:
    return replay_runs([run], specs if specs is not None else [SPEC], None, strict)


def by_name(records: list[TrialReplay]) -> dict[str, TrialReplay]:
    return {r.trial.split("/")[-1]: r for r in records}


# -- the core: blocked or not, split by outcome ---------------------------------------------------


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs" / "synthetic"
    make_trial(root, "F15-s00-faulted-r0", "F15", "faulted", "silent_wrong", [f"cd /work && {AWK}"])
    make_trial(
        root,
        "F15-s01-faulted-r0",
        "F15",
        "faulted",
        "pass",
        [
            "cd /work; head -3 expr.tsv",
            "python3 -c \"print(sum(2**float(l.split()[1])-1 for l in open('expr.tsv')))\"",
        ],
    )
    make_trial(
        root,
        "F15-s02-faulted-r0",
        "F15",
        "faulted",
        "pass",
        ["cd /work && head -3 expr.tsv && wc -l expr.tsv"],
    )
    make_trial(root, "F15-s00-clean-r0", "F15", "clean", "pass", [f"cd /work && {AWK}"])
    make_trial(root, "F15-s01-clean-r0", "F15", "clean", "silent_wrong", [f"cd /work && {AWK}"])
    make_trial(root, "F1-s00-faulted-r0", "F1", "faulted", "protocol_fail", ["ls /work"], files={})
    return root


def test_the_trap_is_blocked_and_correct_data_costs_nothing(run):
    records = by_name(replay_one(run))

    assert records["F15-s00-faulted-r0"].blocked and records["F15-s00-faulted-r0"].first_block == 0
    assert records["F15-s00-faulted-r0"].blocking_rules == (RULE,)
    assert records["F15-s01-faulted-r0"].blocked and records["F15-s01-faulted-r0"].first_block == 1
    assert not records["F15-s02-faulted-r0"].blocked  # only looked at the file
    assert not records["F15-s00-clean-r0"].blocked and not records["F15-s01-clean-r0"].blocked
    assert not records["F1-s00-faulted-r0"].blocked
    assert not any(r.crashed or r.problem for r in records.values())


def test_summary_splits_each_family_and_twin_by_outcome(run):
    summary = summarise(replay_one(run))

    assert list(summary) == [("F1", "faulted"), ("F15", "clean"), ("F15", "faulted")]
    faulted = summary[("F15", "faulted")]
    assert (faulted["pass"].n, faulted["pass"].blocked) == (2, 1)
    assert (faulted["silent_wrong"].n, faulted["silent_wrong"].blocked) == (1, 1)
    assert faulted["silent_wrong"].rate == 1.0 and faulted["other"].rate is None
    clean = summary[("F15", "clean")]
    assert (clean["pass"].n, clean["pass"].blocked, clean["silent_wrong"].blocked) == (1, 0, 0)
    assert (
        summary[("F1", "faulted")]["other"].n == 1
    )  # protocol_fail is neither pass nor silent_wrong


def test_the_table_reads_like_the_issue_asks(run):
    records = replay_one(run)
    text = replay.render(summarise(records), records, ["broken.yaml: ValueError"])

    lines = {
        line.split()[0] + " " + line.split()[1]: line
        for line in text.splitlines()
        if line.startswith("F")
    }
    assert "1/2 50%" in lines["F15 faulted"] and "1/1 100%" in lines["F15 faulted"]
    assert "0/1 0%" in lines["F15 clean"]
    assert "trials 6;" in text and "matrix.log_scale_state 2" in text
    assert "skipped spec: broken.yaml: ValueError" in text and "no declaration" in text


def test_family_order_is_numeric_not_alphabetical():
    records = [
        TrialReplay("r", f"m/{f}-{i}", "m", f, "clean", "pass")
        for i, f in enumerate(["F10", "F2", "F1", "F16", "F9"])
    ]
    assert [family for family, _ in summarise(records)] == ["F1", "F2", "F9", "F10", "F16"]


# -- how commands become calls ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        AWK,
        "awk '{print $1}' /work/expr.tsv",
        "cd /work/sub && awk '{print $1}' ../expr.tsv",
        "cd sub; cd .. ; python3 -c \"open('expr.tsv')\"",
        "bash -c \"cd /work && awk '{print}' expr.tsv\"",
        "timeout 30 nice awk '{print}' expr.tsv",
        "for f in missing.tsv expr.tsv; do awk '{print $1}' $f; done",
        "f=expr.tsv; awk '{print $1}' \"$f\" | head",
        "echo $(awk '{print $1}' expr.tsv | wc -l)",
        "awk '{print $1}' <(cat expr.tsv)",
        "cat expr.tsv | python3 -c \"import sys; open('expr.tsv')\"",
    ],
)
def test_commands_the_agent_wrote_reach_the_engine_as_calls(tmp_path, command):
    root = tmp_path / "r"
    make_trial(
        root,
        "t-r0",
        "F15",
        "faulted",
        "silent_wrong",
        [command],
        files={"expr.tsv": matrix_text(True), "sub/keep.txt": "x\n"},
    )

    (record,) = replay_one(root)

    assert record.blocked, command


def test_sandbox_path_mapping_only_touches_the_mount_point():
    work = Path("/runs/x/trials/m/t/work")
    assert map_sandbox("/work/expr.tsv", work) == "/runs/x/trials/m/t/work/expr.tsv"
    assert map_sandbox("open('/work/a'), '/work'", work) == f"open('{work}/a'), '{work}'"
    assert map_sandbox("/workflow/a /work.bak ./work/a ~/work/a x/work", work) == (
        "/workflow/a /work.bak ./work/a ~/work/a x/work"
    )


def test_stdin_forms_are_given_to_the_engine_unless_strict(tmp_path):
    root = tmp_path / "r"
    make_trial(
        root, "redirect-r0", "F15", "faulted", "silent_wrong", ["awk '{print $1}' < expr.tsv"]
    )
    make_trial(
        root,
        "heredoc-r0",
        "F15",
        "faulted",
        "silent_wrong",
        ["python3 - <<'EOF'\nimport csv\nrows = list(csv.reader(open('expr.tsv')))\nEOF"],
    )
    make_trial(
        root, "plain-r0", "F15", "faulted", "silent_wrong", ["python3 -c \"open('expr.tsv')\""]
    )

    loose = by_name(replay_one(root))
    strict = by_name(replay_one(root, strict=True))

    assert [loose[k].blocked for k in ("redirect-r0", "heredoc-r0", "plain-r0")] == [True] * 3
    assert [strict[k].blocked for k in ("redirect-r0", "heredoc-r0", "plain-r0")] == [
        False,
        False,
        True,
    ]


def test_directories_that_are_elsewhere_or_unknown(tmp_path):
    root = tmp_path / "r"
    make_trial(
        root,
        "elsewhere-r0",
        "F15",
        "faulted",
        "silent_wrong",
        ["cd /nonexistent-replay-dir && awk '{print}' expr.tsv"],
    )
    make_trial(
        root,
        "unknown-r0",
        "F15",
        "faulted",
        "silent_wrong",
        ["cd $SOMEWHERE && awk '{print}' expr.tsv"],
    )
    make_trial(
        root,
        "subshell-r0",
        "F15",
        "faulted",
        "silent_wrong",
        ["(cd /tmp/t && ls); awk '{print}' expr.tsv"],
    )

    records = by_name(replay_one(root))

    assert not records[
        "elsewhere-r0"
    ].blocked  # a file in the agent's /tmp is not in this workspace
    assert records["unknown-r0"].blocked  # unknowable directory: the work directory is assumed
    assert records["subshell-r0"].blocked  # the cd did not leak


def test_declarations_in_the_command_are_honoured_and_the_process_environment_is_not(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BIOWARDEN_DECLARE", "matrix_state=log_transformed")
    root = tmp_path / "r"
    make_trial(root, "bare-r0", "F15", "faulted", "silent_wrong", [AWK])
    make_trial(
        root,
        "env-r0",
        "F15",
        "faulted",
        "pass",
        [f"BIOWARDEN_DECLARE='matrix_state=log_transformed' {AWK}"],
    )
    make_trial(
        root, "flag-r0", "F15", "faulted", "pass", [AWK + " --declare matrix_state=log_transformed"]
    )
    make_trial(
        root, "wrong-r0", "F15", "faulted", "pass", [AWK + " --declare matrix_state=raw_counts"]
    )
    make_trial(root, "garbled-r0", "F15", "faulted", "pass", [AWK + " --declare novalue"])

    records = by_name(replay_one(root))

    assert records[
        "bare-r0"
    ].blocked  # the replaying shell's BIOWARDEN_DECLARE did not leak into the calls
    assert not records["env-r0"].blocked and not records["flag-r0"].blocked
    assert records["wrong-r0"].blocked
    assert "malformed" in records["garbled-r0"].problem


def test_relative_run_paths_work(tmp_path, monkeypatch):
    make_trial(tmp_path / "runs" / "rel", "t-r0", "F15", "faulted", "silent_wrong", [AWK])
    monkeypatch.chdir(tmp_path)

    (record,) = replay_runs([Path("runs/rel")], [SPEC], None)

    assert record.blocked and record.run == "rel"


def test_openai_style_transcripts_are_read(tmp_path):
    transcript = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": AWK})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "0"},
    ]
    make_trial(tmp_path / "r", "t-r0", "F15", "faulted", "silent_wrong", [], transcript=transcript)

    (record,) = replay_one(tmp_path / "r")

    assert record.n_commands == 1 and record.blocked


# -- robustness ------------------------------------------------------------------------------------


@checker("replay_test_crashes")
def _crashes(call, match, ctx):
    raise RuntimeError("boom")


@checker("replay_test_warns")
def _warns(call, match, ctx):
    return CheckResult("unknown", "cannot_tell", "cannot tell")


def rule_for(check: str, rule_id: str) -> dict:
    return {
        "id": rule_id,
        "slot": "units",
        "kind": "fact",
        "tier": "T1",
        "applies": {"file_patterns": [r"[\w./-]+\.tsv"]},
        "check": check,
        "message": "m",
    }


def test_a_crashed_checker_is_counted_as_crashed_and_never_as_a_block(tmp_path):
    spec = Spec.model_validate(
        {
            "name": "t",
            "rules": [
                rule_for("replay_test_crashes", "t.crash"),
                rule_for("replay_test_warns", "t.warn"),
            ],
        }
    )
    root = tmp_path / "r"
    make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", [AWK])

    (record,) = replay_runs([root], [spec], None)

    assert record.crashed and not record.blocked and record.warned
    assert "checker_crashed" in record.codes


def test_a_rule_the_engine_could_not_check_is_a_crash_not_a_verdict(tmp_path):
    """The engine turns a fault of a rule (here an invalid pattern) into a `layer_error` outcome
    that only warns; the replay counts it as an error and keeps it out of `fired`."""
    broken = {**rule_for("replay_test_warns", "t.bad"), "applies": {"file_patterns": ["("]}}
    spec = Spec.model_validate({"name": "t", "rules": [broken]})
    root = tmp_path / "r"
    make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", [AWK, AWK])

    (record,) = replay_runs([root], [spec], None)

    assert record.crashed and not record.blocked and not record.warned and record.problem == ""
    assert record.n_calls == 2 and record.codes == ("layer_error",)
    assert record.crashed_rules == ("t.bad",) and record.fired == ()


def test_an_exception_out_of_the_engine_is_an_error_and_the_replay_goes_on(tmp_path):
    class Exploding:
        def evaluate(self, call):
            raise RuntimeError("boom")

    root = tmp_path / "r"
    trial = make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", [AWK, AWK])

    record = replay.replay_trial(trial, Exploding(), "r")  # type: ignore[arg-type]

    assert record.crashed and not record.blocked and record.problem == "engine: RuntimeError"
    assert record.n_calls == 2 and record.codes == ("engine_error",)


def test_a_warning_only_verdict_is_not_a_block(tmp_path):
    spec = Spec.model_validate({"name": "t", "rules": [rule_for("replay_test_warns", "t.warn")]})
    root = tmp_path / "r"
    make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", [AWK])

    (record,) = replay_runs([root], [spec], None)

    assert record.warned and not record.blocked and not record.crashed


def test_unreadable_trials_are_reported_not_fatal(tmp_path):
    root = tmp_path / "r"
    broken = make_trial(root, "broken-r0", "F15", "faulted", "pass", [AWK])
    (broken / "transcript.json").write_text("{not json")
    missing = make_trial(root, "missing-r0", "F15", "faulted", "pass", [AWK])
    (missing / "transcript.json").unlink()
    make_trial(root, "odd-r0", "F15", "faulted", "pass", [], transcript={"role": "assistant"})
    (root / "trials" / "vendor__model-a" / "unfinished-r0").mkdir()  # no score.json: not a trial
    (root / "trials" / "stray.txt").write_text("x")

    records = by_name(replay_one(root))

    assert set(records) == {"broken-r0", "missing-r0", "odd-r0"}
    assert (
        records["broken-r0"].problem.startswith("transcript") and not records["broken-r0"].blocked
    )
    assert records["missing-r0"].problem.startswith("transcript")
    assert records["odd-r0"].n_commands == 0 and records["odd-r0"].problem


def test_odd_commands_do_not_stop_the_replay(tmp_path):
    root = tmp_path / "r"
    commands = [
        "echo 'unterminated",
        "awk '",
        "cat <<EOF\nnever closed",
        "(((",
        "$(",
        "for f in",
        AWK,
    ]
    make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", commands)

    (record,) = replay_one(root)

    assert record.blocked and record.first_block == 6 and record.unparsed >= 3


def test_an_empty_or_absent_set_of_specs_replays_to_zero_blocks(tmp_path):
    make_trial(tmp_path / "r", "t-r0", "F15", "faulted", "silent_wrong", [AWK])
    (record,) = replay_runs([tmp_path / "r"], [], None)
    assert not record.blocked and record.n_calls >= 1


def test_broken_specs_are_skipped_and_listed(tmp_path, monkeypatch):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    good = (Path(spec_module.SPEC_DIR) / "expression_matrix.yaml").read_text()
    (specs_dir / "good.yaml").write_text(good)
    (specs_dir / "garbage.yaml").write_text("rules: [: not yaml")
    (specs_dir / "nochecker.yaml").write_text(
        good.replace("name: expression_matrix", "name: nochecker")
        .replace("id: matrix.log_scale_state", "id: nochecker.rule")
        .replace("check: expression_matrix_state", "check: checker_that_does_not_exist")
    )
    monkeypatch.setattr(spec_module, "SPEC_DIR", specs_dir)
    monkeypatch.setattr(replay, "SPEC_DIR", specs_dir)

    specs, skipped = usable_specs()

    assert [s.name for s in specs] == ["expression_matrix"]
    assert any(line.startswith("garbage.yaml") for line in skipped)
    assert any(line.startswith("nochecker") and "unknown checkers" in line for line in skipped)
    assert [s.name for s in usable_specs(["nothing"])[0]] == []


def test_the_packaged_specs_that_exist_all_replay_without_error(tmp_path):
    specs, skipped = usable_specs()  # whatever other authors have packaged so far
    assert "expression_matrix" in {s.name for s in specs} and isinstance(skipped, list)
    root = tmp_path / "r"
    make_trial(
        root,
        "a-r0",
        "F15",
        "faulted",
        "silent_wrong",
        [AWK, "bcftools view -R regions.bed -H variants.vcf.gz"],
    )
    make_trial(
        root,
        "b-r0",
        "F15",
        "clean",
        "pass",
        ["samtools faidx ref.fa chr1:1-10", "bedtools intersect -a a.bed -b b.bed"],
    )

    records = replay_runs([root], specs, None)  # tools unknown: crashes are counted, never raised

    assert len(records) == 2 and by_name(records)["a-r0"].n_calls == 2
    assert by_name(records)["a-r0"].blocked  # at least the matrix rule
    assert sum(g.n for groups in summarise(records).values() for g in groups.values()) == 2


def test_two_runs_are_combined(tmp_path):
    make_trial(tmp_path / "one", "t-r0", "F15", "faulted", "silent_wrong", [AWK])
    make_trial(tmp_path / "two", "t-r0", "F15", "faulted", "pass", ["head expr.tsv"])

    records = replay_runs([tmp_path / "one", tmp_path / "two"], [SPEC], None)

    assert [(r.run, r.blocked) for r in records] == [("one", True), ("two", False)]


# -- what the record says about the rules ----------------------------------------------------------


def test_every_rule_that_did_not_pass_is_named_with_its_effect(run):
    records = by_name(replay_one(run))

    blocked = records["F15-s00-faulted-r0"]
    assert blocked.fired and all(
        f.startswith(f"{RULE}:") and f.endswith(":block") for f in blocked.fired
    )
    assert (
        blocked.blocking_rules == (RULE,) and blocked.n_unknown == 0 and blocked.crashed_rules == ()
    )
    assert records["F15-s02-faulted-r0"].fired == ()  # a command that only looked at the file
    assert records["F15-s00-clean-r0"].fired == ()


def test_warn_only_and_unknown_verdicts_are_recorded_beside_blocks(tmp_path):
    spec = Spec.model_validate(
        {
            "name": "t",
            "rules": [
                rule_for("replay_test_warns", "t.warn"),
                rule_for("replay_test_crashes", "t.crash"),
            ],
        }
    )
    root = tmp_path / "r"
    make_trial(root, "t-r0", "F15", "faulted", "silent_wrong", [AWK])

    (warned,) = replay_runs([root], [spec, SPEC], None)

    assert warned.blocked and warned.warned is False  # the matrix rule blocks: a block wins
    assert "t.warn:cannot_tell:warn" in warned.fired and any(
        f.endswith(":block") for f in warned.fired
    )
    assert warned.n_unknown == 1
    assert warned.crashed and warned.crashed_rules == ("t.crash",)
    assert not any(f.startswith("t.crash") for f in warned.fired)  # a crash is an error, not a fire

    (only_warn,) = replay_runs([root], [spec], None)
    assert only_warn.warned and not only_warn.blocked
    assert only_warn.fired == ("t.warn:cannot_tell:warn",)


def test_the_trials_identity_and_exposure_travel_with_the_record(tmp_path):
    extra = {
        "seed": 3, "arm": "bare", "contract": "bare", "tier": "implicit", "route": "forced",
        "exposed": True, "signature": False,
    }  # fmt: skip
    root = tmp_path / "r"
    make_trial(root, "full-r0", "F15", "faulted", "silent_wrong", [AWK], extra=extra)
    make_trial(root, "bare-r0", "F15", "faulted", "silent_wrong", [AWK])
    trial = make_trial(root, "task-r0", "F15", "faulted", "silent_wrong", [AWK])
    (trial / "task.json").write_text(
        json.dumps(
            {"family": "F15", "twin": "faulted", "seed": "4", "tier": "explicit", "route": "free"}
        )
    )

    records = by_name(replay_one(root))

    full = records["full-r0"]
    assert (full.seed, full.arm, full.contract, full.tier, full.route) == (
        3,
        "bare",
        "bare",
        "implicit",
        "forced",
    )
    assert full.exposed is True and full.signature is False
    plain = records["bare-r0"]
    assert plain.seed is None and plain.exposed is None and plain.arm == ""
    from_task = records["task-r0"]
    assert (from_task.seed, from_task.tier, from_task.route) == (4, "explicit", "free")


def test_the_json_records_carry_the_new_fields(run, tmp_path):
    out = tmp_path / "out.json"
    proc = cli("--run", str(run), "--spec", "expression_matrix", "--json", str(out), cwd=tmp_path)

    assert proc.returncode == 0, proc.stderr
    records = json.loads(out.read_text())
    blocked = next(r for r in records if r["trial"].endswith("F15-s00-faulted-r0"))
    assert (
        blocked["fired"] and blocked["fired"][0].startswith(RULE) and blocked["crashed_rules"] == []
    )
    assert {"seed", "arm", "contract", "tier", "route", "exposed", "signature", "n_unknown"} <= set(
        blocked
    )


# -- the command line ----------------------------------------------------------------------------


def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-m", "biowarden.bench.admission.replay", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_cli_prints_the_table_in_isolated_mode(run, tmp_path):
    proc = cli(
        "--run",
        str(run),
        "--spec",
        "expression_matrix",
        "--json",
        str(tmp_path / "out.json"),
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    assert "specs: expression_matrix" in proc.stdout
    assert "pass blocked" in proc.stdout and "silent_wrong blocked" in proc.stdout
    faulted = next(
        line for line in proc.stdout.splitlines() if line.startswith("F15") and "faulted" in line
    )
    assert "1/2 50%" in faulted and "1/1 100%" in faulted
    records = json.loads((tmp_path / "out.json").read_text())
    assert len(records) == 6 and sum(r["blocked"] for r in records) == 2


def test_cli_strict_flag_and_repeated_runs(run, tmp_path):
    make_trial(
        tmp_path / "second", "t-r0", "F15", "faulted", "silent_wrong", ["awk '{print}' < expr.tsv"]
    )

    both = cli(
        "--run",
        str(run),
        "--run",
        str(tmp_path / "second"),
        "--spec",
        "expression_matrix",
        cwd=tmp_path,
    )
    strict = cli(
        "--run",
        str(run),
        "--run",
        str(tmp_path / "second"),
        "--spec",
        "expression_matrix",
        "--strict",
        cwd=tmp_path,
    )

    assert (
        "trials 7;" in both.stdout
        and "trials blocked by rule: matrix.log_scale_state 3" in both.stdout
    )
    assert "strict argv" in strict.stdout and "matrix.log_scale_state 2" in strict.stdout


def test_cli_rejects_a_directory_without_trials_and_an_unknown_spec(tmp_path, run):
    (tmp_path / "empty").mkdir()
    missing = cli("--run", str(tmp_path / "empty"), cwd=tmp_path)
    unknown = cli("--run", str(run), "--spec", "no_such_spec", cwd=tmp_path)

    assert missing.returncode == 2 and "no trials directory" in missing.stderr
    assert unknown.returncode == 2 and "no usable packaged spec" in unknown.stderr
