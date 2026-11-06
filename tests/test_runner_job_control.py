"""Job control of the runner: one durable record per planned trial, a loud job, opaque paths.

Regressions for the stage-2 review (R-2, R-3, R-4, R-7, R-9, A-5, A-6, A-12, S-5, S-7, S-12, D-6,
FC-2). Almost everything here runs with a stub in place of the builder and the agent loop, so
nothing calls an API and nothing needs the sandbox; the tests that do need real bubblewrap say so.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from biowarden.bench import score as scoring
from biowarden.bench.runner import cli, sandbox
from biowarden.bench.runner import layer as layer_module
from biowarden.bench.runner import trial as trial_module
from biowarden.bench.runner.agent import AgentConfig, Trajectory
from biowarden.bench.runner.cli import (
    code_digest,
    config_differences,
    job_ledger,
    load_records,
    main,
    plan,
    report,
    rescore,
    tool_manifest,
)
from biowarden.bench.runner.trial import (
    MAX_ATTEMPTS,
    Budget,
    RunInfo,
    TrialSpec,
    check_neutral_root,
    make_writable,
    move_tree,
    neutral_root,
    note_unrun,
    opaque_work_dir,
    record_crash,
    remove_tree,
    run_trial,
    write_json_atomic,
)
from biowarden.bench.tasks import Task, ToolEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-haiku-5-5"
NEUTRAL_VAR = "BIOWARDEN_NEUTRAL_ROOT"
OPAQUE = re.compile(r"[0-9a-f]{16}")
INFO = RunInfo("c" * 64, "r" * 64, "t" * 64, "job-1")


# --- stubs: a builder and an agent that need neither tools nor a sandbox ----------------------


class Stubs:
    """Replaces the family builder and the agent loop of `run_trial`."""

    def __init__(self, monkeypatch):
        self.builds: list[Path] = []  # the work directory each build was given
        self.works: list[Path] = []  # the work directory each agent was given
        self.fail_build: Exception | None = None
        self.fail_drive: Exception | None = None
        self.answer: int | None = 3
        self.traj = dict(stop="end_turn", turns=2, commands=1, usd=0.01)
        self.messages: list = []
        monkeypatch.setattr(trial_module, "build_task", self.build)
        monkeypatch.setattr(trial_module, "_drive", self.drive)

    def build(self, family, seed, twin, work, tools, tier="explicit", route="free"):
        self.builds.append(work)
        if self.fail_build is not None:
            raise self.fail_build
        (work / "README.md").write_text("the task")
        truth = {"kind": "int", "expected": 3, "naive": 5}
        return Task("t", family, "slot", twin, seed, "do it", truth, ())

    def drive(self, spec, client, tools, cfg, task, work, trial_dir):
        self.works.append(work)
        if self.fail_drive is not None:
            raise self.fail_drive
        if self.answer is not None:
            (work / "result.json").write_text(json.dumps({"answer": self.answer}))
        return Trajectory(messages=list(self.messages), **self.traj)


@pytest.fixture
def neutral(monkeypatch):
    # hex, not the test name: a neutral root may not spell a word that names a trial or an arm
    base = Path(tempfile.gettempdir()) / secrets.token_hex(6)
    monkeypatch.setenv(NEUTRAL_VAR, str(base / "neutral"))
    yield base / "neutral"
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def stubs(monkeypatch, neutral):
    return Stubs(monkeypatch)


def spec_of(family="F1", seed=0, twin="faulted", **knobs) -> TrialSpec:
    return TrialSpec(MODEL, family, seed, twin, contract="bare", **knobs)


def go(spec, root, budget=None, info=None):
    return run_trial(spec, root, None, None, AgentConfig(MODEL), budget or Budget(5), info)


def trial_dir(root, spec) -> Path:
    return root.resolve() / "trials" / spec.trial_id


# --- plan() ---------------------------------------------------------------------------------


def test_a_model_family_or_seed_listed_twice_is_planned_once():
    specs = list(plan(["m", "m"], ["F1", "F2", "F1"], [0, 1, 0, 1, 1], reps=1))

    ids = [s.trial_id for s in specs]
    assert len(specs) == 1 * 2 * 2 * 2 and len(set(ids)) == len(ids)


def test_plan_is_seed_major_with_a_fixed_family_order_inside_each_seed():
    families = [f"F{n}" for n in range(1, 9)]

    specs = list(plan(["m"], families, [0, 1], reps=1))

    seeds = [s.seed for s in specs]
    assert seeds == sorted(seeds)  # a run cut short by the budget keeps every seed balanced
    order = {
        seed: [s.family for s in specs if s.seed == seed and s.twin == "clean"] for seed in (0, 1)
    }
    assert all(sorted(o) == sorted(families) for o in order.values())
    # fixed: the plan of today and of tomorrow list the same trials in the same order, and the
    # order does not depend on how the families were typed
    assert order[0] == ["F1", "F5", "F7", "F2", "F3", "F8", "F6", "F4"]
    assert order[1] == ["F4", "F8", "F7", "F1", "F5", "F2", "F6", "F3"]
    assert list(plan(["m"], list(reversed(families)), [0, 1], reps=1)) == specs


def test_plan_and_analysis_share_the_trial_ids():
    spec = next(iter(plan(["deepseek/x"], ["F9"], [3], 1, "implicit", "forced", "bare")))

    assert spec.trial_id == "deepseek__x/F9-s03-clean-implicit-forced-bare-r0"


# --- one durable record per trial ------------------------------------------------------------


def test_a_builder_failure_is_persisted_as_an_infra_error_with_its_traceback(tmp_path, stubs):
    stubs.fail_build = FileNotFoundError("builder used a relative path")
    spec = spec_of()

    record = go(spec, tmp_path / "run", info=INFO)

    on_disk = json.loads((trial_dir(tmp_path / "run", spec) / "score.json").read_text())
    assert on_disk == record
    assert record["outcome"] == "infra_error" and record["stop"] == "build_failed"
    assert record["reason"].startswith("build failed: FileNotFoundError")
    assert "Traceback" in record["traceback"] and "relative path" in record["traceback"]
    assert record["attempt"] == 1 and record["usd"] == 0.0 and record["job_id"] == "job-1"
    assert list(neutral_root().iterdir()) == []  # no work directory left under the neutral root


def test_an_infra_error_is_rerun_until_it_has_had_its_attempts_and_then_stays(tmp_path, stubs):
    stubs.fail_build = RuntimeError("boom")
    spec = spec_of()

    attempts = [go(spec, tmp_path)["attempt"] for _ in range(MAX_ATTEMPTS)]
    final = go(spec, tmp_path)

    assert attempts == [1, 2, 3] and MAX_ATTEMPTS == 3
    assert final["attempt"] == 3 and final["outcome"] == "infra_error"
    assert len(stubs.builds) == 3  # the fourth call found the final record and ran nothing


def test_an_unexpected_exception_after_the_agent_ran_keeps_its_spend(tmp_path, stubs, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("the scorer cannot read this answer")

    monkeypatch.setattr(trial_module, "score_result", broken)
    stubs.traj["usd"] = 0.0123
    budget = Budget(5)
    spec = spec_of()

    record = go(spec, tmp_path, budget)

    assert record["outcome"] == "infra_error" and record["stop"] == "crashed"
    assert record["reason"].startswith("crashed: ValueError: the scorer")
    assert "score_result" in record["traceback"] or "broken" in record["traceback"]
    assert record["usd"] == 0.0123 and budget.spent == 0.0123  # the money was spent
    assert (trial_dir(tmp_path, spec) / "work" / "result.json").exists()  # the evidence is kept


def test_an_exception_while_the_agent_works_is_persisted_too(tmp_path, stubs):
    stubs.fail_drive = RuntimeError("the sandbox fell over")

    record = go(spec_of(), tmp_path)

    assert record["outcome"] == "infra_error" and "the sandbox fell over" in record["reason"]
    assert list(neutral_root().iterdir()) == []


def test_an_interrupt_is_not_swallowed_and_leaves_no_orphan_directory(tmp_path, stubs):
    stubs.fail_drive = KeyboardInterrupt()
    spec = spec_of()

    with pytest.raises(KeyboardInterrupt):
        go(spec, tmp_path)

    assert list(neutral_root().iterdir()) == []  # the work directory is gone from /tmp
    assert not (trial_dir(tmp_path, spec) / "score.json").exists()  # and no record claims a result
    stubs.fail_drive = None
    assert go(spec, tmp_path)["outcome"] == "pass"  # the next run starts it again


@pytest.mark.parametrize("stop", ["credit_exhausted", "circuit_open", "api_error"])
def test_a_credit_or_breaker_stop_is_an_infra_error_that_carries_its_stop_code(
    tmp_path, stubs, stop
):
    stubs.traj.update(stop=stop)
    stubs.answer = None

    record = go(spec_of(), tmp_path)

    assert record["outcome"] == "infra_error" and record["stop"] == stop


def test_a_skipped_trial_has_no_score_json_but_a_line_in_unrun_jsonl_and_a_topup_resumes_it(
    tmp_path, stubs
):
    spent = Budget(0.01)
    spent.charge(0.02)
    spec = spec_of()

    skipped = go(spec, tmp_path, spent, INFO)

    assert skipped["outcome"] == "skipped_budget"
    assert not trial_dir(tmp_path, spec).exists() and not stubs.builds
    lines = [json.loads(x) for x in (tmp_path / "unrun.jsonl").read_text().splitlines()]
    assert lines[0]["trial"] == spec.trial_id and lines[0]["reason"] == "skipped_budget"
    assert lines[0]["job_id"] == "job-1"

    resumed = go(spec, tmp_path, Budget(5))  # after a top-up the trial is simply run
    assert resumed["outcome"] == "pass" and (trial_dir(tmp_path, spec) / "score.json").exists()


def test_a_truncated_score_json_is_moved_aside_and_the_trial_rerun(tmp_path, stubs):
    spec = spec_of()
    folder = trial_dir(tmp_path, spec)
    folder.mkdir(parents=True)
    (folder / "score.json").write_text('{"outcome": "pass", "tru')  # a run killed while writing

    record = go(spec, tmp_path)

    assert record["outcome"] == "pass" and record["attempt"] == 1
    kept = list((tmp_path / "quarantine").iterdir())
    assert len(kept) == 1 and kept[0].read_text() == '{"outcome": "pass", "tru'
    assert json.loads((folder / "score.json").read_text()) == record


def test_a_finished_trial_is_returned_as_it_is_and_not_run_again(tmp_path, stubs):
    spec = spec_of()
    first = go(spec, tmp_path)

    again = go(spec, tmp_path)

    assert again == first and len(stubs.builds) == 1


def test_score_json_is_replaced_whole_or_not_at_all(tmp_path, monkeypatch):
    path = tmp_path / "score.json"
    write_json_atomic(path, {"outcome": "pass"})

    with pytest.raises(TypeError):  # half the new content is written, then the object fails
        write_json_atomic(path, {"outcome": "silent_wrong", "bad": object()})
    assert json.loads(path.read_text()) == {"outcome": "pass"}
    assert [p.name for p in tmp_path.iterdir()] == ["score.json"]  # and no temporary file stays

    def refused(*args):
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError, match="no space"):
        write_json_atomic(path, {"outcome": "protocol_fail"})
    assert json.loads(path.read_text()) == {"outcome": "pass"}
    assert [p.name for p in tmp_path.iterdir()] == ["score.json"]


def test_read_only_files_and_directories_in_an_old_trial_do_not_stop_the_rerun(tmp_path, stubs):
    spec = spec_of()
    folder = trial_dir(tmp_path, spec)
    locked = folder / "work" / "locked"
    locked.mkdir(parents=True)
    (locked / "f.txt").write_text("x")
    (locked / "f.txt").chmod(0o400)
    locked.chmod(0o500)  # nobody can delete its entries
    (folder / "work").chmod(0o500)
    write_json_atomic(folder / "score.json", {"outcome": "infra_error", "attempt": 1})

    record = go(spec, tmp_path)

    assert record["outcome"] == "pass" and record["attempt"] == 2


def test_remove_tree_never_follows_a_link_out_of_the_tree(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    outside.chmod(0o500)
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "link").symlink_to(outside)
    (tree / "sub" / "link2").symlink_to(outside / "keep.txt")

    make_writable(tree)
    assert stat.S_IMODE(outside.stat().st_mode) == 0o500  # the link target was left alone
    remove_tree(tree)

    assert not tree.exists() and (outside / "keep.txt").read_text() == "keep"
    outside.chmod(0o700)


def test_record_crash_persists_an_infra_error_but_never_replaces_a_finished_record(tmp_path):
    spec = spec_of()

    crashed = record_crash(spec, tmp_path, RuntimeError("lost"), INFO)
    second = record_crash(spec, tmp_path, RuntimeError("lost again"), INFO)
    done = trial_dir(tmp_path, spec) / "score.json"
    done.write_text(json.dumps({"outcome": "pass", "attempt": 1}))
    third = record_crash(spec, tmp_path, RuntimeError("late"), INFO)

    assert crashed["attempt"] == 1 and second["attempt"] == 2
    assert (
        crashed["outcome"] == "infra_error" and crashed["reason"] == "crashed: RuntimeError: lost"
    )
    assert (
        third == {"outcome": "pass", "attempt": 1}
        and json.loads(done.read_text())["outcome"] == "pass"
    )


# --- the record's provenance fields -----------------------------------------------------------


def test_every_record_carries_the_attempt_the_start_time_and_the_job_digests(tmp_path, stubs):
    stubs.messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "bash",
                    "input": {"command": "bcftools view -H -R regions.tsv variants.vcf.gz"},
                }
            ],
        }
    ]

    record = go(spec_of(), tmp_path, info=INFO)

    assert record["attempt"] == 1 and record["job_id"] == "job-1"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["started"])
    assert (record["code_digest"], record["run_config_sha"]) == ("c" * 64, "r" * 64)
    assert record["tool_manifest_sha"] == "t" * 64
    assert record["exposed"] is True and record["exposed_issued"] is True  # recorded in every cell
    assert record["scorer"] == scoring.SCORER_VERSION == 3
    assert record["pin_match"] is None and record["finish_reasons"] == {}


def test_without_a_job_the_provenance_fields_are_null_not_missing(tmp_path, stubs):
    record = go(spec_of(), tmp_path)

    for field in ("job_id", "code_digest", "run_config_sha", "tool_manifest_sha"):
        assert field in record and record[field] is None


def test_exposure_in_the_record_counts_commands_that_ran_and_commands_issued_apart(tmp_path, stubs):
    refused = "BLOCKED, the command did not run. FIX: declare it"
    stubs.messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "bash",
                    "input": {"command": "bcftools view -H -R regions.tsv variants.vcf.gz"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": refused}],
        },
    ]

    record = go(spec_of(), tmp_path)

    assert record["exposed"] is False and record["exposed_issued"] is True


# --- the opaque work directory (FC-2) --------------------------------------------------------


def test_the_agent_works_in_a_randomly_named_directory_that_is_moved_to_the_trial(tmp_path, stubs):
    spec = TrialSpec(MODEL, "F61", 3, "faulted", tier="implicit", route="forced", contract="bare")

    record = go(spec, tmp_path)

    (used,) = stubs.works
    assert used.parent == neutral_root() and OPAQUE.fullmatch(used.name)
    assert stubs.builds == [used]  # built where the agent worked
    final = trial_dir(tmp_path, spec) / "work"
    assert final.is_dir() and not final.is_symlink() and (final / "result.json").exists()
    assert (final / "README.md").read_text() == "the task"
    assert not used.exists() and list(neutral_root().iterdir()) == []
    mapping = json.loads((trial_dir(tmp_path, spec) / "task.json").read_text())["work_dir"]
    assert mapping == {"opaque": str(used), "final": str(final)}
    assert record["outcome"] == "pass"  # scored from the opaque directory, before the move


def test_two_trials_get_two_different_directories_and_the_mode_is_private(neutral):
    first, second = opaque_work_dir(), opaque_work_dir()

    assert first != second and first.parent == second.parent == neutral.resolve()
    assert stat.S_IMODE(first.stat().st_mode) == 0o700


def test_the_default_neutral_root_is_the_system_temp_dir_and_layer_py_agrees(monkeypatch, tmp_path):
    monkeypatch.delenv(NEUTRAL_VAR, raising=False)
    assert neutral_root() == Path(tempfile.gettempdir()).resolve() == layer_module.neutral_root()

    monkeypatch.setenv(NEUTRAL_VAR, str(tmp_path / "elsewhere"))
    assert neutral_root() == layer_module.neutral_root() == (tmp_path / "elsewhere").resolve()


def test_across_filesystems_the_work_directory_is_copied_and_the_source_removed(
    tmp_path, monkeypatch
):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.txt").write_text("a")
    (src / "result.json").write_text("{}")
    (src / "result.json").chmod(0o400)  # a read-only file
    (src / "sub").chmod(0o500)  # and a directory nobody can write to
    (src / "link").symlink_to("sub/a.txt")
    (src / "dangling").symlink_to("/no/such/place")
    os.mkfifo(src / "pipe")  # copying it would block forever

    def other_device(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "rename", other_device)
    move_tree(src, dst)

    assert not src.exists()
    assert (dst / "sub" / "a.txt").read_text() == "a" and (dst / "result.json").read_text() == "{}"
    assert (dst / "link").is_symlink() and os.readlink(dst / "link") == "sub/a.txt"
    assert os.readlink(dst / "dangling") == "/no/such/place"
    assert not (dst / "pipe").exists()
    remove_tree(dst)


def test_a_failed_copy_leaves_the_source_untouched_and_no_partial_destination(
    tmp_path, monkeypatch
):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.txt").write_text("a")

    def other_device(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    def failing_copy(*args, **kwargs):
        raise OSError(errno.EIO, "disk error")

    monkeypatch.setattr(os, "rename", other_device)
    monkeypatch.setattr(trial_module.shutil, "copytree", failing_copy)
    with pytest.raises(OSError, match="disk error"):
        move_tree(src, dst)

    assert (src / "a.txt").read_text() == "a" and not dst.exists()


def test_a_work_directory_that_cannot_be_moved_is_kept_and_the_trial_is_an_infra_error(
    tmp_path, stubs, monkeypatch
):
    def refused(*args, **kwargs):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(trial_module, "move_tree", refused)
    spec = spec_of()

    record = go(spec, tmp_path)

    (used,) = stubs.works
    assert record["outcome"] == "infra_error" and f"kept at {used}" in record["reason"]
    assert (used / "result.json").exists()  # the data are still where the agent left them


def test_a_refused_launch_leaves_nothing_behind(tmp_path, stubs, monkeypatch):
    def refuse(*args, **kwargs):
        raise layer_module.PromotionMismatch("the table changed")

    monkeypatch.setattr(trial_module, "_run", refuse)
    spec = spec_of()

    with pytest.raises(layer_module.PromotionMismatch):
        go(spec, tmp_path)

    assert list(neutral_root().iterdir()) == [] and not trial_dir(tmp_path, spec).exists()


def test_a_neutral_root_that_cannot_run_programs_is_refused(neutral, monkeypatch):
    assert check_neutral_root() == neutral.resolve() and list(neutral.iterdir()) == []
    real_access = os.access

    def noexec(path, mode, **kwargs):
        return False if mode == os.X_OK else real_access(path, mode, **kwargs)

    monkeypatch.setattr(os, "access", noexec)
    with pytest.raises(OSError, match="cannot be run.*BIOWARDEN_NEUTRAL_ROOT"):
        check_neutral_root()
    assert list(neutral.iterdir()) == []


# --- real bubblewrap: nothing the agent can read names the trial -------------------------------


@pytest.fixture(scope="module")
def real_tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not sandbox.available():
        pytest.skip("needs samtools, bcftools, bedtools and bubblewrap")
    return env


def tool_use(command, call_id="toolu_1"):
    body = {"type": "tool_use", "id": call_id, "name": "bash", "input": {"command": command}}
    return SimpleNamespace(model_dump=lambda mode="json": body, **body)


class ScriptedClient:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.messages = self

    def create(self, **kwargs):
        return self.steps.pop(0)


def reply(stop, *blocks):
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, usage=usage, model=MODEL, stop_details=None
    )


PROBE = (
    "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < \"$f\"; echo; done; cat /proc/self/mountinfo"
)


@pytest.mark.parametrize("arm", ["bare", "block-nonote"])
def test_nothing_the_agent_reads_in_proc_names_the_trial(real_tools, arm, monkeypatch):
    # hex, not the test name: the base directory must not spell a word the probe looks for
    base = Path(tempfile.gettempdir()) / secrets.token_hex(6)
    base.mkdir()
    try:
        monkeypatch.setenv(NEUTRAL_VAR, str(base / "neutral"))
        spec = TrialSpec(MODEL, "F1", 0, "faulted", arm=arm)
        client = ScriptedClient(reply("tool_use", tool_use(PROBE)), reply("end_turn"))

        record = run_trial(
            spec, base / "runs", client, real_tools, AgentConfig(MODEL), Budget(5), INFO
        )

        folder = base / "runs" / "trials" / spec.trial_id
        transcript = json.loads((folder / "transcript.json").read_text())
        seen = json.dumps(transcript).lower()
        assert record["commands"] == 1 and "/work" in seen
        arm_words = {word for name in cli.ARMS for word in (name, *name.split("-"))}
        models = {"claude", "haiku", "deepseek", "gpt-oss", "openai", "qwen", "mimo", "glm"}
        for word in ("faulted", "clean", *sorted(arm_words), *sorted(models)):
            assert word not in seen, word
        assert (folder / "work").is_dir() and not (folder / "work").is_symlink()
        assert list((base / "neutral").iterdir()) == []  # shims, socket and work all gone
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --- run: the job as a whole ------------------------------------------------------------------


class FakeRunner:
    """Stands in for run_trial inside the pool: writes a record like the real one would."""

    def __init__(self):
        self.calls: list[str] = []
        self.script: dict[str, object] = {}  # trial id -> Exception | "skip" | record fields
        self.seen_files: list[set[str]] = []

    def __call__(self, spec, root, clients, tools, cfg, budget, info=None):
        done = Path(root) / "trials" / spec.trial_id / "score.json"
        if done.exists():
            cached = json.loads(done.read_text())
            if cached["outcome"] != "infra_error" or cached.get("attempt", 1) >= MAX_ATTEMPTS:
                return cached  # finished trials are not run again
        self.calls.append(spec.trial_id)
        self.seen_files.append({p.name for p in Path(root).iterdir()})
        action = self.script.get(spec.trial_id, {})
        if isinstance(action, Exception):
            raise action
        if action == "skip":
            note_unrun(Path(root), spec, "skipped_budget")
            return {"trial": spec.trial_id, "outcome": "skipped_budget", "usd": 0.0}
        record = {
            "trial": spec.trial_id,
            **asdict(spec),
            "outcome": "pass",
            "delivered": True,
            "stop": "end_turn",
            "turns": 2,
            "usd": 0.01,
            "attempt": 1,
            "exposed": False,
            "job_id": info.job_id if info else None,
            **action,
        }
        folder = Path(root) / "trials" / spec.trial_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / "score.json", record)
        budget.charge(record["usd"])
        return record


@pytest.fixture
def job(tmp_path, monkeypatch, neutral):
    tool_dir = tmp_path / "tooldir" / "bin"
    tool_dir.mkdir(parents=True)
    runner = FakeRunner()
    tools = ToolEnv((tool_dir,))
    monkeypatch.setattr(ToolEnv, "discover", classmethod(lambda cls, root=None: tools))
    monkeypatch.setattr(sandbox, "available", lambda: True)
    monkeypatch.setattr(cli, "get_key", lambda name: "test-key")
    monkeypatch.setattr(cli, "run_trial", runner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return SimpleNamespace(
        runner=runner, out=tmp_path / "out", tools=tools, tool_dir=tool_dir, tmp=tmp_path
    )


def run_cli(job, *extra, families=("F1", "F2"), seeds="0-1", models=(MODEL,)):
    argv = ["run", "--out", str(job.out), "--models", *models, "--families", *families]
    return main([*argv, "--seeds", seeds, "--contract", "bare", "--workers", "2", *extra])


def jobs_lines(job):
    return [json.loads(x) for x in (job.out / "jobs.jsonl").read_text().splitlines()]


def test_plan_and_run_files_exist_before_the_first_trial_starts(job, capsys):
    status = run_cli(job)

    assert status == 0 and len(job.runner.calls) == 2 * 2 * 2
    assert all({"plan.json", "run.json"} <= files for files in job.runner.seen_files)
    plan_json = json.loads((job.out / "plan.json").read_text())
    expected = [s.trial_id for s in plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare")]
    assert plan_json["ids"] == expected  # the order the runner runs them in
    assert set(plan_json["specs"]) == set(expected) == set(job.runner.calls)
    run_json = json.loads((job.out / "run.json").read_text())
    config = run_json["config"]
    assert config["agent"]["effort"] == "medium" and config["agent"]["max_turns"] == 30
    assert config["agent"]["max_tokens"] > 0 and config["agent"]["max_trial_usd"] == 0.75
    assert set(config["prompts"]) == {"full", "bare", "layer_note", "careful_note"}
    assert re.fullmatch(r"[0-9a-f]{64}", config["code"]["digest"])
    assert config["promotion"]["sha256"] and config["scorer"] == 3
    assert run_json["run_config_sha"] and run_json["jobs"][0]["job_id"].startswith("local-")
    assert "ledger: " in capsys.readouterr().out


def test_the_job_ends_with_one_ledger_line_per_run(job, capsys):
    assert run_cli(job) == 0

    (line,) = jobs_lines(job)
    assert line["planned"] == 8 and line["scored"] == {"pass": 8} and line["absent"] == 0
    assert (line["infra"], line["skipped_budget"], line["build_failed"], line["aborted"]) == (
        0,
        0,
        0,
        None,
    )
    assert line["usd"] == pytest.approx(0.08)
    assert f"ledger: {json.dumps(line, sort_keys=True)}" in capsys.readouterr().out


def test_a_planned_trial_without_a_record_makes_the_job_fail_and_is_counted(job, capsys):
    first = next(iter(plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare"))).trial_id
    job.runner.script[first] = "skip"  # the budget skipped it: no score.json

    status = run_cli(job)

    (line,) = jobs_lines(job)
    assert status == 3 and line["absent"] == 1 and line["skipped_budget"] == 1
    assert line["scored"] == {"pass": 7}
    assert "INCOMPLETE" in capsys.readouterr().err


def test_an_infra_error_with_attempts_left_makes_the_job_fail_loudly(job):
    first = next(iter(plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare"))).trial_id
    job.runner.script[first] = {"outcome": "infra_error", "stop": "api_error", "attempt": 1}
    assert run_cli(job) == 3

    job.runner.script[first] = {"outcome": "infra_error", "stop": "api_error", "attempt": 3}
    assert run_cli(job) == 0  # three attempts: excluded and counted, nothing left to rerun
    assert jobs_lines(job)[-1]["infra"] == 1 and jobs_lines(job)[-1]["infra_to_rerun"] == 0


def test_one_exception_in_one_trial_never_stops_the_pool(job):
    ids = [s.trial_id for s in plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare")]
    job.runner.script[ids[2]] = RuntimeError("html where json was expected")

    status = run_cli(job)

    assert status == 3 and sorted(job.runner.calls) == sorted(ids)  # every trial was started
    crashed = json.loads((job.out / "trials" / ids[2] / "score.json").read_text())
    assert crashed["outcome"] == "infra_error" and crashed["attempt"] == 1
    assert crashed["reason"] == "crashed: RuntimeError: html where json was expected"
    (line,) = jobs_lines(job)
    assert line["scored"] == {"pass": 7} and line["infra"] == 1 and line["infra_to_rerun"] == 1


def test_exhausted_credit_stops_the_pool_and_the_job_says_so(job, capsys):
    ids = [s.trial_id for s in plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare")]
    job.runner.script[ids[0]] = {"outcome": "infra_error", "stop": "credit_exhausted"}

    status = run_cli(job, "--workers", "1")

    (line,) = jobs_lines(job)
    assert status == 4 and line["aborted"].startswith("credit_exhausted")
    assert job.runner.calls == [ids[0]]  # nothing was started after the stop code
    assert line["absent"] == len(ids) - 1 and line["infra"] == 1
    assert "STOP: credit_exhausted" in capsys.readouterr().out

    job.runner.script.clear()  # the account is topped up: the same command resumes
    assert run_cli(job, "--workers", "1") == 0
    assert job.runner.calls == [ids[0], *ids]  # the infra_error is rerun, the rest is new


def test_a_final_credit_record_left_by_an_earlier_run_does_not_stop_this_one(job):
    ids = [s.trial_id for s in plan([MODEL], ["F1", "F2"], [0, 1], 1, contract="bare")]
    final = {"outcome": "infra_error", "stop": "credit_exhausted", "attempt": MAX_ATTEMPTS}
    job.runner.script[ids[0]] = final
    assert run_cli(job, "--workers", "1") == 4  # the first run is stopped by its own record

    job.runner.script.clear()
    status = run_cli(job, "--workers", "1")

    assert status == 0  # its three attempts are used up: excluded and counted, not run again
    assert job.runner.calls == [ids[0], *ids[1:]] and jobs_lines(job)[-1]["infra"] == 1


def test_a_changed_promotion_table_stops_the_pool_without_one_record_per_trial(job, monkeypatch):
    def mismatch(spec, *args, **kwargs):
        raise layer_module.PromotionMismatch("106 rules have no entry")

    monkeypatch.setattr(cli, "run_trial", mismatch)

    status = run_cli(job, "--workers", "1")

    (line,) = jobs_lines(job)
    assert status == 4 and "promotion table" in line["aborted"] and line["absent"] == 8
    assert not list((job.out / "trials").glob("*/*/score.json"))  # no infra_error for each trial


def test_resuming_with_another_configuration_is_refused_and_names_the_difference(job, capsys):
    assert run_cli(job) == 0
    before = (job.out / "plan.json").read_text()
    calls = len(job.runner.calls)

    status = run_cli(job, "--effort", "high")

    err = capsys.readouterr().err
    assert status == 2 and "agent.effort: 'medium' != 'high'" in err
    assert len(job.runner.calls) == calls and (job.out / "plan.json").read_text() == before


def test_resuming_with_the_same_configuration_is_accepted_and_adds_a_job_entry(job):
    assert run_cli(job) == 0
    assert run_cli(job, "--max-usd", "9", "--workers", "1") == 0  # budget and workers may differ

    run_json = json.loads((job.out / "run.json").read_text())
    assert len(run_json["jobs"]) == 2 and len(job.runner.calls) == 8  # the second run found all
    assert len(json.loads((job.out / "plan.json").read_text())["jobs"]) == 2


def test_a_directory_with_records_but_no_run_json_is_not_resumed(job, capsys):
    stale = job.out / "trials" / "m" / "F1-s00-clean-r0"
    stale.mkdir(parents=True)
    (stale / "score.json").write_text(json.dumps({"outcome": "pass"}))

    assert run_cli(job) == 2

    assert "no run.json" in capsys.readouterr().err and not job.runner.calls


def test_other_arms_share_a_directory_and_plan_json_lists_the_union(job):
    assert run_cli(job) == 0
    assert run_cli(job, "--arm", "note") == 0

    plan_json = json.loads((job.out / "plan.json").read_text())
    assert len(plan_json["ids"]) == 16 == len(set(plan_json["ids"]))
    arms = {spec["arm"] for spec in plan_json["specs"].values()}
    assert arms == {"bare", "note"}
    assert [j["n"] for j in plan_json["jobs"]] == [8, 8]


def test_exclude_families_and_repeated_seeds(job):
    status = run_cli(job, "--exclude-families", "F2", families=("F1", "F2", "F3"), seeds="0-1,1")

    assert status == 0 and len(job.runner.calls) == 2 * 2 * 2  # F1 and F3, seeds 0 and 1
    assert {c.split("/")[1].split("-")[0] for c in job.runner.calls} == {"F1", "F3"}
    assert len(set(job.runner.calls)) == len(job.runner.calls)


def test_a_run_whose_plan_is_empty_is_refused(job, capsys):
    assert run_cli(job, "--exclude-families", "F1", "F2") == 2

    assert "plan is empty" in capsys.readouterr().err


def test_the_code_must_come_from_the_declared_snapshot(job, capsys):
    elsewhere = job.tmp / "no-such-snapshot"

    assert run_cli(job, "--snapshot", str(elsewhere)) == 2
    assert "not from the snapshot" in capsys.readouterr().err and not job.runner.calls

    assert run_cli(job, "--snapshot", str(ROOT)) == 0  # the package is imported from under ROOT
    run_json = json.loads((job.out / "run.json").read_text())
    assert run_json["jobs"][0]["snapshot"] == str(ROOT)


def test_a_layer_arm_needs_a_promotion_table_that_matches_the_specs(job, capsys):
    table = job.tmp / "promotion.json"
    table.write_text("{}")

    assert run_cli(job, "--arm", "block-nonote", "--promotion", str(table)) == 2
    assert "does not match the specs" in capsys.readouterr().err and not job.runner.calls

    assert run_cli(job, "--promotion", str(table)) == 0  # a bare run only records the table


def test_the_workers_are_capped_outside_a_scheduler_job(job, capsys):
    assert run_cli(job, "--workers", "8") == 0

    assert re.search(r"workers reduced from 8 to [12]\b", capsys.readouterr().out)


@pytest.mark.parametrize(
    "name", ["haiku-run", "bare", "faulted-twin", "stage2-F12", "block-arm", "seed7", "DeepSeek"]
)
def test_a_neutral_root_whose_path_names_a_trial_an_arm_or_a_model_is_refused(monkeypatch, name):
    base = Path(tempfile.gettempdir()) / secrets.token_hex(6)
    monkeypatch.setenv(NEUTRAL_VAR, str(base / name))
    try:
        with pytest.raises(OSError, match="sandbox can read that path"):
            check_neutral_root()
        assert not base.exists()  # nothing was made there
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_a_neutral_root_in_the_job_is_checked_before_anything_is_written(job, monkeypatch, capsys):
    monkeypatch.setenv(NEUTRAL_VAR, str(job.tmp / "claude-haiku"))

    assert run_cli(job) == 2

    assert "'claude'" in capsys.readouterr().err
    assert not (job.out / "run.json").exists() and not job.runner.calls


def test_a_neutral_root_that_cannot_run_programs_refuses_the_job(job, monkeypatch, capsys):
    real_access = os.access

    def noexec(path, mode, **kwargs):
        return False if mode == os.X_OK else real_access(path, mode, **kwargs)

    monkeypatch.setattr(os, "access", noexec)

    assert run_cli(job) == 2

    assert "BIOWARDEN_NEUTRAL_ROOT" in capsys.readouterr().err and not job.runner.calls


# --- what the manifest and the digests are made of ---------------------------------------------


def test_the_code_digest_follows_the_files_and_ignores_bytecode(tmp_path):
    package = tmp_path / "pkg"
    (package / "sub" / "__pycache__").mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    (package / "sub" / "b.yaml").write_text("k: v\n")
    first = code_digest(package)

    (package / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (package / "a.pyc").write_bytes(b"\x01")
    assert code_digest(package) == first  # caches and bytecode do not count

    (package / "sub" / "b.yaml").write_text("k: w\n")
    changed = code_digest(package)
    assert changed != first

    (package / "sub" / "b.yaml").rename(package / "sub" / "c.yaml")
    assert code_digest(package) not in (first, changed)  # a rename counts
    assert re.fullmatch(r"[0-9a-f]{64}", code_digest())  # the package this test imports


def test_config_differences_names_each_changed_key():
    old = {"agent": {"effort": "medium", "max_turns": 30}, "scorer": 3, "gone": 1}
    new = {"agent": {"effort": "high", "max_turns": 30}, "scorer": 3, "added": 2}

    assert config_differences(old, new) == [
        "added: <absent> != 2",
        "agent.effort: 'medium' != 'high'",
        "gone: 1 != <absent>",
    ]
    assert config_differences(old, old) == []


def test_the_tool_manifest_records_versions_locks_and_changes_with_either(tmp_path):
    tool_dir = tmp_path / "tools" / "bin"
    tool_dir.mkdir(parents=True)
    script = tool_dir / "samtools"
    script.write_text('#!/bin/sh\necho "samtools 9.9"\necho "Using htslib 9.9"\n')
    script.chmod(0o755)
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "bio-a.tsv").write_text("numpy\t2.2.6\n")
    tools = ToolEnv((tool_dir,))

    manifest = tool_manifest(tools, locks, None)

    assert manifest["versions"] == {"samtools": "samtools 9.9"}  # only the tools that exist
    assert list(manifest["env_locks"]) == ["bio-a.tsv"] and manifest["tools_json_sha256"] is None
    assert manifest["sdk"]["anthropic"] and manifest["sdk"]["openai"]
    sha = cli.canonical_sha(manifest)

    script.write_text('#!/bin/sh\necho "samtools 9.10"\n')
    assert cli.canonical_sha(tool_manifest(tools, locks, None)) != sha
    script.write_text('#!/bin/sh\necho "samtools 9.9"\necho "Using htslib 9.9"\n')
    (locks / "bio-a.tsv").write_text("numpy\t2.3.0\n")
    assert cli.canonical_sha(tool_manifest(tools, locks, None)) != sha


# --- reading the directory: report, loaders, rescore --------------------------------------------


def write_record(root, spec, **fields):
    record = {
        "trial": spec.trial_id,
        **asdict(spec),
        "outcome": "pass",
        "delivered": True,
        "flagged": False,
        "exposed": True,
        "usd": 0.01,
        **fields,
    }
    folder = root / "trials" / spec.trial_id
    folder.mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder / "score.json", record)
    return record


def test_report_groups_by_arm_and_shows_silent_over_planned(tmp_path):
    root = tmp_path / "run"
    bare = [TrialSpec(MODEL, f"F{n}", 0, "faulted", contract="bare") for n in range(1, 6)]
    layer = [
        TrialSpec(MODEL, f"F{n}", 0, "faulted", contract="bare", arm="note") for n in range(1, 6)
    ]
    root.mkdir()
    write_json_atomic(
        root / "plan.json",
        {
            "ids": [s.trial_id for s in bare + layer],
            "specs": {s.trial_id: asdict(s) for s in bare + layer},
        },
    )
    for i, spec in enumerate(bare):  # 3 silent of 5 planned, one of them undelivered-looking
        write_record(root, spec, outcome="silent_wrong" if i < 3 else "pass")
    for i, spec in enumerate(layer[:4]):  # the fifth layer trial is absent
        write_record(root, spec, outcome="silent_wrong" if i == 0 else "pass")
    sink = tmp_path / "report.txt"

    with open(sink, "w") as out:
        summary = report(root, out=out)

    cell, cell_note = cli.cell_of(asdict(bare[0])), cli.cell_of(asdict(layer[0]))
    assert cell != cell_note  # the arm is part of the cell label
    plain = summary[f"{cell}|faulted|{MODEL}"]
    arm = summary[f"{cell_note}|faulted|{MODEL}"]
    assert (plain["planned"], plain["silent"], plain["silent_planned"]) == (5, 3, 0.6)
    assert (arm["planned"], arm["silent"], arm["n"], arm["silent_planned"]) == (5, 1, 4, 0.2)
    text = sink.read_text()
    assert "sil/plan%" in text and "1 absent of 10 planned" in text
    assert cell_note in text


def test_report_without_a_plan_still_works_and_tolerates_bad_records(tmp_path, monkeypatch):
    root = tmp_path / "run"
    good = TrialSpec(MODEL, "F1", 0, "clean", contract="bare")
    torn = TrialSpec(MODEL, "F1", 1, "clean", contract="bare")
    gone = TrialSpec(MODEL, "F1", 2, "clean", contract="bare")
    write_record(root, good)
    write_record(root, torn)
    write_record(root, gone)
    (root / "trials" / torn.trial_id / "score.json").write_text('{"outcome": "pa')
    (root / "trials" / "m" / "F9-s00-clean-r0").mkdir(parents=True)
    (root / "trials" / "m" / "F9-s00-clean-r0" / "score.json").write_text("[1, 2]")
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.parent.name == gone.trial_id.split("/")[1] and self.name == "score.json":
            raise FileNotFoundError(self)  # a rerun removed the directory after the glob
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    unreadable: list[Path] = []

    records = load_records(root, unreadable)
    with open(os.devnull, "w") as out:
        summary = report(root, out=out)

    assert [r["trial"] for r in records] == [good.trial_id]
    assert len(unreadable) == 2  # the torn record and the one that is not an object
    only = summary[f"{cli.cell_of(asdict(good))}|clean|{MODEL}"]
    assert only["n"] == 1 and only["planned"] is None and only["silent_planned"] is None


def test_the_ledger_tells_scored_infra_skipped_build_failed_and_absent_apart(tmp_path):
    root = tmp_path / "run"
    specs = [TrialSpec(MODEL, f"F{n}", 0, "clean", contract="bare") for n in range(1, 9)]
    write_record(root, specs[0], outcome="pass")
    write_record(root, specs[1], outcome="silent_wrong")
    write_record(root, specs[2], outcome="infra_error", stop="api_error", attempt=1)
    write_record(root, specs[3], outcome="infra_error", stop="build_failed", attempt=3)
    (root / "trials" / specs[4].trial_id).mkdir(parents=True)
    (root / "trials" / specs[4].trial_id / "score.json").write_text("{trunc")
    note_unrun(root, specs[5], "skipped_budget")
    # specs[6] and specs[7] were never reached

    ledger = job_ledger(root, specs, "job-9", 0.1234, None)

    assert ledger["planned"] == 8 and ledger["scored"] == {"pass": 1, "silent_wrong": 1}
    assert (ledger["infra"], ledger["infra_to_rerun"], ledger["build_failed"]) == (2, 1, 1)
    assert (ledger["skipped_budget"], ledger["absent"], ledger["unreadable"]) == (1, 3, 1)
    assert cli.exit_status(ledger) == 3
    assert cli.exit_status({**ledger, "aborted": "credit_exhausted"}) == 4
    clean = {**ledger, "absent": 0, "infra_to_rerun": 0, "unreadable": 0}
    assert cli.exit_status(clean) == 0


def test_rescore_skips_what_it_cannot_read_instead_of_stopping(tmp_path, capsys):
    root = tmp_path / "run"
    no_work = TrialSpec(MODEL, "F1", 0, "clean", contract="bare")
    torn = TrialSpec(MODEL, "F1", 1, "clean", contract="bare")
    no_task = TrialSpec(MODEL, "F1", 2, "clean", contract="bare")
    for spec in (no_work, torn, no_task):
        write_record(root, spec, stop="end_turn")
    (root / "trials" / torn.trial_id / "score.json").write_text("{trunc")
    (root / "trials" / no_task.trial_id / "work").mkdir()

    assert rescore(root) == 0

    err = capsys.readouterr().err
    assert f"{no_work.trial_id.split('/')[1]}: no work directory" in err
    assert f"{torn.trial_id.split('/')[1]}: unreadable score.json" in err
    assert f"{no_task.trial_id.split('/')[1]}: FileNotFoundError" in err
