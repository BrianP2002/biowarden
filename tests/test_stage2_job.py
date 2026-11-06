"""tools/stage2_job.sbatch run under bash against a stand-in for the runner.

The stand-in (FAKE_CLI) replaces bench/runner/cli.py in a copy of the package that plays the
snapshot: it logs the arguments of each call, writes the plan and trial records a real run leaves,
and ends with a status the test scripts. Everything else is the real template, preflight.py,
freeze_hashes.py and stage2.py. No SLURM, no API, no sandbox.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from biowarden.bench import stage2

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools" / "stage2_job.sbatch"

FAKE_CLI = '''
"""Stand-in for the runner CLI (tests/test_stage2_job.py)."""
import argparse
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

ALIASES = {
    "haiku": "claude-haiku-5-5",
    "dsflash": "deepseek/deepseek-v4.1-flash",
    "gptoss": "openai/gpt-oss-120b",
}
Spec = namedtuple("Spec", "trial_id model family seed twin rep arm")


def parse_seeds(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def plan(models, families, seeds, reps, tier="explicit", route="free", contract="full",
         harness="minimal", arm="bare"):
    for rep in range(reps):
        for seed in dict.fromkeys(seeds):
            for family in sorted(set(families)):
                if family == "F4" and (contract, tier, route) != ("full", "explicit", "free"):
                    continue
                for twin in ("clean", "faulted"):
                    for model in dict.fromkeys(models):
                        name = f"{family}-s{seed:02d}-{twin}-{contract}-{tier}-{route}-{arm}-r{rep}"
                        yield Spec(model.replace("/", "__") + "/" + name, model, family, seed,
                                   twin, rep, arm)


def main(argv=None):
    p = argparse.ArgumentParser()
    run = p.add_subparsers(dest="cmd", required=True).add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--families", nargs="+", default=None)
    run.add_argument("--exclude-families", nargs="+", default=[])
    run.add_argument("--seeds", required=True)
    run.add_argument("--reps", type=int, default=1)
    run.add_argument("--tier")
    run.add_argument("--route")
    run.add_argument("--contract")
    run.add_argument("--arm")
    run.add_argument("--effort")
    run.add_argument("--max-turns", type=int)
    run.add_argument("--max-usd", type=float)
    run.add_argument("--max-trial-usd", type=float)
    run.add_argument("--workers", type=int)
    run.add_argument("--snapshot")
    run.add_argument("--env-locks")
    args = p.parse_args(argv)
    log = Path(os.environ["FAKE_LOG"])
    script = json.loads(Path(os.environ["FAKE_SCRIPT"]).read_text())
    calls = len(log.read_text().splitlines()) if log.exists() else 0
    step = script[calls] if calls < len(script) else {}
    with log.open("a") as handle:
        handle.write(json.dumps({**vars(args), "argv": sys.argv[1:]}) + "\\n")
    if args.families is None:
        from biowarden.bench.tasks.families import all_families
        args.families = list(all_families())
    families = [f for f in args.families if f not in set(args.exclude_families)]
    models = [ALIASES.get(m, m) for m in args.models]
    specs = list(plan(models, families, parse_seeds(args.seeds), args.reps, args.tier,
                      args.route, args.contract, "minimal", args.arm))
    out = Path(args.out)
    absent = step.get("absent", 0)
    for spec in specs[absent:]:
        folder = out / "trials" / spec.trial_id
        folder.mkdir(parents=True, exist_ok=True)
        row = {"trial": spec.trial_id, "outcome": "pass", "attempt": 1, "turns": 3, "usd": 0.001}
        (folder / "score.json").write_text(json.dumps(row))
    plan_file = out / "plan.json"
    ids = json.loads(plan_file.read_text())["ids"] if plan_file.exists() else []
    ids += [s.trial_id for s in specs if s.trial_id not in ids]
    plan_file.write_text(json.dumps({"format": 1, "ids": ids}))
    usd = step.get("usd", 0.01)
    ledger = {"job_id": "slurm-" + os.environ["SLURM_JOB_ID"], "planned": len(specs),
              "absent": absent, "usd": usd}
    with (out / "jobs.jsonl").open("a") as handle:
        handle.write(json.dumps(ledger) + "\\n")
    print("ledger: " + json.dumps(ledger))
    return step.get("rc", 3 if absent else 0)


if __name__ == "__main__":
    sys.exit(main())
'''

FRESH = ["F2", "F3", "F5", "F6"]


def sets_dict() -> dict:
    families = ["F2", "F10", "F3", "F5", "F6"]
    return {
        "format": 2,
        "primary": ["F2", "F10"],
        "q1": ["F3"],
        "q2": ["F5"],
        "q3": ["F6"],
        "excluded": [],
        "seeds": {f: list(stage2.registry_seeds(f)) for f in families},
        "default_seeds": {f: list(stage2.registry_seeds(f)[:3]) for f in [*families, "F4"]},
        "default_only": ["F4"],
    }


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("repo")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for dest in (base / "src", base / "runs" / "stage2-code" / "src"):
        shutil.copytree(ROOT / "src" / "biowarden", dest / "biowarden", ignore=ignore)
        (dest / "biowarden" / "bench" / "runner" / "cli.py").write_text(FAKE_CLI)
    for rel in (
        "tools/stage2_job.sbatch",
        "tools/preflight.py",
        "tools/env_lock.py",
        "docs/preregistration/freeze_hashes.py",
        "pyproject.toml",
    ):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, base / rel)
    shutil.copytree(ROOT / "envs" / "locks", base / "envs" / "locks")
    (base / "docs" / "preregistration" / "stage2-sets.json").write_text(json.dumps(sets_dict()))
    return base


class Job:
    def __init__(self, repo: Path, tmp_path: Path, script: list[dict] | None = None, **env: str):
        self.repo = repo
        self.out = f"runs/job-{tmp_path.name}"
        self.fake_log = tmp_path / "calls.jsonl"
        tmp_path.mkdir(parents=True, exist_ok=True)
        script_file = tmp_path / "script.json"
        script_file.write_text(json.dumps(script or []))
        self.env = {
            **{
                k: v
                for k, v in os.environ.items()
                if k != "PYTHONPATH" and not k.startswith("SLURM_")
            },
            "SLURM_SUBMIT_DIR": str(repo),
            "SLURM_JOB_ID": "4242",
            # a root of its own: jobs of different tests run side by side under pytest-xdist
            "BIOWARDEN_NEUTRAL_ROOT": f"/tmp/bw-job-{uuid.uuid4().hex[:8]}",
            "PYTHON": sys.executable,
            "MODEL": "haiku",
            "CELL": "primary",
            "OUT": self.out,
            "CAP": "10",
            "FREEZE": "none",
            "ENV_CHECK": "0",
            "SCAN": "0",
            "FAKE_LOG": str(self.fake_log),
            "FAKE_SCRIPT": str(script_file),
            **env,
        }
        self.env = {k: v for k, v in self.env.items() if v is not None}

    def run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.repo / "tools" / "stage2_job.sbatch")],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    @property
    def calls(self) -> list[dict]:
        if not self.fake_log.exists():
            return []
        return [json.loads(line) for line in self.fake_log.read_text().splitlines()]

    @property
    def log(self) -> str:
        return (self.repo / self.out / "job-4242.log").read_text()


def shape(call: dict) -> tuple:
    return (call["arm"], call["reps"], call["seeds"])


def test_every_chunk_runs_each_arm_and_the_order_is_reversed_in_odd_chunks(repo, tmp_path):
    job = Job(repo, tmp_path, CHUNK="2")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr
    # three chunks of at most two seeds; in each, a call per seed block (fresh families, then the
    # five that agents have seen); the replicate on P comes last
    chunks = [("1000,1001", "1020,1021"), ("1002,1003", "1022,1023"), ("1004", "1024")]
    expected = []
    for number, seeds in enumerate(chunks):
        arms = ["bare", "block-nonote", "warn-nonote"]
        order = arms if number % 2 == 0 else arms[::-1]
        for arm in order:
            expected += [(arm, 1, s) for s in seeds]
        expected += [("bare", 2, s) for s in seeds]
    assert [shape(c) for c in job.calls] == expected
    assert "DONE (stage2)" in done.stdout
    assert "DONE (stage2)" in job.log  # the per-job log has the same story


def test_every_call_is_accepted_by_the_real_runner_command_line(repo, tmp_path, monkeypatch):
    from biowarden.bench.runner import cli

    job = Job(repo, tmp_path, CHUNK="5")
    assert job.run().returncode == 0
    seen = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: seen.append(args) or 0)

    for call in job.calls:
        assert cli.main(call["argv"]) == 0  # an unknown flag or a bad choice would exit 2

    assert len(seen) == len(job.calls) == 8
    first = seen[0]
    assert (first.models, first.arm, first.effort, first.max_turns) == (
        ["haiku"], "bare", "medium", 30,
    )  # fmt: skip
    assert first.snapshot == "runs/stage2-code" and first.env_locks == "envs/locks"
    assert {a.arm for a in seen} == {"bare", "block-nonote", "warn-nonote"}  # aa runs as bare
    assert sorted({a.reps for a in seen}) == [1, 2]


def test_the_family_lists_come_from_the_frozen_sets_file(repo, tmp_path):
    job = Job(repo, tmp_path, CHUNK="5")

    assert job.run().returncode == 0
    by = {(c["arm"], c["seeds"]): c["families"] for c in job.calls if c["reps"] == 1}

    assert by[("bare", "1000,1001,1002,1003,1004")] == FRESH
    assert by[("block-nonote", "1020,1021,1022,1023,1024")] == ["F10"]
    assert by[("warn-nonote", "1000,1001,1002,1003,1004")] == ["F2", "F3"]  # P and Q1
    assert by[("warn-nonote", "1020,1021,1022,1023,1024")] == ["F10"]
    replicate = [c for c in job.calls if c["reps"] == 2]
    assert [c["families"] for c in replicate] == [["F2"], ["F10"]]  # P only


def test_every_call_carries_the_pinned_settings(repo, tmp_path):
    job = Job(repo, tmp_path, CHUNK="5", WORKERS="6", MAX_TRIAL_USD="0.5", AA="0")

    assert job.run().returncode == 0

    assert len(job.calls) == 6  # no replicate: three arms, two seed blocks
    for call in job.calls:
        assert call["out"] == job.out and call["models"] == ["haiku"]
        assert (call["contract"], call["tier"], call["route"]) == ("bare", "implicit", "forced")
        assert (call["effort"], call["max_turns"], call["max_trial_usd"]) == ("medium", 30, 0.5)
        assert (call["workers"], call["snapshot"]) == (6, "runs/stage2-code")
        assert call["env_locks"] == "envs/locks"
        assert call["exclude_families"] == []


def test_a_call_that_ends_with_status_3_is_run_again_with_the_rest_of_the_cap(repo, tmp_path):
    script = [{"rc": 3, "absent": 2, "usd": 1.5}]  # the first call leaves two trials open
    job = Job(repo, tmp_path, script, CHUNK="5", CAP="10")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr
    first, second, third = job.calls[:3]
    assert shape(first) == shape(second) and first["families"] == second["families"]
    assert shape(third) != shape(second)
    assert first["max_usd"] == 10.0 and second["max_usd"] == pytest.approx(8.5)
    assert third["max_usd"] == pytest.approx(8.49)  # 10 - 1.5 - 0.01
    assert len(job.calls) == 9  # eight calls, one of them twice
    assert "try 2" in job.log


def test_after_three_tries_the_job_goes_on_and_ends_incomplete(repo, tmp_path):
    script = [{"rc": 3, "absent": 2}] * 3  # three tries, three times two trials left open
    job = Job(repo, tmp_path, script, CHUNK="5", AA="0")

    done = job.run()

    assert done.returncode == 1
    assert len(job.calls) == 3 + 5  # the first call three times, the five others once each
    assert "FAILED chunk 0 bare seeds 1000,1001,1002,1003,1004: status 3" in job.log
    assert "INCOMPLETE" in job.log and "DONE" not in job.log
    assert "bare: 2 absent" in done.stdout  # and the data say which


def test_status_4_stops_the_job(repo, tmp_path):
    job = Job(repo, tmp_path, [{}, {"rc": 4}], CHUNK="5")

    done = job.run()

    assert done.returncode == 1
    assert len(job.calls) == 2  # nothing is started after credit ran out or the API went down
    assert "status 4" in job.log and "INCOMPLETE" in job.log


def test_the_cap_covers_all_the_calls_of_the_job_together(repo, tmp_path):
    script = [{"usd": 0.04}] * 10
    job = Job(repo, tmp_path, script, CHUNK="5", CAP="0.05")

    done = job.run()

    assert done.returncode == 1
    assert [round(c["max_usd"], 2) for c in job.calls] == [0.05, 0.01]
    assert "STOP: the cap of 0.05 USD is spent" in job.log
    assert "INCOMPLETE" in job.log


def test_success_is_what_the_records_say_not_what_the_runner_returned(repo, tmp_path):
    job = Job(
        repo, tmp_path, [{"rc": 0, "absent": 1}], CHUNK="5", AA="0"
    )  # exit 0, one trial missing

    done = job.run()

    assert done.returncode == 1
    assert len(job.calls) == 6  # every call ended with status 0 ...
    assert "FAILED" not in job.log
    assert "bare: 1 absent" in done.stdout  # ... and the data check caught it
    assert "INCOMPLETE" in job.log


def test_a_registered_trial_that_was_never_planned_is_found(repo, tmp_path):
    # nothing runs the warn-nonote arm: the frozen plan has trials for it, plan.json has none
    job = Job(repo, tmp_path, CHUNK="5", AA="0")
    out = repo / job.out
    assert job.run().returncode == 0
    plan = json.loads((out / "plan.json").read_text())
    plan["ids"] = [i for i in plan["ids"] if "warn-nonote" not in i]
    (out / "plan.json").write_text(json.dumps(plan))
    verify = subprocess.run(
        [sys.executable, "tools/preflight.py", "verify", "--out", job.out, "--model", "haiku",
         "--cell", "primary", "--sets", "docs/preregistration/stage2-sets.json"],
        cwd=repo,
        env={**job.env, "PYTHONPATH": str(repo / "runs" / "stage2-code" / "src")},
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert verify.returncode == 1
    assert "warn-nonote: 30 not in plan.json" in verify.stdout


def test_a_sets_file_without_families_does_not_make_an_empty_job_a_success(repo, tmp_path):
    empty = {**sets_dict(), "primary": [], "q1": [], "q2": [], "q3": []}
    sets = tmp_path / "empty-sets.json"
    sets.write_text(json.dumps(empty))
    job = Job(repo, tmp_path, SETS=str(sets), CHUNK="5")

    done = job.run()

    assert done.returncode == 1 and not job.calls
    assert "DONE" not in job.log


def test_the_default_cell_has_three_seeds_f4_and_no_layer_warning_arm(repo, tmp_path):
    job = Job(repo, tmp_path, CELL="default", CHUNK="1")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr
    arms = [c["arm"] for c in job.calls]
    assert arms == ["bare", "bare", "block-nonote", "block-nonote"] + [
        "block-nonote", "block-nonote", "bare", "bare"
    ] + ["bare", "bare", "block-nonote", "block-nonote"]  # fmt: skip
    assert {c["reps"] for c in job.calls} == {1}  # no replicate outside the primary cell
    assert (job.calls[0]["contract"], job.calls[0]["tier"], job.calls[0]["route"]) == (
        "full", "explicit", "free",
    )  # fmt: skip
    assert "F4" in job.calls[0]["families"]
    assert {c["seeds"] for c in job.calls} == {"1000", "1001", "1002", "1020", "1021", "1022"}


def test_the_pilot_is_one_bare_call_over_every_family_but_the_excluded(repo, tmp_path):
    job = Job(repo, tmp_path, CELL="pilot", EXCLUDE="F34 F35")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr
    assert len(job.calls) == 1
    call = job.calls[0]
    assert (call["arm"], call["seeds"], call["reps"]) == ("bare", "0-4", 1)
    assert call["families"] is None  # all of them: no list to go stale
    assert call["exclude_families"] == ["F34", "F35"]
    assert (call["contract"], call["tier"], call["route"]) == ("bare", "implicit", "forced")
    assert "chunks: 1 of 1 seed(s)" in job.log


def test_the_job_refuses_to_start_outside_sbatch_or_without_its_settings(repo, tmp_path):
    outside = Job(repo, tmp_path, SLURM_SUBMIT_DIR=None).run()
    assert outside.returncode != 0 and "run this script with sbatch" in outside.stderr

    no_cap = Job(repo, tmp_path, CAP=None).run()
    assert no_cap.returncode != 0 and "USD this job may spend" in no_cap.stderr

    wrong = Job(repo, tmp_path, CELL="held-out").run()
    assert wrong.returncode == 2 and "CELL must be pilot, primary or default" in wrong.stderr
    assert not Job(repo, tmp_path).calls  # nothing ran in any of them


def test_a_snapshot_that_can_be_written_to_gets_a_warning_a_read_only_one_does_not(repo, tmp_path):
    snapshot = repo / "runs" / "stage2-code"
    writable = Job(repo, tmp_path / "w", CHUNK="5", AA="0")
    assert writable.run().returncode == 0
    assert "WARNING: the snapshot can be written to" in writable.log

    def lock(mode: str) -> None:
        subprocess.run(["chmod", "-R", mode, str(snapshot)], check=True)

    lock("a-w")
    try:
        sealed = Job(repo, tmp_path / "r", CHUNK="5", AA="0")
        done = sealed.run()
        assert done.returncode == 0, done.stdout + done.stderr
        assert "can be written to" not in sealed.log
    finally:
        lock("u+w")


def test_the_package_must_come_from_the_snapshot(repo, tmp_path):
    job = Job(repo, tmp_path, SNAP="runs/not-a-snapshot")

    done = job.run()

    assert done.returncode == 1
    assert "is not imported from runs/not-a-snapshot" in job.log
    assert not job.calls


def test_the_seed_registry_scan_stops_a_job_whose_seeds_have_been_run(repo, tmp_path):
    for earlier in (repo / "runs").glob("job-*"):  # the other tests' fake runs
        shutil.rmtree(earlier)
    clean = Job(repo, tmp_path / "clean", SCAN="1", CHUNK="5", AA="0")
    assert clean.run().returncode == 0  # its own directory does not count

    old = repo / "runs" / "earlier-run" / "trials" / "some__model" / "F2-s1001-faulted-r0"
    old.mkdir(parents=True)
    try:
        job = Job(repo, tmp_path, SCAN="1")
        done = job.run()
        assert done.returncode == 1 and not job.calls
        assert "a registered (family, seed) has been run before" in job.log
        assert "F2 seed 1001" in done.stdout
    finally:
        shutil.rmtree(repo / "runs" / "earlier-run")


def test_the_freeze_file_is_verified_before_the_first_trial(repo, tmp_path):
    frozen = repo / "docs" / "preregistration" / "freeze-stage2.txt"
    freeze = [
        sys.executable, "docs/preregistration/freeze_hashes.py", "--stage2",
        "--code-root", "runs/stage2-code", "--repo-root", ".",
        "--sets", "docs/preregistration/stage2-sets.json",
    ]  # fmt: skip
    score = repo / "runs" / "stage2-code" / "src" / "biowarden" / "bench" / "score.py"
    original = score.read_text()
    try:
        made = subprocess.run(
            [*freeze, "--write", str(frozen)], cwd=repo, capture_output=True, text=True, check=False
        )
        assert made.returncode == 0, made.stderr

        job = Job(repo, tmp_path, FREEZE="docs/preregistration/freeze-stage2.txt", CHUNK="5")
        done = job.run()
        assert done.returncode == 0, done.stdout + done.stderr
        assert "verified" in job.log and len(job.calls) == 8

        score.write_text(original + "\n# a change after the freeze\n")  # the snapshot drifts
        again = Job(repo, tmp_path / "again", FREEZE="docs/preregistration/freeze-stage2.txt")
        done = again.run()
        assert done.returncode == 1 and not again.calls
        assert "differs from the snapshot" in done.stdout or "no longer match" in done.stdout
        assert "differ from docs/preregistration/freeze-stage2.txt" in again.log
    finally:
        score.write_text(original)
        frozen.unlink(missing_ok=True)


# ---- audit of the stage-2 fixes ------------------------------------------------------------


def write_freeze(repo: Path, name: str, *sets: str) -> str:
    args = [sys.executable, "docs/preregistration/freeze_hashes.py", "--stage2"]
    args += ["--code-root", "runs/stage2-code", "--repo-root", "."]
    for item in sets:
        args += ["--sets", item]
    done = subprocess.run(
        [*args, "--write", name, "--overwrite"], cwd=repo, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return name


def test_a_torn_ledger_line_stops_the_job_instead_of_giving_every_call_the_whole_cap(
    repo, tmp_path
):
    job = Job(repo, tmp_path, CHUNK="5", CAP="0.05", AA="0")
    ledger = repo / job.out / "jobs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"job_id": "slurm-4242", "usd": 0.01}\n{"job_id": "slurm-42')  # a kill

    done = job.run()

    assert done.returncode == 1, done.stdout + done.stderr
    assert len(job.calls) == 1  # the first call ran; the spend could not be read, so no second
    assert "the spend of this job cannot be read" in job.log and "DONE" not in job.log


def test_a_held_out_job_refuses_a_freeze_that_does_not_list_the_sets_file(repo, tmp_path):
    code_only = write_freeze(repo, f"freeze-code-{tmp_path.name}.txt")
    job = Job(repo, tmp_path, FREEZE=code_only)

    done = job.run()

    assert done.returncode == 1
    assert job.calls == []  # nothing ran: the code-only freeze (the pilot's) cannot vouch for sets
    assert "does not list" in job.log and "second freeze" in job.log


def test_a_held_out_job_runs_when_the_freeze_lists_the_sets_file(repo, tmp_path):
    second = write_freeze(
        repo, f"freeze-second-{tmp_path.name}.txt", "docs/preregistration/stage2-sets.json"
    )
    job = Job(repo, tmp_path, FREEZE=second, CHUNK="5")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr
    assert len(job.calls) == 8


def test_the_pilot_needs_no_sets_file_and_only_the_code_freeze(repo, tmp_path):
    code_only = write_freeze(repo, f"freeze-pilot-{tmp_path.name}.txt")
    job = Job(repo, tmp_path, CELL="pilot", FREEZE=code_only, CHUNK="5")

    done = job.run()

    assert done.returncode == 0, done.stdout + done.stderr


def test_the_sandboxed_work_directories_get_a_root_of_their_own_that_is_removed_afterwards(
    repo, tmp_path
):
    # the default root, /tmp/biowarden-<job id>: an id of its own, so no other job's root is checked
    job = Job(
        repo,
        tmp_path,
        CHUNK="5",
        AA="0",
        SLURM_JOB_ID=str(os.getpid()),
        BIOWARDEN_NEUTRAL_ROOT=None,
    )

    assert job.run().returncode == 0

    assert "BIOWARDEN_NEUTRAL_ROOT" in (repo / "tools" / "stage2_job.sbatch").read_text()
    assert not Path(f"/tmp/biowarden-{os.getpid()}").exists()  # the trap removed it at the end
