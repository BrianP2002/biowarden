"""Families F101-F112 against the real python-sc: the naive route exits 0 with an empty stderr
(the trap is silent) and lands on the recorded naive answer; the reference script lands on the
truth; the worker used by the admission cases agrees with a real `python-sc -c`.

Every run here is a separate `python-sc -c` process (the way an agent runs it), so this file is
slower than test_family_sc.py: one faulted workspace per family.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biowarden.bench.admission.cases_sc_common import worker_for
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.extra.f101_de_use_raw import RUN_HOOKS, last_json

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"F{n}" for n in range(101, 113)]
SPECS = load()
# a warning that tells the user something is wrong with the data would make the trap loud
INFORMATIVE = re.compile(
    r"raw count data|already log|not unique|are not unique|non-integers|overflow|outside range|"
    r"Trying to modify|warn|error",
    re.IGNORECASE,
)


# mudata 0.3.10 prints this deprecation notice whenever a MuData is built or read; it says nothing
# about the cells
MUDATA_NOTICE = re.compile(
    r"^.*FutureWarning: From 0\.4 \.update\(\).*$|^\s+self\._update_attr.*$", re.MULTILINE
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-sc" not in env.real_bins(["python-sc"]):
        pytest.skip("python-sc is not available")
    return env


class Recorder:
    """Run solver snippets as separate processes and remember (returncode, stderr)."""

    def __init__(self, tools):
        self.tools = tools
        self.runs: list[tuple[int, str]] = []

    def __call__(self, code, cwd):
        proc = self.tools.run(["python-sc", "-c", code], cwd)
        self.runs.append((proc.returncode, proc.stderr))
        return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture
def recorder(tools):
    rec = Recorder(tools)
    RUN_HOOKS.append(rec)
    yield rec
    RUN_HOOKS.pop()


@pytest.mark.parametrize("family", IDS)
def test_naive_route_is_silent_and_lands_on_the_recorded_answer(tools, tmp_path, recorder, family):
    spec = SPECS[family]
    work = tmp_path / "faulted"
    task = build_task(family, 0, "faulted", work, tools, "implicit", "free")
    got = spec.naive(task, work, tools)
    assert recorder.runs, "the naive solver ran no python-sc script"
    for code, stderr in recorder.runs:
        assert code == 0
        assert not INFORMATIVE.search(MUDATA_NOTICE.sub("", stderr)), stderr[-400:]
    recorded = {**task.truth, "expected": task.truth["naive"]}
    assert answer_matches(recorded, got["answer"])
    assert not answer_matches(task.truth, got["answer"])


@pytest.mark.parametrize("family", IDS)
def test_reference_route_lands_on_the_truth_on_both_twins(tools, tmp_path, recorder, family):
    spec = SPECS[family]
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        task = build_task(family, 1, twin, work, tools, "explicit", "free")
        got = spec.reference(task, work, tools)
        assert answer_matches(task.truth, got["answer"]), (family, twin)
    assert all(code == 0 for code, _ in recorder.runs)


def test_the_admission_worker_agrees_with_a_real_python_sc(tools, tmp_path):
    work = tmp_path / "f"
    task = build_task("F105", 2, "faulted", work, tools, "implicit", "free")
    code = (
        "import json, pandas as pd, anndata as ad\n"
        "a = ad.read_h5ad('cells.h5ad')\n"
        "ann = pd.read_csv('annotations.csv', index_col=0)\n"
        "a.obs['ct'] = ann['celltype'].values\n"
        "print(json.dumps(int((a.obs['ct'] == a.obs['ct'].iloc[0]).sum())))\n"
    )
    rc, out, _ = worker_for(tools).run(code, work)
    proc = tools.run(["python-sc", "-c", code], work)
    assert rc == proc.returncode == 0 and last_json(out) == last_json(proc.stdout)
    rc, out, err = worker_for(tools).run("raise SystemExit(3)", work)
    assert rc == 3
    rc, out, err = worker_for(tools).run("1 / 0", work)
    assert rc == 1 and "ZeroDivisionError" in err
    assert task.truth["naive"] != task.truth["expected"]
