"""Worker sizing from cgroup limits (fake cgroup trees, no real limits needed)."""

from __future__ import annotations

import pytest

from biowarden.bench.runner.resources import (
    LOGIN_NODE_WORKERS,
    MIB,
    MIN_WORKER_MB,
    cgroup_memory,
    on_scheduler_job,
    plan_workers,
)


@pytest.fixture
def scheduler_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")


@pytest.fixture
def login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def v1_tree(tmp_path, limit, usage, leaf_limit=None):
    root = tmp_path / "cg"
    leaf = root / "memory/user.slice/user-1.slice/session-9.scope"
    leaf.mkdir(parents=True)
    parent = leaf.parent
    (parent / "memory.limit_in_bytes").write_text(str(limit))
    (parent / "memory.usage_in_bytes").write_text(str(usage))
    (leaf / "memory.limit_in_bytes").write_text(str(leaf_limit or 1 << 62))  # no limit at the leaf
    (leaf / "memory.usage_in_bytes").write_text("1")
    proc = tmp_path / "cgroup"
    proc.write_text("11:memory:/user.slice/user-1.slice/session-9.scope\n4:cpu:/other\n")
    return proc, root


def test_the_limit_is_read_from_the_tightest_ancestor_cgroup(tmp_path):
    proc, root = v1_tree(tmp_path, limit=6 * 1024 * MIB, usage=2 * 1024 * MIB)

    assert cgroup_memory(proc, root) == (6 * 1024 * MIB, 2 * 1024 * MIB)


def test_workers_shrink_to_fit_the_free_memory(tmp_path, scheduler_job):
    proc, root = v1_tree(tmp_path, limit=6 * 1024 * MIB, usage=2 * 1024 * MIB)

    # 70% of 6 GiB = 4301 MiB, minus 2048 MiB in use = 2253 MiB free: three 700 MiB workers
    assert plan_workers(48, 700, proc_cgroup=proc, root=root) == 3
    assert plan_workers(2, 700, proc_cgroup=proc, root=root) == 2  # never above the request
    assert plan_workers(48, 90_000, proc_cgroup=proc, root=root) == 1  # always at least one


def test_without_a_limit_the_request_stands(tmp_path, scheduler_job):
    proc = tmp_path / "cgroup"
    proc.write_text("")

    assert plan_workers(48, 700, proc_cgroup=proc, root=tmp_path) == 48


def test_a_worker_is_never_counted_for_less_than_the_floor(tmp_path, scheduler_job):
    proc, root = v1_tree(tmp_path, limit=6 * 1024 * MIB, usage=2 * 1024 * MIB)

    # 2253 MiB free: 5 workers of 400 MiB. The old estimate of 80 MiB would have said 28.
    assert MIN_WORKER_MB >= 400
    assert plan_workers(48, 80, proc_cgroup=proc, root=root) == 2253 // MIN_WORKER_MB == 5
    assert plan_workers(48, MIN_WORKER_MB, proc_cgroup=proc, root=root) == 5


def test_outside_a_scheduler_job_at_most_two_workers_run(tmp_path, login_node):
    proc = tmp_path / "cgroup"
    proc.write_text("")  # no memory limit at all
    big, root = v1_tree(tmp_path, limit=512 * 1024 * MIB, usage=0)

    assert not on_scheduler_job() and LOGIN_NODE_WORKERS == 2
    assert plan_workers(48, 700, proc_cgroup=proc, root=tmp_path) == 2
    assert plan_workers(48, 700, proc_cgroup=big, root=root) == 2  # plenty of memory, still two
    assert plan_workers(1, 700, proc_cgroup=big, root=root) == 1  # the request below the cap stands


def test_on_a_scheduler_job_the_login_cap_does_not_apply(tmp_path, scheduler_job):
    big, root = v1_tree(tmp_path, limit=512 * 1024 * MIB, usage=0)

    assert on_scheduler_job()
    assert plan_workers(8, 700, proc_cgroup=big, root=root) == 8
