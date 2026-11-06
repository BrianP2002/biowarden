"""Size the worker pool to the memory this process may actually use.

Login nodes cap each user's cgroup (6 GiB and 4 CPU cores on the cluster this was built on), and the
kernel kills processes in that cgroup when it is exceeded. A worker is an agent loop plus the
sandboxed shell, tools and interpreters it starts (a few hundred MB at the peak); OpenCode is a
~200 MB executable that holds a few hundred MB of RSS per run on top, so dozens of concurrent runs
on a login node get the whole session killed. The runner therefore reads the cgroup limit and
current usage, caps the number of workers, runs at most two workers outside a scheduler job, and
refuses the heavy harness there unless told otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

UNLIMITED = 1 << 60  # cgroup v1 reports "no limit" as a huge number
MIB = 1024 * 1024
LOGIN_NODE_WORKERS = 2  # the most workers outside a scheduler job (SLURM_JOB_ID unset)
MINIMAL_WORKER_MB = 400  # a worker of the minimal harness at its peak (80 MB undercounted it)
OPENCODE_WORKER_MB = 700
MIN_WORKER_MB = MINIMAL_WORKER_MB  # no worker is planned for less than this


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def cgroup_memory(
    proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")
) -> tuple[int | None, int | None]:
    """(limit, usage) in bytes of the tightest memory cgroup above this process, if any."""
    limit: int | None = None
    usage: int | None = None
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None, None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base, limit_file, usage_file = root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):  # cgroup v1
            base, limit_file, usage_file = (
                root / "memory",
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            )
        else:
            continue
        parts = Path(path.lstrip("/")).parts
        for depth in range(len(parts), -1, -1):
            directory = base.joinpath(*parts[:depth])
            value = _read_int(directory / limit_file)
            if value is not None and value < UNLIMITED and (limit is None or value < limit):
                limit, usage = value, _read_int(directory / usage_file)
    return limit, usage


def on_scheduler_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def plan_workers(
    requested: int,
    per_worker_mb: int,
    fraction: float = 0.7,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    root: Path = Path("/sys/fs/cgroup"),
) -> int:
    """The largest worker count <= requested that fits in `fraction` of the cgroup's free memory.

    A worker is never counted for less than MIN_WORKER_MB, and outside a scheduler job at most
    LOGIN_NODE_WORKERS run, whatever the memory.
    """
    if not on_scheduler_job():
        requested = min(requested, LOGIN_NODE_WORKERS)
    limit, usage = cgroup_memory(proc_cgroup, root)
    if limit is None:
        return requested
    free = max(0, int(limit * fraction) - (usage or 0))
    return max(1, min(requested, free // (max(per_worker_mb, MIN_WORKER_MB) * MIB)))
