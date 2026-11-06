"""Run the regression suites of bedtools, samtools and bcftools with and without the layer.

    python tools/real_code_scan.py fetch SOURCES             # download, check, unpack the sources
    python tools/real_code_scan.py run SOURCES --out FILE    # run the suites, write the aggregate
    python tools/real_code_scan.py summary FILE              # print the tables of a result file

A tool's own test suite is the longest list of calls its authors call correct: each one runs on a
tiny input with a known answer. `run` starts every suite in a scratch directory, twice as it is (the
second run tells a flaky test from a disturbed one) and once under `biowarden` in its default mode,
where every rule blocks, and compares the runs:

    a test that passes without the layer and fails with it was disturbed, by a wrong block or by a
    shim that is not transparent; the ledger of the guarded run says which;
    the ledger also counts the calls the layer checked, how many it blocked or warned about, and by
    which rules.

The bedtools tests are shell scripts, one per subcommand, and are compared test by test (their
`set -e` line is dropped, so that a blocked call fails its own test instead of ending the script).
The samtools and bcftools tests are one Perl script each (`test/test.pl`); it calls the program next
to the `test` directory, so a link to the program found on PATH (under the guard, the shim) is put
there, and the runs are compared by the totals the script prints. Its `cmd`, which builds inputs and
expected outputs, ends the script on the first failure; it is changed to print the failure, so that
a blocked call fails the tests that need it and the script goes on (the same change in both runs).

The sources are not part of the repository: `fetch` downloads the tagged releases and compares the
digest of each unpacked `test` tree with the one pinned below. Each run happens under `bwrap` (the
whole file system read-only but the scratch directory, no network) when it can be started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

TIMEOUT = 1800  # seconds for one run of one suite or unit
MAX_ARGS = 16  # words of a blocked call that are kept
MAX_WORD = 120  # characters of one of them


class Suite(NamedTuple):
    name: str
    version: str
    url: str
    top: str  # the directory the archive unpacks to
    test_tree: str  # sha256 of its `test` tree (tree_digest)


SUITES = {
    s.name: s
    for s in (
        Suite(
            "bedtools",
            "2.31.1",
            "https://github.com/arq5x/bedtools2/archive/refs/tags/v2.31.1.tar.gz",
            "bedtools2-2.31.1",
            "905fdd3cb35bae57b25a4a772b0e39e5ba8b58d225b80f97c71bde6fee86e59f",
        ),
        Suite(
            "samtools",
            "1.24",
            "https://github.com/samtools/samtools/releases/download/1.24/samtools-1.24.tar.bz2",
            "samtools-1.24",
            "3a499ef2d45f99db83a5de1dc5aa6194c6fbf1f38e4fd59f80a73927ee79a6b4",
        ),
        Suite(
            "bcftools",
            "1.24",
            "https://github.com/samtools/bcftools/releases/download/1.24/bcftools-1.24.tar.bz2",
            "bcftools-1.24",
            "050b5058df74422d5660697e384d84a190ec797d77970e8bc0b5f2c7c20eb375",
        ),
    )
}
SUMMARY_LINE = re.compile(
    r"^\s+(total|passed|failed|expected failure|unexpected pass)\s+\.\.\s+(\d+)"
)
LABEL = re.compile(r"^\s*([A-Za-z0-9_.\-/]+?)\.\.\.(.*)$")
# the `cmd` of test.pl that builds an input or an expected output ends the whole script on the first
# failure; a blocked call would then end the run instead of failing its own test
CMD_DIES = re.compile(
    r'if \( \$ret \) \{ error\("The command failed( \[\$ret\])?: \$cmd\\n"[^;]*\); \}'
)


# ---- the sources ------------------------------------------------------------------------------


def tree_digest(tree: Path) -> str:
    """One digest for the names, sizes and contents of every file below `tree`."""
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*") if p.is_file() and not p.is_symlink()):
        digest.update(f"{path.relative_to(tree).as_posix()}\0{path.stat().st_size}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def safe_unpack(archive: Path, dest: Path) -> None:
    """Unpack below `dest`; a member outside it, a device or a link that leaves it is refused."""
    dest = dest.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"{archive.name}: {member.name} leaves the target directory")
            if member.isdev() or member.isfifo():
                raise ValueError(f"{archive.name}: {member.name} is not a file or directory")
            if member.islnk() or member.issym():
                base = target.parent if member.issym() else dest
                link = (base / member.linkname).resolve()
                if dest != link and dest not in link.parents:
                    raise ValueError(f"{archive.name}: {member.name} links outside")
        tar.extractall(dest, filter="data")


def fetch(sources: Path, names: Iterable[str]) -> int:
    sources.mkdir(parents=True, exist_ok=True)
    for name in names:
        suite = SUITES[name]
        target = sources / suite.top
        if target.is_dir():
            shutil.rmtree(target)
        with tempfile.TemporaryDirectory(prefix="biowarden-fetch-") as scratch:
            archive = Path(scratch) / "archive"
            with urllib.request.urlopen(suite.url, timeout=120) as reply:  # noqa: S310
                archive.write_bytes(reply.read())
            safe_unpack(archive, sources)
        found = tree_digest(target / "test")
        if found != suite.test_tree:
            print(f"{name}: the test tree differs from the pinned one ({found})", file=sys.stderr)
            return 1
        print(f"{name} {suite.version}: unpacked, test tree {found[:12]}")
    return 0


# ---- the environment --------------------------------------------------------------------------


def tool_environment(tools_json: Path | None) -> dict[str, str]:
    """The environment the suites run in: the benchmark's tool directories, or this process's."""
    env = dict(os.environ)
    if tools_json is not None:
        from biowarden.bench.tasks.tools import ToolEnv

        tools = ToolEnv.from_json(tools_json)
        env.update(tools.process_env())
        env["PATH"] = tools.path_var("/usr/bin:/bin")
    env.update(HOME="/tmp", TMPDIR="/tmp", LC_ALL="C")
    for name in ("BIOWARDEN_DECLARE", "BIOWARDEN_SOCK", "BIOWARDEN_REAL_PATH", "BT"):
        env.pop(name, None)
    return env


def find_bwrap() -> str | None:
    """A working `bwrap` (from BIOWARDEN_BWRAP or PATH), or None."""
    for candidate in (os.environ.get("BIOWARDEN_BWRAP"), shutil.which("bwrap")):
        if not candidate:
            continue
        probe = [candidate, "--ro-bind", "/", "/", "--unshare-all", "--", "true"]
        if subprocess.run(probe, capture_output=True, check=False).returncode == 0:
            return candidate
    return None


# ---- one run ----------------------------------------------------------------------------------


class Run(NamedTuple):
    code: int | None  # None: out of time
    seconds: float
    stdout: str
    stderr: str
    ledger: list[dict]


def run_command(
    command: Sequence[str],
    *,
    workdir: Path,
    guarded: bool,
    env: dict[str, str],
    bwrap: str | None,
    extra_env: dict[str, str],
) -> Run:
    """Start `command` in `workdir` (a directory below a private scratch directory), alone or
    under biowarden; in a sandbox the scratch directory is /tmp and the rest is read-only."""
    scratch = workdir.parent
    inside = Path("/tmp") if bwrap else scratch
    cwd = inside / workdir.relative_to(scratch)
    ledger = inside / "ledger.jsonl"
    argv = [str(c) for c in command]
    if guarded:
        guard = [sys.executable, "-m", "biowarden.engine.guard", "--ledger", str(ledger)]
        argv = [*guard, "--", *argv]
    if bwrap:
        argv = [
            bwrap, "--ro-bind", "/", "/", "--bind", str(scratch), "/tmp", "--dev", "/dev",
            "--proc", "/proc", "--unshare-all", "--die-with-parent", "--chdir", str(cwd),
            "--", *argv,
        ]  # fmt: skip
    started = time.monotonic()
    try:
        done = subprocess.run(
            argv,
            cwd=None if bwrap else cwd,
            env={**env, **extra_env},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=TIMEOUT,
            check=False,
        )
        code, out, err = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        code, out, err = None, _text(exc.stdout), _text(exc.stderr)
    seconds = time.monotonic() - started
    rows = []
    path = scratch / "ledger.jsonl"
    if guarded and path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return Run(code, seconds, out, err, rows)


def _text(value: bytes | str | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


# ---- the suites -------------------------------------------------------------------------------


def bedtools_units(source: Path) -> list[str]:
    tests = source / SUITES["bedtools"].top / "test"
    return sorted(
        d.name
        for d in tests.iterdir()
        if (d / f"test-{d.name}.sh").is_file() and d.name != "negativecontrol"
    )


def prepare(suite: str, unit: str | None, source: Path, scratch: Path) -> tuple[Path, list[str]]:
    """Copy what one run needs into `scratch` and return its working directory and command."""
    tests = source / SUITES[suite].top / "test"
    if suite == "bedtools":
        workdir = scratch / "ws"
        shutil.copytree(tests / str(unit), workdir)
        script = (workdir / f"test-{unit}.sh").read_text().splitlines()
        if script and script[0].strip() == "set -e;":
            script = script[1:]  # a blocked call fails its test, it does not end the script
        (workdir / "run.sh").write_text("\n".join(script) + "\n")
        return workdir, ["bash", "run.sh"]
    workdir = scratch / "ws"
    shutil.copytree(tests, workdir / "test")
    script = workdir / "test" / "test.pl"
    patched, count = CMD_DIES.subn(
        r'if ( $ret ) { print STDERR "The command failed\1: $cmd\\n"; }', script.read_text()
    )
    if count != 1:
        raise ValueError(f"{suite}: expected one `cmd` that dies on failure, found {count}")
    script.write_text(patched)
    link = f'ln -sf "$(command -v {suite})" {suite} && exec perl test/test.pl'
    return workdir, ["bash", "-c", link]


def parse_labels(stdout: str) -> dict[str, str]:
    """`label...ok` / `label...fail` of a bedtools test script -> label: ok | fail | unfinished.

    A failing test prints its diff between the label and the `fail`, so the result may be on a
    later line; a label that meets the next label first is unfinished."""
    results: dict[str, str] = {}
    current: str | None = None
    for line in stdout.splitlines():
        found = LABEL.match(line)
        if found:
            if current is not None and results[current] == "pending":
                results[current] = "unfinished"
            current, rest = found.group(1), found.group(2).strip()
            results[current] = rest if rest in ("ok", "fail") else "pending"
        elif (
            current is not None and results[current] == "pending" and line.strip() in ("ok", "fail")
        ):
            results[current] = line.strip()
    if current is not None and results[current] == "pending":
        results[current] = "unfinished"
    return results


def parse_totals(text: str) -> dict[str, int]:
    """The summary test.pl prints: total, passed, failed, expected failure, unexpected pass."""
    totals: dict[str, int] = {}
    for line in text.splitlines():
        found = SUMMARY_LINE.match(line)
        if found:
            totals[found.group(1)] = int(found.group(2))
    return totals


def clip(word: str) -> str:
    return word if len(word) <= MAX_WORD else word[:MAX_WORD] + "…"


def summarize_ledger(rows: Sequence[dict]) -> dict:
    """What the layer did with the calls of a guarded run."""
    firing: Counter[str] = Counter()
    blocked = []
    for row in rows:
        fired = [o for o in row["outcomes"] if o["effective"] != "none"]
        for outcome in fired:
            firing[f"{outcome['rule']}|{outcome['code']}|{outcome['effective']}"] += 1
        if row["action"] != "allow":
            blocked.append(
                {
                    "action": row["action"],
                    "tool": row["tool"],
                    "argv": [clip(w) for w in row["argv"][:MAX_ARGS]],
                    "rules": sorted({o["rule"] for o in fired}),
                    "codes": sorted({o["code"] for o in fired}),
                }
            )
    return {
        "calls": len(rows),
        "by_tool": dict(Counter(r["tool"] for r in rows)),
        "by_action": dict(Counter(r["action"] for r in rows)),
        "firing": dict(firing),
        "non_allowed": blocked,
    }


def merge_ledgers(parts: Iterable[dict]) -> dict:
    total: dict = {"calls": 0, "by_tool": Counter(), "by_action": Counter(), "firing": Counter()}
    total["non_allowed"] = []
    for part in parts:
        total["calls"] += part["calls"]
        for key in ("by_tool", "by_action", "firing"):
            total[key].update(part[key])
        total["non_allowed"].extend(part["non_allowed"])
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in total.items()}


def run_suite(
    name: str, source: Path, env: dict[str, str], bwrap: str | None, only: Sequence[str] = ()
) -> dict:
    units: list[str | None] = list(bedtools_units(source)) if name == "bedtools" else [None]
    if only:
        units = [u for u in units if u in only]
    extra = {"BT": "bedtools"} if name == "bedtools" else {}
    per_unit = []
    for unit in units:
        runs: dict[str, Run] = {}
        for label, guarded in (("base1", False), ("base2", False), ("guarded", True)):
            with tempfile.TemporaryDirectory(prefix="biowarden-scan-") as scratch:
                workdir, command = prepare(name, unit, source, Path(scratch))
                runs[label] = run_command(
                    command, workdir=workdir, guarded=guarded, env=env, bwrap=bwrap, extra_env=extra
                )
        per_unit.append(compare(name, unit, runs))
        print(f"{name} {unit or ''}: {per_unit[-1]['verdict']}", flush=True)
    return aggregate(name, per_unit)


def compare(name: str, unit: str | None, runs: dict[str, Run]) -> dict:
    base1, base2, guarded = runs["base1"], runs["base2"], runs["guarded"]
    entry: dict = {
        "unit": unit or name,
        "exit": {"base": base1.code, "guarded": guarded.code},
        "seconds": {
            "base": round(min(base1.seconds, base2.seconds), 2),
            "guarded": round(guarded.seconds, 2),
        },
        "ledger": summarize_ledger(guarded.ledger),
    }
    if name == "bedtools":
        b1, b2, g = (parse_labels(r.stdout) for r in (base1, base2, guarded))
        stable = {k for k, v in b1.items() if v == "ok" and b2.get(k) == "ok"}
        entry["tests"] = len(b1)
        entry["baseline_ok"] = len(stable)
        entry["flaky"] = sorted(k for k in b1 if k not in stable and b1[k] != b2.get(k))
        entry["disturbed"] = sorted(k for k in stable if g.get(k) != "ok")
        entry["guarded_missing"] = sorted(k for k in b1 if k not in g)
    else:
        t1, t2, tg = (parse_totals(r.stdout + "\n" + r.stderr) for r in (base1, base2, guarded))
        entry["totals"] = {"base": t1, "base_again": t2, "guarded": tg}
        entry["tests"] = t1.get("total", 0)
        entry["baseline_ok"] = min(t1.get("passed", 0), t2.get("passed", 0))
        entry["flaky"] = [] if t1 == t2 else ["totals differ between the two baseline runs"]
        entry["disturbed"] = max(0, entry["baseline_ok"] - tg.get("passed", 0))
        entry["guarded_missing"] = [] if tg.get("total") == t1.get("total") else ["total differs"]
    count = len(entry["disturbed"]) if isinstance(entry["disturbed"], list) else entry["disturbed"]
    entry["verdict"] = f"{entry['baseline_ok']} ok without the layer, {count} disturbed"
    return entry


def aggregate(name: str, per_unit: list[dict]) -> dict:
    ledger = merge_ledgers(u["ledger"] for u in per_unit)
    base = sum(u["seconds"]["base"] for u in per_unit)
    guarded = sum(u["seconds"]["guarded"] for u in per_unit)
    calls = ledger["calls"]
    disturbed = sum(
        len(u["disturbed"]) if isinstance(u["disturbed"], list) else u["disturbed"]
        for u in per_unit
    )
    return {
        "version": SUITES[name].version,
        "units": len(per_unit),
        "tests": sum(u["tests"] for u in per_unit),
        "baseline_ok": sum(u["baseline_ok"] for u in per_unit),
        "disturbed": disturbed,
        "flaky": sum(len(u["flaky"]) for u in per_unit),
        "calls_checked": calls,
        "calls_by_tool": ledger["by_tool"],
        "calls_by_action": ledger["by_action"],
        "firing": ledger["firing"],
        "non_allowed": ledger["non_allowed"],
        "seconds_base": round(base, 1),
        "seconds_guarded": round(guarded, 1),
        "ms_added_per_call": round(1000 * (guarded - base) / calls, 1) if calls else None,
        "disturbed_tests": {u["unit"]: u["disturbed"] for u in per_unit if u["disturbed"]},
        "units_detail": per_unit,
    }


def describe(env: dict[str, str]) -> dict:
    """The versions the result was measured with."""
    from biowarden.engine.spec import packaged_specs

    def first_line(*command: str) -> str:
        try:
            done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        except OSError:
            return "not found"
        return (
            (done.stdout or done.stderr).strip().splitlines()[0]
            if done.stdout or done.stderr
            else ""
        )

    return {
        "rules": sum(len(spec.rules) for spec in packaged_specs()),
        "mode": "promoted",
        "python": sys.version.split()[0],
        "bedtools": first_line("bedtools", "--version"),
        "samtools": first_line("samtools", "--version"),
        "bcftools": first_line("bcftools", "--version"),
    }


def compact(result: dict) -> dict:
    """The aggregate that is published: no per-unit detail, one entry per distinct blocked call."""
    suites = {}
    for name, suite in result["suites"].items():
        kept = {k: v for k, v in suite.items() if k not in ("units_detail", "non_allowed")}
        calls: Counter[tuple] = Counter()
        for item in suite["non_allowed"]:
            calls[
                (
                    item["action"],
                    item["tool"],
                    tuple(item["argv"]),
                    tuple(item["rules"]),
                    tuple(item["codes"]),
                )
            ] += 1
        kept["non_allowed"] = [
            {
                "action": a,
                "tool": t,
                "argv": list(argv),
                "rules": list(rules),
                "codes": list(codes),
                "times": n,
            }
            for (a, t, argv, rules, codes), n in sorted(calls.items())
        ]
        suites[name] = kept
    return {k: v for k, v in result.items() if k != "suites"} | {"suites": suites}


# ---- the tables -------------------------------------------------------------------------------


def summary(result: dict) -> str:
    lines = [
        "| Suite | Version | Tests passing without the layer | Disturbed by the layer "
        "| Calls checked | Not allowed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, suite in result["suites"].items():
        refused = sum(v for k, v in suite["calls_by_action"].items() if k != "allow")
        lines.append(
            f"| {name} | {suite['version']} | {suite['baseline_ok']:,} of {suite['tests']:,} "
            f"| {suite['disturbed']} | {suite['calls_checked']:,} | {refused} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="real_code_scan", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "run"):
        step = sub.add_parser(name)
        step.add_argument("sources", type=Path, help="directory that holds the unpacked sources")
        step.add_argument("--suite", action="append", choices=sorted(SUITES), default=[])
    run = sub.choices["run"]
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--tools", type=Path, help="tools.json of the benchmark (else this PATH)")
    run.add_argument("--unit", action="append", default=[], help="only this bedtools subcommand")
    run.add_argument("--no-sandbox", action="store_true")
    run.add_argument(
        "--compact", type=Path, help="also write the aggregate without per-unit detail"
    )
    sub.add_parser("summary").add_argument("result", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(summary(json.loads(args.result.read_text())))
        return 0
    names = args.suite or sorted(SUITES)
    if args.command == "fetch":
        return fetch(args.sources, names)
    bwrap = None if args.no_sandbox else find_bwrap()
    env = tool_environment(args.tools)
    result = {
        "schema": 1,
        "sandbox": bool(bwrap),
        "environment": describe(env),
        "suites": {n: run_suite(n, args.sources, env, bwrap, args.unit) for n in names},
    }
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.compact:
        args.compact.write_text(json.dumps(compact(result), indent=1, sort_keys=True) + "\n")
    print(summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
