"""docs/preregistration/freeze_hashes.py: the stage-1 mode is unchanged, the stage-2 mode freezes.

The stage-2 tests build a small repository and snapshot in a temporary directory: one file per
group member, as many as the group's expected size asks for.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "preregistration" / "freeze_hashes.py"
EMPTY = hashlib.sha256(b"").hexdigest()
PKG = "src/biowarden"

# The groups and the algorithm of the stage-1 script, as they were when `freeze.txt` was written.
OLD_GROUPS = {
    "scorer": ["src/biowarden/bench/score.py"],
    "analysis": ["src/biowarden/bench/analysis.py"],
    "stage1": ["src/biowarden/bench/stage1.py"],
    "engine": ["src/biowarden/engine/*.py", "src/biowarden/engine/rules/*.py"],
    "specs": ["src/biowarden/engine/specs/*.yaml"],
    "promotion": ["src/biowarden/engine/specs/promotion.json"],
    "tasks": [
        "src/biowarden/bench/tasks/families.py",
        "src/biowarden/bench/tasks/common.py",
        "src/biowarden/bench/tasks/instance.py",
        "src/biowarden/bench/tasks/extra/f07_regions_vs_targets.py",
        "src/biowarden/bench/tasks/extra/f10_indel_bed.py",
        "src/biowarden/bench/tasks/extra/f13_phred_offset.py",
        "src/biowarden/bench/tasks/extra/f15_log_scale.py",
    ],
    "runner": ["src/biowarden/bench/runner/*.py"],
}


def old_output(root: Path) -> str:
    lines = []
    for name, patterns in OLD_GROUPS.items():
        h = hashlib.sha256()
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                h.update(path.relative_to(root).as_posix().encode())
                h.update(path.read_bytes())
        lines.append(f"{name:<10}{h.hexdigest()}")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def fh():
    spec = importlib.util.spec_from_file_location("freeze_hashes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["freeze_hashes"] = module
    spec.loader.exec_module(module)
    return module


def put(root: Path, rel: str, text: str | None = None) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rel if text is None else text)


# -- stage 1 ------------------------------------------------------------------------------------


def legacy_tree(root: Path) -> None:
    for patterns in OLD_GROUPS.values():
        for pattern in patterns:
            if "*" not in pattern:
                put(root, pattern)
    for rel in (
        "engine/a.py", "engine/b.py", "engine/rules/r1.py", "engine/specs/s1.yaml",
        "engine/specs/s2.yaml", "bench/runner/x.py", "bench/runner/y.py",
    ):  # fmt: skip
        put(root, f"{PKG}/{rel}")
    (root / PKG / "bench" / "stage1.py").unlink()  # stage1.py was absent from some snapshots


def test_the_stage_1_table_and_algorithm_are_unchanged(fh, tmp_path, capsys):
    legacy_tree(tmp_path)

    assert fh.LEGACY_GROUPS == OLD_GROUPS
    assert fh.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out == old_output(tmp_path)
    # the empty group still prints the empty-input digest here: four published freeze files have it
    assert f"stage1    {EMPTY}\n" in out


def test_the_stage_1_mode_reproduces_the_published_freeze_files(fh, capsys):
    snapshot = ROOT / "runs" / "stage1-code2"
    published = ROOT / "docs" / "preregistration" / "freeze-rerun.txt"
    if not (snapshot / "src").is_dir() or not published.is_file():
        pytest.skip("the stage-1 snapshot is not on this machine")
    if (snapshot / "src" / "science_harness").is_dir():
        pytest.skip(
            "the stage-1 snapshot predates the rename; the first commit's tool reproduces it"
        )

    assert fh.main([str(snapshot)]) == 0

    assert capsys.readouterr().out == published.read_text()


def test_the_stage_1_mode_is_the_default_and_takes_no_stage_2_flags(fh, tmp_path, capsys):
    legacy_tree(tmp_path)
    for argv in (["--code-root", "x"], ["--write", "f"], ["--sets", "f"], ["--overwrite"]):
        with pytest.raises(SystemExit) as exc:
            fh.main([str(tmp_path), *argv])
        assert exc.value.code == 2
        assert "need --stage2" in capsys.readouterr().err


# -- the stage-2 command line ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--stage2"], "needs --code-root and --repo-root"),
        (["--stage2", "--code-root", "a"], "needs --code-root and --repo-root"),
        (["--stage2", "--repo-root", "a"], "needs --code-root and --repo-root"),
        (["--stage2", "x", "--code-root", "a", "--repo-root", "b"], "not a positional root"),
        (["--stage2", "--code-root", "a", "--repo-root", "b", "--bogus"], "unrecognized"),
        (["--stage", "--code-root", "a", "--repo-root", "b"], "unrecognized"),  # no abbreviations
        (["--stage2", "--code-root", "a", "--repo-root", "b", "--write", "f", "--verify", "g"],
         "not allowed"),
    ],
)  # fmt: skip
def test_the_command_line_is_strict(fh, argv, message, capsys):
    with pytest.raises(SystemExit) as exc:
        fh.main(argv)

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_a_script_run_of_stage_2_without_its_roots_exits_with_2():
    import subprocess

    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--stage2"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 2 and EMPTY not in done.stdout  # not eight empty digests, exit 0


# -- a small repository and its snapshot ----------------------------------------------------------


def build_repo(base: Path) -> None:
    """A repository with as many files per group as the groups' expected sizes ask for."""
    pkg = f"{PKG}"
    put(base, f"{pkg}/__init__.py")
    put(base, f"{pkg}/keys.py")
    put(base, f"{pkg}/bench/score.py")
    for name in ("analysis", "pilot", "stage2"):
        put(base, f"{pkg}/bench/{name}.py")
    put(base, f"{pkg}/bench/admission/replay.py")
    put(base, f"{pkg}/bench/admission/cases.py")  # support
    put(base, f"{pkg}/bench/tasks/tools.py")
    put(base, f"{pkg}/bench/tasks/family_rules.json", "{}")
    for i in range(109):
        put(base, f"{pkg}/bench/tasks/extra/f{i:03d}.py")
    for i in range(14):
        put(base, f"{pkg}/bench/runner/r{i:02d}.py")
    for i in range(60):
        put(base, f"{pkg}/engine/e{i:02d}.py")
    for i in range(20):
        put(base, f"{pkg}/engine/rules/r{i:02d}.py")
    put(base, f"{pkg}/ledger/ledger.py")
    put(base, f"{pkg}/contracts/loader.py")
    for i in range(51):
        put(base, f"{pkg}/engine/specs/s{i:02d}.yaml")
    put(base, f"{pkg}/engine/specs/promotion.json", "{}")
    put(base, "pyproject.toml")
    for tool in ("stage2_job.sbatch", "preflight.py", "env_lock.py"):
        put(base, f"tools/{tool}")
    for name in ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "venv-pip-freeze.txt", "tool-versions.txt"):
        put(base, f"envs/locks/{name}")


def snapshot_of(base: Path, name: str = "snap") -> Path:
    dest = base / name
    shutil.copytree(base / "src", dest / "src")
    return dest


@pytest.fixture
def repo(tmp_path):
    base = tmp_path / "repo"
    build_repo(base)
    return base


def freeze(fh, repo: Path, *extra: str, snap: Path | None = None, write: Path | None = None):
    snap = snap or snapshot_of(repo)
    argv = ["--stage2", "--code-root", str(snap), "--repo-root", str(repo), *extra]
    return fh.main(argv + (["--write", str(write)] if write else []))


def test_a_complete_repository_is_frozen_with_a_manifest_and_group_digests(fh, repo, tmp_path):
    out = tmp_path / "freeze.txt"

    assert freeze(fh, repo, write=out) == 0
    text = out.read_text()

    lines = text.splitlines()
    heads = dict(line.split(None, 1) for line in lines[1 : lines.index("--")])
    assert lines[0].startswith("# stage-2 freeze")
    assert list(heads) == [
        "scorer", "analysis", "engine", "specs", "promotion", "tasks", "runner", "support",
        "config", "env", "code", "root", "files",
    ]  # fmt: skip
    assert "sets" not in heads  # freeze 1 has none
    files = [line.split("  ", 1) for line in lines[lines.index("--") + 1 :]]
    paths = [p for _, p in files]
    assert paths == sorted(paths) and int(heads["files"]) == len(paths)
    assert all(len(d) == 64 for d, _ in files)
    assert not any(p.startswith("/") or "__pycache__" in p or p.endswith(".pyc") for p in paths)
    assert {"pyproject.toml", "tools/stage2_job.sbatch", "envs/locks/a.tsv"} <= set(paths)
    assert {"tools/preflight.py", "tools/env_lock.py"} <= set(paths)  # what the template runs
    assert f"{PKG}/bench/tasks/family_rules.json" in paths
    assert EMPTY not in text and str(tmp_path) not in text  # no empty hash, no path of this machine


def test_the_digests_do_not_depend_on_where_the_repository_is(fh, tmp_path):
    texts = []
    for name in ("one", "two/deeper"):
        base = tmp_path / name / "repo"
        build_repo(base)
        out = tmp_path / f"{name.replace('/', '_')}.txt"
        assert freeze(fh, base, write=out) == 0
        texts.append(out.read_text())

    assert texts[0] == texts[1]


def test_the_group_digests_are_the_sha256_of_their_lines(fh, repo, tmp_path):
    out = tmp_path / "freeze.txt"
    freeze(fh, repo, write=out)
    heads, files = fh.parse(out.read_text())

    def digest(paths):
        lines = [f"{files[p]}  {p}" for p in sorted(paths)]
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    assert heads["scorer"] == digest([f"{PKG}/bench/score.py"])
    assert heads["promotion"] == digest([f"{PKG}/engine/specs/promotion.json"])
    assert heads["env"] == digest([p for p in files if p.startswith("envs/locks/")])
    assert heads["root"] == digest(files)
    assert (
        heads["code"]
        == hashlib.sha256(
            "\n".join(
                f"{files[p]}  {p[len(PKG) + 1 :]}" for p in sorted(files) if p.startswith(PKG + "/")
            ).encode()
        ).hexdigest()
    )
    assert (
        files[f"{PKG}/bench/score.py"]
        == hashlib.sha256(b"src/biowarden/bench/score.py").hexdigest()
    )
    # every file is in exactly one group, and no group but `sets` is empty
    assert {fh.group_of(p) for p in files} == set(fh.GROUPS) - {"sets"}


def test_code_is_the_digest_the_runner_stamps_on_every_record(fh, repo, tmp_path):
    runner = pytest.importorskip("biowarden.bench.runner.cli")
    out = tmp_path / "freeze.txt"
    freeze(fh, repo, write=out)

    heads, _ = fh.parse(out.read_text())

    assert heads["code"] == runner.code_digest(repo / PKG)


def test_the_real_package_and_environment_files_make_a_valid_freeze(fh, tmp_path):
    # the repository itself, frozen against itself: every group has what the freeze asks for
    freeze_ = fh.build(ROOT, ROOT, [])

    assert freeze_.problems() == []
    assert not any("__pycache__" in p for p in freeze_.files)
    assert all(freeze_.groups[g] for g in fh.EXPECTED)
    assert len(freeze_.members["tasks"]) >= 111  # the 110 builders and helpers + family_rules.json
    assert f"{PKG}/bench/stage2.py" in freeze_.members["analysis"]
    assert f"{PKG}/bench/admission/replay.py" in freeze_.members["analysis"]
    assert f"{PKG}/bench/tasks/tools.py" in freeze_.members["tasks"]
    heads, _ = fh.parse(freeze_.render())  # an empty __init__.py has the empty hash, a group never
    assert EMPTY not in heads.values()


# -- what makes it exit with 2 ------------------------------------------------------------------


def test_a_missing_root_or_a_root_without_the_package_is_refused(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)

    assert freeze(fh, repo, snap=tmp_path / "nowhere") == 2
    assert "--code-root" in capsys.readouterr().err
    (tmp_path / "bare").mkdir()
    assert freeze(fh, tmp_path / "bare", snap=snap) == 2
    assert f"no {PKG}" in capsys.readouterr().err
    assert freeze(fh, repo, snap=tmp_path / "bare") == 2


def test_a_group_without_files_prints_missing_not_the_empty_hash(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)
    shutil.rmtree(repo / "envs")
    out = tmp_path / "freeze.txt"

    assert freeze(fh, repo, snap=snap, write=out) == 2

    shown = capsys.readouterr()
    assert "env       MISSING" in shown.out
    assert "env: MISSING (no files)" in shown.err
    assert EMPTY not in shown.out + shown.err
    assert not out.exists()  # nothing is written


def test_a_group_below_its_expected_size_is_refused(fh, repo, tmp_path, capsys):
    for i in range(5):
        (repo / PKG / "bench" / "tasks" / "extra" / f"f{i:03d}.py").unlink()
    snap = snapshot_of(repo)
    out = tmp_path / "freeze.txt"

    assert freeze(fh, repo, snap=snap, write=out) == 2

    assert "tasks: 106 files, expected at least 111" in capsys.readouterr().err
    assert not out.exists()


def test_a_required_file_cannot_be_missing_even_if_the_group_is_big_enough(
    fh, repo, tmp_path, capsys
):
    (repo / PKG / "bench" / "tasks" / "family_rules.json").unlink()
    put(repo, f"{PKG}/bench/tasks/extra/extra_one.py")  # 111 files again
    snap = snapshot_of(repo)

    assert freeze(fh, repo, snap=snap, write=tmp_path / "f.txt") == 2

    assert f"{PKG}/bench/tasks/family_rules.json: MISSING (required)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "rel", ["bench/stage2.py", "bench/pilot.py", "bench/admission/replay.py", "bench/score.py",
            "bench/tasks/tools.py", "engine/specs/promotion.json"],
)  # fmt: skip
def test_each_file_the_review_found_missing_is_required(fh, repo, rel, tmp_path, capsys):
    (repo / PKG / rel).unlink()
    snap = snapshot_of(repo)

    assert freeze(fh, repo, snap=snap) == 2

    assert f"{PKG}/{rel}: MISSING (required)" in capsys.readouterr().err


def test_the_live_src_must_be_the_snapshot(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)
    (repo / PKG / "bench" / "score.py").write_text("changed after the snapshot")
    (repo / PKG / "bench" / "new.py").write_text("only in the repository")
    (repo / PKG / "engine" / "e00.py").unlink()  # only in the snapshot

    assert freeze(fh, repo, snap=snap, write=tmp_path / "f.txt") == 2

    err = capsys.readouterr().err
    assert "differs from the snapshot" in err
    assert "src: changed: src/biowarden/bench/score.py" in err
    assert "src: new: src/biowarden/bench/new.py" in err
    assert "src: missing: src/biowarden/engine/e00.py" in err
    assert not (tmp_path / "f.txt").exists()


def test_a_symbolic_link_in_a_frozen_tree_is_refused(fh, repo, tmp_path, capsys):
    (tmp_path / "elsewhere.py").write_text("code that is not under the frozen directory")
    (repo / PKG / "link.py").symlink_to(tmp_path / "elsewhere.py")

    assert freeze(fh, repo, snap=snapshot_of(repo, "snap2")) == 2

    err = capsys.readouterr().err
    assert "link.py" in err and "symbolic link" in err


def test_caches_are_not_part_of_the_freeze(fh, repo, tmp_path):
    one, two = tmp_path / "one.txt", tmp_path / "two.txt"
    freeze(fh, repo, write=one)
    cache = repo / PKG / "bench" / "__pycache__"
    cache.mkdir()
    (cache / "score.cpython-312.pyc").write_bytes(b"\0\1")
    (cache / "index.dat").write_bytes(b"not a .pyc, but inside a cache directory")
    (repo / PKG / "stray.pyc").write_bytes(b"\0")

    assert freeze(fh, repo, snap=snapshot_of(repo, "snap2"), write=two) == 0

    assert one.read_text() == two.read_text()


# -- verify ----------------------------------------------------------------------------------------


def verify(fh, repo: Path, record: Path, snap: Path, *extra: str) -> int:
    argv = ["--stage2", "--code-root", str(snap), "--repo-root", str(repo), "--verify", str(record)]
    return fh.main([*argv, *extra])


def test_verify_accepts_what_was_frozen_and_names_what_changed(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)
    record = tmp_path / "freeze.txt"
    assert freeze(fh, repo, snap=snap, write=record) == 0
    assert verify(fh, repo, record, snap) == 0
    assert "verified" in capsys.readouterr().out

    # the snapshot and the repository drift together (a deliberate change after the freeze)
    for tree in (snap, repo):
        (tree / PKG / "bench" / "score.py").write_text("the scorer was edited")
        (tree / PKG / "engine" / "e01.py").unlink()
        put(tree, f"{PKG}/engine/e99.py")
    (repo / "tools" / "stage2_job.sbatch").write_text("the template was edited")
    (repo / "envs" / "locks" / "a.tsv").write_text("a package moved")

    assert verify(fh, repo, record, snap) == 2

    err = capsys.readouterr().err
    assert "the files no longer match" in err
    for expected in (
        "changed: src/biowarden/bench/score.py",
        "changed: tools/stage2_job.sbatch",
        "changed: envs/locks/a.tsv",
        "missing: src/biowarden/engine/e01.py",
        "new: src/biowarden/engine/e99.py",
        "scorer: recorded",
        "config: recorded",
        "env: recorded",
        "root: recorded",
    ):
        assert expected in err, expected


def test_verify_fails_when_the_snapshot_is_not_what_the_repository_holds(
    fh, repo, tmp_path, capsys
):
    snap = snapshot_of(repo)
    record = tmp_path / "freeze.txt"
    freeze(fh, repo, snap=snap, write=record)
    (snap / PKG / "bench" / "score.py").write_text("the running copy was edited")

    assert verify(fh, repo, record, snap) == 2

    assert "differs from the snapshot" in capsys.readouterr().err


def test_verify_notices_a_freeze_file_that_was_edited_by_hand(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)
    record = tmp_path / "freeze.txt"
    freeze(fh, repo, snap=snap, write=record)
    text = record.read_text()
    head, _, rest = text.partition("scorer    ")
    record.write_text(head + "scorer    " + "0" * 64 + rest[64:])  # a forged group digest

    assert verify(fh, repo, record, snap) == 2

    assert "scorer: recorded " + "0" * 64 in capsys.readouterr().err


def test_verify_without_a_freeze_file_or_with_garbage_is_refused(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)

    assert verify(fh, repo, tmp_path / "missing.txt", snap) == 2
    assert "--verify" in capsys.readouterr().err
    garbage = tmp_path / "garbage.txt"
    garbage.write_text("scorer abc\n--\nnot a manifest line\n")
    assert verify(fh, repo, garbage, snap) == 2
    assert "unreadable line" in capsys.readouterr().err


# -- freeze 2: the sets file, the plan manifest, the draft ----------------------------------------


def test_the_sets_files_are_in_the_manifest_and_found_again_by_verify(fh, repo, tmp_path, capsys):
    put(repo, "docs/preregistration/stage2-sets.json", '{"format": 2}')
    put(repo, "docs/preregistration/stage2-plan.json", '{"entries": []}')
    snap = snapshot_of(repo)
    record = tmp_path / "freeze.txt"

    both = ["--sets", "docs/preregistration/stage2-sets.json"]
    both += ["--sets", "docs/preregistration/stage2-plan.json"]
    assert freeze(fh, repo, *both, snap=snap, write=record) == 0

    heads, files = fh.parse(record.read_text())
    assert "docs/preregistration/stage2-sets.json" in files and "sets" in heads
    assert fh.group_of("docs/preregistration/stage2-plan.json") == "sets"
    # no --sets on the command line: the freeze file says which files it froze
    assert verify(fh, repo, record, snap) == 0
    capsys.readouterr()
    (repo / "docs/preregistration/stage2-sets.json").write_text('{"format": 2, "p": []}')
    assert verify(fh, repo, record, snap) == 2
    assert "changed: docs/preregistration/stage2-sets.json" in capsys.readouterr().err
    (repo / "docs/preregistration/stage2-sets.json").unlink()
    assert verify(fh, repo, record, snap) == 2
    assert "sets: MISSING docs/preregistration/stage2-sets.json" in capsys.readouterr().err


def test_a_sets_file_must_exist_have_content_and_lie_inside_the_repository(
    fh, repo, tmp_path, capsys
):
    snap = snapshot_of(repo)
    put(repo, "docs/empty.json", " \n")
    (tmp_path / "outside.json").write_text("{}")

    assert freeze(fh, repo, "--sets", "docs/none.json", snap=snap) == 2
    assert "sets: MISSING docs/none.json" in capsys.readouterr().err
    assert freeze(fh, repo, "--sets", "docs/empty.json", snap=snap) == 2
    assert "docs/empty.json is empty" in capsys.readouterr().err
    assert freeze(fh, repo, "--sets", str(tmp_path / "outside.json"), snap=snap) == 2
    assert "not inside --repo-root" in capsys.readouterr().err
    assert freeze(fh, repo, "--sets", "pyproject.toml", snap=snap) == 2
    assert "already part of the manifest as config" in capsys.readouterr().err


def test_a_different_freeze_file_is_not_replaced_without_overwrite(fh, repo, tmp_path, capsys):
    snap = snapshot_of(repo)
    record = tmp_path / "freeze.txt"
    assert freeze(fh, repo, snap=snap, write=record) == 0
    first = record.read_text()
    assert freeze(fh, repo, snap=snap, write=record) == 0  # the same content: nothing to protect

    (repo / "pyproject.toml").write_text("an edit after the freeze")
    assert freeze(fh, repo, snap=snap, write=record) == 2
    assert "pass --overwrite" in capsys.readouterr().err
    assert record.read_text() == first
    assert freeze(fh, repo, "--overwrite", snap=snap, write=record) == 0
    assert record.read_text() != first


def test_without_write_the_manifest_goes_to_stdout(fh, repo, capsys):
    assert freeze(fh, repo) == 0

    out = capsys.readouterr().out
    assert out.startswith("# stage-2 freeze") and "\n--\n" in out


# ---- audit of the stage-2 fixes ------------------------------------------------------------


def test_a_snapshot_whose_package_is_a_link_to_the_live_tree_is_refused(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "freeze_hashes",
        Path(__file__).resolve().parents[1] / "docs/preregistration/freeze_hashes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = Path(__file__).resolve().parents[1]
    snap = tmp_path / "snap"
    (snap / "src").mkdir(parents=True)
    (snap / "src" / "biowarden").symlink_to(repo / "src" / "biowarden")

    with pytest.raises(module.Refused) as caught:
        module.build(snap, repo, [])

    assert any("is a link" in line for line in caught.value.problems)


def test_the_tasks_group_must_hold_every_builder_and_the_family_rule_map():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "freeze_hashes",
        Path(__file__).resolve().parents[1] / "docs/preregistration/freeze_hashes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = Path(__file__).resolve().parents[1] / "src/biowarden/bench/tasks"
    present = [
        p
        for p in tasks.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]

    assert module.EXPECTED["tasks"] == len(present)  # a deleted builder would fall below the floor
