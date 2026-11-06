"""The per-trial layer rig, the central exposure helpers and the gateway tool (no API calls).

Regression tests for the stage-2 review: a fault inside the layer never fails a command (E-1), the
promotion table is checked against the specs at launch (E-5), the summary records what an analysis
needs (E-6, R-8), nothing the sandbox binds names the trial (FC-2), a tool lookup is not exposure
(FB-7, FD-6, FA-6), statements split in order (FE-4) and the gateway refuses --declare (E-9).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from biowarden.bench.runner import layer as layer_module
from biowarden.bench.runner import sandbox
from biowarden.bench.runner.exposure import (
    _segments,
    drop_discovery,
    exposure,
    issued_commands_of,
    statements,
    strip_discovery,
    trial_exposure,
)
from biowarden.bench.runner.layer import (
    ARMS,
    NEUTRAL_ROOT_VAR,
    UNPROMOTED_ALLOWED,
    PromotionMismatch,
    check_promotion,
    opaque_dir,
    start_layer,
    unpromoted_rules,
)
from biowarden.bench.tasks import ToolEnv
from biowarden.engine import Promotion
from biowarden.engine.decide import PACKAGED_PROMOTION
from biowarden.engine.spec import packaged_specs

ROOT = Path(__file__).resolve().parents[1]
LAYER_ARMS = [arm for arm, setting in ARMS.items() if setting is not None]


@pytest.fixture
def neutral(tmp_path, monkeypatch):
    """A neutral root of this test's own, so that it can see what a rig puts there."""
    root = tmp_path / "neutral"
    socket = root / ("0" * 16) / layer_module.SOCKET_NAME
    short = None
    if len(os.fsencode(socket)) > layer_module.SOCKET_PATH_MAX:
        # a long temp directory (xdist workers, a high pytest counter, macOS) overflows the path
        short = Path(tempfile.mkdtemp(prefix="biowarden-", dir="/tmp"))
        root = short / "neutral"
    monkeypatch.setenv(NEUTRAL_ROOT_VAR, str(root))
    yield root
    if short is not None:
        shutil.rmtree(short, ignore_errors=True)


@pytest.fixture
def fake_tools(tmp_path):
    """A tool environment without a real program: one empty bin directory outside the repo."""
    bin_dir = tmp_path / "tools" / "bin"
    bin_dir.mkdir(parents=True)
    return ToolEnv((bin_dir,), ())


@pytest.fixture(scope="module")
def real_tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not sandbox.available():
        pytest.skip("needs the real tools and bubblewrap")
    return env


def rig_for(tmp_path, tools, arm="blockall", **kwargs):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    rig = start_layer(arm, tmp_path / "trial", work, tools, **kwargs)
    assert rig is not None
    return rig


def full_table(specs):
    return {r.id: {"mode": "block", "spec_hash": s.content_hash()} for s in specs for r in s.rules}


def write_table(tmp_path, table):
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps(table))
    return str(path)


# --- a fault inside the layer fails open (E-1) ---------------------------------------------------


def test_a_fault_in_the_precheck_lets_the_command_run_and_is_recorded(
    tmp_path, neutral, fake_tools, monkeypatch
):
    def boom(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "/root/x.tsv")

    monkeypatch.setattr(layer_module, "precheck", boom)
    rig = rig_for(tmp_path, fake_tools)
    try:
        checked = rig.layer.precheck("python3 -c \"open('/root/x.tsv')\"")
        summary = rig.summary()
    finally:
        rig.close()

    assert (checked.action, checked.text) == ("allow", "")  # the agent is told nothing
    assert len(summary["errors"]) == 1
    assert summary["errors"][0].startswith("precheck: PermissionError")


def test_commands_that_used_to_raise_on_the_host_no_longer_escape_the_layer(
    tmp_path, neutral, fake_tools
):
    # a host-side stat of a path the agent chose raised PermissionError / ENAMETOOLONG (E-1)
    long_name = "a" * 300
    rig = rig_for(tmp_path, fake_tools)
    try:
        for command in (
            "python3 -c \"open('/root/x.tsv')\"",
            f"python3 -c \"open('{long_name}.tsv')\"",
        ):
            assert rig.layer.precheck(command).action in ("allow", "warn")
        errors = rig.summary()["errors"]
    finally:
        rig.close()

    assert all(e.startswith("precheck: ") and len(e) < 400 for e in errors)


def test_a_command_without_a_fault_still_gets_its_real_decision(tmp_path, neutral, fake_tools):
    rig = rig_for(tmp_path, fake_tools)
    try:
        refused = rig.layer.precheck("bcftools view -H -R regions.tsv variants.vcf.gz")
        fine = rig.layer.precheck("ls -la")
    finally:
        rig.close()

    assert refused.action == "block" and "BLOCKED, the command did not run" in refused.text
    assert (fine.action, fine.text) == ("allow", "")


# --- the summary (E-5, E-6, R-8) -----------------------------------------------------------------


def test_the_summary_carries_the_digests_the_firing_rules_and_the_health_counts(
    tmp_path, neutral, fake_tools
):
    rig = rig_for(tmp_path, fake_tools)
    try:
        rig.layer.precheck("bcftools view -H -R regions.tsv variants.vcf.gz")
        summary = rig.summary()
    finally:
        rig.close()

    assert summary["actions"]["block"] == 1 and summary["errors"] == []
    assert summary["ledger_errors"] == 0 and summary["crashed"] == 0
    assert summary["touched_layer"] == 0
    assert all(re.fullmatch(r"[\w.]+:\w+:(warn|block)", key) for key in summary["fired"])
    assert "bcftools.region_semantics:declaration_missing:block" in summary["fired"]
    assert sorted(summary["fired_counts"]) == summary["fired"]
    assert summary["spec_hashes"] == {s.name: s.content_hash() for s in packaged_specs()}
    assert summary["promotion_sha"] == hashlib.sha256(PACKAGED_PROMOTION.read_bytes()).hexdigest()
    assert summary["unpromoted"] == sorted(UNPROMOTED_ALLOWED)


def test_the_summary_counts_crashed_checkers_and_survives_bad_ledger_lines(
    tmp_path, neutral, fake_tools
):
    def decision(code, effective="block", **extra):
        outcome = {"rule": "r.x", "code": code, "effective": effective}
        return json.dumps(
            {"event": "engine_decision", "action": "block", "outcomes": [outcome], **extra}
        )

    rig = rig_for(tmp_path, fake_tools)
    lines = [
        decision("checker_crashed"),
        decision("layer_error", "warn", source="gateway"),
        decision("fine", "none"),
        json.dumps({"event": "something_else"}),
        "{not json",
        json.dumps({"event": "engine_decision"}),  # no action, no outcomes
        json.dumps({"event": "engine_decision", "action": "maybe", "outcomes": []}),
        json.dumps({"event": "engine_decision", "action": "allow", "outcomes": [{"rule": "x"}]}),
    ]
    rig.ledger_path.write_text("\n".join(lines) + "\n")
    try:
        summary = rig.summary()
    finally:
        rig.close()

    assert summary["crashed"] == 2  # checker_crashed and layer_error
    assert summary["ledger_errors"] == 4
    assert summary["actions"] == {"allow": 0, "warn": 0, "block": 3}
    assert summary["fired"] == ["r.x:checker_crashed:block", "r.x:layer_error:warn"]
    assert summary["by_source"]["gateway"]["block"] == 1


@pytest.mark.parametrize("arm", ["block-nonote", "warn-nonote", "block-shim"])
def test_commands_that_name_the_layer_are_counted(tmp_path, neutral, fake_tools, arm):
    naming = [
        "ls -la /opt/biowarden/bin",
        "cat /opt/biowarden/bin/biowarden-shim",
        "env | grep BIOWARDEN_",
        "python3 -c \"import socket; socket.socket(socket.AF_UNIX).connect('engine.sock')\"",
        "ls biowarden-shim*",
    ]
    quiet = ["ls -la", "samtools --version", "cat README.md; echo biowarden"]
    rig = rig_for(tmp_path, fake_tools, arm)  # block-shim has no pre-check, and still counts
    try:
        for command in [*naming, *quiet]:
            rig.layer.precheck(command)
        touched = rig.summary()["touched_layer"]
    finally:
        rig.close()

    assert touched == len(naming)


def test_the_gateway_arm_has_no_shim_to_touch(tmp_path, neutral, fake_tools):
    rig = rig_for(tmp_path, fake_tools, "block-gateway")
    try:
        summary = rig.summary()
        assert rig.layer is None and rig.shim_dir is None and rig.gateway is not None
    finally:
        rig.close()

    assert summary["touched_layer"] == 0


# --- the promotion table against the specs (E-5) ------------------------------------------------


def test_unpromoted_rules_lists_absent_entries_and_entries_measured_on_another_spec():
    specs = packaged_specs()
    table = full_table(specs)
    assert unpromoted_rules(specs, Promotion(table)) == []

    absent = specs[0].rules[0].id
    stale = specs[1].rules[0].id
    unhashed = specs[2].rules[0].id
    del table[absent]
    table[stale] = {"mode": "block", "spec_hash": "0" * 64}
    table[unhashed] = {"mode": "block"}  # no hash: trusted for any version, as Promotion does

    assert unpromoted_rules(specs, Promotion(table)) == sorted([absent, stale])


def test_the_packaged_table_leaves_out_exactly_the_allowed_rules():
    # when a rule is promoted (or one is added), set UNPROMOTED_ALLOWED to the new truth
    found = unpromoted_rules(packaged_specs(), Promotion.packaged())

    assert found == sorted(UNPROMOTED_ALLOWED)


def test_a_job_can_check_the_table_once_before_its_workers_start(tmp_path):
    specs = packaged_specs()
    table = full_table(specs)
    victim = specs[0].rules[0].id
    del table[victim]
    path = write_table(tmp_path, table)

    checked = check_promotion(path, allow_unpromoted=[victim])

    assert checked.unpromoted == [victim] and isinstance(checked.promotion, Promotion)
    assert checked.sha == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with pytest.raises(PromotionMismatch):
        check_promotion(path)  # not the registered allow-list
    assert issubclass(PromotionMismatch, ValueError)  # a handler for ValueError still sees it
    assert check_promotion().sha == hashlib.sha256(PACKAGED_PROMOTION.read_bytes()).hexdigest()


def test_every_layer_arm_starts_with_the_packaged_table(tmp_path, neutral, fake_tools):
    for arm in LAYER_ARMS:
        rig = rig_for(tmp_path / arm, fake_tools, arm)
        try:
            assert (rig.layer is not None) is ARMS[arm].shims
            assert (rig.gateway is not None) is ARMS[arm].gateway
            assert rig.summary()["unpromoted"] == sorted(UNPROMOTED_ALLOWED)
        finally:
            rig.close()


def test_the_launch_fails_unless_the_unpromoted_rules_are_the_allowed_ones(
    tmp_path, neutral, fake_tools
):
    specs = packaged_specs()
    table = full_table(specs)
    victim = specs[0].rules[0].id
    del table[victim]
    path = write_table(tmp_path, table)

    with pytest.raises(PromotionMismatch, match="no current entry") as caught:
        rig_for(tmp_path, fake_tools, promotion_path=path)  # the default allows another rule
    assert victim in str(caught.value)
    assert not neutral.exists() or not list(neutral.iterdir())  # a refused launch leaves nothing

    for allowed in ([victim], (victim,), {victim}):
        rig = rig_for(tmp_path, fake_tools, promotion_path=path, allow_unpromoted=allowed)
        try:
            assert rig.summary()["unpromoted"] == [victim]
        finally:
            rig.close()
    rig = rig_for(tmp_path, fake_tools, promotion_path=path, allow_unpromoted=None)  # record only
    try:
        assert rig.summary()["unpromoted"] == [victim]
    finally:
        rig.close()


def test_a_rule_that_is_allowed_to_lack_an_entry_but_has_one_fails_the_launch(
    tmp_path, neutral, fake_tools
):
    path = write_table(tmp_path, full_table(packaged_specs()))  # every rule promoted

    with pytest.raises(PromotionMismatch, match="allowed to lack an entry") as caught:
        rig_for(tmp_path, fake_tools, promotion_path=path, allow_unpromoted=["mdtraj.resid_index"])
    assert "mdtraj.resid_index" in str(caught.value)


def test_an_entry_measured_on_another_version_of_the_spec_fails_the_launch(
    tmp_path, neutral, fake_tools
):
    specs = packaged_specs()
    table = full_table(specs)
    for rule in UNPROMOTED_ALLOWED:
        table.pop(rule, None)
    stale = next(r.id for s in specs for r in s.rules if r.id not in UNPROMOTED_ALLOWED)
    table[stale]["spec_hash"] = "f" * 64
    path = write_table(tmp_path, table)

    with pytest.raises(PromotionMismatch, match="no current entry") as caught:
        rig_for(tmp_path, fake_tools, promotion_path=path)
    assert stale in str(caught.value)


def test_the_arms_without_a_layer_start_nothing_and_read_no_table(tmp_path, neutral, fake_tools):
    for arm in ("bare", "note", "careful"):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        assert start_layer(arm, tmp_path / "t", work, fake_tools, "/no/such/table.json") is None
    assert not neutral.exists()
    with pytest.raises(ValueError, match="arm must be one of"):
        start_layer("nonsense", tmp_path / "t", tmp_path, fake_tools)


# --- nothing the sandbox binds names the trial (FC-2) ---------------------------------------------


def test_shims_and_socket_live_under_the_neutral_root_with_random_names(
    tmp_path, neutral, fake_tools
):
    trial = tmp_path / "trials" / "claude-haiku-5-5" / "F1-s00-faulted-block-nonote-r0"
    work = tmp_path / "work"
    work.mkdir()
    rig = start_layer("block-nonote", trial, work, fake_tools)
    shim_dir, sock = rig.layer.shim_dir, rig.layer.sock
    try:
        root = neutral.resolve()
        assert shim_dir.parent == root and sock.parent.parent == root and sock.name == "e.sock"
        assert re.fullmatch(r"[0-9a-f]{16}", shim_dir.name)
        assert re.fullmatch(r"[0-9a-f]{16}", sock.parent.name)
        assert shim_dir != sock.parent and trial not in shim_dir.parents
        assert (shim_dir / "biowarden-shim").is_file()
        assert (trial / "biowarden").is_dir() and not (trial / "biowarden" / "shims").exists()
        assert rig.tmp == sock.parent and rig.shim_dir == shim_dir
    finally:
        rig.close()

    assert not shim_dir.exists() and not sock.parent.exists()  # removed when the rig closes
    assert list(neutral.iterdir()) == []


def test_two_rigs_never_share_a_directory(tmp_path, neutral, fake_tools):
    first = rig_for(tmp_path / "a", fake_tools)
    second = rig_for(tmp_path / "b", fake_tools)
    try:
        assert first.layer.shim_dir != second.layer.shim_dir
        assert first.layer.sock != second.layer.sock
    finally:
        first.close()
        second.close()


def test_the_neutral_root_defaults_to_the_system_temp_dir(monkeypatch):
    monkeypatch.delenv(NEUTRAL_ROOT_VAR, raising=False)
    made = opaque_dir(layer_module.neutral_root())
    try:
        assert made.parent == Path(tempfile.gettempdir()).resolve()
        assert made.stat().st_mode & 0o77 == 0  # private
    finally:
        made.rmdir()


def test_a_neutral_root_too_long_for_a_unix_socket_is_refused(tmp_path, monkeypatch, fake_tools):
    monkeypatch.setenv(NEUTRAL_ROOT_VAR, str(tmp_path / ("d" * 90)))

    with pytest.raises(ValueError, match="too long for a unix socket"):
        rig_for(tmp_path, fake_tools)


def test_a_launch_that_fails_halfway_leaves_no_directory_behind(
    tmp_path, neutral, fake_tools, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("engine would not start")

    monkeypatch.setattr(layer_module, "Engine", broken)

    with pytest.raises(RuntimeError, match="would not start"):
        rig_for(tmp_path, fake_tools)
    assert list(neutral.iterdir()) == []


def test_a_neutral_root_that_cannot_run_the_shim_is_refused_at_launch(
    tmp_path, neutral, fake_tools, monkeypatch
):
    real_access = os.access

    def access(path, mode, **kwargs):  # what a noexec mount answers for an executable file
        return False if str(path).endswith("biowarden-shim") else real_access(path, mode, **kwargs)

    monkeypatch.setattr(os, "access", access)

    with pytest.raises(OSError, match="cannot be executed.*BIOWARDEN_NEUTRAL_ROOT"):
        rig_for(tmp_path, fake_tools)
    assert list(neutral.iterdir()) == []


PROBE = (
    "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < \"$f\"; echo; done; cat /proc/self/mountinfo"
)


def test_nothing_the_agent_can_read_in_proc_names_the_trial(real_tools, monkeypatch):
    # the work directory is FP-2's: here it is made the way the runner makes it (a random name
    # under the neutral root, linked from the trial directory)
    base = Path(tempfile.gettempdir()) / secrets.token_hex(6)  # hex: spells no forbidden word
    base.mkdir()
    try:
        monkeypatch.setenv(NEUTRAL_ROOT_VAR, str(base / "neutral"))
        trial = base / "runs" / "trials" / "claude-haiku-5-5" / "F1-s00-faulted-block-nonote-r0"
        trial.mkdir(parents=True)
        (trial / "work").symlink_to(opaque_dir(layer_module.neutral_root()))
        rig = start_layer("block-nonote", trial, trial / "work", real_tools)
        try:
            out = sandbox.run_command(
                trial / "work", real_tools, PROBE, max_chars=10**7, layer=rig.layer
            )
            seen = out.stdout + out.stderr
            # the probe is not blind: a shim directory kept in the trial directory would show
            leaky = trial / "biowarden" / "shims"
            shutil.copytree(rig.layer.shim_dir, leaky, symlinks=True)
            peek = sandbox.run_command(
                trial / "work",
                real_tools,
                PROBE,
                max_chars=10**7,
                layer=replace(rig.layer, shim_dir=leaky),
            )
            leaked = peek.stdout + peek.stderr
            shim_name = rig.layer.shim_dir.name
        finally:
            rig.close()
    finally:
        shutil.rmtree(base, ignore_errors=True)

    assert out.exit_code == 0 and "/opt/biowarden/bin" in seen and shim_name in seen
    arm_words = {word for arm in ARMS for word in (arm, *arm.split("-"))}
    models = {"claude", "haiku", "deepseek", "gpt-oss", "openai", "qwen", "mimo", "glm"}
    for word in ("faulted", "clean", *sorted(arm_words), *sorted(models)):
        assert word not in seen.lower(), word
    assert "faulted" in leaked


# --- tool lookups are not exposure (FB-7, FD-6, FA-6) ---------------------------------------------


@pytest.mark.parametrize(
    ("command", "kept"),
    [
        ("which fastp", ""),
        ("type samtools", ""),
        ("whereis blastn", ""),
        ("man samtools-view | head", ""),
        ("command -v bcftools >/dev/null && echo yes", "echo yes"),
        ("fastp --version", ""),
        ("blastn -version", ""),
        ("diamond blastp --help | head -30", ""),
        ("env FOO=1 which samtools", ""),
        ("samtools --version | head -1; bcftools --version | head -1; bedtools --version", ""),
        ("which fastp && fastp -i a.fq -o b.fq --cut_right", "fastp -i a.fq -o b.fq --cut_right"),
        ("cat README.md; ls -la; which python-sc", "cat README.md; ls -la"),
        ("which -a bcftools || echo missing", "echo missing"),
        (
            "cd /work && which minimap2; minimap2 --version 2>&1 | head; cat g.fai",
            "cd /work ; cat g.fai",
        ),
        ("for t in samtools bcftools; do $t --version; done", "for t in samtools bcftools; done"),
        (
            "cat <<EOF > f.txt\nwhich x\nEOF\nwhich y\nsamtools view -c a.bam",
            "cat <<EOF > f.txt\nwhich x\nEOF\nsamtools view -c a.bam",
        ),
        ("cat <<-EOF\n\twhich x\n\tEOF\nwhich y\nls", "cat <<-EOF\n\twhich x\n\tEOF\nls"),
    ],
)
def test_a_tool_lookup_is_dropped_and_the_rest_of_the_command_is_kept(command, kept):
    assert strip_discovery(command) == kept


@pytest.mark.parametrize(
    "command",
    [
        "fastp -i r.fq -o o.fq -h report.html",  # a bare -h is fastp's --html, never help
        "samtools view -h x.bam | head",  # and samtools view's "keep the header"
        "bcftools view --no-version -Oz in.vcf",
        "samtools faidx ref.fa chr1:1-10 --version-id 3",
        "samtools sort in.bam 2>&1 | tail -2",
        "bcftools view -H -R r.tsv v.vcf.gz >| out.txt",
        "$(which samtools) view -c a.bam",  # a real call through a looked-up path
        'echo "a && which b"',  # quoted text is not split
        "which=3; type=4; echo $which $type",  # assignments, not lookups
        'python-sc -c "\nimport scanpy as sc\ntype = 5\nwhich x\n"',  # a program inside quotes
        "python3 - <<'EOF'\ntype = 3\nwhich = 4\nEOF",  # a here-document body
        "# which x\nsamtools flagstat a.bam",
        "python3 - <<'EOF' | head\nimport sys\nfoo --help\nEOF",  # a body stays with its command
        "bash <<EOF && which q\nwhich inside\nEOF",
        "echo $((1 << 2)) && cat <<< 'which a'",
        "echo 'unbalanced",
        "which \\\n fastp",  # a line that does not parse is left alone
        "",
    ],
)
def test_a_command_with_no_lookup_comes_back_character_for_character(command):
    assert strip_discovery(command) == command


def test_odd_input_never_fails_and_the_segments_always_add_up_to_the_command():
    # agents write anything: quotes left open, stray brackets, here-documents with no end
    rng = random.Random(7)
    pieces = ["'", '"', "`", "\\", "#", ";", "|", "||", "&&", "<<", "<<<", "<<EOF", "<<-'EOF'"]
    pieces += ["EOF", "(", ")", "[", "]", "{", "}", "\n", " ", "$(", "2>&1", ">|", "\x00", "é"]
    pieces += ["--help", "--version", "-h", "which", "type", "command -v", "x", "a=b", "do"]
    for _ in range(3000):
        command = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 40)))
        assert "".join(op + text for op, text in _segments(command)) == command
        kept = strip_discovery(command)
        assert isinstance(kept, str) and len(kept) <= len(command)
        assert all(isinstance(s, str) and s.strip() == s for s in statements(command))


def test_drop_discovery_removes_the_commands_that_were_only_lookups():
    commands = ["which fastp", "fastp --version", "fastp -i a.fq -o b.fq", " ", "ls; which fastp"]

    assert drop_discovery(commands) == ["fastp -i a.fq -o b.fq", "ls"]


def test_every_family_check_sees_the_commands_without_their_lookups(monkeypatch, tmp_path):
    seen = []

    def check(commands, work):
        seen.append(list(commands))
        return bool(commands)

    from biowarden.bench.tasks import extra

    monkeypatch.setattr(extra, "load", lambda: {"FX": SimpleNamespace(exposed=check)})
    found = exposure("FX", ["which x", "x --version; x -i in", "ls"], tmp_path)

    assert seen == [["x -i in", "ls"]] and found == {"exposed": True, "n_commands": 3}
    assert exposure("FX", ["which x", "type x"], tmp_path)["exposed"] is False


def test_lookups_are_not_exposure_for_the_families_the_review_named(tmp_path):
    lookups = ["which fastp", "fastp --version", "fastp --help | head -40", "which blastn"]
    assert not exposure("F83", lookups, tmp_path)["exposed"]  # FD-6: 5 of 5 were `which fastp`
    assert not exposure("F87", lookups, tmp_path)["exposed"]
    assert not exposure("F77", ["which blastn", "blastn -version"], tmp_path)["exposed"]
    # the same families still see a real call, also after a lookup in the same command
    assert exposure("F83", ["fastp -i reads.fastq -o o.fq -q 20"], tmp_path)["exposed"]
    assert exposure("F83", ["which fastp && fastp -i reads.fastq -o o.fq -q 20"], tmp_path)[
        "exposed"
    ]
    assert exposure("F87", ["fastp --version; fastp -i reads.fastq -o o.fq"], tmp_path)["exposed"]


def test_the_builtin_families_keep_their_answers_with_a_lookup_in_front(tmp_path):
    region = "bcftools view -H -R regions.tsv variants.vcf.gz | cut -f3"

    assert exposure("F1", [region], tmp_path)["exposed"]
    assert exposure("F1", [f"which bcftools; bcftools --version; {region}"], tmp_path)["exposed"]
    assert not exposure("F1", ["which bcftools", "bcftools --version"], tmp_path)["exposed"]
    assert exposure("F4", ["samtools faidx ref.fa chr1:1-10"], tmp_path)["exposed"]


# --- statements (FE-4) ----------------------------------------------------------------------------


def test_statements_split_at_newlines_and_semicolons_outside_brackets_in_order():
    command = "a=1; b = foo(1,\n  2); c\nd \\\n e;;  \n"

    assert statements(command) == ["a=1", "b = foo(1,\n  2)", "c", "d  e"]
    assert statements("") == [] and statements(" ; \n ") == []


def test_statements_read_the_program_inside_a_quoted_argument():
    # the pilot agents mostly pass python as `python-sc -c "..."`, several statements in one command
    command = (
        "cd /work; python-sc -c \"\nimport scanpy as sc\nsc.tl.rank_genes_groups(a,\n   'g'); "
        'b.raw = None\nprint(a)\n"'
    )

    assert statements(command) == [
        "cd /work",
        'python-sc -c "',
        "import scanpy as sc",
        "sc.tl.rank_genes_groups(a,\n   'g')",
        "b.raw = None",
        "print(a)",
        '"',
    ]


def test_a_family_can_judge_each_trapped_call_by_its_own_statement():
    # one command, a trapped call next to the safe variant: judged as a whole it is set aside
    command = "python-sc -c \"sc.tl.rank_genes_groups(a, 'g'); b.raw = None\""
    trapped = re.compile(r"rank_genes_groups\(")
    safe = re.compile(r"\.raw\s*=\s*None|use_raw\s*=\s*False")

    assert trapped.search(command) and safe.search(command)
    judged = [s for s in statements(command) if trapped.search(s) and not safe.search(s)]
    assert len(judged) == 1


# --- one exposure call per trial ------------------------------------------------------------------


BLOCKED = "BLOCKED, the command did not run. FIX: state the convention"


def anthropic_turn(*calls):
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": cid, "name": name, "input": args}
            for cid, name, args in calls
        ],
    }


def anthropic_results(**results):
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": text}
            for cid, text in results.items()
        ],
    }


def test_trial_exposure_separates_commands_that_ran_from_commands_the_layer_refused(tmp_path):
    messages = [
        anthropic_turn(
            ("a", "bash", {"command": "bcftools view -T r.bed v.vcf"}),
            ("b", "bash", {"command": "which bcftools"}),
        ),
        anthropic_results(a=BLOCKED, b="[exit code 0]"),
    ]

    found = trial_exposure("F1", messages, tmp_path)

    # the refused -T call did not take the route; it was issued; the lookup counts for neither
    assert found == {"exposed": False, "n_commands": 1, "exposed_issued": True}
    assert issued_commands_of(messages) == ["bcftools view -T r.bed v.vcf", "which bcftools"]


def test_trial_exposure_counts_a_gateway_call_like_a_bash_call(tmp_path):
    messages = [
        anthropic_turn(("c", "run_tool", {"tool": "bcftools", "args": ["view", "-R", "r.bed"]})),
        anthropic_results(c="[exit code 0]"),
    ]

    found = trial_exposure("F1", messages, tmp_path)

    assert found == {"exposed": True, "n_commands": 1, "exposed_issued": True}


def test_trial_exposure_reads_an_openai_style_transcript_and_agrees_without_refusals(tmp_path):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "samtools faidx ref.fa chr1:1-10"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "x", "content": "[exit code 0]"},
    ]

    found = trial_exposure("F4", messages, tmp_path)

    assert found == {"exposed": True, "n_commands": 1, "exposed_issued": True}
    assert trial_exposure("F4", [], tmp_path) == {
        "exposed": False,
        "n_commands": 0,
        "exposed_issued": False,
    }


# --- the gateway refuses --declare inside args (E-9) ---------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["view", "--declare", "region_semantics=overlap", "-R", "r.bed", "v.vcf"],
        ["view", "--declare=region_semantics=overlap", "-R", "r.bed", "v.vcf"],
        ["--declare", "x=y"],
    ],
)
def test_the_gateway_refuses_declare_inside_args_and_never_asks_the_engine(
    tmp_path, neutral, fake_tools, args
):
    rig = rig_for(tmp_path, fake_tools, "block-gateway")
    try:
        text, failed = rig.gateway.run({"tool": "bcftools", "args": args})
        summary = rig.summary()
    finally:
        rig.close()

    assert failed is True and text.startswith("Error:")
    assert "--declare" in text and "`declared`" in text
    assert summary["actions"] == {"allow": 0, "warn": 0, "block": 0}


def test_the_gateway_still_blocks_a_trapped_call_and_refuses_an_unknown_tool(
    tmp_path, neutral, fake_tools
):
    rig = rig_for(tmp_path, fake_tools, "block-gateway")
    try:
        bad = rig.gateway.run({"tool": "bcftools", "args": ["view", "-H", "-R", "r.tsv", "v.vcf"]})
        junk = rig.gateway.run({"tool": "rm", "args": ["-rf", "x"]})
        summary = rig.summary()
    finally:
        rig.close()

    assert bad[1] is True and "BLOCKED" in bad[0]
    assert junk[1] is True and junk[0].startswith("Error: send")
    assert summary["by_source"]["gateway"]["block"] == 1


def test_the_stop_text_follows_the_reporting_contract_through_the_real_layer(
    tmp_path, neutral, real_tools
):
    from biowarden.bench.tasks import build_task

    texts = {}
    for contract in ("full", "bare"):
        trial = tmp_path / contract
        work = trial / "work"
        build_task("F1", 0, "faulted", work, real_tools, "implicit", "forced")
        rig = start_layer("block-gateway", trial, work, real_tools, contract=contract)
        try:
            call = {
                "tool": "bcftools",
                "args": ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"],
            }
            replies = [rig.gateway.run(call)[0] for _ in range(4)]
        finally:
            rig.close()
        texts[contract] = next(t for t in replies if t.startswith("STOP") or "\nSTOP" in t)

    assert "needs_input" in texts["full"] and '"answer": null' not in texts["full"]
    assert "needs_input" not in texts["bare"] and '{"answer": null}' in texts["bare"]


def test_a_refused_run_tool_call_is_not_an_executed_command(tmp_path, neutral, real_tools):
    from biowarden.bench.runner.exposure import trial_exposure
    from biowarden.bench.tasks import build_task

    work = tmp_path / "trial" / "work"
    build_task("F1", 0, "faulted", work, real_tools, "implicit", "forced")
    rig = start_layer("block-gateway", tmp_path / "trial", work, real_tools)
    try:
        reply = rig.gateway.run(
            {"tool": "bcftools", "args": ["view", "-R", "regions.tsv", "--declare", "a=b", "v.gz"]}
        )
    finally:
        rig.close()
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "run_tool",
                        "arguments": '{"tool": "bcftools", "args": ["view", "-R", "regions.tsv"]}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": reply[0]},
    ]

    assert reply[1] is True and "BLOCKED, the command did not run" in reply[0]
    found = trial_exposure("F1", messages, work)
    assert found["exposed"] is False and found["n_commands"] == 0
