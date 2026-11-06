"""Engine-rule fixes of the stage-2 review (FP-6) that cut across rules.

E-2   the fix of a missing region_semantics declaration is two complete commands, overlap first.
E-5   the promotion table covers every rule with the hash of its current spec (python-only fixes
      keep every hash valid); F92 (mdtraj.resid_index) waits for its own admission run (FD-10).
E-8   evaluating a rule never changes the workspace it looks at.
The other fixes are tested beside their rules: FB-5 in test_rules_aln_samtools, E-7 in
test_rules_sc_multimodal and test_rules_iv_bedtools, E-8 (norm) in test_rules_var_bcftools, E-12
in test_rules_var_plink.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

import pytest

from biowarden.bench.admission import Case, known_sources
from biowarden.bench.tasks import ToolEnv
from biowarden.engine import Call, Engine, Promotion, Spec
from biowarden.engine.render import render_block
from biowarden.engine.rules import bcftools_regions as regions
from biowarden.engine.spec import load_packaged, packaged_specs
from biowarden.engine.tools import call_rule_tools

ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = "bcftools.region_semantics"
CWD = Path("/nonexistent-workspace")  # the suggester and the T0 checker read no file


# ---- E-2: two labelled, complete commands -------------------------------------------------------


def semantics_engine() -> Engine:
    data = load_packaged("bcftools_regions").model_dump(mode="json")
    data["rules"] = [r for r in data["rules"] if r["id"] == SEMANTICS]
    return Engine([Spec.model_validate(data)], mode="block", retry_cap=10**9)


def missing_declaration(argv: list[str]) -> str:
    """The text a call without the declaration is blocked with (its first line, the FIX)."""
    decision = semantics_engine().evaluate(Call("bcftools", tuple(argv), CWD, {}))
    (fired,) = decision.firing()
    assert (fired.result.code, decision.action) == ("declaration_missing", "block"), argv
    return render_block(decision).splitlines()[0]


COMMAND = re.compile(r"(?:written: |\) or )(bcftools .*? --declare region_semantics=(\w+))(?= \()")
OVERLAP_OPTIONS = ("--regions-overlap", "--targets-overlap")


def commands_of(fix: str) -> list[tuple[str, str]]:
    """(command, declared value) of every complete command in the fix text, in order."""
    return [(m.group(1), m.group(2)) for m in COMMAND.finditer(fix)]


def without_overlap_options(argv: list[str]) -> list[str]:
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
        elif token in OVERLAP_OPTIONS:
            skip = True
        elif not token.startswith(tuple(o + "=" for o in OVERLAP_OPTIONS)):
            out.append(token)
    return out


SHAPES = [
    "view -H -R r.bed v.vcf.gz",
    "view -H -T r.bed v.vcf.gz",
    "view -H -r chr1:100-200 v.vcf.gz",
    "view -H -t chr1:100-200 v.vcf.gz",
    "view -H --regions chr1 v.vcf.gz",
    "view -H --targets chr1:1-9 v.vcf.gz",
    "view -H --regions-file r.bed v.vcf.gz",
    "view -H --regions-file=r.bed v.vcf.gz",
    "view -H --targets-file r.bed v.vcf.gz",
    "view -H -Rr.bed v.vcf.gz",
    "view -H -Tr.bed v.vcf.gz",
    # an explicit overlap option decides: 1|record|2|variant overlap, 0|pos start only
    "view -H -R r.bed --regions-overlap 1 v.vcf.gz",
    "view -H -R r.bed --regions-overlap record v.vcf.gz",
    "view -H -R r.bed --regions-overlap=2 v.vcf.gz",
    "view -H -R r.bed --regions-overlap variant v.vcf.gz",
    "view -H -R r.bed --regions-overlap 0 v.vcf.gz",
    "view -H -R r.bed --regions-overlap=pos v.vcf.gz",
    "view -H -T r.bed --targets-overlap 1 v.vcf.gz",
    "view -H -T r.bed --targets-overlap=record v.vcf.gz",
    "view -H -T r.bed --targets-overlap 2 v.vcf.gz",
    "view -H -T r.bed --targets-overlap variant v.vcf.gz",
    "view -H -T r.bed --targets-overlap 0 v.vcf.gz",
    "view -H -T r.bed --targets-overlap=pos v.vcf.gz",
    # both kinds at once, and an overlap option of the other kind
    "view -r chr1 -T r.bed v.vcf.gz",
    "view -R a.bed -t chr1:1-9 v.vcf.gz",
    "view -R r.bed --targets-overlap 0 v.vcf.gz",
    "view -T r.bed --regions-overlap 1 v.vcf.gz",
    "view -r chr1 -T r.bed --targets-overlap 2 v.vcf.gz",
    # other subcommands, a plugin, and `call`, which has no --targets-overlap
    "query -f %ID\\t%END\\n -R r.bed v.vcf.gz",
    "query -f %ID\\n -T r.bed v.vcf.gz",
    "filter -i POS>0 -R r.bed v.vcf.gz",
    "stats -T r.bed v.vcf.gz",
    "norm -m-any -T r.bed v.vcf.gz",
    "isec -r chr1 a.vcf.gz b.vcf.gz",
    "+fill-tags -T r.bed v.vcf.gz",
    "call -mv -T r.bed v.vcf.gz",
    "call -mv -R r.bed v.vcf.gz",
    "call -mv -R r.bed --regions-overlap 0 v.vcf.gz",
]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_fix_of_a_missing_declaration_is_overlap_then_start_only_for_this_call(shape):
    argv = shape.split()
    fix = missing_declaration(argv)
    found = commands_of(fix)
    assert [value for _, value in found] == ["overlap", "start_only"], fix
    engine = semantics_engine()
    for command, value in found:
        # complete and copy-paste: the shim reads the call exactly like this and strips --declare
        call = Call.from_argv("bcftools", shlex.split(command)[1:], CWD, env={})
        assert call.declared == {"region_semantics": value}
        chosen, invalid = regions.selections(call.argv)
        assert chosen and not invalid and {s.semantic for s in chosen} == {value}, command
        # and nothing but the overlap options differs from the call that was blocked (`call` has
        # no --targets-overlap, so its overlap command reads -T as -R)
        was = without_overlap_options(argv)
        if argv[0] == "call" and value == "overlap":
            was = [token.replace("-T", "-R") for token in was]
        assert without_overlap_options(list(call.argv)) == was, command
        outcome = engine.evaluate(call)
        assert outcome.action in ("allow", "warn"), command  # a query caveat may warn, never block
        assert all(o.result.verdict != "fail" for o in outcome.outcomes), command


@pytest.mark.parametrize(
    ("shape", "as_written"),
    [
        ("view -H -R r.bed v.vcf.gz", "overlap"),
        ("view -H -r chr1 v.vcf.gz", "overlap"),
        ("view -H -T r.bed v.vcf.gz", "start_only"),
        ("view -H -t chr1:1-9 v.vcf.gz", "start_only"),
        ("view -H -R r.bed --regions-overlap 2 v.vcf.gz", "overlap"),
        ("view -H -R r.bed --regions-overlap 0 v.vcf.gz", "start_only"),
        ("view -H -T r.bed --targets-overlap variant v.vcf.gz", "overlap"),
        ("view -H -T r.bed --targets-overlap pos v.vcf.gz", "start_only"),
        ("view -r chr1 -T r.bed v.vcf.gz", None),  # mixed: both commands change an option
    ],
)
def test_the_command_that_changes_no_option_is_marked_as_the_call_as_written(shape, as_written):
    argv = shape.split()
    fix = missing_declaration(argv)
    (overlap, _), (start_only, _) = commands_of(fix)
    marked = [v for v in ("overlap", "start_only") if f"({v}, your call as written:" in fix]
    assert marked == ([as_written] if as_written else [])
    unchanged = {"overlap": overlap, "start_only": start_only}
    for value, command in unchanged.items():
        same = shlex.split(command)[1:-2] == argv  # only --declare was added
        assert same == (value == as_written), (value, command)


def test_the_fix_says_what_the_call_keeps_what_each_command_means_and_what_declare_is():
    fix = missing_declaration("view -H -T r.bed v.vcf.gz".split())
    assert fix.startswith(
        "FIX: -T r.bed (--targets-overlap defaults to 0) keeps only records whose POS lies in a "
        "region. Replace the bcftools call with one of these two, exactly as written: "
    )
    kept = "a gVCF reference block or a deletion that begins before the region is kept"
    dropped = "a record that begins before the region is dropped"
    assert f"(overlap: {kept})" in fix
    assert f"(start_only, your call as written: {dropped})" in fix
    assert fix.endswith(
        "--declare is not a bcftools option: the sandbox checker reads it and removes it before "
        "bcftools runs. Setting --regions-overlap or --targets-overlap alone does not lift this "
        "block."
    )
    mixed = missing_declaration("view -r chr1 -T r.bed v.vcf.gz".split())
    both = (
        "-r chr1 (--regions-overlap defaults to 1) keeps every record that overlaps a region; "
        "-T r.bed (--targets-overlap defaults to 0) keeps only records whose POS lies in a region."
    )
    assert mixed.startswith(f"FIX: {both} Replace the bcftools call")


def test_an_overlap_value_bcftools_rejects_leaves_the_generic_fix():
    fix = missing_declaration("view -T r.bed --targets-overlap 3 v.vcf.gz".split())
    assert fix == "FIX: re-run with --declare region_semantics=<overlap|start_only>"


# ---- E-5 and FD-10: the promotion table against the specs ---------------------------------------

# every rule is promoted for its current spec hash (mdtraj.resid_index, F92, was admitted last, on
# 40 development seeds: its 300-seed run did not finish)
UNPROMOTED: set[str] = set()


def test_the_promotion_table_covers_every_rule_with_the_hash_of_its_spec():
    promotion = Promotion.packaged()
    rules = {r.id: s.content_hash() for s in packaged_specs() for r in s.rules}
    missing = sorted(set(rules) - set(promotion.entries))
    assert set(missing) <= UNPROMOTED, f"rules without a promotion entry: {missing}"
    stale = [
        rule_id
        for rule_id, entry in promotion.entries.items()
        if entry.get("spec_hash") != rules.get(rule_id)  # no hash at all counts as stale
    ]
    assert not stale, f"entries that are not for the current spec (or for no rule): {stale}"
    assert all(e["mode"] in ("block", "warn") for e in promotion.entries.values())


def test_a_rule_without_an_entry_only_warns_and_an_entry_applies_to_its_own_spec_only():
    promotion = Promotion.packaged()
    hashes = {r.id: s.content_hash() for s in packaged_specs() for r in s.rules}
    for rule_id in UNPROMOTED & set(hashes) - set(promotion.entries):
        assert promotion.mode_of(rule_id, hashes[rule_id]) == "warn"
    for rule_id in set(promotion.entries) & set(hashes):
        assert promotion.mode_of(rule_id, hashes[rule_id]) == promotion.entries[rule_id]["mode"]
        assert promotion.mode_of(rule_id, "0" * 64) == "warn"  # another spec: no evidence


# ---- E-8: evaluating the rules does not touch the workspace -----------------------------------


def tree(root: Path) -> dict[str, tuple[str, int, int, str]]:
    """Name -> (kind, size, mtime_ns, sha256) of everything under `root`."""
    out: dict[str, tuple[str, int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        info = path.lstat()
        if path.is_symlink():
            out[name] = ("link", 0, 0, str(path.readlink()))
        elif path.is_dir():
            out[name] = ("dir", 0, 0, "")
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[name] = ("file", info.st_size, info.st_mtime_ns, digest)
    return out


def sample(cases: list[Case], limit: int = 24) -> list[Case]:
    """At most `limit` cases spread over the list (both twins, right and wrong), and the last."""
    if len(cases) <= limit:
        return cases
    return [cases[i * len(cases) // limit] for i in range(limit)] + [cases[-1]]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


@pytest.fixture(scope="module")
def every_rule(tools):
    specs = packaged_specs()
    return Engine(
        specs,
        mode="block",
        real_bins=tools.real_bins(call_rule_tools(specs)),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )


INDEXES = (".fai", ".bai", ".csi", ".tbi", ".crai", ".gzi")  # files a tool writes when absent


def drop_indexes(cwd: Path) -> None:
    for path in cwd.rglob("*"):
        if path.is_file() and path.name.endswith(INDEXES):
            path.unlink()


@pytest.mark.parametrize("rule_id", sorted(known_sources()))
def test_evaluating_every_rule_leaves_the_workspace_byte_identical(
    rule_id, tools, every_rule, tmp_path
):
    """The calls of each rule's admission cases, judged by the whole engine (E-8).

    Once on the workspace as the family builder left it, once more without the index files a tool
    would write beside its input (`bcftools norm -f ref.fa` makes ref.fa.fai).
    """
    try:
        cases = sample(known_sources()[rule_id](0, tmp_path, tools))
    except FileNotFoundError as exc:  # a program its workspace builder runs is not installed
        pytest.skip(f"{rule_id}: {exc}")
    assert cases, rule_id
    for stage in ("as built", "without indexes"):
        for case in cases:
            before = tree(case.call.cwd)
            every_rule.evaluate(case.call)
            after = tree(case.call.cwd)
            assert after == before, (rule_id, stage, case.note, sorted(set(after) ^ set(before)))
        for cwd in {case.call.cwd for case in cases}:
            drop_indexes(cwd)
