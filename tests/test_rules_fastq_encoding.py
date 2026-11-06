"""Rule fastq.phred_offset: the sniff, the command test, the checker on real argv, admission.

The unit tests run on small FASTQ files written here (plain and gzip) and on F13 workspaces built
by the benchmark; the admission test measures the rule over 300 seeds with calls labelled by
running the real tools (bench/admission/cases_fastq_encoding.py).
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_fastq_encoding import TEMPLATES
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules import fastq_encoding as fe
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "fastq.phred_offset"
KEY = "fastq_quality_offset"

# the same Phred scores in both encodings: Q2 tail, then high qualities
SCORES = [[36, 38, 40, 35, 30, 12, 2, 2], [40, 39, 31, 30, 29, 20, 10, 2]]
PLUS33 = ["".join(chr(q + 33) for q in read) for read in SCORES]  # '#'..'I'
PLUS64 = ["".join(chr(q + 64) for q in read) for read in SCORES]  # 'B'..'h'
SOLEXA = ["h;;hhhhh"]  # Solexa+64 reaches down to ';'
AMBIGUOUS = ["IIIIIIII", "GGHHIIGG"]  # ';'..'J' fits both encodings
ABOVE_H = ["~~~~~~~~"]  # only above 'h': neither Illumina 1.3+ nor Sanger Illumina


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("fastq_encoding")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_fastq(path: Path, quals: list[str], eol: str = "\n") -> Path:
    text = "".join(f"@read{i}{eol}{'A' * len(q)}{eol}+{eol}{q}{eol}" for i, q in enumerate(quals))
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as handle:
            handle.write(text.encode())
    else:
        path.write_bytes(text.encode())
    return path


def outcome(engine: Engine, tool: str, argv: list[str], cwd: Path, declared=None):
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == RULE_ID]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_declares_an_optional_two_valued_key(spec):
    (rule,) = spec.rules
    assert spec.name == "fastq_encoding" and rule.id == RULE_ID
    assert (rule.slot, rule.kind, rule.tier) == ("units", "fact", "T1")
    assert rule.incidents == ["GEN-38"]
    assert rule.declare is not None
    assert (rule.declare.key, rule.declare.values) == (KEY, ["33", "64"])
    assert rule.declare.required is False
    engine_for(spec)  # building the engine imports the rule modules and checks the checker name
    assert rule.check in known_checkers()
    schema = spec.declaration_schema()
    assert schema["properties"][KEY]["enum"] == ["33", "64"] and schema["required"] == []


def test_spec_rejects_a_wrong_shape(spec):
    data = spec.model_dump(mode="json")
    data["rules"][0]["tier"] = "T0"
    with pytest.raises(ValidationError, match="cannot be tier T0"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][0]["applies"]["file_patterns"] = []
    with pytest.raises(ValidationError, match="needs applies.file_patterns"):
        Spec.model_validate(data)


@pytest.mark.parametrize(
    ("name", "should_match"),
    [
        ("reads.fastq", True),
        ("reads.fq", True),
        ("reads.fastq.gz", True),
        ("reads.fq.gz", True),
        ("run/sample_R1.fastq.gz", True),
        ("/abs/dir/x.fq", True),
        ("reads.fastq.gz.tmp", False),
        ("reads.fasta", False),
        ("reads.fqs", False),
        ("reads.fastq.bak", False),
        ("reads.txt", False),
    ],
)
def test_file_pattern_matches_fastq_names(spec, name, should_match):
    (pattern,) = spec.rules[0].applies.file_patterns
    hits = [m.group(0) for m in re.finditer(pattern, f"awk '{{print}}' {name} | head")]
    assert bool(hits) is should_match
    if should_match:
        assert hits == [name]


# ---- the sniff -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "kind"),
    [
        (33, 73, "phred33"),  # '!'..'I'
        (35, 126, "phred33"),  # anything below ';' proves +33, even with a high maximum
        (58, 104, "phred33"),  # 58 is just below ';'
        (59, 104, "phred64"),  # ';'..'h': Solexa+64
        (66, 104, "phred64"),  # 'B'..'h'
        (64, 75, "phred64"),  # 'K' is the first character above 'J'
        (59, 74, "ambiguous"),  # ';'..'J'
        (66, 73, "ambiguous"),
        (63, 105, "ambiguous"),  # above 'h' and nothing below ';'
        (90, 126, "ambiguous"),
    ],
)
def test_classification_boundaries(low, high, kind):
    assert fe.classify(low, high) == kind


def sniff(tmp_path: Path, quals, name="r.fastq", max_reads=1000, eol="\n"):
    path = write_fastq(tmp_path / name, quals, eol)
    return fe.sniff_qualities(Ctx(tmp_path, "T1"), path, max_reads)


@pytest.mark.parametrize("name", ["r.fastq", "r.fq", "r.fastq.gz"])
def test_sniff_reads_plain_and_gzip_files(tmp_path, name):
    got = sniff(tmp_path, PLUS33, name)
    assert (got.kind, chr(got.low), chr(got.high), got.reads) == ("phred33", "#", "I", 2)
    got = sniff(tmp_path, PLUS64, name)
    assert (got.kind, chr(got.low), chr(got.high), got.reads) == ("phred64", "B", "h", 2)


def test_sniff_solexa_ambiguous_and_above_h(tmp_path):
    assert sniff(tmp_path, SOLEXA).kind == "phred64"
    assert sniff(tmp_path, AMBIGUOUS).kind == "ambiguous"
    assert sniff(tmp_path, ABOVE_H).kind == "ambiguous"


def test_sniff_ignores_crlf_and_is_positional(tmp_path):
    assert sniff(tmp_path, PLUS64, eol="\r\n").kind == "phred64"
    # a quality line may start with '@' or '+': only the line position says what it is
    assert sniff(tmp_path, ["@+@+@+!#", "@@@@++++"]).kind == "phred33"
    assert sniff(tmp_path, ["@" + "h" * 6 + "@"]).kind == "phred64"


def test_sniff_is_bounded_by_max_reads(tmp_path):
    quals = [*AMBIGUOUS, *PLUS64]  # the +64 evidence only starts at the third read
    assert sniff(tmp_path, quals, max_reads=2).kind == "ambiguous"
    assert sniff(tmp_path, quals, max_reads=3).kind == "phred64"


def test_sniff_handles_files_shorter_than_a_read_and_non_fastq(tmp_path):
    for text in (
        "",
        "@only_a_header\n",
        "@r\nACGT\n",
        "@r\nACGT\n+\n",
        ">fasta\nACGT\n",
        "\n\n\n\n",
    ):
        path = tmp_path / "x.fastq"
        path.write_text(text)
        got = fe.sniff_qualities(Ctx(tmp_path, "T1"), path, 100)
        assert got.kind == "none", repr(text)
    # a quality line that is not as long as the sequence ends the sniff there
    path = tmp_path / "bad.fastq"
    path.write_text("@r\nACGT\n+\nhhh\n")
    assert fe.sniff_qualities(Ctx(tmp_path, "T1"), path, 100).kind == "none"
    path = tmp_path / "two.fastq"
    path.write_text("@r\nACGT\n+\n#III\n@s\nACGT\n+\nhh\n")
    got = fe.sniff_qualities(Ctx(tmp_path, "T1"), path, 100)
    assert (got.kind, got.reads) == ("phred33", 1)


def test_sniff_of_corrupt_gzip_is_unreadable_and_of_a_partly_readable_one_keeps_what_it_has(
    tmp_path,
):
    junk = tmp_path / "junk.fastq.gz"
    junk.write_bytes(b"\x1f\x8b" + b"not a gzip stream" * 20)
    assert fe.sniff_qualities(Ctx(tmp_path, "T1"), junk, 10).kind == "unreadable"
    good = write_fastq(tmp_path / "ok.fastq.gz", PLUS64 * 400).read_bytes()
    (tmp_path / "cut.fastq.gz").write_bytes(good[: len(good) * 2 // 3])
    got = fe.sniff_qualities(Ctx(tmp_path, "T1"), tmp_path / "cut.fastq.gz", 1000)
    assert got.kind in ("phred64", "unreadable")


# ---- the command test ----------------------------------------------------------------------------


def seen(tool: str, argv: list[str]) -> tuple[bool, bool]:
    """(the command reads qualities, it shows a 64 offset) for a call on reads.fastq."""
    call = Call(tool, tuple(argv), Path("/nonexistent"))
    text = fe.command_text(call, Ctx(Path("/"), "T0"))
    return fe.reads_qualities(tool, text), fe.shows_64_offset(text)


AWK_TABLE = (
    'BEGIN{for(i=33;i<127;i++)o[sprintf("%c",i)]=i} NR%4==0{for(j=1;j<=length($0);j++)'
    "if(o[substr($0,j,1)]-{off}>=30)n++} END{print n}"
)
READS = [
    ("awk", [AWK_TABLE.replace("{off}", "33"), "reads.fastq"]),
    ("gawk", [AWK_TABLE.replace("{off}", "33"), "reads.fastq"]),
    ("mawk", [AWK_TABLE.replace("{off}", "33"), "reads.fastq"]),
    ("awk", ["-v", "off=33", "{x=o[$1]-off}", "reads.fastq"]),
    ("awk", ['NR%4==0{for(i=1;i<=length($0);i++)if(substr($0,i,1)>="?")c++}END{print c}', "r.fq"]),
    ("awk", ['{ if (substr($0,1,1) >= "?") n++ }', "reads.fastq"]),
    ("python3", ["-c", "print(sum(ord(c) >= 63 for c in open('reads.fastq').read()))"]),
    ("python3", ["-c", "print(sum(ord(c)-33>=30 for l in open('reads.fastq') for c in l))"]),
    ("python3.11", ["-c", "q = [ord(c) - 33 for c in open('reads.fastq').read()]"]),
    ("python3", ["-c", "from Bio import SeqIO; SeqIO.parse('reads.fastq', 'fastq')"]),
    ("python3", ["-c", "OFFSET = 33; open('reads.fastq')"]),
    ("python3", ["-c", "import struct; struct.unpack('B', b'x'); open('reads.fastq')"]),
    ("perl", ["-ne", "$n++ if ord($_)-33>=30; END{print $n}", "reads.fastq"]),
    ("Rscript", ["-e", "q <- utf8ToInt(x) - 33; phred <- 1; read('reads.fastq.gz')"]),
    ("samtools", ["import", "reads.fastq", "-o", "reads.bam"]),
    ("bash", ["-c", "samtools import reads.fastq | samtools view -"]),
    ("bash", ["-c", "sed -n '4~4p' reads.fastq | python3 -c 'print(ord(chr(70))-33)'"]),
    ("seqtk", ["fqchk", "reads.fastq"]),
    ("fastp", ["-i", "reads.fastq", "-o", "out.fastq"]),
    ("bwa", ["mem", "ref.fa", "reads.fastq"]),
]
SILENT_READS = [
    ("head", ["-8", "reads.fastq"]),
    ("head", ["-33", "reads.fastq"]),
    ("tail", ["-n", "4", "reads.fastq"]),
    ("cat", ["reads.fastq"]),
    ("wc", ["-l", "reads.fastq"]),
    ("ls", ["-la", "reads.fastq"]),
    ("zcat", ["reads.fastq.gz"]),
    ("grep", ["-c", "^@", "reads.fastq"]),
    ("sed", ["-n", "4~4p", "reads.fastq"]),
    ("file", ["reads.fastq"]),
    ("md5sum", ["reads.fastq"]),
    ("cut", ["-c1-10", "reads.fastq"]),
    ("awk", ["NR%4==0", "reads.fastq"]),
    ("awk", ["END{print NR/4}", "reads.fastq"]),
    ("awk", ["NR%4==2{print length($0)}", "reads.fastq"]),
    ("awk", ["NR>=5 && NR<=8", "reads.fastq"]),
    ("awk", ["NR>=63 && NR<=70", "reads.fastq"]),
    ("python3", ["-c", "q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q), max(q))"]),
    ("python3", ["-c", "print(len(open('reads.fastq').readlines())//4)"]),
    ("perl", ["-ne", "print if $. <= 4", "reads.fastq"]),
    ("bash", ["-c", "zcat reads.fastq.gz | head -n 400 | tail -n 33"]),
    ("bash", ["-c", "awk 'NR%4==0' reads.fastq | head -3"]),
    ("samtools", ["view", "-H", "reads.bam"]),
    ("samtools", ["faidx", "ref.fa"]),
    ("less", ["reads.fastq"]),
]
HANDLES_64 = [
    ("awk", [AWK_TABLE.replace("{off}", "64"), "reads.fastq"]),
    ("awk", ["-v", "off=64", "{x=o[$1]-off}", "reads.fastq"]),
    ("awk", ["{ if ($1 >= 94) n++ }", "reads.fastq"]),
    ("awk", ['{ if (substr($0,1,1) >= "^") n++ } # ord', "reads.fastq"]),
    ("python3", ["-c", "print(sum(ord(c)-64>=30 for c in open('reads.fastq').read()))"]),
    ("python3", ["-c", "o = 33 if min(map(ord, open('reads.fastq').read())) < 59 else 64"]),
    ("python3", ["q.py", "reads.fastq", "--phred64"]),
    ("python3", ["-c", "ord(c) - ord('@'); open('reads.fastq')"]),
    ("python3", ["-c", "SeqIO.parse('reads.fastq', 'fastq-illumina')"]),
    ("python3", ["-c", "SeqIO.parse('reads.fastq', 'fastq-solexa')"]),
    ("seqtk", ["seq", "-Q64", "-V", "reads.fastq"]),
    ("fastp", ["--phred64", "-i", "reads.fastq"]),
]


@pytest.mark.parametrize(("tool", "argv"), READS)
def test_commands_that_read_qualities_as_numbers_are_recognised(tool, argv):
    assert seen(tool, argv)[0] is True


@pytest.mark.parametrize(("tool", "argv"), SILENT_READS)
def test_viewers_and_non_quality_code_are_not_recognised(tool, argv):
    assert seen(tool, argv)[0] is False


@pytest.mark.parametrize(("tool", "argv"), HANDLES_64)
def test_a_written_64_offset_is_seen(tool, argv):
    assert seen(tool, argv)[1] is True


@pytest.mark.parametrize(("tool", "argv"), READS)
def test_a_33_reading_shows_no_64_offset(tool, argv):
    assert seen(tool, argv)[1] is False


def test_the_fastq_name_itself_does_not_trigger_the_keywords():
    # a file called quality_phred64.fastq must neither look like quality code nor like a 64 offset
    assert seen("awk", ["END{print NR}", "quality_phred64.fastq"]) == (False, False)
    assert seen("cat", ["Q30_reads_64.fq.gz"]) == (False, False)


# ---- the checker through the engine --------------------------------------------------------------

COUNT_33 = ("awk", [AWK_TABLE.replace("{off}", "33"), "reads.fastq"])
COUNT_64 = ("awk", [AWK_TABLE.replace("{off}", "64"), "reads.fastq"])
INSPECT = ("head", ["-8", "reads.fastq"])
INSPECT_PY = ("python3", ["-c", "q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q))"])


@pytest.mark.parametrize("name", ["reads.fastq", "reads.fastq.gz"])
def test_phred64_file_fails_until_declared_and_a_wrong_declaration_fails(spec, tmp_path, name):
    write_fastq(tmp_path / name, PLUS64)
    eng = engine_for(spec)
    tool, argv = COUNT_33[0], [COUNT_33[1][0], name]
    decision, result = outcome(eng, tool, argv, tmp_path)
    assert decision.action == "block" and result.code == "phred64_undeclared"
    text = render_block(decision)
    for needle in ("Phred+64", "'B'..'h'", "ASCII 66..104", "31 too high", "'^'", "--declare"):
        assert needle in text, needle
    assert f"{KEY}=64" in text and "Q33" in text
    decision, result = outcome(eng, tool, argv, tmp_path, {KEY: "64"})
    assert decision.action == "allow" and result.verdict == "pass"
    decision, result = outcome(eng, tool, argv, tmp_path, {KEY: "33"})
    assert decision.action == "block" and result.code == "declared_offset_contradicts_data"
    text = render_block(decision)
    assert "you declared fastq_quality_offset=33" in text and "Phred+64" in text


def test_phred33_file_passes_silently_and_a_declaration_of_64_contradicts_it(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", PLUS33)
    eng = engine_for(spec)
    for tool, argv in (COUNT_33, COUNT_64, INSPECT):
        for declared in (None, "33"):
            decision, result = outcome(
                eng, tool, argv, tmp_path, {KEY: declared} if declared else {}
            )
            assert decision.action == "allow", (tool, argv, declared)
            assert result is None or result.verdict == "pass"
    decision, result = outcome(eng, *COUNT_33, tmp_path, {KEY: "64"})
    assert result.code == "declared_offset_contradicts_data"
    assert "'#'..'I'" in render_block(decision) and "below ';'" in render_block(decision)
    # a viewer is not a quality-reading command: a declaration does not make it one
    assert outcome(eng, *INSPECT, tmp_path, {KEY: "64"})[0].action == "allow"


def test_a_64_offset_written_in_the_command_needs_no_declaration(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", PLUS64)
    eng = engine_for(spec)
    for tool, argv in (COUNT_64, HANDLES_64[4], HANDLES_64[5]):
        decision, result = outcome(eng, tool, argv, tmp_path)
        assert decision.action == "allow", (tool, argv)
        assert result is None or result.verdict == "pass"
    # but a declaration that contradicts the data still fails
    assert (
        outcome(eng, *COUNT_64, tmp_path, {KEY: "33"})[1].code == "declared_offset_contradicts_data"
    )


def test_inspection_of_a_phred64_file_stays_silent(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", PLUS64)
    eng = engine_for(spec)
    for tool, argv in SILENT_READS:
        decision, result = outcome(eng, tool, argv, tmp_path)
        assert decision.action == "allow", (tool, argv)
        assert result is None or result.verdict == "pass"
    for declared in ("33", "64"):
        assert outcome(eng, *INSPECT, tmp_path, {KEY: declared})[0].action == "allow"
        assert outcome(eng, *INSPECT_PY, tmp_path, {KEY: declared})[0].action == "allow"


def test_ambiguous_data_warns_unknown_unless_declared(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", AMBIGUOUS)
    eng = engine_for(spec)
    decision, result = outcome(eng, *COUNT_33, tmp_path)
    assert result.verdict == "unknown" and result.code == "offset_not_provable"
    assert decision.action == "warn"  # unknown never blocks
    warning = render_warning(decision)
    assert "'G'..'I'" in warning and "both Phred+33 and Phred+64" in warning
    for declared in ("33", "64"):
        decision, result = outcome(eng, *COUNT_33, tmp_path, {KEY: declared})
        assert decision.action == "allow" and result.verdict == "pass"
    assert outcome(eng, *COUNT_64, tmp_path)[1].verdict == "pass"  # the command shows its offset
    assert outcome(eng, *INSPECT, tmp_path)[0].action == "allow"
    write_fastq(tmp_path / "high.fastq", ABOVE_H)
    argv = [COUNT_33[1][0], "high.fastq"]
    assert outcome(eng, COUNT_33[0], argv, tmp_path)[1].verdict == "unknown"


def test_short_empty_and_non_fastq_files_cost_nothing(spec, tmp_path):
    eng = engine_for(spec)
    for name, text in (
        ("empty.fastq", ""),
        ("header.fastq", "@r\n"),
        ("short.fastq", "@r\nACGT\n+\n"),
        ("fasta.fq", ">chr1\nACGT\n"),
    ):
        (tmp_path / name).write_text(text)
        decision, result = outcome(eng, COUNT_33[0], [COUNT_33[1][0], name], tmp_path)
        assert decision.action == "allow" and result.verdict == "pass", name


def test_corrupt_gzip_is_unknown_not_a_crash(spec, tmp_path):
    (tmp_path / "junk.fastq.gz").write_bytes(b"\x1f\x8b" + b"not a gzip stream" * 20)
    decision, result = outcome(
        engine_for(spec), COUNT_33[0], [COUNT_33[1][0], "junk.fastq.gz"], tmp_path
    )
    assert result.verdict == "unknown" and result.code == "offset_not_provable"
    assert decision.action == "warn"


def test_invalid_declaration_is_refused_by_the_engine(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", PLUS64)
    _, result = outcome(engine_for(spec), *COUNT_33, tmp_path, {KEY: "pre-1.8"})
    assert result.code == "declaration_invalid"


def test_several_files_and_a_script_file(spec, tmp_path):
    write_fastq(tmp_path / "a.fastq", PLUS33)
    write_fastq(tmp_path / "b.fastq.gz", PLUS64)
    eng = engine_for(spec)
    prog = AWK_TABLE.replace("{off}", "33")
    _, result = outcome(eng, "awk", [prog, "a.fastq", "b.fastq.gz"], tmp_path)
    assert result.verdict == "fail" and "b.fastq.gz" in result.message
    _, result = outcome(eng, "awk", [prog, "a.fastq", "b.fastq.gz"], tmp_path, {KEY: "64"})
    assert result.verdict == "fail" and "a.fastq" in result.message  # contradicts the +33 file
    (tmp_path / "count.py").write_text(
        "OFFSET = 33\nq = [ord(c) - OFFSET for c in open(0).read()]\n"
    )
    (tmp_path / "count64.py").write_text("q = [ord(c) - 64 for c in open(0).read()]\n")
    assert outcome(eng, "python3", ["count.py", "b.fastq.gz"], tmp_path)[1].verdict == "fail"
    assert outcome(eng, "python3", ["count64.py", "b.fastq.gz"], tmp_path)[1].verdict == "pass"
    assert outcome(eng, "python3", ["absent.py", "b.fastq.gz"], tmp_path)[1].verdict == "pass"


def test_the_sniff_is_bounded_by_max_reads_in_the_spec(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", [*AMBIGUOUS, *PLUS64])
    data = spec.model_dump(mode="json")
    data["rules"][0]["params"]["max_reads"] = 2
    short = Spec.model_validate(data)
    assert outcome(engine_for(short), *COUNT_33, tmp_path)[1].verdict == "unknown"
    assert outcome(engine_for(spec), *COUNT_33, tmp_path)[1].verdict == "fail"


def test_the_checker_reads_files_so_it_is_tier_t1_and_a_t0_context_refuses(spec, tmp_path):
    write_fastq(tmp_path / "reads.fastq", PLUS64)
    match = Match(spec.rules[0], (tmp_path / "reads.fastq",))
    check = known_checkers()["fastq_phred_offset"]
    call = Call(COUNT_33[0], tuple(COUNT_33[1]), tmp_path)
    with pytest.raises(TierViolation):
        check(call, match, Ctx(tmp_path, "T0"))
    assert check(call, match, Ctx(tmp_path, "T1")).verdict == "fail"
    # a viewer is decided from the text alone: a T0 context would raise if it read the file
    viewer = Call("head", ("-8", "reads.fastq"), tmp_path)
    assert check(viewer, match, Ctx(tmp_path, "T0")).verdict == "pass"


# ---- F13 workspaces built with the real tools ----------------------------------------------------

# commands of the F13 family tests (tests/test_family_encodings.py) that take the trapped route
FAMILY_NAIVE = [
    "awk '" + AWK_TABLE.replace("{off}", "33") + "' reads.fastq",
    "python3 -c \"print(sum(ord(c)-33>=30 for l in open('reads.fastq').read().split('\\n')[3::4] "
    'for c in l))"',
    "python3 - <<'EOF'\nq = [ord(c) - 33 for l in open('reads.fastq') for c in l.strip()]\nEOF",
    "samtools import reads.fastq -o reads.bam",
    "awk 'NR%4==0' reads.fastq | python3 -c \"print(sum(ord(c) >= 63 for c in open(0).read()))\"",
    "awk 'NR%4==0{for(i=1;i<=length($0);i++)if(substr($0,i,1)>=\"?\")c++}END{print c}' reads.fastq",
    'awk \'BEGIN{for(i=33;i<=126;i++)t=t sprintf("%c",i)} NR%4==0{for(i=1;i<=length($0);i++)'
    "if(index(t,substr($0,i,1))>=30)c++}END{print c}' reads.fastq",
]
FAMILY_SAFE = [
    "head -8 reads.fastq; ls -la",
    "python3 -c \"q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q), max(q))\"",
    "awk '" + AWK_TABLE.replace("{off}", "64") + "' reads.fastq",
    "python3 -c \"o = 33 if min(map(ord, open('reads.fastq').read())) < 59 else 64\"",
    "python3 q.py reads.fastq --phred64",
    "python3 - <<'EOF'\nq = [ord(c) - ord('@') for c in open('reads.fastq').read()]\nEOF",
]


@pytest.fixture(scope="module")
def workspaces(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f13")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F13", 5, twin, work, tools, "implicit"), work)
    return out


def test_the_family_naive_commands_fail_on_the_faulted_twin_and_the_safe_ones_do_not(
    spec, workspaces
):
    _, work = workspaces["faulted"]
    eng = engine_for(spec)
    for command in FAMILY_NAIVE:
        decision, result = outcome(eng, "bash", ["-c", command], work)
        assert result is not None and result.verdict == "fail", command
        assert decision.action == "block"
    for command in FAMILY_SAFE:
        decision, result = outcome(eng, "bash", ["-c", command], work)
        assert decision.action == "allow", command
        assert result is None or result.verdict == "pass"


def test_every_call_on_the_clean_twin_passes_without_a_declaration(spec, workspaces):
    _, work = workspaces["clean"]
    eng = engine_for(spec)
    for template in TEMPLATES:
        decision, result = outcome(eng, template.tool, list(template.argv), work)
        assert decision.action == "allow", template.name
        assert result is None or result.verdict == "pass", template.name


def test_on_the_faulted_twin_only_the_33_calls_fail_and_the_true_declaration_passes_all(
    spec, workspaces
):
    _, work = workspaces["faulted"]
    eng = engine_for(spec)
    for template in TEMPLATES:
        argv = list(template.argv)
        _, result = outcome(eng, template.tool, argv, work)
        failed = result is not None and result.verdict == "fail"
        assert failed == (template.offset == "33"), template.name
        _, declared = outcome(eng, template.tool, argv, work, {KEY: "64"})
        assert declared is None or declared.verdict == "pass", template.name
        _, wrong = outcome(eng, template.tool, argv, work, {KEY: "33"})
        assert (wrong is not None and wrong.verdict == "fail") == (not template.inspection)


# ---- admission -----------------------------------------------------------------------------------


def test_admission(spec, tools, tmp_path):
    stats = measure(RULE_ID, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    right_undeclared = stats.cells[("right", False)]
    assert stats.conditional
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert stats.killed() > 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    # every undeclared 33-offset call on the +64 twin is caught: 8 per seed
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8
    # a wrong declaration contradicts the data; a right declaration with botched code passes
    assert stats.cells[("wrong", True)].failed > 0
