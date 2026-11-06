"""engine.shellsplit: bash command strings as agents write them -> simple commands."""

# ruff: noqa: E501  (the realistic commands are one line each, as the agent wrote them)

from __future__ import annotations

import random
import time

import pytest

from biowarden.engine.shellsplit import (
    MAX_COMMANDS,
    Redirect,
    SimpleCommand,
    Split,
    split_command,
)


def cmds(text: str, cwd: str | None = "/work") -> list[tuple[str, tuple[str, ...]]]:
    return [(c.tool, c.argv) for c in split_command(text, cwd).commands]


def tools(text: str, cwd: str | None = "/work") -> list[str]:
    return [c.tool for c in split_command(text, cwd).commands]


# -- basics --------------------------------------------------------------------------------------


def test_a_pipeline_with_a_redirect_and_cd_prefix():
    result = split_command(
        "cd /work && head -3 expr.tsv | awk -F'\\t' '{print $2}' > out.txt", "/work"
    )

    assert [(c.tool, c.argv, c.cwd) for c in result.commands] == [
        ("cd", ("/work",), "/work"),
        ("head", ("-3", "expr.tsv"), "/work"),
        ("awk", ("-F\\t", "{print $2}"), "/work"),
    ]
    assert result.commands[2].redirects == (Redirect(">", "out.txt"),)
    assert result.unparsed == () and result.cwd == "/work"


def test_tool_is_the_basename_of_the_program():
    assert tools("/usr/bin/awk 1 f; ./run.sh; ~/bin/tool x; python3.12 -c 1") == [
        "awk",
        "run.sh",
        "tool",
        "python3.12",
    ]


@pytest.mark.parametrize("sep", [";", "&&", "||", "\n", "&", " ; ", "&&\n"])
def test_every_separator_splits(sep):
    assert tools(f"echo a{sep}echo b") == ["echo", "echo"]


def test_empty_and_blank_input():
    for text in ("", "   ", "\n\n", "# only a comment", ";", "&&"):
        assert split_command(text).commands == ()


# -- quoting -------------------------------------------------------------------------------------


def test_code_strings_are_one_argument_with_quotes_removed():
    awk = "NR>1{s=0;for(i=2;i<=NF;i++)s+=$i; if(s/(NF-1)>100)n++} END{print n+0}"
    py = "import csv\nrows = list(csv.reader(open('expr.tsv'), delimiter='\\t'))\nprint(len(rows))"

    assert cmds(f"awk -F'\\t' '{awk}' expr.tsv") == [("awk", ("-F\\t", awk, "expr.tsv"))]
    assert cmds(f"awk -F\"\\t\" -v thr=100 -v 'label=a b' '{awk}' expr.tsv")[0][1][:5] == (
        "-F\\t",
        "-v",
        "thr=100",
        "-v",
        "label=a b",
    )
    assert cmds(f'python3 -c "{py.replace(chr(34), chr(92) + chr(34))}"') == [
        ("python3", ("-c", py))
    ]
    plain = py.replace("'", '"')
    assert cmds(f"python3 -c '{plain}'") == [("python3", ("-c", plain))]


def test_quote_concatenation_escapes_and_empty_arguments():
    assert cmds("echo 'a b'\"c d\"e\\ f '' \"\"") == [("echo", ("a bc de f", "", ""))]
    assert cmds('echo "say \\"hi\\" \\$HOME \\\\ \\n"') == [("echo", ('say "hi" $HOME \\ \\n',))]
    assert cmds("printf $'a\\tb\\n' x") == [("printf", ("a\tb\n", "x"))]
    assert cmds("echo 'it'\\''s'") == [("echo", ("it's",))]


def test_comments_are_dropped_but_hashes_inside_words_are_not():
    assert cmds("echo a#b $# ${#x} 'x # y' # real comment\necho next") == [
        ("echo", ("a#b", "$#", "${#x}", "x # y")),
        ("echo", ("next",)),
    ]


def test_line_continuations_join_the_command():
    text = "bedtools intersect \\\n  -a variants.bed \\\n  -b regions.bed \\\n  > out.bed"
    result = split_command(text)
    assert [(c.tool, c.argv) for c in result.commands] == [
        ("bedtools", ("intersect", "-a", "variants.bed", "-b", "regions.bed"))
    ]
    assert result.commands[0].redirects == (Redirect(">", "out.bed"),)
    assert cmds('echo "a \\\nb"') == [("echo", ("a b",))]


def test_crlf_input():
    assert tools("echo a\r\necho b\r\n") == ["echo", "echo"]


# -- redirections and here-documents -----------------------------------------------------------


def test_redirections_never_enter_argv():
    result = split_command(
        "sort -k2,2nr expr.tsv > sorted.tsv 2>/dev/null; cat < in.txt >> log 2>&1"
    )

    first, second = result.commands
    assert first.argv == ("-k2,2nr", "expr.tsv")
    assert first.redirects == (Redirect(">", "sorted.tsv"), Redirect(">", "/dev/null", "2"))
    assert second.argv == ()
    assert second.redirects == (
        Redirect("<", "in.txt"),
        Redirect(">>", "log"),
        Redirect(">&", "1", "2"),
    )


def test_redirect_before_the_command_and_ampersand_forms():
    (c,) = split_command("> out.txt echo hi").commands
    assert (c.tool, c.argv, c.redirects) == ("echo", ("hi",), (Redirect(">", "out.txt"),))
    (c,) = split_command("tool &> all.log").commands
    assert c.redirects == (Redirect("&>", "all.log"),)


def test_heredoc_body_is_apart_from_argv_and_following_commands_still_parse():
    text = "cd /work; python3 - <<'EOF'\nimport csv\nprint(open('expr.tsv').read())\nEOF\necho done"
    result = split_command(text)

    py = result.commands[1]
    assert (py.tool, py.argv) == ("python3", ("-",))
    assert py.heredoc == "import csv\nprint(open('expr.tsv').read())\n"
    assert py.redirects == (Redirect("<<", "EOF"),)
    assert result.commands[2].tool == "echo" and result.unparsed == ()


def test_heredoc_variants_tabs_quotes_two_on_a_line_and_redirect_target():
    text = "cat > r.json <<EOF\n{\"a\": 1}\nEOF\ncat <<-'X'\n\tindented\n\tX\ncat <<A <<B\none\nA\ntwo\nB\n"
    first, second, third = split_command(text).commands

    assert first.heredoc == '{"a": 1}\n' and first.redirects[0] == Redirect(">", "r.json")
    assert second.heredoc == "indented\n"
    assert third.heredoc == "one\n"  # the body of the first heredoc


def test_heredoc_delimiter_inside_a_pipeline_and_with_a_terminator_on_its_own_line():
    text = "python3 - <<'PY' | head -3\nprint(1)\nPY\nls"
    result = split_command(text)
    assert [(c.tool, c.heredoc) for c in result.commands] == [
        ("python3", "print(1)\n"),
        ("head", None),
        ("ls", None),
    ]


def test_here_string_is_a_redirect_not_a_heredoc():
    (c,) = split_command("awk '{print $1}' <<< 'chr1 10 20'").commands
    assert c.heredoc is None and c.redirects == (Redirect("<<<", "chr1 10 20"),)


def test_unterminated_heredoc_is_flagged_and_keeps_the_body():
    result = split_command("python3 - <<'PY'\nprint(1)\n")
    assert result.commands[0].heredoc == "print(1)\n"
    assert [u.reason for u in result.unparsed] == ["here-document not terminated"]


# -- substitution ---------------------------------------------------------------------------------


def test_command_substitution_inner_commands_are_returned_nested_first():
    result = split_command("echo $(cat names.txt | wc -l) done")

    assert [(c.tool, c.nested) for c in result.commands] == [
        ("cat", True),
        ("wc", True),
        ("echo", False),
    ]
    assert result.commands[-1].argv == ("$(cat names.txt | wc -l)", "done")


def test_backticks_nesting_and_quoted_substitution():
    assert tools("echo `date +%s`") == ["date", "echo"]
    assert tools('x="$(echo $(basename "$(pwd)"))"; ls') == ["pwd", "basename", "echo", "ls"]
    assert cmds('echo "$(echo "a b")"')[-1] == ("echo", ('$(echo "a b")',))


def test_process_substitution_keeps_the_text_in_the_argument():
    result = split_command("diff <(zcat a.tsv.gz | sort) <(sort b.tsv) > d.txt")

    assert [(c.tool, c.nested) for c in result.commands] == [
        ("zcat", True),
        ("sort", True),
        ("sort", True),
        ("diff", False),
    ]
    assert result.commands[-1].argv == ("<(zcat a.tsv.gz | sort)", "<(sort b.tsv)")


def test_arithmetic_is_not_a_command():
    assert cmds("echo $((1 + 2)) $(( (3+4)*2 )); (( n++ )); echo x") == [
        ("echo", ("$((1 + 2))", "$(( (3+4)*2 ))")),
        ("echo", ("x",)),
    ]


# -- cd tracking and scope -----------------------------------------------------------------------


def test_cd_is_tracked_across_separators_and_is_relative_to_the_current_directory():
    result = split_command("cd /work; ls; cd sub && ls; cd ../other/.. ; ls; cd -; ls", "/")

    assert [(c.tool, c.cwd) for c in result.commands if c.tool == "ls"] == [
        ("ls", "/work"),
        ("ls", "/work/sub"),
        ("ls", "/work"),
        ("ls", "/work/sub"),
    ]
    assert result.cwd == "/work/sub"


def test_cd_does_not_leak_out_of_subshells_substitutions_pipelines_and_background():
    text = (
        "(cd a && ls); ls; echo $(cd b; pwd); ls; cd c | cat; ls; cd d & ls; bash -c 'cd e; ls'; ls"
    )
    where = [(c.tool, c.cwd) for c in split_command(text, "/w").commands]

    assert ("ls", "/w/a") in where
    outer = [cwd for tool, cwd in where if tool == "ls"]
    assert outer.count("/w") == 5 and outer.count("/w/a") == 1 and outer.count("/w/e") == 1


def test_cd_persists_inside_braces_ifs_and_loops():
    assert [c.cwd for c in split_command("{ cd a; }; ls", "/w").commands if c.tool == "ls"] == [
        "/w/a"
    ]
    assert [c.cwd for c in split_command("if true; then cd a; fi; ls", "/w").commands][-1] == "/w/a"
    assert [c.cwd for c in split_command("for d in x y; do cd $d; done; ls", "/w").commands][
        -1
    ] == "/w/x/y"


def test_unknowable_directories_become_none_until_an_absolute_cd():
    result = split_command(
        'cd "$(dirname "$0")"; ls; cd $HOME/x; ls; cd sub; ls; cd /abs; ls', "/w"
    )
    assert [c.cwd for c in result.commands if c.tool == "ls"] == [None, None, None, "/abs"]
    assert [c.cwd for c in split_command("cd; ls", "/w").commands][-1] == "~"
    assert [c.cwd for c in split_command("cd ~/proj; ls", "/w").commands][-1] == "~/proj"


def test_default_directory_is_dot():
    assert [c.cwd for c in split_command("ls; cd d; ls").commands] == [".", ".", "d"]


# -- variables, loops, control flow -------------------------------------------------------------


def test_assigned_variables_are_expanded_outside_single_quotes():
    result = split_command(
        "f=expr.tsv; d=/data; awk '$1>0' $f \"$d/${f}\" 'lit$f'; FOO=1 BAR=x tool a"
    )

    awk, tool = result.commands
    assert awk.argv == ("$1>0", "expr.tsv", "/data/expr.tsv", "lit$f")
    assert (tool.tool, tool.env) == ("tool", (("FOO", "1"), ("BAR", "x")))


def test_for_loop_over_words_runs_the_body_once_per_word():
    result = split_command("for f in a.tsv b.tsv 'c d.tsv'; do awk '{print $1}' $f; done; echo end")

    assert cmds("for f in a.tsv b.tsv; do awk 1 $f; done") == [
        ("awk", ("1", "a.tsv")),
        ("awk", ("1", "b.tsv")),
    ]
    assert [c.argv[-1] for c in result.commands if c.tool == "awk"] == ["a.tsv", "b.tsv", "c d.tsv"]
    assert result.commands[-1].tool == "echo"


def test_for_loop_over_a_glob_or_substitution_is_returned_once_with_the_variable_literal():
    assert cmds("for f in *.tsv; do wc -l $f; done") == [("wc", ("-l", "*.tsv"))]
    result = split_command("for f in $(ls *.tsv); do wc -l $f; done")
    assert [(c.tool, c.argv) for c in result.commands] == [("ls", ("*.tsv",)), ("wc", ("-l", "$f"))]


def test_nested_loops_and_while_read_pipelines():
    text = "for a in 1 2; do for b in x y; do echo $a$b; done; done"
    assert [c.argv[0] for c in split_command(text).commands] == ["1x", "1y", "2x", "2y"]
    text = "cat ids.txt | while read id; do grep $id table.tsv; done > out.txt"
    assert tools(text) == ["cat", "read", "grep"]


def test_if_case_functions_and_tests_are_flattened():
    text = (
        "if [ -f x ]; then echo yes; elif [[ -d y && -f z ]]; then echo dir; else echo no; fi\n"
        "case $1 in a|b) echo ab;; *) echo other;; esac\n"
        "f() { echo in f; }\nfunction g { echo in g; }\nf"
    )
    assert tools(text) == ["[", "echo", "echo", "echo", "echo", "echo", "echo", "echo", "f"]


def test_a_huge_loop_is_bounded():
    words = " ".join(f"w{i}" for i in range(500))
    result = split_command(f"for f in {words}; do echo $f; done")
    assert len(result.commands) <= 24
    assert any("truncated" in u.reason for u in result.unparsed)


# -- wrappers ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,tool,argv,wrappers",
    [
        ("timeout 60 python3 -c 'print(1)'", "python3", ("-c", "print(1)"), ("timeout",)),
        ("timeout -s KILL 10s awk 1 f", "awk", ("1", "f"), ("timeout",)),
        ("nohup python3 job.py &", "python3", ("job.py",), ("nohup",)),
        ("time -p awk 1 f", "awk", ("1", "f"), ("time",)),
        ("nice -n 5 sort f", "sort", ("f",), ("nice",)),
        ("sudo -u bob awk 1 f", "awk", ("1", "f"), ("sudo",)),
        ("env LC_ALL=C sort -n f", "sort", ("-n", "f"), ("env",)),
        ("command awk 1 f", "awk", ("1", "f"), ("command",)),
        ("exec python3 x.py", "python3", ("x.py",), ("exec",)),
        ("xargs -n1 -I{} awk 1 {}", "awk", ("1", "{}"), ("xargs",)),
        ("timeout 5 nice stdbuf -oL awk 1 f", "awk", ("1", "f"), ("timeout", "nice", "stdbuf")),
    ],
)
def test_wrappers_are_removed_so_the_tool_is_the_program_that_runs(text, tool, argv, wrappers):
    (c,) = split_command(text).commands
    assert (c.tool, c.argv, c.wrappers) == (tool, argv, wrappers)


def test_env_assignments_of_the_wrappers_are_recorded():
    (c,) = split_command("env -i LC_ALL=C X=1 awk 1 f").commands
    assert c.tool == "awk" and dict(c.env) == {"LC_ALL": "C", "X": "1"}


def test_command_v_is_a_lookup_not_a_wrapper():
    (c,) = split_command("command -v awk").commands
    assert (c.tool, c.argv) == ("command", ("-v", "awk"))


def test_bash_c_and_eval_are_kept_and_split_again():
    result = split_command("bash -lc \"cd sub && awk '{print $1}' expr.tsv\"", "/w")

    assert [(c.tool, c.nested, c.cwd) for c in result.commands] == [
        ("bash", False, "/w"),
        ("cd", True, "/w"),
        ("awk", True, "/w/sub"),
    ]
    assert result.commands[0].argv[0] == "-lc"
    assert tools('eval "awk 1 f | sort"') == ["eval", "awk", "sort"]
    assert tools("sh script.sh") == ["sh"]  # a script file is not inline code


# -- unparsed text and robustness ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,reason",
    [
        ("echo 'abc", "unterminated single quote"),
        ('echo "abc', "unterminated double quote"),
        ("echo $(abc", "unterminated $("),
        ("echo `abc", "unterminated backtick"),
        ("echo ${abc", "unterminated ${"),
        ("echo a )", "unmatched )"),
        ("(echo a", "unclosed ("),
        ("echo a >", "redirection without a target"),
        ("for f in a b; do echo $f", "for without done"),
    ],
)
def test_odd_input_is_flagged_and_the_rest_is_returned(text, reason):
    result = split_command(text)

    assert reason in [u.reason for u in result.unparsed]
    assert result.commands  # the text before the problem is still returned


def test_a_for_header_without_a_body_is_flagged():
    result = split_command("for f in a b")
    assert [u.reason for u in result.unparsed] == ["for without do"] and result.commands == ()


def test_text_after_a_problem_is_still_parsed():
    result = split_command("echo ok; echo 'broken")
    assert [c.argv for c in result.commands] == [("ok",), ("broken",)]


def test_nesting_depth_and_command_count_are_capped():
    deep = "echo " + "$(echo " * 40 + "x" + ")" * 40
    result = split_command(deep)
    assert any("too deep" in u.reason for u in result.unparsed)
    many = "; ".join(["echo a"] * (MAX_COMMANDS + 50))
    result = split_command(many)
    assert len(result.commands) == MAX_COMMANDS
    assert any("too many" in u.reason for u in result.unparsed)


def test_a_very_long_command_is_fast():
    script = "x = 1\n" * 20000
    started = time.perf_counter()
    result = split_command(f"python3 - <<'EOF'\n{script}EOF\nawk '{'a' * 100000}' f")
    assert time.perf_counter() - started < 3
    assert [c.tool for c in result.commands] == ["python3", "awk"]


@pytest.mark.parametrize(
    "text",
    [
        "$(" * 5000,
        "'" * 5001,
        '"' * 5001,
        "(" * 3000,
        "\\" * 5000,
        "for f in a; do " * 500 + "echo $f;" + "done;" * 500,
        "echo " + "$(echo " * 3000 + "x" + ")" * 3000,
        "cat <<A " * 2000 + "\n",
        "case x in a) " * 2000,
        "a | " * 20000 + "b",
        "${" * 3000,
        "echo a\n" * 50000,
    ],
)
def test_pathological_input_is_bounded_in_time_and_never_raises(text):
    started = time.perf_counter()
    result = split_command(text, "/w")
    assert time.perf_counter() - started < 5
    assert isinstance(result, Split) and len(result.commands) <= MAX_COMMANDS
    assert len(result.unparsed) < 50  # one report per kind of problem, not one per character


def test_non_string_input_does_not_crash():
    for value in (None, 12, b"echo hi", ["echo"]):
        assert isinstance(split_command(value), Split)  # type: ignore[arg-type]


REALISTIC = [
    "cd /work && ls -la && cat README.md",
    "cd /work; head -5 expr.tsv; echo ---; awk -F'\\t' 'NR==1{print NF-1\" samples\"} END{print NR-1\" genes\"}' expr.tsv",
    "cd /work && awk -F'\\t' 'NR>1{s=0;for(i=2;i<=NF;i++)s+=$i; if(s/(NF-1)>100)c++}END{print c+0}' expr.tsv",
    "cat /work/README.md; echo ---; ls -la /work",
    "cd /work && python3 - <<'EOF'\nimport csv, math\nrows = list(csv.reader(open('expr.tsv'), delimiter='\\t'))\nprint(sum(2**float(x) - 1 for r in rows[1:] for x in r[1:]))\nEOF",
    'cd /work && cat > result.json <<\'EOF\'\n{"status":"ok","answer":27,"flags":[],"method":"back-transformed"}\nEOF\ncat result.json && python3 -m json.tool result.json',
    "cd /work && bcftools view -R regions.bed -H variants.vcf.gz | cut -f3 | python3 -c 'import sys,json; print(json.dumps({\"answer\":[l.strip() for l in sys.stdin]}))' > result.json && cat result.json",
    "zcat variants.vcf.gz | grep -v '^#' | awk -v OFS='\\t' '{print $1,$2-1,$2,$3}' | bedtools intersect -a stdin -b regions.bed -wa | cut -f4 | sort -u | wc -l",
    "samtools faidx ref.fa $(awk '{print $1\":\"$2\"-\"$3}' regions.txt | tr '\\n' ' ') 2>&1 | head -20",
    "for r in $(cat regions.txt); do samtools faidx ref.fa $r | tail -n +2 | tr -d '\\n'; echo; done",
    "while read -r chrom start end; do bcftools view -H -r ${chrom}:${start}-${end} variants.vcf.gz | wc -l; done < regions.bed",
    "paste <(cut -f1 a.tsv) <(cut -f2 b.tsv) | sort -k2,2nr | head -n 5 > top.tsv; diff top.tsv expected.tsv || echo differs",
    "export LC_ALL=C; sort -t$'\\t' -k1,1 -k2,2n regions.bed > sorted.bed && bedtools merge -i sorted.bed -c 4 -o distinct",
    "[ -f ref.fa.fai ] || samtools faidx ref.fa; test -s variants.vcf.gz && echo ok",
    "perl -F'\\t' -lane 'next if $.==1; $s=0; $s+=2**$F[$_]-1 for 1..$#F; $n++ if $s/$#F>100; END{print $n+0}' expr.tsv",
    "Rscript -e 'x <- read.table(\"expr.tsv\", header=TRUE, row.names=1); print(sum(rowMeans(2^x-1) > 100))'",
    "python3 -c \"\nimport gzip\nwith gzip.open('/work/variants.vcf.gz','rt') as f:\n    for line in f:\n        if not line.startswith('#'):\n            print(line.split('\\t')[2])\n\" | head",
    "mkdir -p /tmp/t && cd /tmp/t && cp /work/*.bed . && ls; cd /work && rm -rf /tmp/t",
    "(cd /work && ls) > listing.txt; wc -l < listing.txt",
    'echo \'{"answer": ["v04", "v05"]}\' > /work/result.json; cat /work/result.json',
]


@pytest.mark.parametrize("text", REALISTIC)
def test_realistic_agent_commands_parse_without_unparsed_text(text):
    result = split_command(text, "/work")
    assert result.unparsed == (), result.unparsed
    assert result.commands


def test_realistic_command_details():
    out = split_command(REALISTIC[7]).commands
    assert [c.tool for c in out] == ["zcat", "grep", "awk", "bedtools", "cut", "sort", "wc"]
    assert out[2].argv == ("-v", "OFS=\t".replace("\t", "\\t"), "{print $1,$2-1,$2,$3}")
    assert out[3].argv == ("intersect", "-a", "stdin", "-b", "regions.bed", "-wa")
    out = split_command(REALISTIC[8]).commands
    assert [c.tool for c in out] == ["awk", "tr", "samtools", "head"]
    assert out[2].argv[:2] == ("faidx", "ref.fa") and out[2].argv[2].startswith("$(awk")
    out = split_command(REALISTIC[10]).commands
    assert out[-1].tool == "wc" and out[0].tool == "read"
    assert out[1].argv == ("view", "-H", "-r", "${chrom}:${start}-${end}", "variants.vcf.gz")
    out = split_command(REALISTIC[17]).commands
    assert [(c.tool, c.cwd) for c in out if c.tool in ("ls", "rm")] == [
        ("ls", "/tmp/t"),
        ("rm", "/work"),
    ]
    assert split_command(REALISTIC[12]).commands[1].argv == (
        "-t\t",
        "-k1,1",
        "-k2,2n",
        "regions.bed",
    )


def test_truncating_realistic_commands_anywhere_never_raises():
    for text in REALISTIC:
        for end in range(0, len(text) + 1, 3):
            result = split_command(text[:end], "/work")
            assert isinstance(result.commands, tuple) and isinstance(result.unparsed, tuple)
            assert all(isinstance(c, SimpleCommand) for c in result.commands)


def test_random_metacharacter_soup_never_raises_or_hangs():
    rng = random.Random(7)
    alphabet = list("ab $\\'\"`(){}[]<>|&;#\n\t=*?~!-0123") + [
        "<<",
        "$(",
        "<(",
        "EOF\n",
        "for ",
        "do ",
        "done",
        "case ",
        "esac",
        "if ",
    ]
    started = time.perf_counter()
    for _ in range(4000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        result = split_command(text, rng.choice(["/work", ".", None]))
        assert isinstance(result, Split)
    assert time.perf_counter() - started < 20


def test_mutated_realistic_commands_never_raise():
    rng = random.Random(11)
    for _ in range(1500):
        text = list(rng.choice(REALISTIC))
        for _ in range(rng.randint(1, 4)):
            i = rng.randrange(len(text))
            text[i] = rng.choice(list("'\"`$(){}<>|&;\\\n"))
        assert isinstance(split_command("".join(text)), Split)


def test_program_keeps_the_word_as_written_so_a_call_by_path_can_be_told_from_a_call_by_name():
    split = split_command(
        "/usr/bin/awk 1 f; ./run.sh; env X=1 /opt/x/bcftools view; python3.12 -c 1"
    )

    assert [(c.tool, c.program) for c in split.commands] == [
        ("awk", "/usr/bin/awk"),
        ("run.sh", "./run.sh"),
        ("bcftools", "/opt/x/bcftools"),
        ("python3.12", "python3.12"),
    ]
    assert [c.program for c in split_command("time nohup samtools view -H x.bam").commands] == [
        "samtools"
    ]
