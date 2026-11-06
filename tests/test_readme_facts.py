"""README.md: its numbers, links, paths and commands are checked against the code and the results.

The README states what the studies found, and a number that drifts from `docs/results/` or from
the rule specs costs the repository its credibility, so each one is recomputed here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.families import all_families
from biowarden.engine.spec import packaged_specs
from helpers import console_blocks

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
FLAT = " ".join(re.sub(r"(?m)^> ?", "", README).split())  # wrapped by hand: compare on one line
RESULTS = json.loads((ROOT / "docs" / "results" / "stage2-results.json").read_text())

MODELS = ["claude-haiku-5-5", "deepseek/deepseek-v4.1-flash", "openai/gpt-oss-120b"]
NAMES = ["Claude Haiku 5.5", "DeepSeek V4.1 Flash", "gpt-oss-120b"]
HARDEST_CELL = ["bare", "implicit", "forced"]

GALLERY_FAMILY = {"featureCounts": "F43", "plink": "F59", "blastp": "F78"}


def pct(share: float) -> int:
    return int(share * 100 + 0.5)


def silent(model: str, arm: str, subset: str) -> float:
    """Share of faulted tasks that ended silent, hardest cell, over a set of families."""
    (row,) = (
        r
        for r in RESULTS["safety"]
        if (r["model"], r["arm"], r["set"], r["twin"]) == (model, arm, subset, "faulted")
        and r["cell"] == HARDEST_CELL
    )
    return row["silent_planned"]


def table_rows(header: str, text: str = README) -> list[list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


# ---- the stage-2 numbers -------------------------------------------------------------------------


def test_the_headline_in_the_intro_is_the_analysis():
    bare = [pct(silent(m, "bare", "P")) for m in MODELS]
    block = [pct(silent(m, "block-nonote", "P")) for m in MODELS]

    assert RESULTS["verdict"] == "EFFECTIVE" and RESULTS["primary"]["n_primary"] == 47
    assert (
        f"cut silent wrong answers from {bare[0]}%, {bare[1]}% and {bare[2]}% to "
        f"{block[0]}%, {block[1]}% and {block[2]}% for three agents, on the 47 task families"
    ) in FLAT


def test_the_figure_description_is_the_analysis():
    left = [
        f"{name} {pct(silent(m, 'bare', 'P'))}%(?: with no layer)?, "
        f"{pct(silent(m, 'warn-nonote', 'P'))}%(?: with a warning)?, "
        f"{pct(silent(m, 'block-nonote', 'P'))}%(?: with a block)?"
        for name, m in zip(NAMES, MODELS, strict=True)
    ]
    right = ", ".join(
        f"{pct(silent(m, 'bare', 'gated'))}% and {pct(silent(m, 'block-nonote', 'gated'))}%"
        for m in MODELS
    )

    for pattern in left:
        assert re.search(pattern, README), pattern
    assert f"all 103 gated families, no layer against block: {right}" in README
    assert RESULTS["primary"]["n_safety"] == 103


def test_the_results_table_is_the_analysis():
    rows = {row[0]: row for row in table_rows("| Model | 47 families")}
    by_model = {m["model"]: m for m in RESULTS["primary"]["models"]}

    for name, model in zip(NAMES, MODELS, strict=True):
        _, primary, wide, verdict = rows[name]
        passed = by_model[model]["passes"]
        loss = by_model[model]["delivered_loss"]["estimate"] * 100
        assert primary == (
            f"{pct(silent(model, 'bare', 'P'))}% → {pct(silent(model, 'block-nonote', 'P'))}%"
        )
        assert wide == (
            f"{pct(silent(model, 'bare', 'gated'))}% → "
            f"{pct(silent(model, 'block-nonote', 'gated'))}%"
        )
        assert verdict.split(":")[0] == ("all five met" if passed else "not met")
        if not passed:
            assert f"{loss:.1f} points more tasks ended with no answer (bar: 5)" in verdict
    assert RESULTS["primary"]["n_primary"] == 47 and RESULTS["primary"]["n_safety"] == 103


def test_the_note_under_the_table_is_the_registration():
    draft = " ".join((ROOT / "docs" / "preregistration" / "stage2-draft.md").read_text().split())

    assert "registered at freeze 2 on 2026-11-01" in draft
    assert "2026-11-01, a SHA-256 manifest; no external registry" in FLAT
    assert "the rules were written for these families" in FLAT.lower()


def test_the_declaration_count_in_the_readme_is_the_specs():
    specs = packaged_specs()
    required = sum(1 for s in specs for r in s.rules if r.declare and r.declare.required)

    assert required == 9 and "9 of the 108 rules ask for one" in FLAT


def test_the_stage_1_ladder_of_the_results_page_is_the_pooled_column_of_the_report():
    report = (ROOT / "docs" / "preregistration" / "stage1-report.md").read_text()
    page = (ROOT / "docs" / "results" / "README.md").read_text()
    pooled = {}
    for line in report.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 8 and cells[-1].endswith("%"):
            pooled[cells[0]] = int(cells[-1].rstrip("%"))
    arms = {  # the arm of the report's ladder -> its code in the table of the results page
        "no layer": "bare",
        "prompt: check formats": "careful",
        "layer, generic block": "block-generic",
        "typed tool + bash": "block-gateway",
        "layer, full warning, no prompt sentence": "warn-nonote",
        "layer, full block, no prompt sentence": "block-nonote",
        "layer, full block, prompt sentence": "block",
    }
    shown = {
        code: int(share)
        for code, share in re.findall(r"\(`([a-z-]+)`\) \| (?:yes|no) \| (\d+)%", page)
    }

    assert {code: pooled[arm] for arm, code in arms.items()} == {c: shown[c] for c in arms.values()}
    assert "6 models, 5 families" in report and "seeds 1000-1019" in report
    assert "500 pooled" in report


# ---- counts ----------------------------------------------------------------------------------


def test_rule_and_family_counts():
    rules = sum(len(spec.rules) for spec in packaged_specs())
    families = len(all_families())

    assert {int(n) for n in re.findall(r"(\d+) rules(?! of the last row)", FLAT)} == {rules}
    assert f"{families} task families, each a clean and a faulted twin" in FLAT
    assert f"total {families}." in (ROOT / "docs" / "families.md").read_text()


def test_the_domain_table_is_the_generated_one():
    """tools/make_supported_tools_doc.py counts the rules and families; its test checks the page."""
    page = (ROOT / "docs" / "supported-tools.md").read_text()
    *domains, total = table_rows("| Domain | Tools | Rules | Families |", page)
    rows = table_rows("| Domain | Tools | Rules | Families |")

    assert total[0] == "Total" and (int(total[2]), int(total[3])) == (108, 105)
    assert len(rows) == len(domains) == 8
    for row, generated in zip(rows, domains, strict=True):
        (name, tools, n_rules, n_families), (full, all_tools, *counts) = row, generated
        assert name.split()[0] == full.split()[0], name
        assert [n_rules, n_families] == counts, name
        shown = [tool for tool in tools.split(", ") if tool != "…"]
        assert shown == all_tools.split(", ")[: len(shown)], name  # the same names, same order
        assert ("…" in tools) == (len(shown) < len(all_tools.split(", "))), name


# ---- links, paths, commands ------------------------------------------------------------------


def test_relative_links_images_and_anchors_resolve():
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", README)
    targets += re.findall(r'(?:src|srcset)="([^"]+)"', README)
    headings = re.findall(r"^#+ (.+)$", README, re.M)
    slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}

    for target in targets:
        if not target.startswith(("http://", "https://", "mailto:")):
            assert (ROOT / target).exists(), target
    for anchor in re.findall(r"\]\(#([^)]+)\)", README):
        assert anchor in slugs, anchor
    for alt in re.findall(r'<img alt="([^"]*)"', README):
        assert len(alt) > 40, "a figure needs a description that carries its numbers"


def test_the_readme_is_concise():
    """A human reader gets the point from a short page; the detail lives in docs/."""
    assert len(README.splitlines()) <= 138


def test_the_smoke_test_command_in_the_docs_uses_flags_the_runner_has():
    command = next(
        line.strip()
        for line in (ROOT / "docs" / "reproduce.md").read_text().splitlines()
        if line.startswith("python -I -m biowarden.bench.runner.cli run --out runs/x ")
    )
    flags = re.findall(r"--[a-z][a-z-]*", command)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    help_text = subprocess.run(
        [sys.executable, "-m", "biowarden.bench.runner.cli", "run", "--help"],
        env=env,
        capture_output=True,
        text=True,
    ).stdout

    assert flags and all(flag in help_text for flag in flags)
    assert re.search(r"--families (F\d+)", command).group(1) in all_families()
    assert "BIOWARDEN_ANTHROPIC_API_KEY" in (ROOT / ".env.example").read_text()


def test_the_guard_has_the_modes_the_readme_names():
    help_text = subprocess.run(
        [sys.executable, "-m", "biowarden.engine.guard", "--help"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    ).stdout

    assert "--mode {warn,block,promoted}" in help_text
    assert "--mode warn" in README and "--declare" in README


def test_the_figure_palette_is_credited():
    assert "MoMAColors" in README and "MetBrewer" in README
    assert "MoMAColors" in (ROOT / "tools" / "make_figures.py").read_text()


# ---- the catches -----------------------------------------------------------------------------


@pytest.mark.slow
def test_each_catch_in_the_readme_is_what_the_layer_prints_for_that_command(tmp_path):
    """Builds the faulted workspace of the family behind each pair and runs the command in it."""
    tools = ToolEnv.discover(ROOT)
    if tools is None:
        pytest.skip("the tools of the families are not available")
    env = {**os.environ, **tools.process_env(), "PYTHONPATH": str(ROOT / "src")}
    lines = console_blocks(README)[1]
    pairs = list(zip(lines[0::2], lines[1::2], strict=True))

    assert len(pairs) >= 3
    for command_line, fix in pairs:
        command = re.split(r"\s{2,}#", command_line.removeprefix("$ "))[0]
        program = command.split()[0]
        if not tools.real_bins([program]):
            pytest.skip(f"{program} is not available")
        work = tmp_path / program
        build_task(GALLERY_FAMILY[program], 1000, "faulted", work, tools, "implicit", "forced")
        guard = [sys.executable, "-m", "biowarden.engine.guard", "--", "bash", "-c", command]
        out = subprocess.run(guard, cwd=work, env=env, capture_output=True, text=True)

        assert out.returncode == 97, command
        assert out.stderr.splitlines()[0].startswith(fix.removesuffix(" …")), command


def test_the_readme_points_to_the_two_later_checks():
    assert "docs/real-code-scan.md" in README and "docs/blind-coverage.md" in README
    assert "of the calls in the test suites" not in FLAT  # no figure of the scan is quoted here
