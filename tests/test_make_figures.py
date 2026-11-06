"""tools/make_figures.py: the numbers behind the bar chart (the drawing needs matplotlib)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mf():
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "tools" / "make_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_figures"] = module
    spec.loader.exec_module(module)
    return module


def row(family="F1", outcome="silent_wrong", **kw):
    base = {
        "family": family,
        "twin": "faulted",
        "outcome": outcome,
        "contract": "bare",
        "tier": "implicit",
        "route": "forced",
        "model": "a/b",
        "arm": "bare",
        "seed": 1000,
    }
    return base | kw


def test_the_module_loads_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # an import of it would now fail
    spec = importlib.util.spec_from_file_location("mf2", ROOT / "tools" / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.share([row()])[0] == 100.0


def test_a_bar_is_the_mean_over_families_not_over_trials(mf):
    rows = (
        [row("F1", "silent_wrong")] * 2 + [row("F1", "pass")] * 6  # F1: 2 of 8 silent
        + [row("F7", "silent_wrong")] * 2  # F7: 2 of 2 silent
    )  # fmt: skip

    value, low, high, n = mf.share(rows)

    assert n == 10
    assert value == pytest.approx(62.5)  # (25% + 100%) / 2, not 4 of 10
    assert low <= value <= high
    assert low >= 25.0 and high <= 100.0  # a resample of two families stays between them


def test_the_interval_resamples_families_and_is_the_same_every_time(mf):
    rows = []
    for family, silent in (("F1", 0), ("F7", 1), ("F10", 2), ("F13", 4), ("F15", 4)):
        rows += [row(family, "silent_wrong")] * silent + [row(family, "pass")] * (4 - silent)

    first, again = mf.share(rows), mf.share(rows)

    assert first == again
    value, low, high, _ = first
    assert value == pytest.approx(100 * (0 + 0.25 + 0.5 + 1 + 1) / 5)
    # a trial-level binomial interval on these 20 trials would be much narrower than this one
    assert high - low > 30
    # with one family there is nothing to resample: no whisker
    one = mf.share([row("F1"), row("F1", "pass")])
    assert one[0] == one[1] == one[2] == 50.0


def test_the_interval_never_leaves_the_bar_it_belongs_to(mf):
    # the percentile interval of a lopsided sample can sit beside the mean; the whisker may not
    rows = [row("F1", "silent_wrong")] + [row(f, "pass") for f in ("F7", "F10", "F13", "F15")]
    value, low, high, _ = mf.share(rows)
    assert low <= value <= high


def write_run(root: Path, rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        folder = root / "trials" / "m" / f"t{i:03d}"
        folder.mkdir(parents=True)
        (folder / "score.json").write_text(json.dumps(r))


def test_load_keeps_the_faulted_primary_cell_trials_of_the_asked_families(mf, tmp_path):
    keep = [row("F1"), row("F7", "pass"), row("F10", harness="minimal")]
    drop = [
        row("F2"),  # another family
        row("F1", twin="clean"),
        row("F1", outcome="infra_error"),
        row("F1", outcome="skipped_budget"),
        row("F1", contract="full"),  # another cell
        row("F1", route="free"),
        row("F1", rep=1),  # the A/A replicate is not a trial of the contrast
        row("F1", harness="opencode"),
    ]
    write_run(tmp_path / "run", keep + drop)

    rows = mf.load(str(tmp_path / "run"))

    assert sorted((r["family"], r["outcome"]) for r in rows) == [
        ("F1", "silent_wrong"),
        ("F10", "silent_wrong"),
        ("F7", "pass"),
    ]
    assert [r["family"] for r in mf.load(str(tmp_path / "run"), ["F2"])] == ["F2"]
    assert mf.load(str(tmp_path / "nowhere")) == []


def test_the_families_can_come_from_a_frozen_sets_file(mf, tmp_path):
    sets = {"primary": ["F10", "F2"], "q1": ["F3"], "q2": ["F5"], "q3": ["F6"], "extra": ["F9"]}
    path = tmp_path / "sets.json"
    path.write_text(json.dumps(sets))

    assert mf.families_from_sets(str(path), "primary") == ["F2", "F10"]  # in family order
    assert mf.families_from_sets(str(path), "bite") == ["F2", "F3", "F10"]
    assert mf.families_from_sets(str(path), "gated") == ["F2", "F3", "F5", "F6", "F10"]
    assert mf.families_from_sets(str(path), "q3") == ["F6"]
    with pytest.raises(ValueError, match="not one of"):
        mf.families_from_sets(str(path), "extra")


def test_the_stage_1_families_are_the_default(mf):
    from biowarden.bench.stage2 import STAGE1_FAMILIES

    assert tuple(STAGE1_FAMILIES) == ("F1", "F7", "F10", "F13", "F15")
    import inspect

    assert inspect.signature(mf.load).parameters["families"].default == STAGE1_FAMILIES


def test_the_stage_2_panel_titles_count_the_families_of_the_sets_file(mf, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    sets = {"primary": ["F1", "F7"], "q1": ["F3"], "q2": ["F5"], "q3": ["F6"]}
    (tmp_path / "sets.json").write_text(json.dumps(sets))
    write_run(tmp_path / "run", [row("F1", model=mf.STAGE2_MODELS[0])])
    drawn = []
    monkeypatch.setattr(mf, "save", lambda fig, stem: drawn.append(fig))

    assert mf.draw_stage2(str(tmp_path / "run"), str(tmp_path / "sets.json"), "unused") == 0

    titles = [ax.get_title(loc="left") for ax in drawn[0].axes]
    assert titles == ["Traps that bite and have a rule (2 families)", "All 5 gated families"]
    assert drawn[0]._suptitle is None  # the README caption carries the title and the footnote


def test_main_needs_a_sets_file_for_a_stratum_and_some_trials(mf, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        mf.main(["--stratum", "primary"])
    assert exc.value.code == 2
    assert mf.main([str(tmp_path / "empty"), str(tmp_path / "fig")]) == 2
    assert "no faulted trials of 5 families" in capsys.readouterr().err


def test_the_chart_is_written_when_matplotlib_is_there(mf, tmp_path):
    pytest.importorskip("matplotlib")
    rows = []
    for arm, silent in (("bare", 4), ("block-nonote", 1)):
        for family in ("F1", "F7"):
            rows += [row(family, arm=arm, model="a/b", seed=i) for i in range(silent)]
            rows += [
                row(family, "pass", arm=arm, model="a/b", seed=10 + i) for i in range(5 - silent)
            ]
    write_run(tmp_path / "run", rows)

    assert mf.main([str(tmp_path / "run"), str(tmp_path / "fig")]) == 0
    assert (tmp_path / "fig.svg").stat().st_size > 1000
    assert (tmp_path / "fig.png").stat().st_size > 1000
