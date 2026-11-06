"""tools/make_diagrams.py: the diagrams state the code's numbers, use the theme's colours and fit.

The drawing needs matplotlib; the numbers, the contrast of the text and the README text do not.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from typing import get_args

import pytest

from biowarden.bench.runner.layer import ARMS
from biowarden.engine.decide import EVALUATE_SECONDS, PACKAGED_PROMOTION, Engine, Mode
from biowarden.engine.shim_client import EXIT_BLOCKED
from biowarden.engine.spec import packaged_specs

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("layers", "decision")


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def md():
    return load("make_diagrams")


@pytest.fixture(scope="module")
def mf(md):
    return md._figures()


def test_the_numbers_are_read_from_the_code(md):
    facts = md.facts()

    assert facts["rules"] == str(sum(len(spec.rules) for spec in packaged_specs()))
    assert facts["exit"] == str(EXIT_BLOCKED)
    assert facts["seconds"] == f"{EVALUATE_SECONDS:.0f}"
    assert facts["third"] == "3rd"  # retry_cap is 2: the third identical block gets the STOP


def test_the_readme_describes_the_diagrams_with_the_same_numbers(md):
    readme = " ".join((ROOT / "README.md").read_text().split())
    facts = md.facts()

    assert f"uses {facts['rules']} rules" in readme
    assert f"exit {facts['exit']} and the fix text" in readme
    assert f"blocked a {facts['third']} time ends in a STOP" in readme
    for stem in STEMS:
        for suffix in ("", "-dark"):
            assert f"docs/figures/{stem}{suffix}.svg" in readme
            assert (ROOT / "docs" / "figures" / f"{stem}{suffix}.svg").is_file()


def test_the_captions_state_the_modes_and_arms_of_the_code():
    readme = " ".join((ROOT / "README.md").read_text().split())
    rules = sum(len(spec.rules) for spec in packaged_specs())
    promoted = json.loads(PACKAGED_PROMOTION.read_text())
    block, warn = ARMS["block-nonote"], ARMS["warn-nonote"]

    assert get_args(Mode) == ("off", "warn", "block", "promoted")
    assert inspect.signature(Engine.__init__).parameters["mode"].default == "promoted"
    assert "Modes: `off`, `warn`, `block` and `promoted`, the default" in readme
    assert len(promoted) == rules and {e["mode"] for e in promoted.values()} == {"block"}
    assert f"all {rules} shipped rules are promoted, so it blocks" in readme
    # in Results the bars are no layer, --mode warn and the default mode, with shims and pre-check
    assert (block.mode, warn.mode) == ("promoted", "warn")
    assert all(arm.shims and arm.precheck and arm.hide_real for arm in (block, warn))
    assert "the layer in `--mode warn`, the layer in its default mode" in readme
    assert "measured with PATH shims plus a whole-command pre-check, in a sandbox" in readme


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_role_reads_well_and_uses_a_colour_of_the_theme(md, mf, mode):
    mf.use_theme("starry-night", mode)
    theme = {*mf.COLOUR.values(), *mf.EDGE.values(), mf.PAGE}

    for name, role in md.roles(mf).items():
        ink = md.text_on(role.fill, mf.INK, mf.PAGE)
        assert md.contrast(role.fill, ink) >= 4.2, (mode, name)
        assert {role.fill, role.edge} <= theme, (mode, name)


def test_a_text_that_does_not_fit_stops_the_drawing(md, mf, tmp_path):
    pytest.importorskip("matplotlib")
    mf.use_theme("starry-night", "light")
    canvas = md.Canvas(mf, 100)
    canvas.box(10, 10, 60, 30, "A label far too long for its box", "", "world")

    with pytest.raises(ValueError, match="does not fit"):
        canvas.save(str(tmp_path / "x"))


def test_the_diagrams_are_drawn_in_the_colours_of_the_charts(md, mf, tmp_path):
    pytest.importorskip("matplotlib")

    assert md.main(["--outdir", str(tmp_path)]) == 0
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        mf.use_theme("starry-night", mode)
        allowed = {c.lower() for c in (*mf.COLOUR.values(), *mf.EDGE.values(), mf.PAGE, mf.INK)}
        for stem in STEMS:
            svg = (tmp_path / f"{stem}{suffix}.svg").read_text()
            used = set(re.findall(r"#[0-9a-fA-F]{6}", svg.lower()))

            assert (tmp_path / f"{stem}{suffix}.png").stat().st_size > 5000
            assert "<text" not in svg and "<path" in svg  # the text is drawn as paths
            assert 'width="597.6pt"' in svg and "<dc:date>" not in svg
            assert used <= allowed, (stem, mode, sorted(used - allowed))
