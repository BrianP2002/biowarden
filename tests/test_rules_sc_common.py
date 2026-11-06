"""Shared readers of the sc rules: the state of a matrix, the .h5ad profile, code text, and the
helpers the other test_rules_sc_*.py files import (small objects built with anndata in python-sc).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f101_de_use_raw import write_h5ad
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, TierViolation
from biowarden.engine.rules import sc_common as sc
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-sc" not in env.real_bins(["python-sc"]):
        pytest.skip("python-sc is not available")
    return env


def small_counts(n_cells: int = 24, n_genes: int = 30, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(1.0, 0.7, size=n_genes))
    return rng.poisson(base[None, :] * rng.uniform(0.7, 1.4, size=(n_cells, 1))).astype(np.int32)


def log_cp10k(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / totals * 1e4).astype(np.float32)


def names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


def build(work: Path, name: str, tools: ToolEnv, X, **kw) -> Path:
    n_cells, n_genes = X.shape
    obs = kw.pop("obs_names", None) or names("C", n_cells)
    var = kw.pop("var_names", None) or names("G", n_genes)
    write_h5ad(work, name, X, obs, var, tools=tools, **kw)
    return work / name


def engine_for(spec_name: str) -> Engine:
    return Engine([load_packaged(spec_name)], mode="block", retry_cap=10**9)


def verdict(
    engine: Engine,
    rule_id: str,
    tool: str,
    argv: list[str],
    cwd: Path,
    declared: dict[str, str] | None = None,
    context: tuple[str, ...] = (),
):
    """The rule's CheckResult for one call, or None when the rule did not apply."""
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}, context))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return own[0].result if own else None


# ---- classify ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "state"),
    [
        ([0, 1, 2, 5, 40], "counts"),
        ([0.0, 1.0, 3.0], "counts"),
        ([0.0, 0.5, 2.25, 6.1], "log_normalized"),
        ([0.0, 12.5, 480.25, 9000.0], "normalized"),
        ([-1.2, 0.3, 4.0], "scaled"),
        ([0.0, 0.0], "empty"),
        ([0.0, 30.5, 2.0], "ambiguous"),
    ],
)
def test_classify_states(values, state):
    assert sc.classify(np.array(values)).state == state


def test_classify_uses_the_log1p_marker_for_the_grey_zone():
    assert sc.classify(np.array([0.0, 30.5, 2.0]), log1p_marked=True).state == "log_normalized"


def test_classify_ignores_non_finite_values():
    assert sc.classify(np.array([np.nan, 1.0, 2.0, np.inf])).state == "counts"


def test_says_names_the_range():
    assert "integer counts (max 40)" in sc.classify(np.array([0, 40])).says()
    assert "negatives" in sc.classify(np.array([-1.0, 2.5])).says()


# ---- read_h5ad ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def objects(tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("sc-common")
    counts = small_counts()
    logged = log_cp10k(counts)
    out = {
        "counts": build(work, "counts.h5ad", tools, counts.astype(np.float32)),
        "log": build(work, "log.h5ad", tools, logged, uns={"log1p": {"base": None}}),
        "log_nomark": build(work, "log_nomark.h5ad", tools, logged),
        "layer": build(
            work, "layer.h5ad", tools, logged, layers={"counts": counts.astype(np.float32)}
        ),
        "raw_cp10k": build(
            work, "raw_cp10k.h5ad", tools, logged, raw={"X": np.expm1(logged).astype(np.float32)}
        ),
        "dense": build(work, "dense.h5ad", tools, logged, sparse=False),
        "int16": build(work, "int16.h5ad", tools, counts.astype(np.int16)),
        "obs": build(
            work,
            "obs.h5ad",
            tools,
            counts.astype(np.float32),
            obs={"group": ["a", "b"] * 12, "label": ["ctrl"] * 6 + ["x"] * 18},
            var={"symbol": names("S", 30)},
        ),
    }
    out["work"] = work
    return out


def test_profile_of_counts_log_and_dense(objects):
    counts = sc.read_h5ad(objects["counts"])
    assert counts.x.state == "counts" and counts.x.sparse and counts.shape == (24, 30)
    assert counts.obs_names[:2] == ["C000", "C001"] and counts.var_names[-1] == "G029"
    log = sc.read_h5ad(objects["log"])
    assert log.x.state == "log_normalized" and log.log1p_marked
    assert not sc.read_h5ad(objects["log_nomark"]).log1p_marked
    dense = sc.read_h5ad(objects["dense"])
    assert dense.x.state == "log_normalized" and not dense.x.sparse
    assert dense.x.dtype == "float32"


def test_profile_of_layers_raw_and_dtype(objects):
    layer = sc.read_h5ad(objects["layer"])
    assert layer.layers["counts"].state == "counts" and layer.counts_source() == "layers['counts']"
    raw = sc.read_h5ad(objects["raw_cp10k"])
    assert raw.raw is not None and raw.raw.state == "normalized"
    assert raw.raw_var_names[0] == "G000" and raw.counts_source() is None
    assert sc.read_h5ad(objects["int16"]).x.dtype == "int16"


def test_columns_and_values(objects):
    info = sc.read_h5ad(objects["obs"])
    assert {"group", "label"} <= set(info.obs_columns) and "symbol" in info.var_columns
    assert sc.obs_column(objects["obs"], "label")[:7] == ["ctrl"] * 6 + ["x"]
    assert sc.var_column(objects["obs"], "symbol")[0] == "S000"
    assert sc.obs_column(objects["obs"], "missing") is None


def test_a_file_that_is_not_an_h5ad_gives_none(tmp_path):
    bad = tmp_path / "bad.h5ad"
    bad.write_bytes(b"not hdf5 at all")
    assert sc.read_h5ad(bad) is None
    assert sc.read_h5ad(tmp_path / "missing.h5ad") is None
    other = tmp_path / "plain.h5"
    import h5py

    with h5py.File(other, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    assert sc.read_h5ad(other) is None


# ---- code text ---------------------------------------------------------------------------------


def test_code_text_reads_scripts_context_and_arguments(tmp_path):
    (tmp_path / "run.py").write_text("print('from the script')\n")
    call = Call("python-sc", ("run.py", "-c", "print(1)"), tmp_path, {}, ("earlier words",))
    ctx = Ctx(tmp_path, "T1")
    text = sc.code_text(call, ctx)
    assert "from the script" in text and "earlier words" in text and "print(1)" in text


def test_scripts_need_tier_t1(tmp_path):
    (tmp_path / "run.py").write_text("x = 1\n")
    call = Call("python-sc", ("run.py",), tmp_path)
    with pytest.raises(TierViolation):
        sc.code_text(call, Ctx(tmp_path, "T0"))


def test_strip_comments_and_find_files(tmp_path):
    assert "use_raw" not in sc.strip_comments("a = 1  # use_raw=False\nb = 2\n")
    (tmp_path / "cells.h5ad").write_bytes(b"x")
    found = sc.find_files("read('cells.h5ad') and ./cells.h5ad", tmp_path, sc._NAME_H5AD)
    assert found == [tmp_path / "cells.h5ad"]


def test_gzip_script_is_not_needed_but_gzip_tables_read(tmp_path):
    path = tmp_path / "t.csv.gz"
    path.write_bytes(gzip.compress(b"a,b\n1,2\n"))
    assert list(Ctx(tmp_path, "T1").lines(path)) == ["a,b", "1,2"]


# ---- admission ---------------------------------------------------------------------------------


def check_admission(rule_id: str, spec_name: str, tools: ToolEnv, tmp_path: Path):
    """Measure a rule over ADMISSION_SEEDS seeds with calls labelled by running the code and assert
    the admission gates: no failure on correct calls (declared or not, the rule is conditional),
    at least one wrong call caught, no crash, a promotion that follows from the numbers."""
    stats = measure(rule_id, [load_packaged(spec_name)], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared, right_plain = stats.right_declared, stats.cells[("right", False)]
    wrong_declared, wrong_plain = stats.cells[("wrong", True)], stats.cells[("wrong", False)]
    assert stats.conditional and stats.admitted(), stats.cells
    assert right_declared.n >= MIN_RIGHT_CALLS and right_plain.n >= MIN_RIGHT_CALLS
    assert right_declared.failed == 0 and right_plain.failed == 0
    assert wrong_declared.n > 0 and wrong_plain.n > 0 and stats.killed() > 0
    assert sum(c.crashed for c in stats.cells.values()) == 0
    assert (stats.recall() or 0) >= 0.4
    assert promotion_is_consistent(stats)
    return stats
