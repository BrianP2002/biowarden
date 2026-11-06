"""Rule sc.mudata_obs_union (family F106 of the sc pack): the cells of a MuData are the UNION of
the cells of its modalities.

Verified here (mudata 0.3.10, anndata 0.11.4, python-sc): an RNA modality with cells c0..c9 and an
ATAC modality with cells c3..c12 give `MuData.n_obs == 13`, `shape == (13, 14)`,
`len(obs_names) == 13`, `obs.shape == (13, 2)`, while 7 cells have both profiles and each modality
reports 10; `mdata.obsm['rna']` and `mdata.obsm['atac']` hold the presence masks (their `&` sums to
7). Reading prints a FutureWarning about `.update()` pulling obs columns, nothing about cells.

What the rule reads. A fact rule over every existing `*.h5mu` named in a command or in the `*.py`
script it runs (h5py only: the obs index of each modality at `/mod/<name>/obs/_index`). It applies
when two or more modalities do not hold the same cells. It then looks at the code: the count of the
MuData's own cells (`m.n_obs`, `m.obs_names`, `m.shape`, `m.obs.shape`, `len(m)`, or the same on
`read_h5mu(...)` directly, where `m` is a name assigned from `read_h5mu` / `MuData(`) without any
sign of an intersection (`&`, `intersection`, `isin`, `obsm`, `intersect_obs`, a modality indexed by
name) fails. Per-modality counts (`m['rna'].n_obs`, `m.mod['rna'].obs_names`) are not a MuData
count and pass: they may be wrong for the task, which the rule cannot know.

Declaration `mudata_cells` (union | intersection): which cell set the count is meant to describe.
`intersection` with a union count in the code fails; `union` passes (the union is meant).
"""

from __future__ import annotations

import re
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.sc_common import (
    _index_names,
    code_text,
    data_files,
    require_h5py,
    strip_comments,
)

H5MU_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.h5mu(?![\w])")
# A name assigned from a read/MuData call. Anchored at the start of a word and bounded in every
# part, so a 40 kB line of letters costs a scan, not a quadratic search (E-7).
MAKES_MUDATA = re.compile(
    r"(?<!\w)(\w{1,64})\s{0,40}=\s{0,40}[\w.]{0,80}\b(?:read_h5mu|MuData|read)\s{0,40}\("
)
INTERSECTS = re.compile(
    r"intersect|&|\bobsm\b|\.isin\(|\.mod\s*\[|\bmod\s*\[[^\]]{1,100}\]"
    r"|\[\s*['\"][\w-]+['\"]\s*\]\s*\.\w*obs"
)
CHAINED = re.compile(r"read_h5mu\s*\([^)]{0,300}\)\s*\.(?:n_obs|obs_names|shape|obs)\b[^\n;]*")
MAX_HANDLES = 64  # a script has a few MuData names; each one costs a scan of the text


def modality_cells(path: Path) -> dict[str, list[str]] | None:
    """Modality name -> obs names of an .h5mu, or None when it is not one."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if "mod" not in f:
                return None
            return {name: _index_names(f["mod"][name]["obs"]) for name in f["mod"]}
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        return None


def counts_mudata_cells(text: str) -> str | None:
    """The first expression that counts the cells of the MuData itself, or None."""
    chained = CHAINED.search(text)
    if chained:
        return chained.group(0)[:80]
    for name in list(dict.fromkeys(MAKES_MUDATA.findall(text)))[:MAX_HANDLES]:
        own = re.search(
            rf"\b{re.escape(name)}\.(?:n_obs|obs_names|shape|obs)\b[^\n;]*"
            rf"|\blen\(\s*{re.escape(name)}(?:\.obs(?:_names)?)?\s*\)",
            text,
        )
        if own:
            return own.group(0)[:80]
    return None


@checker("sc_mudata_obs_union")
def sc_mudata_obs_union(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    counted = counts_mudata_cells(text)
    if counted is None:
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    handled = bool(INTERSECTS.search(text))
    for path in data_files(call, match, ctx, H5MU_NAME, (".h5mu",)):
        mods = modality_cells(path)
        if not mods or len(mods) < 2:
            continue
        sets = {name: set(cells) for name, cells in mods.items()}
        both = set.intersection(*sets.values())
        union = set.union(*sets.values())
        if len(both) == len(union):
            continue
        sizes = ", ".join(f"{name}: {len(cells)}" for name, cells in sets.items())
        if declared == "union":
            continue
        if declared == "intersection" and not handled:
            return CheckResult(
                "fail",
                "declared_intersection_but_union_count",
                f"you declared {key}=intersection but `{counted}` counts the cells of any "
                f"modality of {path.name} ({len(union)}), not the {len(both)} in all of them.",
                expected="intersection",
                got="union",
                fix=(
                    "intersect the modalities' obs_names, or use mdata.obsm['rna'] & "
                    "mdata.obsm['atac']",
                ),
            )
        if declared is None and not handled:
            return CheckResult(
                "fail",
                "mudata_n_obs_is_the_union",
                f"`{counted}` counts the cells of the MuData {path.name}, which is the union of "
                f"its modalities ({sizes}): {len(union)} cells, of which {len(both)} are in every "
                "modality.",
                fix=(
                    "intersect: len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names)), "
                    f"and declare --declare {key}=intersection",
                    f"or declare {key}=union if the cells of any modality are meant",
                ),
            )
    return passed()
