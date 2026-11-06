"""Shared readers for the single-cell rules (sc pack): .h5ad through h5py, code text, file finding.

Nothing here imports scanpy or anndata (the engine runs on the host). An .h5ad is read with h5py
(plus numpy, which h5py needs) and a reader that cannot make sense of a file returns None, so the
rules answer `unknown` instead of crashing.

What is read, following the anndata on-disk format 0.1.0 / 0.2.0 (anndata 0.8 to 0.11):
  X, raw/X, layers/<name>   a dense dataset, or a group with data / indices / indptr (csr / csc);
                            only a bounded sample of the non-zero values is looked at
  obs, var                  the index (attribute `_index` names the dataset), categorical columns as
                            groups with categories and codes, string columns as datasets
  raw/var, uns              the index of raw and the key names of uns

State of a matrix, decided from the sample of values (`classify`):
  counts            every value is an integer (the sample holds at least one value above 0)
  scaled            a negative value (z-scores, centred data)
  log_normalized    non-integer, no negative value, maximum <= LOG_MAX (log1p of counts per 10,000
                    peaks near 9.3, log1p of CPM near 13.8, log2 of CPM near 20)
  normalized        non-integer, no negative value, maximum >= NORM_MIN (CP10K, CPM, TPM)
  ambiguous         a maximum between the two; `uns['log1p']` settles it as log_normalized
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, is_regular_file, passed

LOG_MAX = 20.0
NORM_MIN = 50.0
SAMPLE_ROWS = 300  # rows (cells) of X looked at
SAMPLE_VALUES = 200_000  # stored values read at most
MAX_NAMES = 20_000
MAX_SCRIPTS = 5
H5AD_NAME = r"(?<![\w./~+-])[\w./~+-]+\.h5ad(?![\w])"
PY_SCRIPT_NAME = r"(?<![\w./~+-])[\w./~+-]+\.py(?![\w.])"
PYTHONS = ["python-sc", "python3", "python"]

_NAME_H5AD = re.compile(H5AD_NAME)
_NAME_SCRIPT = re.compile(PY_SCRIPT_NAME)


# -- code text -----------------------------------------------------------------------------------


def script_texts(call: Call, ctx: Ctx, tokens: tuple[str, ...] | None = None) -> list[str]:
    """Text of the `*.py` files named in the call (at most MAX_SCRIPTS, 3000 lines each)."""
    found: list[str] = []
    seen: set[Path] = set()
    for token in tokens if tokens is not None else (*call.argv, *call.context):
        for hit in _NAME_SCRIPT.finditer(token):
            path = ctx.resolve(hit.group(0))
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                found.append("\n".join(ctx.lines(path, limit=3000)))
            except (OSError, EOFError, zlib.error):
                continue
            if len(found) >= MAX_SCRIPTS:
                return found
    return found


def code_text(call: Call, ctx: Ctx) -> str:
    """Everything the command says: tool, arguments, here-document / earlier words, scripts."""
    parts = [call.tool, *call.argv, *call.context, *script_texts(call, ctx)]
    return "\n".join(parts)


def strip_comments(text: str) -> str:
    """Drop `# ...` comments (a '#' inside a quoted string is rare in these scripts)."""
    return re.sub(r"(?m)(?<![\"'\w])#.*$", "", text)


def find_files(
    text: str, cwd: Path, pattern: re.Pattern[str], root: Path | None = None
) -> list[Path]:
    """Existing regular files named in `text` that match `pattern`, in order of appearance.

    With `root` (the trial directory), a name that leads outside it is not a file.
    """
    out: list[Path] = []
    for hit in pattern.finditer(text):
        path = Path(hit.group(0))
        path = path if path.is_absolute() else cwd.absolute() / path
        if is_regular_file(path, root) and path not in out:
            out.append(path)
    return out


def data_files(
    call: Call, match: Match, ctx: Ctx, pattern: re.Pattern[str], suffixes: tuple[str, ...]
) -> list[Path]:
    """Data files of the command: matched by the engine, or named inside the code/scripts."""
    files = [p for p in match.files if p.name.lower().endswith(suffixes)]
    for path in find_files(code_text(call, ctx), ctx.cwd, pattern, ctx.root):
        if path not in files:
            files.append(path)
    return files


def h5ad_files(call: Call, match: Match, ctx: Ctx) -> list[Path]:
    return data_files(call, match, ctx, _NAME_H5AD, (".h5ad",))


# -- matrix state --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    state: str  # counts | scaled | log_normalized | normalized | ambiguous | empty
    minimum: float = 0.0
    maximum: float = 0.0
    values: int = 0
    log1p_marked: bool = False
    dtype: str = ""  # numpy name of the stored values, for example float32 or int16
    sparse: bool = False

    def says(self) -> str:
        """The state in words with the observed range."""
        if self.state == "counts":
            return f"integer counts (max {self.maximum:g})"
        if self.state == "scaled":
            return f"scaled values with negatives (range {self.minimum:.2f} to {self.maximum:.2f})"
        if self.state == "log_normalized":
            return f"log-scale values (non-integer, max {self.maximum:.2f})"
        if self.state == "normalized":
            return f"normalised, not logged values (non-integer, max {self.maximum:.0f})"
        if self.state == "ambiguous":
            return f"non-integer values with max {self.maximum:.1f}"
        return "no values"


def classify(values: Any, log1p_marked: bool = False) -> Sample:
    """State of a sample of values (a numpy array of the stored / non-zero entries)."""
    import numpy as np

    v = np.asarray(values, dtype="float64").ravel()
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.any(v != 0):
        return Sample("empty", 0.0, 0.0, int(v.size), log1p_marked)
    lo, hi = float(v.min()), float(v.max())
    if lo < 0:
        state = "scaled"
    elif bool(np.all(np.abs(v - np.round(v)) < 1e-6)):
        state = "counts"
    elif hi <= LOG_MAX:
        state = "log_normalized"
    elif hi >= NORM_MIN:
        state = "normalized"
    else:
        state = "log_normalized" if log1p_marked else "ambiguous"
    return Sample(state, lo, hi, int(v.size), log1p_marked)


def _decode(array: Any) -> list[str]:
    out: list[str] = []
    for item in array:
        out.append(item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item))
    return out


def _read_matrix(node: Any, log1p_marked: bool) -> Sample | None:
    """Sample a dense dataset or a csr / csc group."""
    import h5py
    import numpy as np

    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            return None
        rows = max(1, min(node.shape[0], SAMPLE_ROWS, SAMPLE_VALUES // max(node.shape[1], 1)))
        return replace(classify(node[:rows], log1p_marked), dtype=node.dtype.name)
    if not isinstance(node, h5py.Group) or not {"data", "indptr"} <= set(node):
        return None
    kind = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", ""))
    kind = kind.decode() if isinstance(kind, bytes) else str(kind)
    data = node["data"]
    limit = min(data.shape[0], SAMPLE_VALUES)
    if kind.startswith("csr"):
        indptr = np.asarray(node["indptr"][: SAMPLE_ROWS + 1])
        limit = min(limit, int(indptr[-1]))
    return replace(classify(data[:limit], log1p_marked), dtype=data.dtype.name, sparse=True)


def _shape(node: Any) -> tuple[int, int] | None:
    import h5py

    if isinstance(node, h5py.Dataset) and node.ndim == 2:
        return int(node.shape[0]), int(node.shape[1])
    if isinstance(node, h5py.Group):
        shape = node.attrs.get("shape", node.attrs.get("h5sparse_shape"))
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return None


def _index_names(group: Any) -> list[str]:
    import h5py

    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    if key in group and isinstance(group[key], h5py.Dataset):
        return _decode(group[key][:MAX_NAMES])
    for fallback in ("index", "_index", "obs_names", "var_names"):
        if fallback in group and isinstance(group[fallback], h5py.Dataset):
            return _decode(group[fallback][:MAX_NAMES])
    return []


def _columns(group: Any) -> list[str]:
    order = group.attrs.get("column-order")
    if order is not None:
        return _decode(list(order)) if len(order) else []
    return [k for k in group if k != group.attrs.get("_index", "_index")]


def column_values(group: Any, name: str) -> list[str] | None:
    """Values of one obs/var column as strings (categoricals decoded), or None."""
    import h5py

    if name not in group:
        return None
    node = group[name]
    if isinstance(node, h5py.Group) and {"categories", "codes"} <= set(node):
        cats = _decode(node["categories"][:])
        codes = node["codes"][:MAX_NAMES]
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]
    if isinstance(node, h5py.Dataset) and node.ndim == 1:
        return _decode(node[:MAX_NAMES].tolist())
    return None


@dataclass(frozen=True)
class H5ad:
    path: Path
    shape: tuple[int, int] | None
    x: Sample
    raw: Sample | None = None
    raw_shape: tuple[int, int] | None = None
    raw_var_names: list[str] = field(default_factory=list)
    layers: dict[str, Sample] = field(default_factory=dict)
    uns_keys: frozenset[str] = frozenset()
    obs_columns: list[str] = field(default_factory=list)
    var_columns: list[str] = field(default_factory=list)
    obs_names: list[str] = field(default_factory=list)
    var_names: list[str] = field(default_factory=list)

    @property
    def log1p_marked(self) -> bool:
        return "log1p" in self.uns_keys

    def counts_source(self) -> str | None:
        """Where integer counts are kept besides X: a layer name or 'raw', else None."""
        for name, sample in self.layers.items():
            if sample.state == "counts":
                return f"layers['{name}']"
        if self.raw is not None and self.raw.state == "counts":
            return "raw.X"
        return None


def read_h5ad(path: Path) -> H5ad | None:
    """Profile of an .h5ad, or None when it is not one (or not the anndata layout)."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            if "X" not in f or "obs" not in f or "var" not in f:
                return None
            uns_keys = frozenset(f["uns"].keys()) if "uns" in f else frozenset()
            marked = "log1p" in uns_keys
            x = _read_matrix(f["X"], marked)
            if x is None:
                return None
            raw = raw_shape = None
            raw_names: list[str] = []
            if "raw" in f and "X" in f["raw"]:
                raw = _read_matrix(f["raw"]["X"], marked)
                raw_shape = _shape(f["raw"]["X"])
                raw_names = _index_names(f["raw"]["var"]) if "var" in f["raw"] else []
            layers: dict[str, Sample] = {}
            if "layers" in f:
                for name in f["layers"]:
                    sample = _read_matrix(f["layers"][name], marked)
                    if sample is not None:
                        layers[name] = sample
            return H5ad(
                path=path,
                shape=_shape(f["X"]),
                x=x,
                raw=raw,
                raw_shape=raw_shape,
                raw_var_names=raw_names,
                layers=layers,
                uns_keys=uns_keys,
                obs_columns=_columns(f["obs"]),
                var_columns=_columns(f["var"]),
                obs_names=_index_names(f["obs"]),
                var_names=_index_names(f["var"]),
            )
    except (OSError, KeyError, ValueError, TypeError, AttributeError, ImportError):
        return None


def obs_column(path: Path, name: str) -> list[str] | None:
    """Values of obs[name] of an .h5ad as strings (None when missing or unreadable)."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return column_values(f["obs"], name)
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        return None


def var_column(path: Path, name: str) -> list[str] | None:
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return column_values(f["var"], name)
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        return None


# -- verdict helpers -----------------------------------------------------------------------------


def unknown(code: str, message: str, fix: tuple[str, ...] = ()) -> CheckResult:
    return CheckResult("unknown", code, message, fix=fix)


def require_h5py() -> CheckResult | None:
    """None when h5py can be imported, else the `unknown` verdict a rule returns (it only warns)."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        return unknown(
            "h5py_missing",
            "the .h5ad files of this command cannot be inspected because h5py is not installed.",
            fix=("pip install h5py in the environment that runs the checking layer",),
        )
    return None


def no_data(text: str) -> CheckResult:
    return passed(note=text)


def quoted(text: str) -> str:
    return text.replace("\n", " ")[:120]
