"""Draw the README bar chart from trial records.

    python tools/make_figures.py [runs/stage1] [docs/figures/silent_errors] [--sets FILE]
                                 [--stratum primary]
    python tools/make_figures.py runs/stage1 STEM --panel per-model  # the per-model chart
    python tools/make_figures.py RUN STEM --sets FILE --stage2      # the stage-2 layout
                                 [--theme starry-night] [--modes light,dark]

Needs matplotlib (the `figures` extra: `pip install matplotlib` in any scratch environment, with
`src` on PYTHONPATH). Writes `<stem>.svg` and `<stem>.png`.
Left: the open-weight models pooled, share of faulted tasks that end in a silent wrong answer per
condition. Right: the same per model for four conditions. A bar is the mean over families of each
family's share (every family counts once, as in the registered analysis); a whisker is the 95%
percentile interval of that mean when the families are resampled with replacement (the family is
the unit of inference: its trials, seeds and models are not independent draws).
The families are the five of stage 1 unless --sets names a frozen sets file, in which case they are
its stratum (primary, q1, q2, q3, bite or gated).
Only aggregates are drawn; no trial log leaves the machine.

Colours (`--theme starry-night`, the default): the palettes VanGogh (Vincent van Gogh, The Starry
Night, 1889) and Picabia (Francis Picabia, The Spring, 1912) of MoMAColors by Blake R. Mills (MIT),
after works in the Museum of Modern Art, New York; the dark mode borrows two tints from the
Hokusai3 and Hiroshige palettes of MetBrewer (CC0), after works in the Metropolitan Museum of Art.
Warm browns are the conditions without a specific message; the blue ramp, pale to midnight, is the
layer, and the hatched bar is the same block reached through a typed tool next to an unchecked bash.
`--modes light,dark` writes `<stem>.svg/.png` and `<stem>-dark.svg/.png` (the SVG has a
transparent page; the PNG has the page colour), for a `<picture>` element in the README.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from collections.abc import Sequence

from biowarden.bench.stage2 import STAGE1_FAMILIES, bootstrap_ci, fam_no

STRATA = {  # a name for a set of families in the sets file: the lists it is the union of
    "primary": ("primary",),
    "q1": ("q1",),
    "q2": ("q2",),
    "q3": ("q3",),
    "bite": ("primary", "q1"),
    "gated": ("primary", "q1", "q2", "q3"),
}
CELL = ("bare", "implicit", "forced")
SKIP = ("infra_error", "skipped_budget")
THEMES = {
    "starry-night": {
        "light": {
            "page": "#ffffff", "ink": "#0C1F4B", "muted": "#666879", "grid": "#e2d8d6",
            "axis": "#a5a6ae",
            "colour": {
                "bare": "#9f7064", "careful": "#e6bcac", "block-gateway": "#0C1F4B",
                "block-generic": "#8ebacd", "warn-nonote": "#246893", "block-nonote": "#0C1F4B",
            },
            "edge": {"careful": "#9f7064", "block-generic": "#246893"},
        },
        "dark": {
            "page": "#0d1117", "ink": "#e2d8d6", "muted": "#a5a6ae", "grid": "#3d3d47",
            "axis": "#666879",
            "colour": {
                "bare": "#9f7064", "careful": "#c99582", "block-gateway": "#aadce0",
                "block-generic": "#246893", "warn-nonote": "#5a97c1", "block-nonote": "#aadce0",
            },
            "edge": {},
        },
    },
    "classic": {  # the colours of the first release
        "light": {
            "page": "#ffffff", "ink": "#1f2933", "muted": "#6b7683", "grid": "#e6e9ed",
            "axis": "#9aa1a9",
            "colour": {
                "bare": "#8a93a0", "careful": "#bcc3cc", "block-gateway": "#d9a441",
                "block-generic": "#9cc6e8", "warn-nonote": "#4a90d9", "block-nonote": "#1f4e79",
            },
            "edge": {},
        },
    },
}  # fmt: skip
PAGE, INK, MUTED, GRID, AXIS = "#ffffff", "#1f2933", "#6b7683", "#e6e9ed", "#9aa1a9"
COLOUR: dict[str, str] = {}
EDGE: dict[str, str] = {}
HATCH = {"block-gateway": "////"}  # lines in the page colour: the block, through a typed tool


def use_theme(name: str, mode: str) -> None:
    """Make the colours of `name` in `mode` (light or dark) the ones the drawing functions use."""
    global PAGE, INK, MUTED, GRID, AXIS
    if name not in THEMES or mode not in THEMES[name]:
        raise ValueError(f"theme {name!r} has no {mode!r} mode (themes: {', '.join(THEMES)})")
    t = THEMES[name][mode]
    PAGE, INK, MUTED, GRID, AXIS = t["page"], t["ink"], t["muted"], t["grid"], t["axis"]
    COLOUR.clear()
    COLOUR.update(t["colour"])
    EDGE.clear()
    EDGE.update(t["edge"])


use_theme("starry-night", "light")
LABEL = {
    "bare": "no layer",
    "careful": "prompt:\n\u201ccheck\nformats\u201d",
    "block-gateway": "block as\ntyped tool\n+ bash",
    "block-generic": "generic\nmessage",
    "warn-nonote": "specific\nwarning",
    "block-nonote": "specific\nblock",
}
BASELINES = ["bare", "careful"]
LAYER = ["block-generic", "block-gateway", "warn-nonote", "block-nonote"]
PER_MODEL = ["bare", "careful", "block-gateway", "block-nonote"]
NAMES = {
    "deepseek/deepseek-v4.1-flash": "DeepSeek\nV4.1 Flash",
    "z-ai/glm-5.3-flash": "GLM 5.3\nFlash",
    "qwen/qwen3.8-flash": "Qwen 3.8\nFlash",
    "xiaomi/mimo-v2.6-flash": "MiMo V2.6\nFlash",
    "openai/gpt-oss-120b": "gpt-oss\n120B",
    "claude-haiku-5-5": "Claude\nHaiku 5.5",
}


def families_from_sets(path: str, stratum: str) -> list[str]:
    """The families of `stratum` in a frozen sets file (format 2), in family order."""
    with open(path) as handle:
        sets = json.load(handle)
    if stratum not in STRATA:
        raise ValueError(f"stratum {stratum!r} is not one of {', '.join(STRATA)}")
    return sorted({f for key in STRATA[stratum] for f in sets[key]}, key=fam_no)


def load(run: str, families: Sequence[str] = STAGE1_FAMILIES) -> list[dict]:
    wanted = set(families)
    rows = []
    for path in sorted(glob.glob(f"{run}/trials/*/*/score.json")):
        with open(path) as handle:
            rows.append(json.load(handle))
    return [
        r
        for r in rows
        if r["family"] in wanted
        and r["twin"] == "faulted"
        and r["outcome"] not in SKIP
        and r.get("rep", 0) == 0  # not the A/A replicate
        and (r.get("contract"), r.get("tier"), r.get("route")) == CELL
        and r.get("harness", "minimal") == "minimal"
    ]


def share(rows: list[dict], draws: int = 10_000) -> tuple[float, float, float, int]:
    """(percent, low, high, trials): the mean over families of the share of silent wrong answers,
    with the percentile interval of the mean over families resampled with replacement."""
    count: dict[str, int] = defaultdict(int)
    silent: dict[str, int] = defaultdict(int)
    for r in rows:
        count[r["family"]] += 1
        silent[r["family"]] += r["outcome"] == "silent_wrong"
    values = [silent[f] / count[f] for f in sorted(count, key=fam_no)]
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values, draws)
    return 100 * mean, 100 * min(low, mean), 100 * max(high, mean), len(rows)


def swatch(arm: str):
    """A legend patch that matches the bar, edge included."""
    from matplotlib.patches import Rectangle

    edge = EDGE.get(arm)
    hatch = HATCH.get(arm)
    return Rectangle(
        (0, 0),
        1,
        1,
        facecolor=COLOUR[arm],
        edgecolor=PAGE if hatch else edge or COLOUR[arm],  # a hatch is drawn in the edge colour
        linewidth=0.8 if edge else 0,
        hatch=hatch,
    )


def bar(ax, x: float, width: float, rows: list[dict], arm: str, value_size: float = 0) -> float:
    value, low, high, _ = share(rows)
    edge = EDGE.get(arm)
    hatch = HATCH.get(arm)
    ax.bar(
        x, value, width, color=COLOUR[arm], edgecolor=PAGE if hatch else edge or COLOUR[arm],
        linewidth=0.8 if edge else 0, zorder=3, hatch=hatch,
    )  # fmt: skip
    for colour, weight, cap in ((PAGE, 2.6, 3.4), (INK, 0.9, 2.5)):  # a halo keeps the whisker
        ax.errorbar(  # visible on a bar of the ink's own colour
            x, value, yerr=[[value - low], [high - value]], color=colour, capsize=cap,
            lw=weight, capthick=weight, zorder=4,
        )  # fmt: skip
    if value_size:
        ax.text(
            x,
            high + 2.2,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=value_size,
            weight="bold",
            zorder=5,
        )
    return value


def style(ax) -> None:
    ax.set_facecolor("none")  # the page (or the PNG's page colour) shows through
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelcolor=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def save(fig, stem: str) -> None:
    """Opaque files in the page colour of the mode: if a theme swap ever fails, the figure stays
    readable (a transparent one with dark ink would vanish on a dark page)."""
    fig.savefig(f"{stem}.svg", facecolor=PAGE)
    fig.savefig(f"{stem}.png", dpi=160, facecolor=PAGE)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "sans-serif", "svg.fonttype": "path", "hatch.linewidth": 1.4}
    )
    return plt


def _split(rows: list[dict]):
    pooled: dict[str, list[dict]] = defaultdict(list)
    per: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        arm = r.get("arm", "bare")
        per[(r["model"], arm)].append(r)
        if "/" in r["model"]:
            pooled[arm].append(r)
    return pooled, per


def draw_pooled(rows: list[dict], stem: str) -> None:
    """The open-weight models pooled: two baselines, then the layer in four forms."""
    plt = _plt()
    pooled, _ = _split(rows)
    fig, ax = plt.subplots(figsize=(8.3, 3.7))
    xs = dict(zip(BASELINES + LAYER, [0, 1.2, 2.7, 3.7, 4.7, 5.7], strict=True))
    values = {a: bar(ax, xs[a], 0.72, pooled[a], a, 10.5) for a in xs if pooled.get(a)}
    ax.set_xticks([xs[a] for a in xs])
    ax.set_xticklabels([LABEL[a] for a in xs], fontsize=9.5)
    for names, text in ((BASELINES, "Baselines"), (LAYER, "Checking layer")):
        x0, x1 = xs[names[0]] - 0.42, xs[names[-1]] + 0.42
        ax.plot([x0, x1], [107, 107], color=MUTED, lw=1, clip_on=False)
        ax.text((x0 + x1) / 2, 109, text, ha="center", va="bottom", color=MUTED, fontsize=10)
    top, end = values["bare"], values["block-nonote"]
    x_arrow = xs["block-nonote"] + 0.6
    ax.annotate(
        "",
        xy=(x_arrow, end + 1),
        xytext=(x_arrow, top),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.1},
        annotation_clip=False,
    )
    ax.plot([xs["careful"] + 0.45, x_arrow], [top, top], ls=(0, (3, 3)), color=MUTED, lw=0.9)
    ax.text(
        x_arrow + 0.08, (top + end) / 2, f"\u2212{round(top) - round(end)}\npts",
        ha="left", va="center", color=INK, fontsize=10, weight="bold",
    )  # fmt: skip
    ax.set_xlim(-0.6, 6.9)
    ax.set_ylabel("faulted tasks ending in a silent wrong answer (%)", fontsize=9.5, color=INK)
    style(ax)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.22)
    save(fig, stem)


def draw_per_model(rows: list[dict], stem: str) -> None:
    """Every model separately: no layer, prompt note, typed tool and specific block."""
    plt = _plt()
    _, per = _split(rows)
    fig, ax = plt.subplots(figsize=(8.3, 3.7))
    models = [m for m in NAMES if any((m, a) in per for a in PER_MODEL)]
    width = 0.2
    offsets = [(k - (len(PER_MODEL) - 1) / 2) * (width + 0.015) for k in range(len(PER_MODEL))]
    for i, model in enumerate(models):
        for off, arm in zip(offsets, PER_MODEL, strict=True):
            if per.get((model, arm)):
                bar(ax, i + off, width, per[(model, arm)], arm, 8.5 if arm == "block-nonote" else 0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES[m] for m in models], fontsize=9.5)
    ax.legend(
        handles=[swatch(a) for a in PER_MODEL],
        labels=[LABEL[a].replace("\n", " ") for a in PER_MODEL],
        fontsize=9.5,
        frameon=False,
        labelcolor=INK,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        handlelength=1,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    ax.set_ylabel("faulted tasks ending in a silent wrong answer (%)", fontsize=9.5, color=INK)
    style(ax)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.87, bottom=0.2)
    save(fig, stem)


STAGE2_MODELS = ["claude-haiku-5-5", "deepseek/deepseek-v4.1-flash", "openai/gpt-oss-120b"]
STAGE2_LABEL = {
    "bare": "no layer",
    "warn-nonote": "specific warning",
    "block-nonote": "specific block",
}


def draw_stage2(run: str, sets_path: str, stem: str) -> int:
    """Stage 2, per model: no layer, specific warning and specific block on the families where a
    trap bites the agents and a rule exists (P); no layer and block on every gated family (the
    warning arm ran on the bite families only). A bar is the mean over families of each family's
    share of faulted trials ending in a silent wrong answer. No title or footer: the README
    caption carries them."""
    primary = families_from_sets(sets_path, "primary")
    gated = families_from_sets(sets_path, "gated")
    panels = [
        (f"Traps that bite and have a rule ({len(primary)} families)", primary, list(STAGE2_LABEL)),
        (f"All {len(gated)} gated families", gated, ["bare", "block-nonote"]),
    ]
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5), gridspec_kw={"wspace": 0.1})
    for ax, (name, families, arms) in zip(axes, panels, strict=True):
        rows = load(run, families)
        width = 0.24 if len(arms) == 3 else 0.32
        for i, model in enumerate(STAGE2_MODELS):
            for k, arm in enumerate(arms):
                mine = [r for r in rows if r["model"] == model and r.get("arm", "bare") == arm]
                if mine:
                    bar(ax, i + (k - (len(arms) - 1) / 2) * (width + 0.02), width, mine, arm, 9.5)
        ax.set_xticks(range(len(STAGE2_MODELS)))
        ax.set_xticklabels([NAMES[m] for m in STAGE2_MODELS], fontsize=9.5)
        ax.set_title(name, loc="left", fontsize=10.5, color=INK, pad=24)
        style(ax)
        ax.legend(
            handles=[swatch(a) for a in arms],
            labels=[STAGE2_LABEL[a] for a in arms],
            fontsize=9,
            frameon=False,
            labelcolor=INK,
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=len(arms),
            handlelength=1,
            columnspacing=1.1,
            handletextpad=0.4,
        )
    axes[1].set_yticklabels([])
    axes[0].set_ylabel("silent wrong answers on faulted tasks (%)", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.8, bottom=0.14)
    save(fig, stem)
    print(f"wrote {stem}.svg/.png: stage 2, {len(primary)} and {len(gated)} families")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run", nargs="?", default="runs/stage1", help="a run directory")
    parser.add_argument("stem", nargs="?", default="docs/figures/silent_errors")
    parser.add_argument(
        "--sets", help="a frozen sets file: draw its families, not the stage-1 five"
    )
    parser.add_argument("--stratum", choices=list(STRATA), help="which families (default: primary)")
    parser.add_argument("--stage2", action="store_true", help="the stage-2 layout (needs --sets)")
    parser.add_argument(
        "--panel",
        default="pooled",
        choices=["pooled", "per-model"],
        help="stage 1: the pooled baselines chart or the per-model chart",
    )
    parser.add_argument("--theme", default="starry-night", choices=list(THEMES))
    parser.add_argument(
        "--modes", default="light,dark", help="light, dark or both (comma separated)"
    )
    args = parser.parse_args(argv)
    if args.stratum and not args.sets:
        parser.error("--stratum needs --sets")
    if args.stage2 and not args.sets:
        parser.error("--stage2 needs --sets")
    modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in modes if m not in THEMES[args.theme]]
    if bad:
        parser.error(f"theme {args.theme} has no mode {', '.join(bad)}")
    families = (
        families_from_sets(args.sets, args.stratum or "primary") if args.sets else STAGE1_FAMILIES
    )
    rows = [] if args.stage2 else load(args.run, families)
    if not args.stage2 and not rows:
        print(f"no faulted trials of {len(families)} families in {args.run}", file=sys.stderr)
        return 2
    for mode in modes:
        use_theme(args.theme, mode)
        stem = args.stem if mode == "light" else f"{args.stem}-{mode}"
        if args.stage2:
            draw_stage2(args.run, args.sets, stem)
        else:
            (draw_pooled if args.panel == "pooled" else draw_per_model)(rows, stem)
            print(
                f"wrote {stem}.svg/.png from {len(rows)} faulted trials of {len(families)} families"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
