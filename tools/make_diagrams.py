"""Draw the two diagrams of the README: where the checking layer sits, and how one call is decided.

    python tools/make_diagrams.py [--outdir docs/figures] [--modes light,dark]

Needs matplotlib (the `figures` extra) and the package (`pip install -e .`). Writes `layers`,
`decision` and, for the dark mode, `layers-dark` and `decision-dark`, each as `.svg` and `.png`.

The colours are those of the charts (`THEMES` in tools/make_figures.py), used by role: warm browns
are what lies outside the layer (the caller, the real tools, the data; in the dark mode they are
outlines only, as the dim "no layer" bars of the charts), the blue ramp is the layer (front ends:
midnight, the engine and what it decides: steel, rules and ledger: pale steel). Text is drawn as
paths, in the font of the charts, so the files look the same everywhere. Every number in the text
is read from the code (the rule count, the exit status of a block, the retry cap, the evaluation
limit), so the diagrams cannot drift from it, and a text that does not fit its box stops the
drawing instead of spilling over.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH = 597.6  # points: 8.3 in, as the charts; a README shows it at 830 px
PAD = 4.5  # the room a text keeps to the edge of its box
MARGIN = 3  # the room a text on a line keeps to the edge of the figure


def _figures():
    """tools/make_figures.py as a module: its theme is the one source of the colours."""
    spec = importlib.util.spec_from_file_location("make_figures", HERE / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_figures", module)
    spec.loader.exec_module(module)
    return module


def facts() -> dict[str, str]:
    """The numbers the diagrams state, from the code they describe."""
    from biowarden.engine.decide import EVALUATE_SECONDS, Engine
    from biowarden.engine.shim_client import EXIT_BLOCKED
    from biowarden.engine.spec import packaged_specs

    cap = inspect.signature(Engine.__init__).parameters["retry_cap"].default
    ordinal = {1: "2nd", 2: "3rd", 3: "4th"}.get(cap, f"{cap + 1}th")
    return {
        "rules": str(sum(len(spec.rules) for spec in packaged_specs())),
        "exit": str(EXIT_BLOCKED),
        "third": ordinal,
        "seconds": f"{EVALUATE_SECONDS:.0f}",
    }


@dataclass(frozen=True)
class Role:
    fill: str
    edge: str


def luminance(colour: str) -> float:
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def roles(mf) -> dict[str, Role]:
    """The colours of the boxes by role, all from the theme in use (no colour of its own)."""
    c = mf.COLOUR
    dark = luminance(mf.PAGE) < 0.2
    return {
        # outside the layer: a pale brown fill, or in the dark mode an outline, so that the layer
        # is what stands out (the charts do the same with their dim "no layer" bars)
        "world": Role(mf.PAGE if dark else c["careful"], c["bare"]),
        "entry": Role(c["block-nonote"], c["block-nonote"]),  # the front ends
        "engine": Role(c["warn-nonote"], c["warn-nonote"]),  # the engine and what it decides
        "store": Role(c["block-generic"], mf.EDGE.get("block-generic", c["warn-nonote"])),
        "note": Role(mf.PAGE, c["warn-nonote"]),
    }


def text_on(fill: str, ink: str, page: str) -> str:
    """Whichever of the two text colours of the theme reads better on this fill."""
    return max((ink, page), key=lambda t: contrast(fill, t))


class Canvas:
    """A drawing board in points, y downwards, in the colours of one mode of the theme."""

    def __init__(self, mf, height: float, label_pt: float = 10.6, sub_pt: float = 9.4):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "path"})
        self.mf, self.height, self.label_pt, self.sub_pt = mf, height, label_pt, sub_pt
        self.fig = plt.figure(figsize=(WIDTH / 72, height / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        c = mf.COLOUR
        self.page, self.ink = mf.PAGE, mf.INK
        self.roles = roles(mf)
        self.unchecked = c["bare"]  # the colour of "no layer"
        self.layer = c["block-nonote"]  # the colour of "the layer"
        self.fits: list[tuple[object, float, float, float, float, str]] = []
        self.free: list[tuple[object, str]] = []

    def text_colour(self, fill: str) -> str:
        return text_on(fill, self.ink, self.page)

    def box(self, x, y, w, h, label, sub="", role="world", number=None, label_pt=None, sub_pt=None):
        from matplotlib.patches import Circle, FancyBboxPatch

        r = self.roles[role]
        label_pt, sub_pt = label_pt or self.label_pt, sub_pt or self.sub_pt
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=5",
                fc=r.fill,
                ec=r.edge,
                lw=1.2,
                zorder=2,
            )  # fmt: skip
        )
        ink = self.text_colour(r.fill)
        n_label, n_sub = label.count("\n") + 1, (sub.count("\n") + 1 if sub else 0)
        lh1, lh2 = label_pt * 1.22, sub_pt * 1.28
        top = y + (h - (n_label * lh1 + n_sub * lh2 + (3 if sub else 0))) / 2
        t1 = self.ax.text(
            x + w / 2, top, label, ha="center", va="top", color=ink, fontsize=label_pt,
            weight="bold", linespacing=1.22, zorder=3,
        )  # fmt: skip
        self.fits.append((t1, x, y, w, h, label))
        if sub:
            t2 = self.ax.text(
                x + w / 2, top + n_label * lh1 + 3, sub, ha="center", va="top", color=ink,
                fontsize=sub_pt, linespacing=1.28, zorder=3,
            )  # fmt: skip
            self.fits.append((t2, x, y, w, h, sub))
        if number is not None:  # a tab on the top edge, so it never covers the text
            cx = x + 14
            self.ax.add_patch(Circle((cx, y), 8, fc=self.page, ec=r.edge, lw=1.3, zorder=4))
            self.ax.text(
                cx, y + 0.3, str(number), ha="center", va="center", color=self.ink,
                fontsize=8.8, weight="bold", zorder=5,
            )  # fmt: skip

    def arrow(self, points, colour=None, dashed=False, lw=1.5, head=7.5):
        """A polyline whose last segment ends in a solid arrowhead."""
        colour = colour or self.ink
        (x0, y0), (x1, y1) = points[-2], points[-1]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        k = min(head, length * 0.8) / length
        base = (x1 - (x1 - x0) * k, y1 - (y1 - y0) * k)
        body = [*points[:-1], base]
        self.ax.plot(
            [p[0] for p in body], [p[1] for p in body], color=colour, lw=lw, zorder=1,
            ls=(0, (3.2, 2.4)) if dashed else "-", solid_capstyle="butt", dash_capstyle="butt",
            solid_joinstyle="miter",
        )  # fmt: skip
        self.ax.annotate(
            "", xy=(x1, y1), xytext=base, zorder=1,
            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": lw, "shrinkA": 0,
                        "shrinkB": 0, "mutation_scale": 9.5},
        )  # fmt: skip

    def line(self, points, colour=None, lw=1.5):
        self.ax.plot(
            [p[0] for p in points], [p[1] for p in points], color=colour or self.ink, lw=lw,
            zorder=1, solid_capstyle="butt", solid_joinstyle="miter",
        )  # fmt: skip

    def note(self, x, y, text, colour=None, ha="center", size=8.8, weight="normal", mask=True):
        """Text on a line: a page-coloured mask keeps the line from running through it."""
        t = self.ax.text(
            x, y, text, ha=ha, va="center", color=colour or self.ink, fontsize=size, weight=weight,
            linespacing=1.28, zorder=6,
            bbox={"fc": self.page, "ec": "none", "pad": 1.8} if mask else None,
        )  # fmt: skip
        self.free.append((t, text))

    def frame(self, x, y, w, h, role=None, colour=None, dashed=True):
        from matplotlib.patches import FancyBboxPatch

        r = self.roles[role] if role else None
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=7",
                fc=r.fill if r else "none",
                ec=r.edge if r else colour,
                lw=1.2,
                ls=(0, (5, 3)) if dashed else "-",
                zorder=0,
            )  # fmt: skip
        )

    def spilled(self) -> list[str]:
        """The texts that do not fit their box, or the figure (with a little room to spare)."""
        self.fig.canvas.draw()
        renderer, inv = self.fig.canvas.get_renderer(), self.ax.transData.inverted()
        bad = []
        for text, x, y, w, h, words in self.fits:
            (x0, y0), (x1, y1) = inv.transform(text.get_window_extent(renderer).get_points())
            if x0 < x + PAD or x1 > x + w - PAD or min(y0, y1) < y + 1 or max(y0, y1) > y + h - 1:
                bad.append(words.replace("\n", " / "))
        for text, words in self.free:
            (x0, _), (x1, _) = inv.transform(text.get_window_extent(renderer).get_points())
            if x0 < MARGIN or x1 > WIDTH - MARGIN:
                bad.append(words.replace("\n", " / "))
        return bad

    def save(self, stem: str):
        import matplotlib.pyplot as plt

        bad = self.spilled()
        if bad:
            raise ValueError(f"{stem}: text does not fit its box: {bad}")
        self.fig.savefig(f"{stem}.svg", facecolor=self.page, metadata={"Date": None})
        self.fig.savefig(f"{stem}.png", dpi=160, facecolor=self.page)
        plt.close(self.fig)


def draw_layers(mf, stem: str) -> None:
    """Where the layer sits: three front ends, one engine, the real tools behind it."""
    f = facts()
    cv = Canvas(mf, 246, label_pt=10.6, sub_pt=9.3)
    ax_, aw = 3, 92  # the caller
    bx, bw = 117, 172  # the front ends
    cx, cw = 315, 124  # the engine and its stores
    dx, dw = 468, 126  # the real world
    entry_h, gap, top = 54, 10, 27
    ys = [top + i * (entry_h + gap) for i in range(3)]
    bottom = ys[-1] + entry_h
    small, eng_y, eng_h = 48, top + 60, 62

    cv.frame(bx - 8, 11, cx + cw - bx + 16, bottom - 11 + 10, colour=cv.layer)
    cv.note(bx + 6, 11, "Checking layer (this repository)", cv.layer, ha="left", weight="bold")
    caller = "Claude Code,\nyour own loop,\na terminal"
    cv.box(ax_, ys[0], aw, bottom - ys[0], "LLM agent\nor user", caller, label_pt=11.4, sub_pt=9.8)
    cv.box(
        bx,
        ys[0],
        bw,
        entry_h,
        "biowarden",
        "PATH shims check calls\nto watched tools by name",
        "entry",
    )
    cv.box(
        bx, ys[1], bw, entry_h, "Whole-command pre-check",
        "checks each command in\na bash string; sandbox only", "entry",
    )  # fmt: skip
    cv.box(bx, ys[2], bw, entry_h, "biowarden-mcp", "only calls made\nthrough run_tool", "entry")
    cv.box(
        cx, ys[0], cw, small, f"{f['rules']} rules", "specs, checkers,\npromotion table", "store"
    )
    cv.box(cx, eng_y, cw, eng_h, "Engine", "match, check, decide;\nnever rewrites a call", "engine")
    cv.box(cx, top + 134, cw, small, "Ledger", "hash chain, optional", "store")
    cv.box(
        dx,
        ys[0],
        dw,
        small,
        "Data files",
        "BAM, VCF, FASTQ, h5ad;\nread by T1, fact rules",
        "world",
    )
    tools = (
        "samtools, bcftools,\nbedtools, awk,\npython3, perl, ...\n\n"
        "run only after the\nengine allows or warns"
    )
    cv.box(dx, eng_y, dw, bottom - eng_y, "Real tools", tools, "world")

    mid = [y + entry_h / 2 for y in ys]
    bus = bx + bw + 13
    ports = [eng_y + eng_h * k for k in (0.26, 0.5, 0.74)]
    for i in range(3):
        cv.arrow([(ax_ + aw, mid[i]), (bx, mid[i])])
    cv.arrow([(bx + bw, mid[0]), (bus, mid[0]), (bus, ports[0]), (cx, ports[0])])
    cv.arrow([(bx + bw, mid[1]), (cx, ports[1])])
    cv.arrow([(bx + bw, mid[2]), (bus, mid[2]), (bus, ports[2]), (cx, ports[2])])
    cv.arrow([(cx + cw / 2, eng_y), (cx + cw / 2, ys[0] + small)])
    cv.arrow([(cx + cw / 2, eng_y + eng_h), (cx + cw / 2, top + 134)])
    cv.arrow([(cx + cw, eng_y + eng_h / 2), (dx, eng_y + eng_h / 2)])
    side, into = cx + cw + 19, ys[0] + small - 11
    cv.arrow([(cx + cw, ports[0]), (side, ports[0]), (side, into), (dx, into)])
    cv.arrow([(dx + dw / 2, eng_y), (dx + dw / 2, ys[0] + small)])

    low = bottom + 24
    left, right = ax_ + aw / 2, dx + dw / 2
    cv.arrow([(left, bottom), (left, low), (right, low), (right, bottom)], cv.unchecked, True, 1.9)
    cv.note(
        (left + right) / 2, low,
        "Not checked: with biowarden alone, an absolute path or a reset PATH;\n"
        "with biowarden-mcp alone, the agent's own Bash",
        weight="bold",
    )  # fmt: skip
    cv.save(stem)


def draw_decision(mf, stem: str) -> None:
    """How one call is decided: match, check, take the most severe effect, then act."""
    f = facts()
    height = 292
    cv = Canvas(mf, height, label_pt=10.4, sub_pt=9.0)
    n, gap = 5, 12
    w = (WIDTH - 6 - gap * (n - 1)) / n
    xs = [3 + i * (w + gap) for i in range(n)]
    h1, y1 = 92, 14
    steps = [
        ("Call arrives", "a process, a bash\nstring or a\nrun_tool call", "world"),
        ("Front end\nbuilds a Call", "program, arguments,\ndirectory,\ndeclarations", "entry"),
        (
            "Match rules",
            "call rules: tool,\nsubcommand and\noptions; fact rules:\nfiles the call names",
            "engine",
        ),
        (
            "Declaration\nand checker",
            "T0: arguments only;\nT1: reads files, may\ndry-run the tool",
            "engine",
        ),
        (
            "Most severe\neffect",
            "verdict × tier ×\nmode × promotion\ntable; unknown\nonly warns",
            "engine",
        ),
    ]
    for i, (label, sub, role) in enumerate(steps):
        cv.box(xs[i], y1, w, h1, label, sub, role, number=i + 1)
    for i in range(n - 1):
        cv.arrow([(xs[i] + w, y1 + h1 / 2), (xs[i + 1], y1 + h1 / 2)])

    w2 = (WIDTH - 6 - gap * 3) / 4
    x2 = [3 + i * (w2 + gap) for i in range(4)]
    y2, h2 = y1 + h1 + 28, 80
    bus = y1 + h1 + 14
    cv.box(
        x2[0],
        y2,
        w2,
        h2,
        "allow / warn",
        "the real tool runs;\nwarn also prints\na warning",
        "world",
    )
    block = (
        f"the call does not run:\nexit {f['exit']} and the FIX\n"
        "text (biowarden-mcp: an\nerror with the FIX)"
    )
    cv.box(x2[1], y2, w2, h2, "block", block, "engine")
    cv.box(
        x2[2], y2, w2, h2, "STOP",
        f"the same call blocked\na {f['third']} time: do not\nretry; it stays\nblocked", "engine",
    )  # fmt: skip
    cv.box(
        x2[3], y2, w2, h2, "Ledger",
        "every evaluation gets\na hash-chained record,\nif --ledger is set", "store",
    )  # fmt: skip
    start = xs[4] + w / 2
    cv.line([(start, y1 + h1), (start, bus), (x2[0] + w2 / 2, bus)])
    cv.arrow([(x2[0] + w2 / 2, bus), (x2[0] + w2 / 2, y2)])
    cv.arrow([(x2[1] + w2 / 2, bus), (x2[1] + w2 / 2, y2)])
    cv.arrow([(x2[3] + w2 / 2, bus), (x2[3] + w2 / 2, y2)], lw=1.1)
    cv.arrow([(x2[1] + w2, y2 + h2 / 2), (x2[2], y2 + h2 / 2)])

    y3 = y2 + h2 + 12
    band = height - y3 - 3
    cv.frame(3, y3, WIDTH - 6, band, role="note", dashed=False)
    note = (
        "Faults. A crashing checker blocks the call (warn mode only warns).\n"
        "A fault of the layer itself (a path it cannot stat, a dry run or the "
        f"{f['seconds']} s limit timing out) is unknown:\n"
        "the call runs, the agent is told nothing, and the ledger has the cause.\n"
        "A shim that cannot reach the engine lets the call run, with a note on stderr."
    )
    t = cv.ax.text(
        WIDTH / 2, y3 + band / 2 + 0.5, note, ha="center", va="center", color=cv.ink,
        fontsize=9.3, linespacing=1.3, zorder=3,
    )  # fmt: skip
    cv.free.append((t, note))
    cv.save(stem)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", default="docs/figures")
    parser.add_argument(
        "--modes", default="light,dark", help="light, dark or both (comma separated)"
    )
    args = parser.parse_args(argv)
    mf = _figures()
    for mode in [m for m in args.modes.split(",") if m]:
        mf.use_theme("starry-night", mode)
        suffix = "" if mode == "light" else f"-{mode}"
        draw_layers(mf, f"{args.outdir}/layers{suffix}")
        draw_decision(mf, f"{args.outdir}/decision{suffix}")
        print(f"wrote {args.outdir}/layers{suffix} and decision{suffix} (.svg, .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
