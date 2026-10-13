"""Generate every data figure of the manuscript and the supplement from the
summary and raw files.

Style: Elsevier double-column layout (cas-dc). A single-column figure is
drawn at the column width of 238 pt (3.30 in) and a full-width figure at the
text width of 494.5 pt (6.84 in); both are placed at 100 %, so the font sizes
below are the printed sizes: 9 pt for labels and legends, 8 pt for tick
labels. The supplement figure is drawn at its own text width (5.40 in).
Times New Roman throughout (text and mathtext), vector PDF with embedded
TrueType fonts, Okabe-Ito colours with a distinct marker and line style per
series, legends outside the plotting area, no title inside the image (panel
letters only).

Each figure passes check_layout() before it is saved: no two texts overlap, no
plotted line or marker touches a text, every text lies inside the canvas, and
every text is Times New Roman of at least 8 pt. A failed check stops the
script.

Usage:  python scripts/make_figures.py --profile full [--png DIR]
Output: figures/*.pdf (full profile) or experiments/results/<profile>/figures/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

ROOT = Path(__file__).resolve().parent.parent
COL = 3.30                         # 238 pt, column width of cas-dc
WIDE = 6.84                        # 494.5 pt, text width of cas-dc
SUPP = 5.40                        # 390 pt, text width of the supplement
LABEL, TICK = 9, 8
MIN_PT = 8

def _serif_font() -> str:
    """Times New Roman where it is installed (the article figures); otherwise a
    metric-compatible substitute, so that the scripts also run on Linux."""
    from matplotlib import font_manager
    for name in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "serif"


FONT = _serif_font()
if FONT != "Times New Roman":
    print(f"note: Times New Roman is not installed, figures use {FONT}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT],
    "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
    "font.size": LABEL, "axes.labelsize": LABEL, "axes.titlesize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.handlelength": 2.0,
    "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.4,
    "axes.axisbelow": True, "axes.titlepad": 4,
})

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000", "grey": "#7F7F7F"}
ATTACK_STYLE = {
    "lr": ("LR", OI["blue"], "o", "-"),
    "svm": ("SVM", OI["sky"], "s", "--"),
    "rf": ("RF", OI["green"], "^", "-."),
    "gb": ("GB", OI["orange"], "v", ":"),
    "mlp": ("MLP", OI["vermillion"], "D", "-"),
    "lr_xor": ("Product LR", OI["purple"], "P", "--"),
}
ATTACKS = list(ATTACK_STYLE)
CODE_COLOURS = [OI["blue"], OI["sky"], OI["green"], OI["orange"], OI["vermillion"], OI["purple"]]
CODE_MARKERS = ["o", "s", "^", "v", "D", "P"]
CODE_LINES = ["-", "--", "-.", "-", "--", "-."]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(rows, keys):
    for r in rows:
        for k in keys:
            if r.get(k, "") != "":
                r[k] = float(r[k])
    return rows


def code_label(code: str) -> str:
    return code.replace(" x", r" $\times$")


def log_budget_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    decades = [10 ** e for e in range(0, 8) if lo <= 10 ** e <= hi]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.xaxis.set_minor_locator(NullLocator())


def chance_line(ax, y=50):
    ax.axhline(y, color="0.35", linewidth=0.7, linestyle=(0, (1, 1)), zorder=1)


# --------------------------------------------------------------------------
# objective layout check
# --------------------------------------------------------------------------
def _texts(fig):
    """Every text that is drawn. Tick labels whose tick lies outside the view
    interval keep a Text object but are not drawn, so they are left out."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 * abs(hi - lo) <= loc <= hi + 1e-9 * abs(hi - lo)):
                    hidden.update({id(tick.label1), id(tick.label2)})
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        out.append(t)
    return out


def _line_points(ax, r):
    """Display coordinates of every plotted line and marker of an axes (and
    of a legend inside the figure), densely sampled, with the marker radius in
    pixels. Points outside the axes are clipped away, as on the page."""
    pts = []
    bbox = ax.bbox
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(np.column_stack(ln.get_data()).astype(float))
        rad = ln.get_markersize() * fig_dpi(ax) / 72 / 2 if ln.get_marker() not in (None, "", "None") else 0
        pts.append((_densify(xy, ln.get_linestyle() not in ("None", "", " ")), rad, True))
    for coll in ax.collections:
        if isinstance(coll, LineCollection) and coll.get_visible():
            for seg in coll.get_segments():
                if len(seg):
                    xy = ax.transData.transform(np.asarray(seg, dtype=float))
                    pts.append((_densify(xy, True), 0, True))
    for sp in ax.spines.values():
        if sp.get_visible():
            path = sp.get_path()
            xy = sp.get_transform().transform(path.vertices)
            pts.append((_densify(xy, True), 0, False))
    clipped = []
    for xy, rad, clip in pts:
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if clip:
            keep = ((xy[:, 0] >= bbox.x0 - 1) & (xy[:, 0] <= bbox.x1 + 1)
                    & (xy[:, 1] >= bbox.y0 - 1) & (xy[:, 1] <= bbox.y1 + 1))
            xy = xy[keep]
        if len(xy):
            clipped.append((xy, rad))
    return clipped


def fig_dpi(ax):
    return ax.figure.dpi


def _densify(xy, connect):
    if not connect or len(xy) < 2:
        return xy
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = int(max(2, np.hypot(*(b - a)) // 0.5))
        out.append(np.linspace(a, b, n)[1:])
    return np.vstack(out)


def check_layout(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _texts(fig)
    boxes = [t.get_window_extent(r) for t in texts]
    problems = []
    W, H = fig.bbox.width, fig.bbox.height
    for t, b in zip(texts, boxes):
        label = repr(t.get_text()[:30])
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
            problems.append(f"text outside canvas: {label}")
        if t.get_fontsize() < MIN_PT - 1e-9:
            problems.append(f"text below {MIN_PT} pt: {label} ({t.get_fontsize()} pt)")
        if FONT.split()[0] not in t.get_fontname():
            problems.append(f"font is not {FONT}: {label} ({t.get_fontname()})")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = boxes[i], boxes[j]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.5 and oy > 0.5:
                problems.append(f"texts overlap: {texts[i].get_text()[:25]!r} and "
                                f"{texts[j].get_text()[:25]!r}")
    pad = 1.5 * fig.dpi / 72                      # 1.5 pt clearance around every text
    holders = list(fig.axes)
    for ax in holders:
        for xy, rad in _line_points(ax, r):
            for t, b in zip(texts, boxes):
                hit = ((xy[:, 0] > b.x0 - pad - rad) & (xy[:, 0] < b.x1 + pad + rad)
                       & (xy[:, 1] > b.y0 - pad - rad) & (xy[:, 1] < b.y1 + pad + rad))
                if hit.any():
                    problems.append(f"line or marker touches text {t.get_text()[:30]!r}")
    for leg in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        lb = leg.get_window_extent(r)
        for ax in fig.axes:
            for xy, rad in _line_points(ax, r):
                hit = ((xy[:, 0] > lb.x0 - rad) & (xy[:, 0] < lb.x1 + rad)
                       & (xy[:, 1] > lb.y0 - rad) & (xy[:, 1] < lb.y1 + rad))
                if hit.any():
                    problems.append("a plotted line runs through a legend")
    problems = sorted(set(problems))
    if problems and FONT in ("Times New Roman", "Liberation Serif"):
        raise SystemExit(f"layout check failed for {name}:\n  " + "\n  ".join(problems))
    w_in, h_in = fig.get_size_inches()
    if problems:
        # a substitute font of other width (for example DejaVu Serif on a bare
        # Linux system) can push a label past the edge; the figure is still
        # written as a preview, the article figures use Times New Roman
        return (f"{name}: written with {FONT}, layout differs from the article figure "
                f"(install Times New Roman or Liberation Serif): " + "; ".join(problems))
    return f"{name}: {w_in:.2f} x {h_in:.2f} in, {len(texts)} texts, layout check passed"


def save(fig, out, name, png_dir):
    msg = check_layout(fig, name)
    fig.savefig(out / f"{name}.pdf")
    if png_dir:
        fig.savefig(png_dir / f"{name}.png", dpi=200)
    plt.close(fig)
    print(msg)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_attack_curves(summ, out, png):
    """Full width: three panels side by side, one legend row above."""
    rows = fnum(read_csv(summ / "attack_summary.csv"),
                ["noisiness", "budget", "mean", "ci_lo", "ci_hi"])
    rows = [r for r in rows if r["dist"] == "uniform" and r["noisiness"] == 0.0]
    panels = [("apuf", "(a) Arbiter PUF"), ("xor2", "(b) 2-XOR arbiter PUF"),
              ("mask_xor2", "(c) Masked 2-XOR arbiter PUF")]
    budgets = sorted({r["budget"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.45), sharey=True)
    for ax, (c, title) in zip(axes, panels):
        for a, (lab, col, mk, ls) in ATTACK_STYLE.items():
            rr = sorted((r for r in rows if r["construction"] == c and r["attack"] == a),
                        key=lambda r: r["budget"])
            x = [r["budget"] for r in rr]
            ax.fill_between(x, [100 * r["ci_lo"] for r in rr], [100 * r["ci_hi"] for r in rr],
                            color=col, alpha=0.18, linewidth=0)
            ax.plot(x, [100 * r["mean"] for r in rr], color=col, marker=mk, linestyle=ls,
                    label=lab, markerfacecolor="white", markeredgewidth=0.8)
        if c == "mask_xor2":
            rr = sorted((r for r in rows if r["construction"] == c
                         and r["attack"] == "oracle_lr_xor"), key=lambda r: r["budget"])
            x = [r["budget"] for r in rr]
            ax.fill_between(x, [100 * r["ci_lo"] for r in rr], [100 * r["ci_hi"] for r in rr],
                            color="0.3", alpha=0.18, linewidth=0)
            ax.plot(x, [100 * r["mean"] for r in rr], color="black", marker="X",
                    linestyle="-", label="Product LR, key known", markersize=4.5)
        chance_line(ax)
        log_budget_axis(ax, budgets[0] * 0.7, budgets[-1] * 1.4)
        ax.set_ylim(40, 102)
        ax.set_title(title, loc="left")
        ax.set_xlabel("CRPs available to the attacker")
    axes[0].set_ylabel("Test accuracy (%)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.17, wspace=0.07)
    save(fig, out, "fig_attack_curves", png)


def fig_stability(summ, out, png):
    """Column width: model recovery for k = 1 to 4 and the masked response."""
    path = summ / "ext_xor_stability_summary.csv"
    if not path.exists():
        return
    rows = fnum(read_csv(path), ["k", "budget", "agreement_mean", "agreement_min",
                                 "agreement_max", "no_key_mean", "no_key_min", "no_key_max"])
    ks = sorted({int(r["k"]) for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    style = {1: (OI["blue"], "o", "-"), 2: (OI["orange"], "s", "--"),
             3: (OI["green"], "^", "-."), 4: (OI["vermillion"], "D", ":")}
    shift = {1: 0.93, 2: 0.977, 3: 1.023, 4: 1.07}       # small horizontal offsets
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    for k in ks:
        rr = sorted((r for r in rows if int(r["k"]) == k), key=lambda r: r["budget"])
        col, mk, ls = style[k]
        x = [r["budget"] * shift[k] for r in rr]
        y = [100 * r["agreement_mean"] for r in rr]
        ax.errorbar(x, y, yerr=[[100 * (r["agreement_mean"] - r["agreement_min"]) for r in rr],
                                [100 * (r["agreement_max"] - r["agreement_mean"]) for r in rr]],
                    color=col, marker=mk, linestyle=ls, capsize=2, elinewidth=0.7,
                    markerfacecolor="white", markeredgewidth=0.8, label=f"$k={k}$")
    nk = {b: [r for r in rows if r["budget"] == b] for b in budgets}
    lo = [100 * min(r["no_key_min"] for r in nk[b]) for b in budgets]
    hi = [100 * max(r["no_key_max"] for r in nk[b]) for b in budgets]
    mean = [100 * float(np.mean([r["no_key_mean"] for r in nk[b]])) for b in budgets]
    ax.fill_between(budgets, lo, hi, color="0.55", alpha=0.25, linewidth=0)
    ax.plot(budgets, mean, color="0.25", marker="x", linestyle="-", markersize=4.5,
            label="masked, no key")
    chance_line(ax)
    log_budget_axis(ax, budgets[0] * 0.72, budgets[-1] * 1.35)
    ax.set_xticks(budgets)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}".replace(",", " ")))
    ax.set_ylim(40, 102)
    ax.set_xlabel("Distinct challenges, each queried 11 times")
    ax.set_ylabel("Agreement or accuracy (%)")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, t in enumerate(l) if t.startswith("$k")] + \
            [i for i, t in enumerate(l) if not t.startswith("$k")]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
              bbox_to_anchor=(0.45, 1.0), ncol=5, columnspacing=0.8, handlelength=1.8)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.16)
    save(fig, out, "fig_stability", png)


def _selected(rows, units):
    out = []
    for u in units:
        cand = [r for r in rows if r["instance"] == u]
        best = max(cand, key=lambda r: (r["val_acc"], -ATTACKS.index(r["attack"])))
        out.append(best["test_acc"])
    return out


def fig_fpga(res, out, png):
    """Column width: horizontal dot plot on a broken accuracy axis. One row per
    condition, two series (measured arbiter chains, 2-XOR pairs). The left
    segment shows the region around chance, the right segment the region of
    high accuracy, so that the spread over instances (about one percentage
    point) stays visible. Every instance is a thin tick and the filled marker
    is the mean over instances."""
    path = res / "fpga_raw.csv"
    if not path.exists():
        return
    rows = fnum(read_csv(path), ["budget", "val_acc", "test_acc"])
    b = max(r["budget"] for r in rows)
    rows = [r for r in rows if r["budget"] == b]
    variants = [("measured", "Measured"), ("mask_only", "Mask only"), ("masked", "Masked"),
                ("key_known", "Masked, key known")]
    series = [("arbiter", "Arbiter chains", OI["blue"], "o", +0.16),
              ("xor2", "2-XOR pairs", OI["vermillion"], "s", -0.16)]
    data = {}
    for kind, *_ in series:
        insts = sorted({r["instance"] for r in rows if r["kind"] == kind})
        for var, _ in variants:
            rr = [r for r in rows if r["kind"] == kind and r["variant"] == var]
            data[(kind, var)] = 100 * np.array([r["test_acc"] for r in rr] if var == "key_known"
                                               else _selected(rr, insts))
    low = np.concatenate([v for v in data.values() if v.mean() < 75])
    high = np.concatenate([v for v in data.values() if v.mean() >= 75])
    lo_lim = (np.floor(low.min() * 2) / 2 - 0.5, np.ceil(low.max() * 2) / 2 + 0.5)
    hi_lim = (np.floor(high.min()) - 0.5, min(100.0, np.ceil(high.max()) + 0.5))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(COL, 2.45), sharey=True,
        gridspec_kw={"width_ratios": [lo_lim[1] - lo_lim[0], hi_lim[1] - hi_lim[0]]})
    n_rows = len(variants)
    handles = []
    for kind, lab, col, mk, off in series:
        n_inst = len(data[(kind, "measured")])
        for i, (var, _) in enumerate(variants):
            vals = data[(kind, var)]
            ax = ax1 if vals.mean() < 75 else ax2
            y = n_rows - 1 - i + off
            ax.plot([vals.min(), vals.max()], [y, y], color=col, linewidth=0.8, zorder=2)
            ax.plot(vals, np.full(len(vals), y), linestyle="none", marker="|", markersize=5.5,
                    markeredgewidth=0.7, color=col, zorder=3)
            ax.plot([vals.mean()], [y], linestyle="none", marker=mk, markersize=4.4, color=col,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        handles.append(Line2D([], [], color=col, marker=mk, markersize=4.4, linestyle="none",
                              markeredgecolor="white", markeredgewidth=0.5,
                              label=f"{lab} ($n={n_inst}$)"))
    ax1.axvline(50, color="0.35", linewidth=0.7, linestyle=(0, (1, 1)), zorder=1)
    ax1.set_xlim(*lo_lim)
    ax2.set_xlim(*hi_lim)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels([v[1] for v in reversed(variants)])
    ax1.set_ylim(-0.6, n_rows - 0.4)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(right=False)
    ax2.tick_params(left=False)
    # break marks: short diagonals at the gap, top and bottom
    d = 0.015
    for ax, x0 in ((ax1, 1.0), (ax2, 0.0)):
        w = d * (ax2.get_position().width / ax.get_position().width) * 1.6
        for y0 in (0.0, 1.0):
            ax.plot([x0 - w, x0 + w], [y0 - 1.6 * d, y0 + 1.6 * d], transform=ax.transAxes,
                    color="black", linewidth=0.6, clip_on=False)
    fig.supxlabel(f"Test accuracy at {int(b):,} CRPs (%)".replace(",", " "), fontsize=LABEL,
                  x=0.65, y=0.035)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=2,
               handlelength=1.0)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.86, bottom=0.175, wspace=0.08)
    save(fig, out, "fig_fpga", png)


def fig_ecc(res, summ, out, png):
    """Full width: two panels side by side, legends below each panel."""
    es = fnum(read_csv(summ / "ecc_summary.csv"),
              ["ber", "failure_weight", "weight_lo", "weight_hi"])
    an = fnum(read_csv(res / "ecc_analytic.csv"), ["ber", "failure"])
    codes = list(dict.fromkeys(r["code"] for r in es))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.1), sharey=True)
    ax = axes[0]
    for code, col, mk, ls in zip(codes, CODE_COLOURS, CODE_MARKERS, CODE_LINES):
        a = [r for r in an if r["code"] == code and r["failure"] > 1e-9]
        ax.plot([100 * r["ber"] for r in a], [r["failure"] for r in a], color=col,
                linestyle=ls, linewidth=0.9)
        p = [r for r in es if r["code"] == code and r["model"] == "iid" and r["failure_weight"] > 0]
        ax.errorbar([100 * r["ber"] for r in p], [r["failure_weight"] for r in p],
                    yerr=[[r["failure_weight"] - r["weight_lo"] for r in p],
                          [r["weight_hi"] - r["failure_weight"] for r in p]],
                    fmt=mk, color=col, markerfacecolor="white", markeredgewidth=0.8,
                    capsize=1.5, elinewidth=0.6, linestyle=ls, label=code_label(code))
    ax.set_title("(a) Independent errors: exact (lines), simulated (markers)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.2)
    ax = axes[1]
    pick = [c for c in codes if c.startswith("BCH(127,36") or c.startswith("RS(48")] or codes[:2]
    mstyle = {"iid": ("independent", "-", "o"), "heterogeneous": ("heterogeneous", "--", "s"),
              "common_mode": ("common mode", ":", "^")}
    for code in pick:
        col = CODE_COLOURS[codes.index(code)]
        for model, (mlab, ls, mk) in mstyle.items():
            p = sorted((r for r in es if r["code"] == code and r["model"] == model
                        and r["failure_weight"] > 0), key=lambda r: r["ber"])
            ax.plot([100 * r["ber"] for r in p], [r["failure_weight"] for r in p], color=col,
                    linestyle=ls, marker=mk, markerfacecolor="white", markeredgewidth=0.8)
    h = [Line2D([], [], color=CODE_COLOURS[codes.index(c)], linewidth=1.6, label=code_label(c))
         for c in pick]
    h += [Line2D([], [], linestyle="none", label=" ")] * (3 - len(pick))
    h += [Line2D([], [], color="0.25", linestyle=ls, marker=mk, markerfacecolor="white",
                 markeredgewidth=0.8, label=mlab) for mlab, ls, mk in mstyle.values()]
    ax.set_title("(b) Sensitivity to the error model", loc="left")
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handlelength=2.2)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3)
        ax.set_xlim(0, 21)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlabel("Mean bit error rate of the weak-PUF reading (%)")
    axes[0].set_ylabel("Key reconstruction failure probability")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.925, bottom=0.345, wspace=0.06)
    save(fig, out, "fig_ecc", png)


def fig_entropy(summ, out, png):
    """Column width: lower bound against the bias, legend above."""
    path = summ / "ext_entropy_summary.csv"
    if not path.exists():
        return
    rows = fnum(read_csv(path), ["bias_p", "entropy_lower", "secret_bits"])
    codes = list(dict.fromkeys(r["code"] for r in rows))
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    for code, col, mk, ls in zip(codes, CODE_COLOURS, CODE_MARKERS, CODE_LINES):
        rr = sorted((r for r in rows if r["code"] == code), key=lambda r: r["bias_p"])
        ax.plot([r["bias_p"] for r in rr], [r["entropy_lower"] for r in rr], color=col,
                marker=mk, linestyle=ls, markerfacecolor="white", markeredgewidth=0.8,
                label=code_label(code))
    ax.axhline(128, color="0.25", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    ax.set_xlim(0.29, 0.51)
    ax.set_ylim(0, 170)
    ax.set_yticks([0, 32, 64, 96, 128, 160])
    ax.set_xlabel("Probability $p_1$ that a weak-PUF bit is one")
    ax.set_ylabel("Lower bound on key entropy (bits)")
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="0.25", linewidth=0.8, linestyle=(0, (4, 2))))
    l.append("128-bit level")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.8,
               columnspacing=0.8)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.72, bottom=0.13)
    save(fig, out, "fig_entropy", png)


def fig_flip(res, out, png):
    """Supplement (single column, 5.40 in)."""
    q = json.loads((res / "quality_summary.json").read_text(encoding="utf-8"))
    series = [("apuf", "Arbiter", OI["blue"], "-"), ("xor2", "2-XOR arbiter", OI["orange"], "--"),
              ("mask_xor2", "Masked 2-XOR", OI["green"], "-.")]
    fig, ax = plt.subplots(figsize=(SUPP, 2.7))
    for c, lab, col, ls in series:
        p = np.array(q[c]["flip_rate"]["profile"])
        sd = np.array(q[c]["flip_rate"]["profile_sd_devices"])
        x = np.arange(1, len(p) + 1)
        ax.fill_between(x, p - sd, p + sd, color=col, alpha=0.18, linewidth=0)
        ax.plot(x, p, color=col, linestyle=ls, label=lab)
    chance_line(ax, 0.5)
    ax.set_xlim(1, len(p))
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_xlabel("Position of the flipped challenge bit")
    ax.set_ylabel("Response flip probability")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.89, bottom=0.15)
    save(fig, out, "fig_flip_profile", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--png", default=None, help="also write PNG previews to this folder")
    args = ap.parse_args()
    res = ROOT / "experiments" / "results" / args.profile
    summ = res / "summary"
    out = ROOT / "figures" if args.profile == "full" else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    png = Path(args.png) if args.png else None
    if png:
        png.mkdir(parents=True, exist_ok=True)
    fig_attack_curves(summ, out, png)
    fig_stability(summ, out, png)
    fig_fpga(res, out, png)
    fig_ecc(res, summ, out, png)
    fig_entropy(summ, out, png)
    fig_flip(res, out, png)
    print("figures written to", out)


if __name__ == "__main__":
    main()
