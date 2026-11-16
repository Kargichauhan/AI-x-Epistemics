"""Figures for the sprint report, drawn from the analysed run data (no numbers typed by hand).

Conventions: thin marks, 2px lines, dots with a white ring, hairline solid grid,
text in ink colours, a legend whenever two or more series share a plot. Series
colours are slots 1-2 of the dataviz reference palette (validated there); the
'no guess' baseline is a neutral grey.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze  # noqa: E402

INK, INK2, MUTED = "#0e1217", "#49505b", "#7b828d"
GRID, AXIS, SURFACE = "#e1e5ea", "#c2c8d0", "#ffffff"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a909a"
HEAT = LinearSegmentedColormap.from_list("blue", ["#eaf1fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

COND = {
    "none": {"con": "disconfirming_evidence", "none": "observation_only", "sup": "new_evidence"},
    "unl": {"con": "unlabelled_summary_plus_disconfirming", "none": "unlabelled_summary", "sup": "unlabelled_summary_plus_evidence"},
    "att": {"con": "attributed_summary_plus_disconfirming", "none": "attributed_summary", "sup": "attributed_summary_plus_evidence"},
}
SERIES = [("none", "No guess (filler line)", GREY), ("unl", "Guess, unlabelled: “It was suggested that…”", BLUE),
          ("att", "Guess, labelled: “The assistant suggested that…”", ORANGE)]
COLS = [("con", "Contrary\nevidence"), ("none", "No new\nevidence"), ("sup", "Supporting\nevidence")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": False,
})


def _clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def cond_stats(data, cond, scale="pp"):
    """Mean over cases of per-case cell means, with a bootstrap CI over cases."""
    key = "mean_pp" if scale == "pp" else "mean_logodds"
    vals = [c[key] for (cid, cn), c in data["cells"].items() if cn == cond and c["n_valid"]]
    s = analyze.summarise(vals)
    return s["mean"], s["ci95_low"], s["ci95_high"]


def contrast(data, name):
    return next(c for c in data["contrasts"] if c["contrast"] == name and "pp" in c)


def fig_matrix(data, path):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    vals = [[cond_stats(data, COND[s][c])[0] for c, _ in COLS] for s, _, _ in SERIES]
    ax.imshow(vals, cmap=HEAT, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#ffffff" if v > 55 else INK)
    ax.set_xticks(range(3), [c.replace("\n", " ") for _, c in COLS])
    ax.set_yticks(range(3), ["No guess", "Guess, unlabelled", "Guess, labelled\n“The assistant”"])
    ax.tick_params(length=0, labelsize=8.5, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(2)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(2)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.set_ticks_position("top")
    fig.savefig(path)
    plt.close(fig)


def fig_interaction(data, path, title=None, scale="pp"):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    xs = range(3)
    for key, label, color in SERIES:
        st = [cond_stats(data, COND[key][c], scale) for c, _ in COLS]
        ax.plot(xs, [m for m, _, _ in st], color=color, linewidth=2, solid_capstyle="round", zorder=3, label=label)
        ax.errorbar(xs, [m for m, _, _ in st], yerr=[[m - lo for m, lo, _ in st], [hi - m for m, _, hi in st]],
                    fmt="none", ecolor=color, elinewidth=1, capsize=0, alpha=0.55, zorder=2)
        ax.scatter(xs, [m for m, _, _ in st], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    eff = {"con": "perseverance_unlabelled", "none": "carryover_unlabelled", "sup": "interpretation_effect_with_evidence"}
    sk = "pp" if scale == "pp" else "logodds"
    labels = []
    for c, name in COLS:
        s = contrast(data, eff[c])[sk]
        d = 1 if scale == "pp" else 2
        labels.append(f"{name}\nguess {s['mean']:+.{d}f}{' pts' if scale == 'pp' else ''}\n[{s['ci95_low']:+.{d}f}, {s['ci95_high']:+.{d}f}]")
    ax.set_xticks(list(xs), labels)
    ax.set_xlim(-0.35, 2.35)
    if scale == "pp":
        ax.set_ylim(0, 100)
        ax.set_ylabel("Stated probability (0–100)")
    else:
        ax.set_ylabel("Stated probability (log-odds)")
    _clean(ax)
    ax.legend(loc="upper left", fontsize=8, handlelength=1.6)
    if title:
        ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)


FOREST_ROWS = [
    ("Effect of the carried-over guess", [
        ("carryover_unlabelled", "No evidence, unlabelled"), ("carryover_attributed", "No evidence, labelled"),
        ("interpretation_effect_with_evidence", "Supporting evidence, unlabelled"),
        ("perseverance_unlabelled", "Contrary evidence, unlabelled"), ("perseverance_attributed", "Contrary evidence, labelled")]),
    ("Effect of the “The assistant” label (unlabelled − labelled)", [
        ("attribution_reduction", "No evidence"), ("attribution_reduction_with_evidence", "Supporting evidence"),
        ("attribution_reduction_with_disconfirming", "Contrary evidence")]),
    ("Evidence responsiveness", [
        ("evidence_effect", "Supporting vs no evidence (gate)"), ("disconfirming_effect", "Contrary vs no evidence (gate)"),
        ("evidence_effect_under_attribution", "Supporting, with labelled guess"),
        ("evidence_preservation", "Change in evidence effect under label")]),
]


def fig_forest(data, path, drift=None):
    rows = []
    for g, items in FOREST_ROWS:
        rows.append(("group", g, None))
        for key, label in items:
            rows.append(("row", label, contrast(data, key)["logodds"]))
    if drift:
        rows.append(("group", "Drift between runs (same prompts, later run)", None))
        for d in drift:
            rows.append(("row", d["condition"].replace("_", " "), d["logodds"]))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * n + 0.6))
    y = n
    for kind, label, st in rows:
        y -= 1
        if kind == "group":
            ax.text(-4.35, y, label, fontweight="bold", fontsize=8.5, color=INK, va="center")
            continue
        ax.text(-4.25, y, label, fontsize=8, color=INK2, va="center")
        ax.plot([st["ci95_low"], st["ci95_high"]], [y, y], color=BLUE, linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([st["mean"]], [y], s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(3.75, y, f"{st['mean']:+.2f} [{st['ci95_low']:+.2f}, {st['ci95_high']:+.2f}]", fontsize=7.5, color=INK2,
                va="center", family="DejaVu Sans Mono")
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=1)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("Difference in stated probability, log-odds (mean over 8 cases, 95% bootstrap CI)")
    _clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.36, right=0.80)
    fig.savefig(path)
    plt.close(fig)


def fig_labels(data, path):
    """Label-strength run: carry-over (vs observation only) for each slot-A wording."""
    items = [("carryover_unlabelled", "“It was suggested that…”"),
             ("carryover_attributed", "“The assistant suggested that…”"),
             (("user_attributed_summary", "observation_only"), "“The user suggested that…”"),
             ("carryover_warned", "“…as an unverified guess, that…”"),
             ("carryover_asserted", "Plain note, no source or hedge")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, scale, xl in ((axes[0], "logodds", "log-odds"), (axes[1], "pp", "probability points")):
        for i, (key, label) in enumerate(items):
            if isinstance(key, tuple):
                st = _ad_hoc(data, key[0], key[1])[scale]
            else:
                st = contrast(data, key)[scale]
            yy = len(items) - 1 - i
            ax.plot([st["ci95_low"], st["ci95_high"]], [yy, yy], color=BLUE, linewidth=1.6, solid_capstyle="round")
            ax.scatter([st["mean"]], [yy], s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=0.9)
        ax.set_xlabel(f"Carry-over vs no guess ({xl})")
        _clean(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(items)), [lab for _, lab in reversed(items)])
    axes[0].tick_params(axis="y", labelsize=8, colors=INK2)
    fig.savefig(path)
    plt.close(fig)


def _ad_hoc(data, a, b):
    """Per-case difference a - b summarised like a contrast (used only for a descriptive figure row)."""
    out = {}
    for scale, key in (("pp", "mean_pp"), ("logodds", "mean_logodds")):
        vals = []
        cases = sorted({cid for (cid, _) in data["cells"]})
        for cid in cases:
            ca, cb = data["cells"].get((cid, a)), data["cells"].get((cid, b))
            if ca and cb and ca["n_valid"] and cb["n_valid"]:
                vals.append(ca[key] - cb[key])
        out[scale] = analyze.summarise(vals)
    return out


REPL_PANELS = [("carryover_unlabelled", "Guess, no evidence\n(unlabelled vs none)"),
               ("attribution_reduction", "Label effect\n(unlabelled − labelled)"),
               ("interpretation_effect_with_evidence", "Guess, supporting evidence\n(vs evidence alone)"),
               ("perseverance_unlabelled", "Guess, contrary evidence\n(vs evidence alone)"),
               ("evidence_effect", "Supporting evidence\n(gate, vs none)"),
               ("disconfirming_effect", "Contrary evidence\n(gate, vs none)")]


def fig_replication(model_data: dict, path):
    models = list(model_data)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.9), sharey=True)
    for ax, (key, title) in zip(axes.flat, REPL_PANELS):
        sts = [contrast(model_data[m], key)["logodds"] for m in models]
        lim = max(1.0, max(max(abs(s["ci95_low"]), abs(s["ci95_high"])) for s in sts) * 1.1)
        for i, s in enumerate(sts):
            yy = len(models) - 1 - i
            ax.plot([s["ci95_low"], s["ci95_high"]], [yy, yy], color=BLUE, linewidth=1.6, solid_capstyle="round")
            ax.scatter([s["mean"]], [yy], s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_title(title, fontsize=8.2, loc="left")
        _clean(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0][0].set_yticks(range(len(models)), list(reversed(models)))
    axes[1][0].set_yticks(range(len(models)), list(reversed(models)))
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=8, colors=INK2)
    fig.supxlabel("Difference in stated probability, log-odds (mean over 8 cases, 95% bootstrap CI)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cases(data, cases, path):
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharey=True, sharex=True)
    for ax, cid in zip(axes.flat, cases):
        for key, _, color in SERIES:
            ys = [data["cells"][(cid, COND[key][c])]["mean_pp"] for c, _ in COLS]
            ax.plot(range(3), ys, color=color, linewidth=1.6)
            ax.scatter(range(3), ys, s=14, color=color, edgecolor=SURFACE, linewidth=0.9, zorder=3)
        ax.set_title(cid.replace("_", " "), fontsize=8, loc="left")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(range(3), ["Contrary", "None", "Support"], fontsize=6.8)
        _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def project_image(data, path, model):
    fig = plt.figure(figsize=(12, 6.3), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    fig.text(0.05, 0.90, "An assistant's guess, carried into a conversation summary,", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.05, 0.835, "makes the next answer more confident. Labelling it doesn't help.", fontsize=19, fontweight="bold", color=INK)
    ax = fig.add_axes([0.07, 0.12, 0.6, 0.58])
    for key, label, color in SERIES:
        ys = [cond_stats(data, COND[key][c])[0] for c, _ in COLS]
        ax.plot(range(3), ys, color=color, linewidth=3, label=label)
        ax.scatter(range(3), ys, s=80, color=color, edgecolor="#ffffff", linewidth=2, zorder=3)
    ax.set_xticks(range(3), [c.replace("\n", " ") for _, c in COLS], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.3, 2.3)
    ax.tick_params(labelsize=11)
    _clean(ax)
    ax.legend(loc="upper left", fontsize=11)
    c1, cp = contrast(data, "carryover_unlabelled")["pp"], contrast(data, "perseverance_unlabelled")["pp"]
    ca = contrast(data, "attribution_reduction")["pp"]
    for i, (big, small) in enumerate([(f"{c1['mean']:+.0f} pts", "guess vs no guess,\nno new evidence"),
                                      (f"{ca['mean']:+.1f} pts", "effect of naming\n“the assistant”"),
                                      (f"{cp['mean']:+.1f} pts", "guess once contrary\nevidence arrives")]):
        y = 0.62 - i * 0.2
        fig.text(0.70, y, big, fontsize=30, fontweight="bold", color=INK)
        fig.text(0.70, y - 0.06, small, fontsize=12, color=INK2, va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)
