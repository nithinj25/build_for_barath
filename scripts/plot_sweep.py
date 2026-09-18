"""Headline sweep figure: python scripts/plot_sweep.py [results/sweep]

Four panels. Colour is always cross_type_sharing, so a series means the same
thing everywhere:

  A same-type hit@10 vs repeat_rate    — sharing does not move it, by
                                          construction; repeat_rate does
  B cross-type hit@10 vs repeat_rate   — sharing 0 is the pre-fix defect; it
                                          carries no cross-type signal until
                                          consistency gets extreme, where
                                          shared style alone supplies some
  C cross-type hit@10 vs MEASURED habit transfer — the interpretable axis:
                                          "sharing 0.5" means nothing to a
                                          reader, "offenders repeat 21% more
                                          often than chance across types" does
  D same-type evidence vs break-even   — the spec's "how consistent must an
                                          offender be" plot; even at high
                                          consistency the evidence sits far
                                          below what the prior demands

Also writes summary.csv: the table view that the palette's contrast warning
obliges, and the file to re-plot from.

Palette: dataviz reference categorical slots 1-5, validated light-mode
(worst adjacent CVD ΔE 9.1, normal-vision 19.6; three slots warn on contrast,
relieved by the direct labels and the CSV).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CROSS = "CROSS_TYPE"


def load(directory: Path) -> list[dict]:
    points = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("rr*_share*.json"))]
    if not points:
        raise SystemExit(f"no sweep points in {directory}")
    return sorted(points, key=lambda r: (r["cross_type_sharing"], r["repeat_rate"]))


def same_type_prior(point: dict) -> float:
    priors = [p["prior_bits"] for k, p in point["pools"].items() if p and k != CROSS and p["prior_bits"]]
    return sum(priors) / len(priors)


def same_type_bits(point: dict) -> float:
    bits = [p["mean_true_pair_evidence_bits"] for k, p in point["pools"].items() if p and k != CROSS]
    return sum(bits) / len(bits)


def write_csv(points: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["repeat_rate", "cross_type_sharing", "alpha",
                    "measured_core_lift_same_type", "measured_core_lift_cross_type",
                    "same_type_hit_at_10", "same_type_recall_at_10", "same_type_pr_auc",
                    "cross_type_hit_at_10", "cross_type_pr_auc",
                    "same_type_mean_true_pair_bits", "cross_type_mean_true_pair_bits",
                    "same_type_mean_prior_bits", "cross_type_prior_bits",
                    "validator_2_truth_mi", "validator_3_recorded_mi"])
        for r in points:
            st, ct, core = r["same_type"]["fs_lr"], r["cross_type"]["fs_lr"], r["core_habit_transfer"]
            w.writerow([r["repeat_rate"], r["cross_type_sharing"], round(r["alpha"], 3),
                        core["mean_lift_same_type"], core["mean_lift_cross_type"],
                        st["hit_at_10"], st["recall_at_10"], st["pr_auc"],
                        ct["hit_at_10"], ct["pr_auc"],
                        round(same_type_bits(r), 3), r["cross_type"]["mean_true_pair_evidence_bits"],
                        round(same_type_prior(r), 2), r["cross_type"]["prior_bits"],
                        r["validators"]["2_truth_mi"]["passed"], r["validators"]["3_recorded_mi"]["passed"]])


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else Path("results/sweep")
    points = load(directory)
    write_csv(points, directory / "summary.csv")
    sharings = sorted({r["cross_type_sharing"] for r in points})
    rr = lambda r: r["repeat_rate"]
    lift = lambda r: r["core_habit_transfer"]["mean_lift_cross_type"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    line_panels = [
        (ax_a, "A · Same-type linkage — sharing does not move it",
         "repeat_rate  (within-type consistency)", "hit@10", lambda r: r["same_type"]["fs_lr"]["hit_at_10"]),
        (ax_b, "B · Cross-type linkage — habit transfer, or extreme consistency",
         "repeat_rate  (within-type consistency)", "hit@10", lambda r: r["cross_type"]["fs_lr"]["hit_at_10"]),
        (ax_d, "D · Same-type evidence vs break-even",
         "repeat_rate  (within-type consistency)", "mean true-pair evidence (bits)", same_type_bits),
    ]
    for ax, title, xlabel, ylabel, value in line_panels:
        style(ax, title, xlabel, ylabel)
        ends = {}
        for i, sharing in enumerate(sharings):
            rows = sorted([r for r in points if r["cross_type_sharing"] == sharing], key=rr)
            xs, ys = [rr(r) for r in rows], [value(r) for r in rows]
            ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"sharing {sharing:g}", zorder=3 + i)
            ends[sharing] = (xs[-1], ys[-1])
        # Label both extremes where the panel separates them, else only the
        # top line; stagger when they nearly coincide so the text cannot collide.
        highs = [y for _, y in ends.values()]
        top, bottom = ends[sharings[-1]], ends[sharings[0]]
        close = abs(top[1] - bottom[1]) < 0.15 * max(max(highs), 1e-9)
        labelled = [(sharings[-1], 0)] if (close and ax is not ax_b) else [(sharings[-1], 7), (sharings[0], -7)]
        for sharing, dy in labelled:
            x, y = ends[sharing]
            note = " · pre-fix defect" if (ax is ax_b and sharing == 0) else ""
            ax.annotate(f"sharing {sharing:g}{note}", (x, y), textcoords="offset points",
                        xytext=(7, dy), color=SECONDARY, fontsize=8, va="center")
        ax.set_xlim(min(map(rr, points)) - 0.03, max(map(rr, points)) + 0.28)
        ax.set_ylim(bottom=0)

    # C: the interpretable axis — measured transfer rather than the latent knob
    style(ax_c, "C · Cross-type linkage vs measured habit transfer",
          "cross-type agreement above chance (measured in truth)", "hit@10")
    for i, sharing in enumerate(sharings):
        rows = [r for r in points if r["cross_type_sharing"] == sharing]
        ax_c.scatter([lift(r) for r in rows], [r["cross_type"]["fs_lr"]["hit_at_10"] for r in rows],
                     s=42, color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=1.2,
                     label=f"sharing {sharing:g}", zorder=3 + i)
    ax_c.set_ylim(bottom=0)

    breakeven = -sum(same_type_prior(r) for r in points) / len(points)
    ax_d.axhline(breakeven, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_d.annotate(f"break-even: {breakeven:.1f} bits needed to clear the prior",
                  (min(map(rr, points)), breakeven), textcoords="offset points", xytext=(2, 6),
                  color=SECONDARY, fontsize=8)
    peak = max(points, key=same_type_bits)
    ax_d.annotate(f"highest observed: {same_type_bits(peak):.2f} bits, at repeat_rate {rr(peak):g}",
                  (0.02, 0.80), xycoords="axes fraction", color=SECONDARY, fontsize=8)
    ax_d.set_ylim(0, breakeven * 1.12)

    handles, labels = ax_a.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=len(sharings), frameon=False,
                        fontsize=9, bbox_to_anchor=(0.5, 0.0))
    for text in legend.get_texts():
        text.set_color(SECONDARY)
    fig.suptitle("Linkage vs offender consistency and cross-type habit sharing",
                 color=INK, fontsize=13, x=0.007, ha="left", y=0.995)
    fig.text(0.007, 0.958, f"{points[0]['n_cases']:,} synthetic cases per point, seed {points[0]['seed']}, "
                           "held-out test offenders; full-pool ranking, no blocking",
             color=SECONDARY, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    for ext in ("png", "svg"):
        fig.savefig(directory / f"sweep.{ext}", dpi=150, facecolor=SURFACE)
    print(f"wrote {directory/'sweep.png'}, {directory/'sweep.svg'}, {directory/'summary.csv'} "
          f"({len(points)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
