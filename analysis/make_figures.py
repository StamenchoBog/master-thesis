"""Generate the thesis figures (Ch. 07 / 08) and LaTeX tables from the committed CSVs.

    python -m analysis.make_figures

Reads results/msc/runs/{summary,paired_stats,utility_reeval}.csv and writes:
  docs/thesis/figures/*.pdf   vector figures (drop into the FINKI LaTeX template)
  docs/thesis/figures/*.png   raster previews
  docs/thesis/tables/*.tex    booktabs tables with Macedonian captions

Design: two-colour Wong colourblind-safe palette (naive = orange, SISA = blue) held
constant across every figure so the results read as one system; medians shown as bars
with all 10 per-seed points overlaid (honest spread + visible no-overlap); Macedonian
axis labels. Captions live in the .tex table files / figure \\caption in the thesis.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "docs/thesis/figures"
TAB = "docs/thesis/tables"
NAIVE, SISA = "#D55E00", "#0072B2"   # Wong palette, colourblind- and print-safe
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # has full Cyrillic coverage
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

summary = pd.read_csv("results/msc/runs/summary.csv")
stats = pd.read_csv("results/msc/runs/paired_stats.csv")
util = pd.read_csv("results/msc/runs/utility_reeval.csv")


def wide(metric):
    """Per-seed naive/sisa arrays for one summary column, aligned by seed."""
    w = summary.pivot_table(index="seed", columns="arm", values=metric).dropna()
    return w["naive"].values, w["sisa"].values


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def paired_bars(metric, ylabel, name, ratio=True, logy=False):
    """Median bar per arm + all 10 seed points overlaid; annotate the median ratio."""
    n, s = wide(metric)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    meds = [np.median(n), np.median(s)]
    ax.bar([0, 1], meds, width=0.6, color=[NAIVE, SISA], alpha=0.35, zorder=1)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate([(n, NAIVE), (s, SISA)]):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color=c,
                   edgecolor="white", linewidth=0.5, s=34, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Наивно", "SISA"])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if ratio and min(meds) > 0:
        r = max(meds) / min(meds)   # magnitude only; caption states which arm is higher
        ax.annotate(f"{r:.1f}×", xy=(0.5, max(meds)), ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
    ax.margins(y=0.15)
    save(fig, name)


# --- H1–H4: physical cost (one visual language) ---
paired_bars("ttr_s",           "Време на закрепнување (s)",        "fig_h1_ttr")
paired_bars("p3_energy_net_wh", "Нето енергија на закрепнување (Wh)", "fig_h2_energy")
paired_bars("p3_throttled_s",  "Траење на термичко забавување (s)", "fig_h3_throttle")
paired_bars("p3_sd_written_mb", "Запишани податоци на SD (MB)",     "fig_h4_sd")

# --- H5: the corrected utility figure (balanced accuracy + specificity) ---
p4 = util[util.phase == "p4"]
fig, ax = plt.subplots(figsize=(5.0, 3.7))
groups = [("Избалансирана\nточност", "balanced_acc"), ("Специфичност\n(бенигни)", "specificity")]
x = np.arange(len(groups)); w = 0.36
for off, arm, c, lab in [(-w / 2, "naive", NAIVE, "Наивно"), (w / 2, "sisa", SISA, "SISA")]:
    meds = [p4[p4.arm == arm][col].median() for _, col in groups]
    ax.bar(x + off, meds, w, color=c, alpha=0.85, label=lab)
    for xi, (_, col) in zip(x + off, groups):
        ax.scatter(xi + np.random.default_rng(1).uniform(-0.08, 0.08, 10),
                   p4[p4.arm == arm][col].values, color="black", s=12, alpha=0.5, zorder=3)
ax.axhline(0.5, ls="--", color="gray", lw=1)
ax.annotate("случаен избор (0.50)", xy=(1.4, 0.5), xytext=(1.4, 0.56),
            ha="right", fontsize=9, color="gray")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("Вредност"); ax.set_ylim(0, 1.0); ax.legend(frameon=False, loc="upper right")
save(fig, "fig_h5_utility")

# --- H6: standing cost (Phase-1 checkpoint I/O) vs recovery time saved ---
overhead = summary[summary.arm == "sisa"]["p1_ckpt_io_s"].median()
n_ttr, s_ttr = wide("ttr_s")
saved = np.median(n_ttr - s_ttr)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.bar([0, 1], [overhead, saved], width=0.6, color=[SISA, "#4C9F70"], alpha=0.8)
for i, v in enumerate([overhead, saved]):
    ax.annotate(f"{v:.1f} s", xy=(i, v), ha="center", va="bottom", fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Стоечки трошок\n(Фаза 1, I/O)", "Заштеда при\nзакрепнување"])
ax.set_ylabel("Време (s)")
save(fig, "fig_h6_cost_of_ownership")

# --- Ch. 08: cross-session robustness (same effect in both subsets) ---
same, cross = [42, 46, 48, 49, 50, 51], [43, 44, 45, 47]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for xpos, seeds, lab in [(0, same, f"Иста сесија\n(n={len(same)})"),
                         (1, cross, f"Различна сесија\n(n={len(cross)})")]:
    sub = summary[summary.seed.isin(seeds)]
    w = sub.pivot_table(index="seed", columns="arm", values="ttr_s").dropna()
    ratios = (w["naive"] / w["sisa"]).values
    ax.scatter([xpos] * len(ratios) + np.random.default_rng(2).uniform(-0.08, 0.08, len(ratios)),
               ratios, color=SISA, s=36, zorder=3)
    ax.hlines(np.median(ratios), xpos - 0.2, xpos + 0.2, color="black", lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Иста сесија\n(n=6)", "Различна сесија\n(n=4)"])
ax.set_ylabel("Однос на време (Наивно / SISA)"); ax.set_ylim(0, 11)
save(fig, "fig_robustness_session")


# --- LaTeX tables ---
def w_row(m):
    r = stats[stats.metric == m].iloc[0]
    return r


MK = {"ttr_s": "Време на закрепнување (s)", "p3_energy_net_wh": "Нето енергија (Wh)",
      "p3_throttled_s": "Термичко забавување (s)", "p3_sd_written_mb": "Запишано на SD (MB)"}
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Спарена статистика за физичките трошоци на закрепнување (N=10 семиња, "
         r"42--51). Медијани по пристап, разлика со 95\% bootstrap интервал на доверба, "
         r"Wilcoxon signed-rank $p$ и Cliff's $\delta$. $p=0.0020$ е долната граница при N=10.}",
         r"\label{tab:paired-cost}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Метрика & Наивно & SISA & Разлика [95\% CI] & $p$ & $\delta$ \\", r"\midrule"]
for m in ["ttr_s", "p3_energy_net_wh", "p3_throttled_s", "p3_sd_written_mb"]:
    r = w_row(m)
    lines.append(f"{MK[m]} & {r.naive_median:g} & {r.sisa_median:g} & "
                 f"{r.diff_median:g} [{r.diff_ci95_lo:g}, {r.diff_ci95_hi:g}] & "
                 f"{r.wilcoxon_p:.4f} & {r.cliffs_delta:+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{TAB}/tab_paired_cost.tex", "w").write("\n".join(lines))
print("  wrote tab_paired_cost.tex")

pm = util[util.phase == "p4"].groupby("arm")[
    ["recall", "specificity", "balanced_acc", "mcc", "roc_auc"]].median()
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Повторна евалуација на глобалниот модел по закрепнување (Фаза 4) врз "
         r"издвоеното глобално тест-множество (100k примери, 96.5\% напади). Медијани по "
         r"пристап. Recall/F1 се тривијални при таков дисбаланс; \emph{избалансираната "
         r"точност}, специфичноста и MCC ја откриваат вистината: SISA колабира во "
         r"мнозинската класа (сè предвидува како напад), додека наивното повторно "
         r"тренирање задржува детекција на бенигни. ROC-AUC е споредлив, па дефектот е "
         r"во калибрацијата, не во информацијата.}",
         r"\label{tab:utility}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Пристап & Recall & Специфичност & Изб.\ точност & MCC & ROC-AUC \\", r"\midrule"]
for arm, lab in [("naive", "Наивно"), ("sisa", "SISA")]:
    r = pm.loc[arm]
    lines.append(f"{lab} & {r.recall:.3f} & {r.specificity:.3f} & {r.balanced_acc:.3f} & "
                 f"{r.mcc:.3f} & {r.roc_auc:.3f} \\\\")
lines.append(r"\midrule")
lines.append(r"Мнозинска класа & 1.000 & 0.000 & 0.500 & 0.000 & 0.500 \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{TAB}/tab_utility.tex", "w").write("\n".join(lines))
print("  wrote tab_utility.tex")

print("Done.")
