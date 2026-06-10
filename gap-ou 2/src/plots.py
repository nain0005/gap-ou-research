"""Plotting. Same four core charts as before, plus cost-sensitivity and
walk-forward charts."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from data import PLOTS_DIR
from stats_tests import categorize

COLORS = {"regime-first": "#2E86AB", "model-first": "#A23B72",
          "signal-first": "#F18F01"}


def make_ordering_plots(df, suffix=""):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = df.copy()
    df["cat"] = df["order"].apply(categorize)

    fig, ax = plt.subplots(figsize=(14, 7))
    ds = df.sort_values("nr", ascending=True)
    ax.barh(ds["order"], ds["nr"], color=[COLORS[categorize(o)] for o in ds["order"]])
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Net Return (%)")
    ax.set_title(f"Net Return by Filter Ordering{suffix}", fontsize=14, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in COLORS.items()])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"01_returns{suffix}.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col, title in zip(axes, ["nr", "wr", "sh", "n"],
                              ["Net Return (%)", "Win Rate (%)", "Sharpe (daily eq.)", "Trades"]):
        sns.boxplot(data=df, x="cat", y=col, hue="cat", palette=COLORS, ax=ax,
                    width=0.6, order=list(COLORS), legend=False)
        sns.stripplot(data=df, x="cat", y=col, color="black", size=4, alpha=0.6,
                      ax=ax, order=list(COLORS))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"02_categories{suffix}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    gd = df[["order", "rA", "rB", "rC", "rD"]].set_index("order")
    gd.columns = ["A: VIX level", "B: Regime", "C: OU model", "D: Signals"]
    gn = gd.div(gd.sum(axis=1).replace(0, 1), axis=0) * 100
    sns.heatmap(gn, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Rejections by Gate (%)", fontweight="bold")
    pd2 = df[["order", "p1", "p2", "p3", "p4"]].set_index("order")
    pd2.columns = ["Pos 1", "Pos 2", "Pos 3", "Pos 4"]
    pn = pd2.div(pd2.sum(axis=1).replace(0, 1), axis=0) * 100
    sns.heatmap(pn, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Rejections by Position (%)", fontweight="bold")
    plt.suptitle(f"Where Trades Die{suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"03_rejections{suffix}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, grp in df.groupby("cat"):
        ax.scatter(grp["wr"], grp["nr"], c=COLORS[cat], label=cat, s=80,
                   edgecolors="black", linewidth=0.5)
        for _, row in grp.iterrows():
            ax.annotate(row["order"], (row["wr"], row["nr"]), fontsize=7,
                        ha="center", va="bottom")
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Net Return (%)")
    ax.set_title(f"Win Rate vs Return{suffix}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"04_scatter{suffix}.png"), dpi=150)
    plt.close()


def make_cost_plot(cost_df):
    """cost_df: columns [cost_bps, order, nr]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for order, grp in cost_df.groupby("order"):
        cat = categorize(order)
        ax.plot(grp["cost_bps"], grp["nr"], marker="o", alpha=0.5,
                color=COLORS[cat], linewidth=1)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Round-trip cost (bps)")
    ax.set_ylabel("Net Return (%)")
    ax.set_title("Cost Sensitivity Across Orderings", fontsize=14, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in COLORS.items()])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_cost_sensitivity.png"), dpi=150)
    plt.close()


def make_walkforward_plot(wf_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(wf_df))
    ax.bar([i - 0.2 for i in x], wf_df["is_nr"], width=0.4,
           label="In-sample (train)", color="#2E86AB", alpha=0.8)
    ax.bar([i + 0.2 for i in x], wf_df["oos_nr"], width=0.4,
           label="Out-of-sample (test)", color="#F18F01", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in wf_df["test_start"]], rotation=45,
                       fontsize=8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Net Return (%)")
    ax.set_title("Walk-Forward: In-Sample vs Out-of-Sample (k_stop selected per window)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_walkforward.png"), dpi=150)
    plt.close()
