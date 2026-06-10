"""Statistical analysis.

Fixes vs original: the single-window Kruskal-Wallis treated 24 permutations
run on the SAME data as independent samples — they are not, so its p-value
was uninterpretable. Replacements:

  1. Friedman test: orderings are re-run on K non-overlapping sub-periods;
     sub-periods act as blocks, ordering-categories as treatments. Correlated
     observations within a block are handled by ranking within blocks.
  2. Moving-block bootstrap on the trade P&L sequence for a CI on the
     best-minus-worst net-return spread (preserves serial dependence).
  3. Order-invariance check (decoupled mode): asserts the trade set is
     identical across orderings — the formal verification that any remaining
     "ordering effect" in P&L was the coupling artifact.
"""

import numpy as np
import pandas as pd
from scipy import stats


def categorize(order: str) -> str:
    return ("regime-first" if order[0] in "AB"
            else "model-first" if order[0] == "C" else "signal-first")


def friedman_across_subperiods(subperiod_results: dict, metric="nr"):
    """subperiod_results: {subperiod_label: DataFrame(order, metric, ...)}.
    Blocks = sub-periods, treatments = ordering categories (mean within block).
    """
    cats = ["regime-first", "model-first", "signal-first"]
    blocks = []
    for label, df in subperiod_results.items():
        d = df.copy()
        d["cat"] = d["order"].apply(categorize)
        means = d.groupby("cat")[metric].mean()
        if not all(c in means.index for c in cats):
            continue
        blocks.append([means[c] for c in cats])
    blocks = np.array(blocks)
    if blocks.shape[0] < 3:
        return None, None, blocks
    stat, p = stats.friedmanchisquare(*blocks.T)
    return stat, p, blocks


def block_bootstrap_spread(trades_best, trades_worst, n_boot=2000,
                           block_len=10, seed=42):
    """Moving-block bootstrap CI for the difference in mean trade P&L between
    the best and worst orderings. Returns (mean_diff, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    a = np.array([t["pnl"] for t in trades_best])
    b = np.array([t["pnl"] for t in trades_worst])
    if len(a) < block_len + 1 or len(b) < block_len + 1:
        return None, None, None

    def resample(x):
        n = len(x)
        k = int(np.ceil(n / block_len))
        starts = rng.integers(0, n - block_len + 1, size=k)
        out = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        return out

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = resample(a).mean() - resample(b).mean()
    return (a.mean() - b.mean(),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def verify_order_invariance(results_by_order: dict) -> bool:
    """Decoupled mode: trade sets must be identical across all orderings."""
    keysets = []
    for order, res in results_by_order.items():
        ks = frozenset((t["t"], str(t["ed"]), str(t["xd"]), round(t["pnl"], 6))
                       for t in res["trades"])
        keysets.append(ks)
    return all(ks == keysets[0] for ks in keysets)
