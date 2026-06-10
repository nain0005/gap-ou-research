"""run_research.py — Gap-Up OU Mean-Reversion: Filter Ordering Research v2

Pipeline:
  [1] Load data, precompute indicators.
  [2] DEV PERIOD, coupled mode: reproduce the original 24-permutation
      experiment -> measures the state-coupling artifact.
  [3] DEV PERIOD, decoupled mode: verify P&L order-invariance; ordering now
      only affects rejection attribution + compute cost (the honest finding).
  [4] Cost sensitivity: coupled-mode spread under 0/5/10/20 bps round-trip.
  [5] Sub-period replication + Friedman test (statistically valid replacement
      for the single-window Kruskal-Wallis) + block-bootstrap CI on the
      best-minus-worst spread.
  [6] Walk-forward over the dev period (k_stop selected per train window).
  [7] HOLDOUT (2025+): evaluated ONCE with the frozen configuration.

Usage:
  pip install -r requirements.txt
  python src/run_research.py            # full pipeline
  python src/run_research.py --no-holdout   # while still iterating
"""

import os
import sys
import time
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (GATES, RESID_WINDOW, EMA_PERIOD, DEV_END, HOLDOUT_START,
                    DEV_SUBPERIODS, COST_BPS, COST_BPS_GRID, K_STOP, END)
from data import load_data, precompute_indicators, RESULTS_DIR
from engine import Engine
from stats_tests import (categorize, friedman_across_subperiods,
                         block_bootstrap_spread, verify_order_invariance)
from walkforward import run_walkforward
from plots import make_ordering_plots, make_cost_plot, make_walkforward_plot


def run_all_orderings(ind, vix_df, vix3m_df, mode, start, end, cost_bps,
                      verbose=True):
    rows, by_order = [], {}
    for i, perm in enumerate(permutations(GATES)):
        order = "".join(perm)
        t0 = time.time()
        eng = Engine(list(perm), mode=mode, cost_bps=cost_bps)
        r = eng.run(ind, vix_df, vix3m_df, start=start, end=end)
        r["order"] = order
        r["time"] = time.time() - t0
        by_order[order] = r
        rows.append({k: v for k, v in r.items() if k not in ("trades", "equity")})
        if verbose:
            print(f"  [{i+1:2d}/24] {order} trades={r['n']:4d} WR={r['wr']:5.1f}% "
                  f"net={r['nr']:+7.2f}% sh={r['sh']:+5.2f} {r['time']:.1f}s")
    return pd.DataFrame(rows), by_order


def main():
    run_holdout = "--no-holdout" not in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    min_len = RESID_WINDOW + EMA_PERIOD + 10

    print("=" * 70)
    print("GAP-UP OU MEAN-REVERSION: FILTER ORDERING RESEARCH v2")
    print("A=VIX_level  B=VIX_regime  C=OU_model  D=Signals")
    print(f"Dev: ...–{DEV_END} | Holdout: {HOLDOUT_START}–{END} | "
          f"Cost: {COST_BPS} bps RT | Exit: sigma-space (k={K_STOP})")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    stocks, vix_data = load_data()
    vix_df, vix3m_df = vix_data.get("VIX"), vix_data.get("VIX3M")
    if vix_df is None:
        print("ERROR: No VIX data"); return
    ind = precompute_indicators(stocks, min_len)
    print(f"  {len(ind)} tickers usable, VIX: {len(vix_df)} days")

    print("\n[2/7] COUPLED mode (original behavior), dev period...")
    df_coupled, by_coupled = run_all_orderings(
        ind, vix_df, vix3m_df, "coupled", None, DEV_END, COST_BPS)
    df_coupled.to_csv(os.path.join(RESULTS_DIR, "summary_coupled_dev.csv"), index=False)
    spread_coupled = df_coupled["nr"].max() - df_coupled["nr"].min()
    make_ordering_plots(df_coupled, suffix="_coupled")

    print("\n[3/7] DECOUPLED mode (pure gates), dev period...")
    df_dec, by_dec = run_all_orderings(
        ind, vix_df, vix3m_df, "decoupled", None, DEV_END, COST_BPS)
    df_dec.to_csv(os.path.join(RESULTS_DIR, "summary_decoupled_dev.csv"), index=False)
    invariant = verify_order_invariance(by_dec)
    spread_dec = df_dec["nr"].max() - df_dec["nr"].min()
    make_ordering_plots(df_dec, suffix="_decoupled")
    print(f"\n  Order-invariance of trade set (decoupled): "
          f"{'VERIFIED' if invariant else 'FAILED — investigate!'}")
    print(f"  Return spread — coupled: {spread_coupled:.2f}pp | "
          f"decoupled: {spread_dec:.2f}pp")
    print(f"  => coupling artifact accounts for "
          f"{spread_coupled - spread_dec:.2f}pp of the originally reported spread.")

    print("\n[4/7] Cost sensitivity (coupled mode, dev period)...")
    cost_rows = []
    for bps in COST_BPS_GRID:
        d, _ = run_all_orderings(ind, vix_df, vix3m_df, "coupled",
                                 None, DEV_END, bps, verbose=False)
        d["cost_bps"] = bps
        cost_rows.append(d[["order", "nr", "sh", "n", "cost_bps"]])
        print(f"  {bps:5.1f} bps: median net {d['nr'].median():+6.2f}%  "
              f"orderings profitable: {(d['nr'] > 0).sum()}/24")
    cost_df = pd.concat(cost_rows, ignore_index=True)
    cost_df.to_csv(os.path.join(RESULTS_DIR, "cost_sensitivity.csv"), index=False)
    make_cost_plot(cost_df)

    print("\n[5/7] Sub-period replication + valid statistics...")
    sub_results = {}
    for (s, e) in DEV_SUBPERIODS:
        d, _ = run_all_orderings(ind, vix_df, vix3m_df, "coupled", s, e,
                                 COST_BPS, verbose=False)
        sub_results[f"{s}–{e}"] = d
        top = d.sort_values("nr", ascending=False).iloc[0]
        print(f"  {s}–{e}: best order {top['order']} ({top['nr']:+.2f}%), "
              f"best category: "
              f"{d.assign(cat=d['order'].apply(categorize)).groupby('cat')['nr'].mean().idxmax()}")
    stat, p, blocks = friedman_across_subperiods(sub_results)
    if stat is not None:
        print(f"\n  Friedman test (blocks=sub-periods, treatments=categories): "
              f"chi2={stat:.3f}, p={p:.4f}")
    best_o = df_coupled.loc[df_coupled["nr"].idxmax(), "order"]
    worst_o = df_coupled.loc[df_coupled["nr"].idxmin(), "order"]
    md, lo, hi = block_bootstrap_spread(by_coupled[best_o]["trades"],
                                        by_coupled[worst_o]["trades"])
    if md is not None:
        sig = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0 — spread not distinguishable from noise"
        print(f"  Block-bootstrap, mean trade P&L diff {best_o} vs {worst_o}: "
              f"{md:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}] ({sig})")

    print("\n[6/7] Walk-forward (k_stop selected per 12m train, applied to 3m test)...")
    wf_df, wf_summary = run_walkforward(ind, vix_df, vix3m_df, list(GATES))
    if len(wf_df):
        wf_df.to_csv(os.path.join(RESULTS_DIR, "walkforward.csv"), index=False)
        make_walkforward_plot(wf_df)
        print(f"\n  Windows: {wf_summary['n_windows']} | "
              f"mean IS {wf_summary['mean_is_nr']:+.2f}% vs "
              f"mean OOS {wf_summary['mean_oos_nr']:+.2f}% "
              f"(decay {wf_summary['is_oos_decay']:+.2f}pp) | "
              f"OOS-positive windows: {wf_summary['oos_positive_windows']}"
              f"/{wf_summary['n_windows']}")

    if run_holdout:
        print(f"\n[7/7] HOLDOUT {HOLDOUT_START}–{END} (frozen config, evaluated once)...")
        eng = Engine(list(GATES), mode="decoupled")
        r = eng.run(ind, vix_df, vix3m_df, start=HOLDOUT_START, end=None)
        print(f"  trades={r['n']} WR={r['wr']:.1f}% net={r['nr']:+.2f}% "
              f"sharpe={r['sh']:+.2f} maxDD={r['mdd']:.2f}%")
        pd.DataFrame([{k: v for k, v in r.items()
                       if k not in ("trades", "equity")}]) \
            .to_csv(os.path.join(RESULTS_DIR, "holdout.csv"), index=False)
    else:
        print("\n[7/7] Holdout SKIPPED (--no-holdout).")

    print("\nDone. See results/ and plots/.")


if __name__ == "__main__":
    main()
