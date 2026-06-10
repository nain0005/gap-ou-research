"""diagnose.py — Gate funnel diagnostic.

Answers: why are there so few trades? For every (day, ticker) pair, evaluates
each precondition and gate INDEPENDENTLY (no early exit), so you can see the
unconditional pass rate of each condition and which one is the binding
constraint.

Usage: python src/diagnose.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (GAP_THRESHOLD, VIX_MIN, VIX_LOOKBACK, RESID_WINDOW,
                    EMA_PERIOD, DEV_END)
from data import load_data, precompute_indicators, RESULTS_DIR
from gates import compute_context, PURE_GATES


def main():
    print("=" * 70)
    print("GATE FUNNEL DIAGNOSTIC")
    print("=" * 70)

    stocks, vix_data = load_data()
    vix_df = vix_data.get("VIX")
    vix3m_df = vix_data.get("VIX3M")

    if vix3m_df is None:
        print("\n*** WARNING: VIX3M.csv MISSING — backwardation falls back to the")
        print("*** VIX>25 proxy, which is FAR more restrictive than real")
        print("*** term-structure backwardation. This alone can collapse trade")
        print("*** counts. Check whether the ^VIX3M download failed.\n")

    ind = precompute_indicators(stocks, RESID_WINDOW + EMA_PERIOD + 10)
    vc = vix_df["Close"].values.astype(float)
    vd = vix_df.index
    v3 = (vix3m_df["Close"].reindex(vd).values.astype(float)
          if vix3m_df is not None else np.full(len(vc), np.nan))

    dev_end = pd.Timestamp(DEV_END)
    warmup = max(RESID_WINDOW + EMA_PERIOD, VIX_LOOKBACK + 1)

    # ── market-level conditions (per day) ──────────────────────────────
    day_rows = []
    for di in range(warmup, len(vd)):
        if vd[di] > dev_end:
            break
        v3i = v3[di] if di < len(v3) else np.nan
        bw = (vc[di] > v3i) if (pd.notna(v3i) and v3i > 0) else (vc[di] > 25.0)
        day_rows.append({
            "date": vd[di],
            "vix_gt_min": vc[di] > VIX_MIN,
            "backwardated": bw,
            "rising": vc[di] > vc[di - VIX_LOOKBACK] if di >= VIX_LOOKBACK else False,
        })
    days = pd.DataFrame(day_rows)
    n_days = len(days)
    print(f"\nMARKET-LEVEL (per day, {n_days} dev-period days):")
    print(f"  VIX > {VIX_MIN}:            {days['vix_gt_min'].mean()*100:5.1f}% of days")
    print(f"  Backwardation:        {days['backwardated'].mean()*100:5.1f}% of days"
          f"  {'(proxy VIX>25!)' if vix3m_df is None else '(VIX > VIX3M)'}")
    print(f"  Both (gates A+B):     {(days['vix_gt_min'] & days['backwardated']).mean()*100:5.1f}% of days")

    # ── candidate-level funnel (per day-ticker) ────────────────────────
    counters = {
        "day_ticker_pairs": 0, "indicator_valid": 0, "gap_up": 0,
        "gA": 0, "gB": 0, "gC": 0, "gD": 0, "all_four": 0,
        "gC_given_AB": 0, "gD_given_ABC": 0, "AB_pairs": 0, "ABC_pairs": 0,
    }
    oversold_misses = []  # how far res_prev was from the oversold threshold

    date_to_di = {d: i for i, d in enumerate(vd)}

    for tk, ii in ind.items():
        d_index = ii["d"]
        for ti in range(RESID_WINDOW + EMA_PERIOD, len(ii["c"])):
            date = d_index[ti]
            if date > dev_end:
                break
            di = date_to_di.get(date)
            if di is None or di < warmup:
                continue
            counters["day_ticker_pairs"] += 1

            pc, to = ii["c"][ti - 1], ii["o"][ti]
            ema_val, rsi_val, rsi_prev = ii["e"][ti], ii["r"][ti], ii["r"][ti - 1]
            if (pc <= 0 or to <= 0 or np.isnan(ema_val) or ema_val <= 0
                    or np.isnan(rsi_val) or np.isnan(rsi_prev)):
                continue
            counters["indicator_valid"] += 1

            gap = (to - pc) / pc
            if gap < GAP_THRESHOLD:
                continue
            counters["gap_up"] += 1

            bar = {
                "ticker": tk, "close": ii["c"][ti], "gap": gap,
                "res": ii["res"][ti], "res_prev": ii["res"][ti - 1],
                "resid_window": ii["res"][max(0, ti - RESID_WINDOW):ti],
                "rsi": rsi_val, "rsi_prev": rsi_prev,
                "vix": vc[di],
                "vix_prev": vc[di - VIX_LOOKBACK] if di >= VIX_LOOKBACK else 0.0,
                "vix3m": v3[di] if di < len(v3) else np.nan,
            }
            ctx = compute_context(bar)
            results = {g: PURE_GATES[g](bar, ctx) for g in "ABCD"}
            for g in "ABCD":
                counters[f"g{g}"] += results[g]
            if all(results.values()):
                counters["all_four"] += 1
            if results["A"] and results["B"]:
                counters["AB_pairs"] += 1
                counters["gC_given_AB"] += results["C"]
                if results["C"]:
                    counters["ABC_pairs"] += 1
                    counters["gD_given_ABC"] += results["D"]
            # how close did gate D's oversold condition come?
            if ctx["ou"] is not None and results["A"] and results["B"] and results["C"]:
                thr = ctx["ou"]["mean"] - ctx["params"]["oversold_sigma"] * ctx["ou"]["sigma"]
                oversold_misses.append(bar["res_prev"] - thr)

    c = counters
    gu = max(c["gap_up"], 1)
    print(f"\nCANDIDATE FUNNEL (per day-ticker pair):")
    print(f"  total pairs:                 {c['day_ticker_pairs']:>8,}")
    print(f"  indicators valid:            {c['indicator_valid']:>8,}")
    print(f"  gap-up > {GAP_THRESHOLD*100:.1f}%:               {c['gap_up']:>8,}"
          f"   <- candidate pool")
    print(f"\nUNCONDITIONAL gate pass rates among gap-up candidates:")
    for g, label in zip("ABCD", ["VIX level", "VIX regime/backwardation",
                                  "OU model fit + HL range", "oversold + RSI"]):
        print(f"  gate {g} ({label:28s}): {c[f'g{g}']/gu*100:5.1f}%")
    print(f"\nSEQUENTIAL funnel (A∧B -> C -> D):")
    print(f"  pass A∧B:                    {c['AB_pairs']:>8,} ({c['AB_pairs']/gu*100:.1f}% of gap-ups)")
    if c["AB_pairs"]:
        print(f"  then pass C:                 {c['ABC_pairs']:>8,} ({c['gC_given_AB']/c['AB_pairs']*100:.1f}% of A∧B)")
    if c["ABC_pairs"]:
        print(f"  then pass D:                 {c['gD_given_ABC']:>8,} ({c['gD_given_ABC']/c['ABC_pairs']*100:.1f}% of A∧B∧C)")
    print(f"  pass ALL FOUR:               {c['all_four']:>8,} eligible entries "
          f"(engine then caps at {2} concurrent)")

    if oversold_misses:
        om = np.array(oversold_misses)
        print(f"\nGATE D oversold condition, among A∧B∧C survivors:")
        print(f"  res_prev - threshold: median {np.median(om):+.4f} "
              f"(negative = oversold enough)")
        print(f"  share oversold enough: {np.mean(om <= 0)*100:.1f}%")
        print("  -> if this share is tiny and the median miss is large, the")
        print("     oversold-sigma thresholds are the binding constraint.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([c]).to_csv(os.path.join(RESULTS_DIR, "diagnostic_funnel.csv"),
                             index=False)
    print("\nSaved results/diagnostic_funnel.csv")

    print("\nINTERPRETATION GUIDE:")
    print("  * 884 trades in v1 came from the 15-MINUTE QuantConnect engine")
    print("    (~26 gate evaluations/day/stock). This daily engine evaluates")
    print("    once/day, so an order-of-magnitude drop is expected, not a bug.")
    print("  * If 'Backwardation' is <20% of days, gate B is doing most of the")
    print("    killing — that is a property of the strategy design (it only")
    print("    trades stressed markets), worth stating in the README.")
    print("  * Any change you make based on this funnel is EXPLORATORY:")
    print("    the 2025+ holdout has been spent and cannot confirm new edits.")


if __name__ == "__main__":
    main()
