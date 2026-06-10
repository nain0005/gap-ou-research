"""Walk-forward analysis.

Rolling scheme over the DEVELOPMENT period only:
  train window: WF_TRAIN_MONTHS — select k_stop (the sigma-space stop width)
                that maximizes net return in-sample
  test window:  WF_TEST_MONTHS  — apply the selected k_stop out-of-sample
  step:         WF_STEP_MONTHS

The concatenated test segments form the walk-forward equity: every trade in
it was taken with a parameter chosen on data strictly preceding it. The gap
between in-sample and walk-forward performance is the honest estimate of
selection overfitting on this parameter.
"""

import numpy as np
import pandas as pd

from config import (WF_TRAIN_MONTHS, WF_TEST_MONTHS, WF_STEP_MONTHS,
                    K_STOP_GRID, COST_BPS, START, DEV_END)
from engine import Engine


def month_edges(start, end, step_months):
    edges = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur <= end:
        edges.append(cur)
        cur = cur + pd.DateOffset(months=step_months)
    return edges


def run_walkforward(ind, vix_df, vix3m_df, gate_order, mode="decoupled",
                    cost_bps=COST_BPS, verbose=True):
    windows = []
    t0 = pd.Timestamp(START) + pd.DateOffset(months=6)   # leave indicator warmup
    dev_end = pd.Timestamp(DEV_END)

    cur = t0
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(months=WF_TRAIN_MONTHS) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=WF_TEST_MONTHS) - pd.Timedelta(days=1)
        if test_end > dev_end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur = cur + pd.DateOffset(months=WF_STEP_MONTHS)

    rows = []
    all_test_trades = []
    for (tr_s, tr_e, te_s, te_e) in windows:
        best_k, best_nr = None, -np.inf
        for k in K_STOP_GRID:
            eng = Engine(gate_order, mode=mode, exit_mode="sigma",
                         k_stop=k, cost_bps=cost_bps)
            r = eng.run(ind, vix_df, vix3m_df, start=tr_s, end=tr_e)
            if r["nr"] > best_nr:
                best_nr, best_k = r["nr"], k
        eng = Engine(gate_order, mode=mode, exit_mode="sigma",
                     k_stop=best_k, cost_bps=cost_bps)
        r_test = eng.run(ind, vix_df, vix3m_df, start=te_s, end=te_e)
        rows.append({
            "train_start": tr_s.date(), "train_end": tr_e.date(),
            "test_start": te_s.date(), "test_end": te_e.date(),
            "k_selected": best_k, "is_nr": best_nr,
            "oos_nr": r_test["nr"], "oos_n": r_test["n"],
            "oos_sh": r_test["sh"], "oos_mdd": r_test["mdd"],
        })
        all_test_trades.extend(r_test["trades"])
        if verbose:
            print(f"  train {tr_s.date()}–{tr_e.date()} -> k*={best_k} "
                  f"(IS {best_nr:+.2f}%) | test {te_s.date()}–{te_e.date()} "
                  f"OOS {r_test['nr']:+.2f}% ({r_test['n']} trades)")

    df = pd.DataFrame(rows)
    if len(df):
        wf_pnls = [t["pnl"] for t in all_test_trades]
        summary = {
            "n_windows": len(df),
            "mean_is_nr": df["is_nr"].mean(),
            "mean_oos_nr": df["oos_nr"].mean(),
            "oos_total_trades": len(all_test_trades),
            "oos_mean_trade_pnl": float(np.mean(wf_pnls)) if wf_pnls else 0.0,
            "oos_positive_windows": int((df["oos_nr"] > 0).sum()),
            "is_oos_decay": df["is_nr"].mean() - df["oos_nr"].mean(),
        }
    else:
        summary = {}
    return df, summary
