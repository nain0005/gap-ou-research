"""Backtest engine.

Changes vs original:
  * Transaction costs: round-trip bps deducted from every trade return.
  * Exit modes: "sigma" (OU-model-consistent: stop at entry_residual -
    k_stop*sigma mapped to price via today's EMA; target = reversion to the
    fitted OU mean) or "fixed_pct" (legacy 1.5%/1.75%).
  * Sharpe computed from the DAILY EQUITY CURVE (annualized sqrt(252)), not
    from per-trade returns — per-trade annualization was statistically wrong
    for overlapping multi-day holds.
  * Fills up to MAX_CONCURRENT_POS from the ranked candidate list (original
    only ever took the single best candidate despite the constant's name).
  * Accepts a [start, end] date range so the same engine drives sub-period
    replication, walk-forward, and the holdout evaluation.
  * Gate logic delegated to gates.py (decoupled or coupled mode).

Documented assumptions:
  * Same-bar stop/target ambiguity resolved conservatively: stop first.
  * Entries at the signal day's close (signal uses same-day close-derived
    residual/RSI, so entry cannot occur before the close).
  * Friday flat rule (no weekend carry) retained from original.
"""

import numpy as np
import pandas as pd

from config import (GAP_THRESHOLD, VIX_LOOKBACK, RESID_WINDOW, EMA_PERIOD,
                    INITIAL_CAP, MAX_HOLD, MAX_CONCURRENT_POS,
                    MONTHLY_LOSS, MONTHLY_PROFIT,
                    EXIT_MODE, K_STOP, SL_PCT, TP_PCT, COST_BPS)
from gates import apply_gates_decoupled, apply_gates_coupled


class Engine:
    def __init__(self, gate_order, mode="decoupled", exit_mode=EXIT_MODE,
                 k_stop=K_STOP, cost_bps=COST_BPS):
        self.gate_order = list(gate_order)
        self.mode = mode
        self.exit_mode = exit_mode
        self.k_stop = k_stop
        self.cost_rt = cost_bps / 10_000.0

    # ────────────────────────────────────────────────────────────────────
    def run(self, ind, vix_df, vix3m_df, start=None, end=None):
        gate_rejections = {g: 0 for g in "ABCD"}
        pos_rejections = {0: 0, 1: 0, 2: 0, 3: 0}
        trades = []
        regime_counts = {"BEST": 0, "OKAY": 0}

        cap = INITIAL_CAP
        positions = {}
        equity = []

        vc = vix_df["Close"].values.astype(float)
        vd = vix_df.index
        v3 = (vix3m_df["Close"].reindex(vd).values.astype(float)
              if vix3m_df is not None else np.full(len(vc), np.nan))

        warmup = max(RESID_WINDOW + EMA_PERIOD, VIX_LOOKBACK + 1)
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None

        cur_month = None
        month_start_cap = cap
        month_halted = False

        for di in range(warmup, len(vd)):
            date = vd[di]
            if start_ts is not None and date < start_ts:
                continue
            if end_ts is not None and date > end_ts:
                break
            dow = date.dayofweek

            mk = (date.year, date.month)
            if cur_month != mk:
                cur_month = mk
                month_start_cap = cap
                month_halted = False
            if not month_halted and month_start_cap > 0:
                mp = (cap - month_start_cap) / month_start_cap
                if mp <= MONTHLY_LOSS or mp >= MONTHLY_PROFIT:
                    month_halted = True
                    for tk in list(positions.keys()):
                        ii = ind.get(tk)
                        if ii is not None:
                            ti = np.searchsorted(ii["d"], date)
                            if 0 < ti < len(ii["c"]):
                                cap = self._close(positions, trades, tk, cap,
                                                  ii["c"][ti], date, "month_halt")
                    positions.clear()

            equity.append({"date": date, "equity": cap})
            if month_halted:
                continue

            # ── manage open positions ──────────────────────────────────
            for tk in list(positions.keys()):
                ii = ind.get(tk)
                if ii is None:
                    continue
                ti = np.searchsorted(ii["d"], date)
                if ti <= 0 or ti >= len(ii["c"]):
                    continue
                p = positions[tk]
                p["dh"] += 1
                lo, hi, cl = ii["l"][ti], ii["h"][ti], ii["c"][ti]
                ema_today = ii["e"][ti]
                residual = ii["res"][ti]

                xp, xr = self._check_exit(p, lo, hi, cl, ema_today, residual, dow)
                if xr:
                    cap = self._close(positions, trades, tk, cap, xp, date, xr)

            # ── new entries ────────────────────────────────────────────
            if len(positions) >= MAX_CONCURRENT_POS:
                continue

            candidates = []
            for tk, ii in ind.items():
                if tk in positions:
                    continue
                ti = np.searchsorted(ii["d"], date)
                if (ti <= 1 or ti >= len(ii["c"])
                        or ti < RESID_WINDOW + EMA_PERIOD):
                    continue
                prev_close, today_open = ii["c"][ti - 1], ii["o"][ti]
                if prev_close <= 0 or today_open <= 0:
                    continue
                gap = (today_open - prev_close) / prev_close
                if gap < GAP_THRESHOLD:
                    continue
                ema_val = ii["e"][ti]
                if np.isnan(ema_val) or ema_val <= 0:
                    continue
                rsi_val, rsi_prev = ii["r"][ti], ii["r"][ti - 1]
                if np.isnan(rsi_val) or np.isnan(rsi_prev):
                    continue

                bar = {
                    "ticker": tk, "close": ii["c"][ti], "gap": gap,
                    "res": ii["res"][ti], "res_prev": ii["res"][ti - 1],
                    "resid_window": ii["res"][max(0, ti - RESID_WINDOW):ti],
                    "rsi": rsi_val, "rsi_prev": rsi_prev,
                    "vix": vc[di],
                    "vix_prev": vc[di - VIX_LOOKBACK] if di >= VIX_LOOKBACK else 0.0,
                    "vix3m": v3[di] if di < len(v3) else np.nan,
                }

                if self.mode == "decoupled":
                    ok, rg, rp, ctx = apply_gates_decoupled(bar, self.gate_order)
                else:
                    ok, rg, rp, ctx = apply_gates_coupled(bar, self.gate_order)

                if not ok:
                    gate_rejections[rg] += 1
                    pos_rejections[rp] += 1
                    continue
                bar["ctx"] = ctx
                candidates.append(bar)

            if candidates:
                candidates.sort(key=lambda x: -x["gap"])
                slots = MAX_CONCURRENT_POS - len(positions)
                for b in candidates[:slots]:
                    ctx = b["ctx"]
                    regime = ctx["regime"] or "OKAY"
                    params = ctx["params"]
                    ou = ctx["ou"] or {}
                    positions[b["ticker"]] = {
                        "ep": b["close"], "ed": vd[di], "dh": 0,
                        "reg": regime, "w": params["pos_weight"],
                        "ou_mean": ou.get("mean", 0.0),
                        "ou_sigma": ou.get("sigma", 0.0),
                        "entry_res": b["res"],
                    }
                    regime_counts[regime] += 1

        return self._summarize(trades, equity, cap, gate_rejections,
                               pos_rejections, regime_counts)

    # ────────────────────────────────────────────────────────────────────
    def _check_exit(self, p, lo, hi, cl, ema_today, residual, dow):
        """Returns (exit_price, exit_reason) or (None, None).
        Same-bar ambiguity: stop checked before target (conservative)."""
        if self.exit_mode == "sigma" and p["ou_sigma"] > 0 and ema_today > 0:
            stop_res = p["entry_res"] - self.k_stop * p["ou_sigma"]
            stop_price = ema_today * (1.0 + stop_res)
            if lo <= stop_price:
                return stop_price, "stop_sigma"
            if residual >= p["ou_mean"]:
                return cl, "mean_reversion"
        else:  # fixed_pct legacy
            sp = p["ep"] * (1 - SL_PCT)
            tp = p["ep"] * (1 + TP_PCT)
            if lo <= sp:
                return sp, "stop_loss"
            if hi >= tp:
                return tp, "take_profit"
            ou_mean = p["ou_mean"]
            if ou_mean < 0 and residual >= ou_mean:
                return cl, "mean_reversion"
            if ou_mean >= 0 and residual >= ou_mean * 0.5:
                return cl, "mean_reversion"
        if p["dh"] >= MAX_HOLD:
            return cl, "max_hold"
        if dow == 4:
            return cl, "friday_exit"
        return None, None

    def _close(self, positions, trades, tk, cap, exit_price, date, reason):
        p = positions[tk]
        gross = exit_price / p["ep"] - 1.0
        net = gross - self.cost_rt
        cap += cap * p["w"] * net
        trades.append({"t": tk, "reg": p["reg"], "ed": p["ed"], "xd": date,
                       "ep": p["ep"], "xp": exit_price,
                       "pnl_gross": gross * 100, "pnl": net * 100,
                       "xr": reason, "dh": p["dh"]})
        del positions[tk]
        return cap

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _summarize(trades, equity, cap, gate_rej, pos_rej, regime_counts):
        n = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = n - wins
        wr = wins / n * 100 if n else 0.0
        aw = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins else 0.0
        al = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if losses else 0.0
        nr = (cap - INITIAL_CAP) / INITIAL_CAP * 100

        sharpe, mdd = 0.0, 0.0
        if len(equity) > 2:
            eq = np.array([e["equity"] for e in equity])
            daily = np.diff(eq) / eq[:-1]
            if np.std(daily) > 0:
                sharpe = float(np.mean(daily) / np.std(daily) * np.sqrt(252))
            peaks = np.maximum.accumulate(eq)
            mdd = float(np.min((eq - peaks) / np.where(peaks > 0, peaks, 1))) * 100

        return {
            "n": n, "w": wins, "l": losses, "wr": wr, "aw": aw, "al": al,
            "nr": nr, "eq": cap, "sh": sharpe, "mdd": mdd,
            "B": regime_counts["BEST"], "O": regime_counts["OKAY"],
            "rA": gate_rej["A"], "rB": gate_rej["B"],
            "rC": gate_rej["C"], "rD": gate_rej["D"],
            "p1": pos_rej[0], "p2": pos_rej[1], "p3": pos_rej[2], "p4": pos_rej[3],
            "trades": trades, "equity": equity,
        }
