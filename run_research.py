"""
run_research.py — Gap-Up OU Mean-Reversion: Filter Ordering Research

Matches the QuantConnect strategy exactly:
  - Tech + Financial large caps (>$10B)
  - Gap-up > 0.5%
  - VIX regime (BEST/OKAY, backwardation required)
  - OU model (β < 0, half-life in range)
  - Oversold + RSI filter
  - SL 1.5%, TP 1.75%, hold 2-3 days, no weekend carry

Permutes gates A/B/C/D across all 24 orderings.

Usage:
    pip install yfinance pandas numpy matplotlib seaborn scipy
    python src/run_research.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from scipy import stats
from collections import deque

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — matches QC strategy exactly
# ══════════════════════════════════════════════════════════════════════════

TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL",
    "CRM", "ADBE", "INTC", "AMD", "QCOM", "NOW", "INTU", "AMAT",
    # Financial Services
    "JPM", "GS", "MS", "WFC", "BLK", "SCHW", "BX", "BAC", "C",
    "USB", "PNC", "AXP", "COF", "SPGI", "MCO", "ICE", "CME",
]

START = "2020-01-01"
END = "2026-04-30"

GAP_THRESHOLD = 0.005
VIX_MIN = 15.0
VIX_LOOKBACK = 6
BACKWARDATION_PROXY = 25.0

REGIME_PARAMS = {
    "BEST": {"pos_weight": 0.50, "oversold_sigma": 0.60,
             "rsi_floor": 15, "rsi_ceiling": 75, "min_hl": 1, "max_hl": 30},
    "OKAY": {"pos_weight": 0.20, "oversold_sigma": 1.0,
             "rsi_floor": 15, "rsi_ceiling": 50, "min_hl": 2, "max_hl": 18},
}

SL_PCT = 0.015
TP_PCT = 0.0175
MIN_HOLD = 2
MAX_HOLD = 3
MAX_POS_PER_DAY = 2
INITIAL_CAP = 1_000_000

EMA_PERIOD = 21
RSI_PERIOD = 21
RESID_WINDOW = 60

MONTHLY_LOSS = -0.10
MONTHLY_PROFIT = 0.15

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

# ══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════

def ema(prices, period):
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < period:
        return out
    out[period - 1] = np.mean(prices[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        out[i] = prices[i] * k + out[i - 1] * (1 - k)
    return out

def rsi(prices, period):
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < period + 1:
        return out
    d = np.diff(prices)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:period])
    al = np.mean(l[:period])
    out[period] = 100.0 - 100.0 / (1.0 + ag / al) if al != 0 else 100.0
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100.0 - 100.0 / (1.0 + ag / al) if al != 0 else 100.0
    return out

def fit_ou(resids):
    arr = np.asarray(resids, dtype=float)
    if len(arr) < 30:
        return None
    d = np.diff(arr)
    lag = arr[:-1]
    A = np.column_stack([np.ones(len(lag)), lag])
    try:
        c, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    except Exception:
        return None
    alpha, beta = float(c[0]), float(c[1])
    if beta >= 0:
        return None
    la = 1.0 + beta
    if la <= 0:
        return None
    hl = -np.log(2.0) / np.log(la)
    if not np.isfinite(hl) or hl <= 0:
        return None
    mu = -alpha / beta
    sig = float(np.std(arr - mu, ddof=1))
    if sig <= 0:
        return None
    return {"alpha": alpha, "beta": beta, "half_life": hl, "mean": mu, "sigma": sig}

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    stocks = {}
    for t in TICKERS:
        fp = os.path.join(DATA_DIR, f"{t}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if not df.empty:
                stocks[t] = df
                continue
        print(f"  Downloading {t}...", end=" ", flush=True)
        try:
            df = yf.download(t, start=START, end=END, interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.to_csv(fp)
                stocks[t] = df
                print("OK")
            else:
                print("EMPTY")
        except Exception as e:
            print(f"FAIL: {e}")

    vix_data = {}
    for sym, label in [("^VIX", "VIX"), ("^VIX3M", "VIX3M")]:
        fp = os.path.join(DATA_DIR, f"{label}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if not df.empty:
                vix_data[label] = df
                continue
        print(f"  Downloading {label}...", end=" ", flush=True)
        try:
            df = yf.download(sym, start=START, end=END, interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.to_csv(fp)
                vix_data[label] = df
                print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

    return stocks, vix_data

# ══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════

class Engine:
    def __init__(self, gate_order):
        self.gate_order = gate_order
        self.gr = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.pr = {0: 0, 1: 0, 2: 0, 3: 0}
        self.trades = []
        self.tc = {"BEST": 0, "OKAY": 0}

    def run(self, stocks, vix_df, vix3m_df):
        self.gr = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.pr = {0: 0, 1: 0, 2: 0, 3: 0}
        self.trades = []
        self.tc = {"BEST": 0, "OKAY": 0}

        cap = INITIAL_CAP
        pos = {}
        eq = []

        # Precompute
        ind = {}
        for t, df in stocks.items():
            if len(df) < RESID_WINDOW + EMA_PERIOD + 10:
                continue
            c = df["Close"].values.astype(float)
            h = df["High"].values.astype(float)
            l = df["Low"].values.astype(float)
            o = df["Open"].values.astype(float)
            e = ema(c, EMA_PERIOD)
            r = rsi(c, RSI_PERIOD)
            res = np.where(e > 0, (c - e) / e, 0.0)
            ind[t] = {"c": c, "h": h, "l": l, "o": o, "e": e, "r": r, "res": res, "d": df.index}

        vc = vix_df["Close"].values.astype(float)
        vd = vix_df.index
        v3 = (vix3m_df["Close"].reindex(vd).values.astype(float)
              if vix3m_df is not None else np.full(len(vc), np.nan))

        cm = None
        msc = cap
        mh = False

        si = max(RESID_WINDOW + EMA_PERIOD, VIX_LOOKBACK + 1)

        for di in range(si, len(vd)):
            date = vd[di]
            dow = date.dayofweek

            mk = (date.year, date.month)
            if cm != mk:
                cm = mk
                msc = cap
                mh = False

            if not mh and msc > 0:
                mp = (cap - msc) / msc
                if mp <= MONTHLY_LOSS or mp >= MONTHLY_PROFIT:
                    mh = True
                    for tk in list(pos.keys()):
                        ii = ind.get(tk)
                        if ii:
                            ti = np.searchsorted(ii["d"], date)
                            if 0 < ti < len(ii["c"]):
                                ep = ii["c"][ti]
                                pnl = (ep / pos[tk]["ep"] - 1) * 100
                                cap += cap * pos[tk]["w"] * (ep / pos[tk]["ep"] - 1)
                                self.trades.append({"t": tk, "reg": pos[tk]["reg"],
                                    "ed": pos[tk]["ed"], "xd": date, "ep": pos[tk]["ep"],
                                    "xp": ep, "pnl": pnl, "xr": "month_halt", "dh": pos[tk]["dh"]})
                    pos.clear()

            eq.append({"date": date, "equity": cap})
            if mh:
                continue

            # Manage positions
            opened_today = 0
            for tk in list(pos.keys()):
                ii = ind.get(tk)
                if not ii:
                    continue
                ti = np.searchsorted(ii["d"], date)
                if ti <= 0 or ti >= len(ii["c"]):
                    continue

                pos[tk]["dh"] += 1
                p = pos[tk]
                lo, hi, cl = ii["l"][ti], ii["h"][ti], ii["c"][ti]
                residual = ii["res"][ti]
                sp = p["ep"] * (1 - SL_PCT)
                tp = p["ep"] * (1 + TP_PCT)
                ou_mean = p.get("om", 0)

                xr = None
                xp = None

                if lo <= sp:
                    xp, xr = sp, "stop_loss"
                elif hi >= tp:
                    xp, xr = tp, "take_profit"
                elif ou_mean < 0 and residual >= ou_mean:
                    xp, xr = cl, "mean_reversion"
                elif ou_mean >= 0 and residual >= ou_mean * 0.5:
                    xp, xr = cl, "mean_reversion"
                elif p["dh"] >= MAX_HOLD:
                    xp, xr = cl, "max_hold"
                elif dow == 4:  # Friday
                    xp, xr = cl, "friday_exit"

                if xr:
                    pnl = (xp / p["ep"] - 1) * 100
                    cap += cap * p["w"] * (xp / p["ep"] - 1)
                    self.trades.append({"t": tk, "reg": p["reg"], "ed": p["ed"],
                        "xd": date, "ep": p["ep"], "xp": xp, "pnl": pnl,
                        "xr": xr, "dh": p["dh"]})
                    del pos[tk]

            if len(pos) >= MAX_POS_PER_DAY:
                continue

            # Find candidates
            cands = []
            for tk, ii in ind.items():
                if tk in pos:
                    continue
                ti = np.searchsorted(ii["d"], date)
                if ti <= 1 or ti >= len(ii["c"]) or ti < RESID_WINDOW + EMA_PERIOD:
                    continue

                pc = ii["c"][ti - 1]
                to = ii["o"][ti]
                tc = ii["c"][ti]
                if pc <= 0 or to <= 0:
                    continue

                gp = (to - pc) / pc
                if gp < GAP_THRESHOLD:
                    continue

                ev = ii["e"][ti]
                if np.isnan(ev) or ev <= 0:
                    continue

                rv = ii["r"][ti]
                rp = ii["r"][ti - 1] if ti > 0 else np.nan
                if np.isnan(rv) or np.isnan(rp):
                    continue

                bar = {
                    "tk": tk, "cl": tc, "lo": ii["l"][ti], "hi": ii["h"][ti],
                    "ev": ev, "res": ii["res"][ti], "res_p": ii["res"][ti - 1],
                    "rw": ii["res"][max(0, ti - RESID_WINDOW):ti],
                    "rv": rv, "rp": rp, "gp": gp, "date": date,
                    "vix": vc[di],
                    "vix_p": vc[di - VIX_LOOKBACK] if di >= VIX_LOOKBACK else 0,
                    "v3": v3[di] if di < len(v3) else np.nan,
                    "regime": None, "rp_params": None, "ou": None,
                }

                ok = True
                for pi, g in enumerate(self.gate_order):
                    passed = self._gate(g, bar)
                    if not passed:
                        self.gr[g] += 1
                        self.pr[pi] += 1
                        ok = False
                        break

                if ok:
                    cands.append(bar)

            if cands and len(pos) < MAX_POS_PER_DAY:
                cands.sort(key=lambda x: -x["gp"])
                b = cands[0]
                reg = b["regime"] or "OKAY"
                rpp = b["rp_params"] or REGIME_PARAMS["OKAY"]
                ou = b["ou"] or {}

                pos[b["tk"]] = {
                    "ep": b["cl"], "ed": date, "dh": 0, "reg": reg,
                    "w": rpp["pos_weight"], "om": ou.get("mean", 0),
                }
                self.tc[reg] += 1

        # Results
        n = len(self.trades)
        w = sum(1 for t in self.trades if t["pnl"] > 0)
        l = n - w
        wr = (w / n * 100) if n > 0 else 0
        aw = np.mean([t["pnl"] for t in self.trades if t["pnl"] > 0]) if w > 0 else 0
        al = np.mean([t["pnl"] for t in self.trades if t["pnl"] <= 0]) if l > 0 else 0
        nr = (cap - INITIAL_CAP) / INITIAL_CAP * 100
        sh = 0
        if n > 1:
            rs = [t["pnl"] / 100 for t in self.trades]
            sh = np.mean(rs) / np.std(rs) * np.sqrt(252) if np.std(rs) > 0 else 0

        mdd = 0
        if eq:
            eqa = np.array([e["equity"] for e in eq])
            pk = np.maximum.accumulate(eqa)
            dd = (eqa - pk) / np.where(pk > 0, pk, 1)
            mdd = float(np.min(dd)) * 100

        return {
            "order": "".join(self.gate_order), "n": n, "w": w, "l": l,
            "wr": wr, "aw": aw, "al": al, "nr": nr, "eq": cap, "sh": sh,
            "mdd": mdd, "B": self.tc["BEST"], "O": self.tc["OKAY"],
            "rA": self.gr["A"], "rB": self.gr["B"], "rC": self.gr["C"], "rD": self.gr["D"],
            "p1": self.pr[0], "p2": self.pr[1], "p3": self.pr[2], "p4": self.pr[3],
        }

    def _gate(self, g, b):
        if g == "A":
            return b["vix"] > VIX_MIN
        elif g == "B":
            vp = b["vix_p"]
            if vp <= 0:
                return False
            rising = b["vix"] > vp
            v3 = b["v3"]
            bw = b["vix"] > v3 if pd.notna(v3) and v3 > 0 else b["vix"] > BACKWARDATION_PROXY
            if not bw:
                return False
            b["regime"] = "BEST" if rising else "OKAY"
            b["rp_params"] = REGIME_PARAMS[b["regime"]]
            return True
        elif g == "C":
            rw = b["rw"]
            if len(rw) < 30:
                return False
            ou = fit_ou(rw)
            if ou is None:
                return False
            rp = b.get("rp_params") or REGIME_PARAMS["OKAY"]
            if not (rp["min_hl"] <= ou["half_life"] <= rp["max_hl"]):
                return False
            b["ou"] = ou
            return True
        elif g == "D":
            ou = b.get("ou")
            if ou is None:
                rw = b["rw"]
                if len(rw) < 30:
                    return False
                ou = fit_ou(rw)
                if ou is None:
                    return False
                b["ou"] = ou
            rp = b.get("rp_params") or REGIME_PARAMS["OKAY"]
            ot = ou["mean"] - rp["oversold_sigma"] * ou["sigma"]
            if b["res_p"] > ot:
                return False
            if not (rp["rsi_floor"] <= b["rv"] < rp["rsi_ceiling"] and b["rv"] > b["rp"]):
                return False
            return True
        return False

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def categorize(o):
    return "regime-first" if o[0] in "AB" else "model-first" if o[0] == "C" else "signal-first"

def make_plots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = {"regime-first": "#2E86AB", "model-first": "#A23B72", "signal-first": "#F18F01"}

    # 1. Returns bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    ds = df.sort_values("nr", ascending=True)
    ax.barh(ds["order"], ds["nr"], color=[colors[categorize(o)] for o in ds["order"]])
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Net Return (%)")
    ax.set_title("Net Return by Filter Ordering", fontsize=14, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in colors.items()])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_returns.png"), dpi=150)
    plt.close()

    # 2. Category boxplots
    df["cat"] = df["order"].apply(categorize)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col, title in zip(axes, ["nr", "wr", "sh", "n"],
            ["Net Return (%)", "Win Rate (%)", "Sharpe", "Trades"]):
        sns.boxplot(data=df, x="cat", y=col, palette=colors, ax=ax, width=0.6,
                    order=["regime-first", "model-first", "signal-first"])
        sns.stripplot(data=df, x="cat", y=col, color="black", size=4, alpha=0.6, ax=ax,
                      order=["regime-first", "model-first", "signal-first"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_categories.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Rejection heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    gd = df[["order", "rA", "rB", "rC", "rD"]].set_index("order")
    gd.columns = ["A: VIX level", "B: Regime", "C: OU model", "D: Signals"]
    gn = gd.div(gd.sum(axis=1).replace(0, 1), axis=0) * 100
    sns.heatmap(gn, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("By Gate (%)", fontweight="bold")

    pd2 = df[["order", "p1", "p2", "p3", "p4"]].set_index("order")
    pd2.columns = ["Pos 1", "Pos 2", "Pos 3", "Pos 4"]
    pn = pd2.div(pd2.sum(axis=1).replace(0, 1), axis=0) * 100
    sns.heatmap(pn, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("By Position (%)", fontweight="bold")
    plt.suptitle("Where Trades Die", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_rejections.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Win rate vs return scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, grp in df.groupby("cat"):
        ax.scatter(grp["wr"], grp["nr"], c=colors[cat], label=cat, s=80, edgecolors="black", linewidth=0.5)
        for _, row in grp.iterrows():
            ax.annotate(row["order"], (row["wr"], row["nr"]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Net Return (%)")
    ax.set_title("Win Rate vs Return", fontsize=14, fontweight="bold")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_scatter.png"), dpi=150)
    plt.close()

    print("  Plots saved to plots/")

# ══════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════

def run_stats(df):
    df["cat"] = df["order"].apply(categorize)
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    for metric, label in [("nr", "Net Return"), ("wr", "Win Rate"), ("sh", "Sharpe")]:
        groups = [g[metric].values for _, g in df.groupby("cat")]
        if all(len(g) >= 2 for g in groups):
            h, p = stats.kruskal(*groups)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
            print(f"\n  {label}: Kruskal-Wallis H={h:.3f} p={p:.4f} {sig}")
            for name, grp in df.groupby("cat"):
                v = grp[metric].values
                print(f"    {name:15s}: mean={np.mean(v):+7.2f} std={np.std(v):5.2f}")

    best_i = df["nr"].idxmax()
    worst_i = df["nr"].idxmin()
    spread = df.loc[best_i, "nr"] - df.loc[worst_i, "nr"]
    print(f"\n  Best:  {df.loc[best_i, 'order']} → {df.loc[best_i, 'nr']:+.2f}%")
    print(f"  Worst: {df.loc[worst_i, 'order']} → {df.loc[worst_i, 'nr']:+.2f}%")
    print(f"  Spread: {spread:.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("GAP-UP OU MEAN-REVERSION: FILTER ORDERING RESEARCH")
    print("A=VIX_level  B=VIX_regime  C=OU_model  D=Signals")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    stocks, vix_data = load_data()
    vix_df = vix_data.get("VIX")
    vix3m_df = vix_data.get("VIX3M")
    if vix_df is None:
        print("ERROR: No VIX data")
        return
    print(f"  {len(stocks)} stocks, VIX: {len(vix_df)} days")

    print("\n[2/4] Running 24 permutations...")
    rows = []
    for i, perm in enumerate(permutations(["A", "B", "C", "D"])):
        order = list(perm)
        os_str = "".join(order)
        t0 = time.time()
        print(f"  [{i+1:2d}/24] {os_str}", end="  ", flush=True)

        eng = Engine(order)
        r = eng.run(stocks, vix_df, vix3m_df)
        el = time.time() - t0
        r["time"] = el

        print(f"trades={r['n']:4d} WR={r['wr']:5.1f}% net={r['nr']:+7.2f}% "
              f"sh={r['sh']:+5.2f} DD={r['mdd']:6.2f}% {el:.1f}s")
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    print("\n[3/4] Generating plots...")
    make_plots(df)

    print("\n[4/4] Statistical analysis...")
    run_stats(df)

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    print(f"{'Order':>6s} {'Cat':>14s} {'N':>5s} {'WR%':>6s} {'Net%':>8s} "
          f"{'Sharpe':>7s} {'DD%':>7s} {'B':>3s} {'O':>3s}")
    print("-" * 65)
    for _, r in df.sort_values("nr", ascending=False).iterrows():
        print(f"{r['order']:>6s} {categorize(r['order']):>14s} {r['n']:>5.0f} "
              f"{r['wr']:>5.1f}% {r['nr']:>+7.2f}% {r['sh']:>+6.2f} "
              f"{r['mdd']:>6.2f}% {r['B']:>3.0f} {r['O']:>3.0f}")

    print("\nDone! Check results/ and plots/")

if __name__ == "__main__":
    main()
