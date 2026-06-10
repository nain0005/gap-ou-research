"""Data loading and per-ticker indicator precomputation."""

import os
import numpy as np
import pandas as pd

from config import (TICKERS, START, END, EMA_PERIOD, RSI_PERIOD)
from indicators import ema, rsi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")


def load_data():
    """Load (or download via yfinance) daily bars for tickers + VIX/VIX3M."""
    try:
        import yfinance as yf
    except ImportError:
        yf = None

    os.makedirs(DATA_DIR, exist_ok=True)
    stocks = {}
    for t in TICKERS:
        fp = os.path.join(DATA_DIR, f"{t}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if not df.empty:
                stocks[t] = df
                continue
        if yf is None:
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
        if yf is None:
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


def precompute_indicators(stocks: dict, min_len: int) -> dict:
    """Compute close/open/high/low arrays, EMA, RSI, and EMA-residuals per ticker."""
    ind = {}
    for t, df in stocks.items():
        if len(df) < min_len:
            continue
        c = df["Close"].values.astype(float)
        e = ema(c, EMA_PERIOD)
        ind[t] = {
            "c": c,
            "h": df["High"].values.astype(float),
            "l": df["Low"].values.astype(float),
            "o": df["Open"].values.astype(float),
            "e": e,
            "r": rsi(c, RSI_PERIOD),
            "res": np.where(e > 0, (c - e) / e, 0.0),
            "d": df.index,
        }
    return ind
