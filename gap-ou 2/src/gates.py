"""Gate logic.

THE CENTRAL FIX of this revision.

In the original code, gate B (VIX regime) had a side effect: it wrote
`regime` / `rp_params` into the candidate dict, and gates C (OU half-life
range) and D (oversold/RSI thresholds) silently fell back to the OKAY
parameter set whenever B had not yet run. Because logical AND is commutative,
*pure* boolean gates would admit an identical trade set under all 24
orderings — so any P&L "ordering effect" in the original was a state-coupling
artifact, not a market phenomenon.

This module supports both modes so the artifact can be quantified:

  decoupled (default): a context phase computes regime, OU fit, and signal
      booleans for every candidate FIRST; gates are then pure predicates over
      that context. Trade set is provably order-invariant; ordering affects
      only (a) rejection attribution and (b) compute cost.

  coupled (legacy): faithful reproduction of the original fallback behavior,
      used to measure how much of the previously reported return spread was
      the coupling artifact.
"""

import numpy as np
import pandas as pd

from config import (VIX_MIN, BACKWARDATION_PROXY, REGIME_PARAMS)
from indicators import fit_ou


def compute_context(bar: dict) -> dict:
    """Compute everything every gate could need, before any gate runs."""
    ctx = {}

    # Regime (gate B's job, but computed unconditionally here)
    vp = bar["vix_prev"]
    rising = bar["vix"] > vp if vp > 0 else False
    v3 = bar["vix3m"]
    backwardated = (bar["vix"] > v3) if (pd.notna(v3) and v3 > 0) \
        else (bar["vix"] > BACKWARDATION_PROXY)
    ctx["vix_prev_valid"] = vp > 0
    ctx["backwardated"] = backwardated
    ctx["regime"] = "BEST" if rising else "OKAY"
    ctx["params"] = REGIME_PARAMS[ctx["regime"]]

    # OU fit (gate C's job)
    rw = bar["resid_window"]
    ctx["ou"] = fit_ou(rw) if len(rw) >= 30 else None

    return ctx


# ── Pure-predicate gates (decoupled mode) ───────────────────────────────────

def gate_A(bar, ctx):
    return bar["vix"] > VIX_MIN


def gate_B(bar, ctx):
    return ctx["vix_prev_valid"] and ctx["backwardated"]


def gate_C(bar, ctx):
    ou = ctx["ou"]
    if ou is None:
        return False
    p = ctx["params"]
    return p["min_hl"] <= ou["half_life"] <= p["max_hl"]


def gate_D(bar, ctx):
    ou = ctx["ou"]
    if ou is None:
        return False
    p = ctx["params"]
    oversold_thresh = ou["mean"] - p["oversold_sigma"] * ou["sigma"]
    if bar["res_prev"] > oversold_thresh:
        return False
    return (p["rsi_floor"] <= bar["rsi"] < p["rsi_ceiling"]
            and bar["rsi"] > bar["rsi_prev"])


PURE_GATES = {"A": gate_A, "B": gate_B, "C": gate_C, "D": gate_D}


def apply_gates_decoupled(bar, gate_order):
    """Returns (passed, rejecting_gate, rejecting_position, ctx)."""
    ctx = compute_context(bar)
    for pos, g in enumerate(gate_order):
        if not PURE_GATES[g](bar, ctx):
            return False, g, pos, ctx
    return True, None, None, ctx


# ── Coupled gates (legacy reproduction) ─────────────────────────────────────

def apply_gates_coupled(bar, gate_order):
    """Faithful reproduction of the original side-effecting gate chain."""
    state = {"regime": None, "params": None, "ou": None}
    for pos, g in enumerate(gate_order):
        if g == "A":
            if not (bar["vix"] > VIX_MIN):
                return False, g, pos, state
        elif g == "B":
            vp = bar["vix_prev"]
            if vp <= 0:
                return False, g, pos, state
            rising = bar["vix"] > vp
            v3 = bar["vix3m"]
            bw = (bar["vix"] > v3) if (pd.notna(v3) and v3 > 0) \
                else (bar["vix"] > BACKWARDATION_PROXY)
            if not bw:
                return False, g, pos, state
            state["regime"] = "BEST" if rising else "OKAY"
            state["params"] = REGIME_PARAMS[state["regime"]]
        elif g == "C":
            rw = bar["resid_window"]
            if len(rw) < 30:
                return False, g, pos, state
            ou = fit_ou(rw)
            if ou is None:
                return False, g, pos, state
            p = state["params"] or REGIME_PARAMS["OKAY"]   # <-- the coupling
            if not (p["min_hl"] <= ou["half_life"] <= p["max_hl"]):
                return False, g, pos, state
            state["ou"] = ou
        elif g == "D":
            ou = state["ou"]
            if ou is None:
                rw = bar["resid_window"]
                if len(rw) < 30:
                    return False, g, pos, state
                ou = fit_ou(rw)
                if ou is None:
                    return False, g, pos, state
                state["ou"] = ou
            p = state["params"] or REGIME_PARAMS["OKAY"]   # <-- the coupling
            ot = ou["mean"] - p["oversold_sigma"] * ou["sigma"]
            if bar["res_prev"] > ot:
                return False, g, pos, state
            if not (p["rsi_floor"] <= bar["rsi"] < p["rsi_ceiling"]
                    and bar["rsi"] > bar["rsi_prev"]):
                return False, g, pos, state
    # fill state for entry sizing if B never set it
    if state["params"] is None:
        state["regime"] = state["regime"] or "OKAY"
        state["params"] = REGIME_PARAMS[state["regime"]]
    return True, None, None, state
