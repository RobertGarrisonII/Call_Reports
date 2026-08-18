#!/usr/bin/env python3
"""run_staleness_bias.py -- MEASURED staleness bias curve for the Tier-1 flow correlation.

The question, and why it is measured rather than argued
-------------------------------------------------------
25 of the 66 real sessions carry a stale ES ladder (top unchanged on 99%+ of 10ms
snapshots; per-session stale fraction 0.14-0.97 at 1s). The Tier-1 statistic
(flow_correlation.table_flow_corr_regimes: per-bar correlation of AR-prefiltered OFI
innovations on 60s bars, Fisher-z, day-level mean) is estimated THROUGH that staleness,
and there are two a-priori stories with OPPOSITE signs:

  * UPWARD: a carried-forward book releases its accumulated flow in one refresh burst,
    and if refreshes cluster when the other leg is active the bursts land on exactly the
    sub-snapshots where the live leg's flow is largest -- co-movement is concentrated,
    not lost, and the per-bar correlation overstates the latent coupling.
  * DOWNWARD (attenuation): between refreshes the stale leg's measured OFI is zero, so
    most sub-snapshots pair live flow with a frozen book and the per-bar correlation is
    pulled toward zero, understating the latent coupling.

Which one wins is an empirical property of the estimator pipeline (masking, CKS OFI,
AR(5) prefilter, 60s bar correlation, Fisher-z), not something to assume -- so this
runner plants a DGP whose cross-leg flow coupling is KNOWN by construction, pushes it
through the REAL Tier-1 code path (flow_correlation.flow_corr_bars, no reimplementation),
and reads the sign off the output.

The planted DGP
---------------
Per simulated day, each leg's order flow is AR(1) (order splitting) driven by a common
factor: f_{j,t} = phi f_{j,t-1} + u_{j,t} with corr(u_spy, u_es) = rho_true (default
0.75, the observed range). The book frames carry constant top prices and quantity
ladders whose CKS OFI telescopes to the planted flow EXACTLY (bid/ask quantities move
by +-f_t/2 with prices fixed, so level-1 OFI = dQ_bid - dQ_ask = f_t) -- the pipeline
therefore sees the planted flow through its own entry point, with nothing else moving.
Constant prices are deliberate: they isolate the FLOW channel (z_ret, rv, spreads are
degenerate and irrelevant to the Tier-1 statistic being measured).

Staleness is the carry-forward mechanism kalman_ecm.staleness_demo_sessions uses: the
ES leg's observed book refreshes with probability (1-s) per grid step and carries
forward otherwise, so measured ES OFI is zero inside a stale run and dumps the
accumulated true flow at each refresh. s in {0, 0.3, 0.5, 0.7, 0.85, 0.95} spans the
observed per-session stale fractions.

What "bias" is measured against
-------------------------------
NOT atanh(rho_true): the per-bar Pearson-then-Fisher-z estimator has its own O(1/n_sub)
finite-bar bias, identical with and without staleness, and letting it into the curve
would confound the staleness effect with an estimator constant. The reference is the
ORACLE day statistic: the same bar machinery (same bars, same coverage floor, same
Fisher-z) applied to the LATENT innovations u the DGP drew -- so
    bias(s) = mean_day [ z_pipeline(s) - z_oracle ]
isolates what staleness does to the pipeline. At s=0 the pipeline sees the planted flow
exactly and its AR prefilter recovers u up to an O(p/T) fitting-shrinkage residual
(~3e-4 z at T=23400 -- three orders below the high-s effect), so the harness's own
unbiasedness check (the gate pins it) is that the measured mean z sits within 2 of ITS
OWN day-level standard errors of the oracle truth. Days are the independent unit: the
SAME latent days are reused across s (common random numbers), so se(bias) -- the SE of
the paired per-day gap -- is far tighter than se(z_meas) and gives the curve its shape
without noise re-drawn per point.

Output: staleness_bias_curve.csv (s, mean measured z, true z, bias, se(bias), ...) and
a .md rendering that states the measured DIRECTION at high s in one sentence.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

import correlation_svar as cs
import flow_correlation as fc

S_GRID_DEFAULT = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)


# ── the planted DGP ───────────────────────────────────────────────────────────
def planted_flow_day(n_steps, rho, phi, rng):
    """One day's latent common-factor flows. Returns (f_spy, f_es, u_spy, u_es):
    u_{j,t} = sqrt(rho) c_t + sqrt(1-rho) e_{j,t} gives corr(u_spy, u_es) = rho by
    construction; f is the AR(1) flow each leg's book actually posts."""
    c = rng.standard_normal(n_steps)
    e1 = rng.standard_normal(n_steps)
    e2 = rng.standard_normal(n_steps)
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
    u1, u2 = a * c + b * e1, a * c + b * e2
    f1, f2 = np.empty(n_steps), np.empty(n_steps)
    f1[0], f2[0] = u1[0], u2[0]
    for t in range(1, n_steps):
        f1[t] = phi * f1[t - 1] + u1[t]
        f2[t] = phi * f2[t - 1] + u2[t]
    return f1, f2, u1, u2


def book_frame_from_flows(f_spy, f_es, base_qty=1e4, dt_s=1.0):
    """Book-like frame whose level-1 CKS OFI equals the planted flow EXACTLY.

    Prices constant, quantities Qb = base + F/2 and Qa = base - F/2 with F the flow's
    cumulative sum: with both prices unchanged the CKS update is dQb - dQa = f_t at
    every step. base_qty keeps the random-walk F from touching zero (sd(F) is a few
    hundred at 23400 steps)."""
    cols = {}
    n = len(f_spy)
    idx = pd.date_range("2024-07-24 09:30:00", periods=n,
                        freq="%dms" % int(dt_s * 1000), tz="America/New_York")
    for a, f, mid, tick in (("SPY", f_spy, 550.0, 0.01), ("ES", f_es, 5500.0, 0.25)):
        F = np.cumsum(f)
        cols[f"{a}_bidprice_1"] = np.full(n, mid - tick / 2)
        cols[f"{a}_askprice_1"] = np.full(n, mid + tick / 2)
        cols[f"{a}_bidquantity_1"] = base_qty + F / 2.0
        cols[f"{a}_askquantity_1"] = base_qty - F / 2.0
    return pd.DataFrame(cols, index=idx)


def apply_carry_forward(df, s, rng, asset="ES"):
    """Carry-forward staleness on one leg, the staleness_demo_sessions mechanism: the
    observed book refreshes with probability (1-s) per grid step and repeats otherwise.
    Measured OFI on the staled leg is zero inside a run and dumps the accumulated flow
    at each refresh -- exactly what a stale ladder does to CKS OFI."""
    if s <= 0:
        return df
    n = len(df)
    fresh = rng.random(n) < (1.0 - s)
    fresh[0] = True
    out = df.copy()
    for c in [c for c in df.columns if c.startswith(asset + "_")]:
        v = np.where(fresh, df[c].to_numpy(float), np.nan)
        out[c] = pd.Series(v, index=df.index).ffill().to_numpy()
    return out


# ── the two day statistics: pipeline (REAL Tier-1 path) and oracle ────────────
def tier1_day_z(df, bar_seconds=60):
    """The Tier-1 day statistic through the REAL code path: flow_corr_bars (masked CKS
    OFI -> AR(5) innovations -> per-bar corr -> Fisher-z), day-level mean z -- the
    quantity table_flow_corr_regimes averages per day."""
    b = fc.flow_corr_bars(df, bar_seconds=bar_seconds, n_levels=1)
    z = b["z_flow"].to_numpy(float)
    return float(np.nanmean(z)) if np.isfinite(z).any() else np.nan


def oracle_day_z(u_spy, u_es, idx, bar_seconds=60):
    """The same bar machinery on the LATENT innovations: same bar grid, same coverage
    floor (mirrors flow_corr_bars's min_sub), same Fisher-z. This is the bias-free
    reference that shares the estimator's finite-bar properties, so pipeline - oracle
    isolates the staleness effect."""
    bar = idx.floor("%ds" % int(bar_seconds))
    dt = float(np.median(np.diff(idx.asi8))) / 1e9 if len(idx) > 1 else 1.0
    cap = max(1.0, float(bar_seconds) / max(dt, 1e-9))
    min_sub = max(10, int(0.25 * cap))
    rho = fc._bar_corr(u_spy, u_es, bar, min_sub)
    z = cs._fisher_z(rho.to_numpy(float))
    return float(np.nanmean(z)) if np.isfinite(z).any() else np.nan


# ── the curve ────────────────────────────────────────────────────────────────
def bias_curve(s_grid=S_GRID_DEFAULT, n_days=24, n_steps=23400, rho=0.75, phi=0.3,
               seed=0, bar_seconds=60, verbose=False):
    """-> DataFrame, one row per staleness level s. Columns: s, z_meas (mean day-level
    pipeline z), se_z_meas (day-level SE of that mean -- the sampling uncertainty of
    the measurement itself), z_true (mean day-level oracle z), z_pop (atanh(rho_true),
    for scale), bias (mean of the per-day pipeline-minus-oracle gap), se_bias
    (day-level SE of that PAIRED gap; much tighter than se_z_meas because the latent
    days are shared across s -- common random numbers -- so day-level DGP noise
    cancels), n_days. Days are the independent unit throughout; staleness draws are
    fresh per (day, s)."""
    rng = np.random.default_rng(seed)
    days = []
    for d in range(int(n_days)):
        f1, f2, u1, u2 = planted_flow_day(int(n_steps), float(rho), float(phi), rng)
        df0 = book_frame_from_flows(f1, f2)
        days.append((df0, oracle_day_z(u1, u2, df0.index, bar_seconds)))
    rows = []
    for s in s_grid:
        gaps, zm, zo = [], [], []
        for d, (df0, z_or) in enumerate(days):
            df = apply_carry_forward(df0, float(s), rng)
            z = tier1_day_z(df, bar_seconds)
            if np.isfinite(z) and np.isfinite(z_or):
                gaps.append(z - z_or)
                zm.append(z)
                zo.append(z_or)
        g = np.asarray(gaps, float)
        zm_a = np.asarray(zm, float)
        rows.append({"s": float(s),
                     "z_meas": float(zm_a.mean()) if len(zm) else np.nan,
                     "se_z_meas": float(zm_a.std(ddof=1) / np.sqrt(len(zm)))
                     if len(zm) > 1 else np.nan,
                     "z_true": float(np.mean(zo)) if len(zo) else np.nan,
                     "z_pop": float(np.arctanh(rho)),
                     "bias": float(g.mean()) if len(g) else np.nan,
                     "se_bias": float(g.std(ddof=1) / np.sqrt(len(g)))
                     if len(g) > 1 else np.nan,
                     "n_days": int(len(g))})
        if verbose:
            r = rows[-1]
            print("  s=%.2f  z_meas=%+.4f  z_true=%+.4f  bias=%+.4f (se %.4f, %d days)"
                  % (r["s"], r["z_meas"], r["z_true"], r["bias"], r["se_bias"],
                     r["n_days"]))
    return pd.DataFrame(rows)


def direction_sentence(curve, s_high=0.85):
    """One sentence stating the MEASURED direction at high s, with its mechanism.
    Written from the sign the data produced -- nothing here presumes it."""
    row = curve.iloc[(curve["s"] - s_high).abs().argmin()]
    b, se = float(row["bias"]), float(row["se_bias"])
    nse = abs(b) / se if se > 0 else np.inf
    if not np.isfinite(b) or (se > 0 and nse < 2):
        return ("At s=%.2f the measured bias (%+.4f z, %.1f se) is not distinguishable "
                "from zero on this sample." % (row["s"], b, nse))
    if b < 0:
        mech = ("DOWNWARD (attenuation): between refreshes the stale leg's measured OFI "
                "is zero, so most sub-snapshots pair live SPY flow with a frozen ES book "
                "and the per-bar correlation is pulled toward zero")
    else:
        mech = ("UPWARD: the refresh bursts concentrate the stale leg's accumulated flow "
                "on the sub-snapshots where they land, and their co-movement with the "
                "live leg outweighs the zeroed stale runs")
    return ("At s=%.2f carry-forward staleness biases the Tier-1 flow-correlation z by "
            "%+.4f (%.0f se) -- %s." % (row["s"], b, nse, mech))


def render_md(curve, rho, n_days, n_steps, seed):
    lines = ["# Staleness bias curve -- Tier-1 flow correlation", "",
             "Planted DGP: two legs' AR(1) order flow with common-factor innovation "
             "correlation rho_true=%.2f; carry-forward staleness on the ES leg (refresh "
             "probability 1-s per 1s grid step, the kalman_ecm.staleness_demo_sessions "
             "mechanism); %d simulated days of %d steps per s, seed %d. The statistic "
             "is the REAL Tier-1 pipeline (flow_correlation.flow_corr_bars: masked CKS "
             "OFI, AR(5) prefilter, 60s-bar correlation, Fisher-z, day-level mean); "
             "'true z' is the same bar machinery on the latent innovations, so 'bias' "
             "isolates the staleness effect from finite-bar estimator bias. Days are "
             "the independent unit for se(bias)." % (rho, n_days, n_steps, seed),
             "",
             "| s | mean measured z | true z | bias | se(bias) | days |",
             "|---|---|---|---|---|---|"]
    for _, r in curve.iterrows():
        lines.append("| %.2f | %+.4f | %+.4f | %+.4f | %.4f | %d |"
                     % (r["s"], r["z_meas"], r["z_true"], r["bias"], r["se_bias"],
                        r["n_days"]))
    lines += ["", direction_sentence(curve), ""]
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        description="Measured staleness bias curve for the Tier-1 flow correlation")
    ap.add_argument("--out", default="", help="output directory (CSV + md written here)")
    ap.add_argument("--n-days", type=int, default=24,
                    help="simulated days per staleness level (>= 20 for a stable mean)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.75,
                    help="planted innovation correlation (observed range ~0.75)")
    ap.add_argument("--phi", type=float, default=0.3,
                    help="AR(1) coefficient of each leg's flow (order splitting)")
    ap.add_argument("--n-steps", type=int, default=23400,
                    help="1s grid steps per simulated day (23400 = full 6.5h session)")
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--s-grid", default=",".join(str(s) for s in S_GRID_DEFAULT),
                    help="comma-separated staleness levels")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")
    s_grid = [float(x) for x in a.s_grid.split(",") if x.strip() != ""]
    print("staleness bias curve: rho_true=%.2f (z_pop=%.4f), %d days x %d steps, "
          "s in {%s}, seed %d"
          % (a.rho, np.arctanh(a.rho), a.n_days, a.n_steps,
             ", ".join("%g" % s for s in s_grid), a.seed))
    curve = bias_curve(s_grid=s_grid, n_days=a.n_days, n_steps=a.n_steps, rho=a.rho,
                       phi=a.phi, seed=a.seed, bar_seconds=a.bar_seconds, verbose=True)
    print()
    print(curve.round(4).to_string(index=False))
    print()
    print(direction_sentence(curve))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        csv_path = os.path.join(a.out, "staleness_bias_curve.csv")
        md_path = os.path.join(a.out, "staleness_bias_curve.md")
        curve.to_csv(csv_path, index=False)
        with open(md_path, "w") as fh:
            fh.write(render_md(curve, a.rho, a.n_days, a.n_steps, a.seed))
        print("wrote %s and %s" % (csv_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
