"""test_surface_excess.py -- guards excess_over_surface (the curated tail vs the fitted PLANE).

(A) Recovery: impact-blind events follow a known bilinear surface T(b,h)=a0+a1 b+a2 h+a3 b h; curated
    events follow the SAME surface PLUS a known excess DELTA. The surface fit on the impact-blind events
    predicts T(b,h), so excess_over_surface recovers mean ~ DELTA (IM-significant), within support.
(B) Null: curated events follow the surface with NO excess -> mean ~ 0, not significant.
(C) Extrapolation flag: curated points placed OUTSIDE the benchmark's (basis,holl) box are reported via
    within_support_frac < 1 -- the surface extrapolates there, and the test says so.
"""
import sys
import numpy as np
import pandas as pd
import onset_response as onr

def T(b, h, a0=0.6, a1=0.25, a2=0.18, a3=-0.12):
    b = np.asarray(b, float); h = np.asarray(h, float)
    return a0 + a1 * b + a2 * h + a3 * b * h

def _panel(bi, hi, t_ib, bc, hc, t_cur):
    n_ib, n_cur = len(bi), len(bc)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_ib + n_cur, freq="2D"),
        "state": np.r_[bi, bc], "state_holl": np.r_[hi, hc], "transmission": np.r_[t_ib, t_cur],
        "impact_blind": [True] * n_ib + [False] * n_cur, "id_strength": 1.0})

def main():
    ok = True
    rng = np.random.default_rng(0)
    g = np.linspace(0.5, 5.0, 6); Bm, Hm = np.meshgrid(g, g); bi, hi = Bm.ravel(), Hm.ravel()   # 36 benchmark events
    t_ib = T(bi, hi) + rng.normal(0, 0.05, len(bi))

    # (A) curated within support, with injected excess DELTA
    DELTA = 0.40
    bc = rng.uniform(1.0, 4.0, 10); hc = rng.uniform(1.0, 4.0, 10)
    t_cur = T(bc, hc) + DELTA + rng.normal(0, 0.05, 10)
    panel = _panel(bi, hi, t_ib, bc, hc, t_cur)
    fit = onr.fit_stress_surface(panel, y_col="transmission")
    ex = onr.excess_over_surface(panel, fit)
    a_ok = abs(ex["mean"] - DELTA) < 0.15 and ex["im"]["p"] < 0.05 and ex["within_support_frac"] > 0.9 and ex["n"] == 10
    print("(A) excess mean=%.3f (true %.2f) IM p=%.4f within_support=%.2f n=%d -> %s"
          % (ex["mean"], DELTA, ex["im"]["p"], ex["within_support_frac"], ex["n"], a_ok))
    ok &= a_ok

    # (B) null: curated follow the surface, no excess
    t_cur0 = T(bc, hc) + rng.normal(0, 0.05, 10)
    panel0 = _panel(bi, hi, t_ib, bc, hc, t_cur0)
    fit0 = onr.fit_stress_surface(panel0, y_col="transmission")
    ex0 = onr.excess_over_surface(panel0, fit0)
    b_ok = abs(ex0["mean"]) < 0.1 and ex0["im"]["p"] > 0.05
    print("(B) null excess mean=%.3f IM p=%.4f (n.s.) -> %s" % (ex0["mean"], ex0["im"]["p"], b_ok))
    ok &= b_ok

    # (C) extrapolation: curated OUTSIDE the benchmark box (basis up to 8, holl up to 8; box is [0.5,5])
    be = rng.uniform(6.0, 8.0, 8); he = rng.uniform(6.0, 8.0, 8)
    t_e = T(be, he) + DELTA + rng.normal(0, 0.05, 8)
    panelE = _panel(bi, hi, t_ib, be, he, t_e)
    fitE = onr.fit_stress_surface(panelE, y_col="transmission")
    exE = onr.excess_over_surface(panelE, fitE)
    c_ok = exE["within_support_frac"] < 0.2 and exE["n"] == 8       # all curated points out of box
    print("(C) extrapolation within_support=%.2f (curated outside box) n=%d -> %s"
          % (exE["within_support_frac"], exE["n"], c_ok))
    ok &= c_ok

    print("surface-excess checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
