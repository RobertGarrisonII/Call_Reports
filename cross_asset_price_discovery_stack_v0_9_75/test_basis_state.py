"""test_basis_state.py -- guards the SPY-ES basis assembled as a conditioning state and its use as
the state in the state-dependent (onset) local-projection IRF.

(A) basis_state recovers the demeaned no-arb basis exactly and is PREDETERMINED (lagged one step):
    we inject a known log-basis path into synthetic SPY/ES mids and check the returned series equals
    the lagged demeaned basis in bps.
(B) fed as `state=` to irf.local_projection, the basis dislocation recovers a KNOWN state-dependent
    response: y = (b0 + b1*s)*shock, so the response at +1 SD dislocation (theta_high) exceeds the
    response at -1 SD (theta_low), recovering b0 (level) and b1 (state slope) -- i.e. transmission is
    stronger when the arbitrage link is already dislocated.
"""
import sys
import numpy as np
import pandas as pd
import liquidity_stress as ls
import irf

EPS = 1e-12

def build_book(T=4000, seed=0):
    rng = np.random.default_rng(seed)
    mid_es = 5000.0 + np.cumsum(rng.normal(0, 0.5, T))            # ES ~ index points
    d = np.zeros(T); rho = 0.97
    for t in range(1, T):
        d[t] = rho * d[t - 1] + rng.normal(0, 2e-4)              # injected log-basis (mean-reverting)
    mid_spy = (mid_es / 10.0) * np.exp(d)                         # logP_SPY - logP_ES = log(1/10) + d
    cols = {}
    for a, mid, tick in (("SPY", mid_spy, 0.01), ("ES", mid_es, 0.25)):
        for i in range(1, 4):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tick
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tick
            cols[f"{a}_bidquantity_{i}"] = 100.0
            cols[f"{a}_askquantity_{i}"] = 100.0
    return pd.DataFrame(cols), d

if __name__ == "__main__":
    df, d = build_book(seed=0)
    z_true = (d - d.mean()) * 1e4                                 # demeaned basis in bps
    print("basis conditioning state + state-dependent onset LP")

    # (A) predetermined recovery
    bs = ls.basis_state(df)                                       # predetermined=True (lagged)
    rec = bs["basis_bps"].to_numpy()
    okA = np.allclose(rec[1:], z_true[:-1], atol=1e-5, rtol=1e-4)
    okabs = np.allclose(bs["abs_basis_bps"].to_numpy()[1:], np.abs(z_true[:-1]), atol=1e-5, rtol=1e-4)
    bs0 = ls.basis_state(df, predetermined=False)["basis_bps"].to_numpy()
    contemp = np.allclose(bs0[1:], z_true[1:], atol=1e-5, rtol=1e-4)   # un-lagged == contemporaneous
    print("  (A) lagged basis == z_{t-1}: %s | |.| matches: %s | predetermined!=contemporaneous: %s"
          % (okA, okabs, (not np.allclose(rec[2:], z_true[2:], atol=1e-3)) and contemp))

    # (B) state-dependent LP with the basis dislocation as the state
    rng = np.random.default_rng(1)
    state = bs["abs_basis_bps"].to_numpy()
    mu, sd = np.nanmean(state), np.nanstd(state)
    s_std = np.nan_to_num((state - mu) / (sd + EPS))
    shock = rng.normal(0, 1, len(state))
    b0, b1 = 1.0, 0.6
    y = (b0 + b1 * s_std) * shock + rng.normal(0, 0.3, len(state))     # stronger response when dislocated
    lp = irf.local_projection(y, shock, state=state, horizons=[0, 1, 2], n_boot=200, block=20, seed=2)
    h0 = lp[lp.horizon == 0].iloc[0]
    theta, tlo, thi = float(h0["theta"]), float(h0["theta_low"]), float(h0["theta_high"])
    slope = (thi - tlo) / 2.0
    print("  (B) h=0: theta=%.3f (true b0=%.1f) | theta_low=%.3f theta_high=%.3f | state-slope=%.3f (true b1=%.1f)"
          % (theta, b0, tlo, thi, slope, b1))

    okB = (thi > tlo + 0.1) and abs(theta - b0) < 0.15 and abs(slope - b1) < 0.2
    print("  state-dependent transmission recovered (high>low, b0/b1 within tol): %s" % okB)
    ok = okA and okabs and contemp and okB
    print("checks: A=%s B=%s -> %s" % (okA and okabs and contemp, okB, ok))
    sys.exit(0 if ok else 1)
