#!/usr/bin/env python3
"""test_svar_lag_artifact.py -- the Table 9 lag order tracks the ROLLING WINDOW, not the data.

Footnote 17 of the paper reports AIC pointing at 60 lags, with 6 used because the full model would
not run at 60. That has always been read as "the criterion wants a long memory and we could not
afford it". It is not. On the current specification the selected lag order is a mechanical function
of `corr_window`, and it appears on data that has NO dynamics of any kind.

The simulation below has a CONSTANT return correlation of 0.6 and iid liquidity variables. There is
nothing for a VAR to find; an honest criterion should choose p ~ 0-1. Instead:

    corr_method   corr_window   BIC-selected p*
    rolling                 8                 8
    rolling                12                12
    rolling                20                20
    rolling                25                25
    rolling                40                 0
    rolling                50                 0
    dcc                    25                 1
    dcc                    50                 1

**p* = corr_window, exactly.** The mechanism is the dependent variable. `dCorr` is the first
difference of a W-bar rolling correlation, so it drops the observation from W bars ago and adds
one now: an MA term at lag W and nowhere else. That is not a slowly-decaying memory a VAR should
approximate -- it is a spike the criterion locates precisely, and reports as the lag order. Above
W ~ 40 the spike's contribution no longer covers the k^2*log(T) cost of the intervening lags and
BIC gives up entirely at p=0. Neither answer is about markets.

The paper samples at 1s and 10ms with a 100-bar window, so the induced spike sits at lag 100 and a
search bounded below it climbs to its own boundary and stops there -- which is what a p*=pmax
"boundary, not optimum" warning is really reporting.

DCC is immune because its conditional correlation is a recursive filter driven by the data rather
than a fixed-width box, so differencing it induces nothing at any particular lag.

**On VECM.** A VECM is not the alternative here and would not fix this. Every variable in the
Table 9 system is already stationary -- differenced spreads, OFI, RV, microprice deviation, and
d-correlation -- so there is no unit root and nothing to cointegrate. The cointegrated object in
this paper is the PRICE system (log SPY, log ES with beta=(1,-1)), and it already has a VECM:
`price_discovery_shares._fit_vecm_fixed` / `panel_vecm`, which is what Hasbrouck information shares
and Gonzalo-Granger require, plus `ecm_sde` and `markov_switching_vecm`. Those are different
objects answering a different question. Table 9 asks how a correlation responds to liquidity
shocks; the VECM asks where price discovery happens. Swapping one for the other would not address
the lag problem, because the lag problem is in the dependent variable's construction.

Run: python test_svar_lag_artifact.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import correlation_svar as cs

TZ = "America/New_York"


def _no_dynamics_frame(n=15000, seed=7, rho=0.6):
    """Constant correlation, iid depth, random-walk mids. Nothing for a VAR to find."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-24 09:30:00", periods=n, freq="s", tz=TZ)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n) * 1e-4
    mid = {"SPY": 550 * np.exp(np.cumsum(z[:, 0])), "ES": 5500 * np.exp(np.cumsum(z[:, 1]))}
    df = pd.DataFrame(index=idx)
    for a in ("SPY", "ES"):
        half = 0.01 if a == "SPY" else 0.25
        for i in range(1, 4):
            df[f"{a}_bidprice_{i}"] = mid[a] - half * i
            df[f"{a}_askprice_{i}"] = mid[a] + half * i
            df[f"{a}_bidquantity_{i}"] = rng.integers(50, 500, n).astype(float)
            df[f"{a}_askquantity_{i}"] = rng.integers(50, 500, n).astype(float)
    return df


def _pick(df, method, W, pmax):
    p, _tab = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method=method,
                                 corr_window=W, criterion="bic", pmax=pmax)
    return p


def check_lag_equals_the_window():
    """The headline: p* == corr_window on data with no dynamics whatsoever."""
    df = _no_dynamics_frame()
    rows, ok = [], True
    for W in (8, 12, 20, 25):
        p = _pick(df, "rolling", W, min(3 * W, 60))
        rows.append((W, p))
        ok &= (p == W)
    print("(1) constant correlation, iid liquidity -- nothing for a VAR to find:")
    for W, p in rows:
        print("      corr_window=%-3d  ->  BIC selects p* = %-3s %s" % (W, p, "== the window" if p == W else ""))
    print("    the selected lag order IS the rolling window, exactly : %s" % ok)
    return bool(ok)


def check_dcc_does_not_do_this():
    """DCC's recursive filter induces no spike, so the criterion reports the truth: p ~ 1."""
    df = _no_dynamics_frame()
    ps = [_pick(df, "dcc", W, min(3 * W, 60)) for W in (12, 25)]
    ok = all(p is not None and p <= 2 for p in ps)
    print("(2) the same data with corr_method='dcc': p* = %s -- small, and independent of the "
          "window, because a recursive conditional correlation has no fixed-width box to difference "
          ": %s" % (", ".join(str(p) for p in ps), ok))
    return ok


def check_large_window_collapses_instead():
    """Above W ~ 40 the spike no longer pays for the intervening lags and BIC quits at 0.

    Worth pinning because it is the OPPOSITE failure and looks reassuring: a small selected lag is
    read as 'the criterion converged', when in fact it means the criterion could not see the only
    structure present. Neither p*=W nor p*=0 is an answer about markets."""
    df = _no_dynamics_frame()
    ps = [(W, _pick(df, "rolling", W, 90)) for W in (40, 50)]
    ok = all(p is not None and p <= 1 for _W, p in ps)
    print("(3) at corr_window=40 and 50 the same estimator returns p* = %s -- it has stopped "
          "finding the spike rather than started being right (%s)"
          % (", ".join(str(p) for _W, p in ps), ok))
    return ok


def check_the_estimator_warns():
    """select_svar_lag must SAY when the selected lag is the window, not leave it to be noticed."""
    df = _no_dynamics_frame()
    p, tab = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method="rolling",
                                corr_window=20, criterion="bic", pmax=50)
    note = cs.lag_diagnosis(p, corr_window=20, pmax=50, corr_method="rolling")
    flags = note["window_artifact"] and not note["at_boundary"]
    says = "corr_window" in note["text"] and "dcc" in note["text"]
    # and a healthy selection is NOT flagged
    clean = cs.lag_diagnosis(3, corr_window=100, pmax=40, corr_method="dcc")
    quiet = not clean["window_artifact"] and not clean["at_boundary"]
    # a boundary solution is still flagged separately
    edge = cs.lag_diagnosis(40, corr_window=100, pmax=40, corr_method="rolling")["at_boundary"]
    ok = flags and says and quiet and edge
    print("(4) p*=%s with corr_window=20 is flagged as a window artifact (%s), the message names "
          "the cause and the fix (%s)" % (p, flags, says))
    print("    a healthy p*=3 under dcc is not flagged (%s), and p*==pmax is still reported as a "
          "boundary (%s) : %s" % (quiet, edge, ok))
    return ok


def check_vecm_is_not_the_alternative():
    """The Table 9 variables are stationary, so there is nothing to cointegrate -- and the VECM
    the price system DOES need is already in the stack, on the right object."""
    import price_discovery_shares as pds
    has_vecm = hasattr(pds, "_fit_vecm_fixed") and hasattr(pds, "panel_vecm")
    df = _no_dynamics_frame(n=3000)
    X, names, ci = cs.build_svar_frame(df, spec="informational", n_levels=3, corr_window=50)
    # every column is a difference, a ratio or a bounded statistic -- none is a price level
    no_levels = not any(n.startswith(("SPY_bid", "SPY_ask", "ES_bid", "ES_ask")) for n in names)
    lhs_is_dcorr = names[ci] == "dCorr"
    ok = has_vecm and no_levels and lhs_is_dcorr
    print("(5) the Table 9 system is %s -- all stationary, no price levels (%s), LHS = %s (%s)"
          % (", ".join(names[:4]) + ", ...", no_levels, names[ci], lhs_is_dcorr))
    print("    the cointegrated PRICE system has its own VECM already (price_discovery_shares."
          "_fit_vecm_fixed / panel_vecm) (%s) : %s" % (has_vecm, ok))
    print("    they answer different questions: 'how does correlation respond to liquidity' vs "
          "'where does price discovery happen'. A VECM would not fix the lag, which is a property")
    print("    of how dCorr is built.")
    return ok


def main():
    checks = [check_lag_equals_the_window, check_dcc_does_not_do_this,
              check_large_window_collapses_instead, check_the_estimator_warns,
              check_vecm_is_not_the_alternative]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("SVAR lag-artifact checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
