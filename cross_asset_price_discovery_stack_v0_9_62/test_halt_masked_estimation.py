#!/usr/bin/env python3
"""test_halt_masked_estimation.py -- halt snapshots must be excluded from the ESTIMATORS, not just the QC.

The 2026-08-05 analysis run reported n_obs = 23,341 = 23,401 - 60 on every session: full session
minus lags, halt included. The QC gate has excluded halt snapshots from its crossed-rate arithmetic
since v0.9.26; no estimator ever did. On the four MWCB days that put 900 snapshots with no valid
midpoint inside every VECM, information share, OFI regression and jump statistic -- and the reopen
gap entered the jump split as one giant "return" (2020-03-16: 44% of common-factor QV under
truncation vs 10.8% under Lee-Mykland; the gap, not jumps).

The fix is three small pieces, tested together here:

  * ``market_halts.mask_frame`` NaNs each leg's market columns inside THAT LEG's windows, after
    alignment (ffill would refill them before it), per leg rather than the union (the ES sole-venue
    minute stays live on the leg that was trading).
  * ``_design_within_day`` keeps finite rows only -- one choke point through which the VECM, both
    information shares, Gonzalo-Granger, lag selection, the windowed panel and the jump split all
    pass. NaN + finite-mask drops the halt, the seam, and every lag window touching either.
  * ``jump_robust`` no longer compresses NaN out of the prices before differencing -- the
    compression spliced the last pre-halt price to the reopen price, manufacturing the halt move as
    one 1-second return.

Run: python test_halt_masked_estimation.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import market_halts as mh
import price_discovery_shares as pds

TZ = "America/New_York"


def _pair_session(n=7000, seed=3, halt=("10:00:00", "10:15:00"), gap_bps=200.0, day="2020-03-09"):
    """A cointegrated SPY/ES pair with a halt window and a REOPEN GAP of ``gap_bps``.

    The gap is the artifact generator: across the halt both mids jump by 2% with no trading in
    between. Unmasked, that move lands inside the estimators; spliced, it lands as ONE return."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
    eff = np.cumsum(rng.normal(0, 3e-5, n))
    a, b = (pd.Timestamp(f"{day} {halt[0]}", tz=TZ), pd.Timestamp(f"{day} {halt[1]}", tz=TZ))
    in_halt = (idx >= a) & (idx <= b)
    eff = eff + np.where(idx > b, gap_bps / 1e4, 0.0)      # the reopen gap
    spy = 280.0 * np.exp(eff + rng.normal(0, 5e-6, n))
    es = 2800.0 * np.exp(eff + rng.normal(0, 5e-6, n))
    spy[in_halt] = spy[np.argmax(in_halt) - 1]             # frozen (stale) through the halt
    es[in_halt] = es[np.argmax(in_halt) - 1]
    df = pd.DataFrame({"SPY_bidprice_1": spy - 0.005, "SPY_askprice_1": spy + 0.005,
                       "SPY_bidquantity_1": 100.0, "SPY_askquantity_1": 100.0,
                       "ES_bidprice_1": es - 0.125, "ES_askprice_1": es + 0.125,
                       "ES_bidquantity_1": 50.0, "ES_askquantity_1": 50.0}, index=idx)
    df.attrs["halt_windows_SPY"] = [(a.isoformat(), b.isoformat())]
    df.attrs["halt_windows_ES"] = [(a.isoformat(), b.isoformat())]
    df.attrs["halt_windows"] = df.attrs["halt_windows_SPY"]
    return df, int(in_halt.sum())


def _mid(df, a):
    return (pd.to_numeric(df[f"{a}_bidprice_1"], errors="coerce").to_numpy(float)
            + pd.to_numeric(df[f"{a}_askprice_1"], errors="coerce").to_numpy(float)) / 2.0


def check_mask_frame_masks_the_right_rows():
    df, n_halt = _pair_session()
    masked, rep = mh.mask_frame(df)
    spy = pd.to_numeric(masked["SPY_bidprice_1"], errors="coerce")
    counts = rep == {"SPY": n_halt, "ES": n_halt} and int(spy.isna().sum()) == n_halt
    outside_intact = spy.notna().sum() == len(df) - n_halt
    recorded = masked.attrs.get("halt_masked") == rep
    ok = counts and outside_intact and recorded
    print("(1) a %d-snapshot halt masks exactly %s rows per leg (%s), leaves the rest intact (%s), "
          "and records itself in attrs (%s) : %s"
          % (n_halt, rep, counts, outside_intact, recorded, ok))
    return ok


def check_per_leg_masking_preserves_the_sole_venue_window():
    """SPY halts for 15 min; ES only for its measured ~1 min inside it. ES's sole-venue minute and
    its post-resume trading must stay live -- masking the union would discard the one leg that was
    trading, which is the exact window §9.4 of the memo calls an exhibit."""
    df, _ = _pair_session()
    df.attrs["halt_windows_ES"] = [("2020-03-09 10:01:00-04:00", "2020-03-09 10:02:00-04:00")]
    masked, rep = mh.mask_frame(df)
    es = pd.to_numeric(masked["ES_bidprice_1"], errors="coerce")
    spy = pd.to_numeric(masked["SPY_bidprice_1"], errors="coerce")
    es_short = rep["ES"] == 61 and int(es.isna().sum()) == 61
    spy_full = rep["SPY"] == 901
    sole = es.loc["2020-03-09 10:00:30-04:00"]
    live = np.isfinite(float(sole))
    ok = es_short and spy_full and live
    print("(2) per-leg windows: SPY masked %d, ES masked %d (%s); ES is LIVE at 10:00:30, inside "
          "the SPY halt but before its own (%s) : %s" % (rep["SPY"], rep["ES"], es_short and spy_full,
                                                         live, ok))
    return ok


def check_design_drops_halt_seam_and_lag_windows():
    df, n_halt = _pair_session()
    masked, _ = mh.mask_frame(df)
    n_lags = 20
    p1, p2 = np.log(_mid(masked, "SPY")), np.log(_mid(masked, "ES"))
    dY, X = pds._design_within_day(p1, p2, n_lags)
    dY0, _X0 = pds._design_within_day(np.log(_mid(df, "SPY")), np.log(_mid(df, "ES")), n_lags)
    # dropped = halt diffs (n_halt+1: both ends of each masked price) + the lag windows behind them
    dropped = len(dY0) - len(dY)
    expect = (n_halt + 1) + n_lags
    right = dropped == expect
    finite = np.isfinite(dY).all() and np.isfinite(X).all()
    ok = right and finite
    print("(3) the design drops %d row(s) against an unmasked baseline -- %d halt diffs + %d "
          "lag-contaminated (%s) -- and everything that remains is finite (%s) : %s"
          % (dropped, n_halt + 1, n_lags, right, finite, ok))
    return ok


def check_vecm_still_estimates_and_no_seam_return():
    df, _ = _pair_session()
    masked, _ = mh.mask_frame(df)
    p1, p2 = np.log(_mid(masked, "SPY")), np.log(_mid(masked, "ES"))
    alpha, Om, resid = pds._fit_vecm_fixed(p1, p2, 10)
    fit_ok = np.isfinite(alpha).all() and np.isfinite(resid).all()
    # the 200 bps reopen gap must NOT appear anywhere in the residuals: the largest residual should
    # be on the scale of the 0.3-bp noise, orders of magnitude below the gap
    biggest = float(np.abs(resid).max())
    no_seam = biggest < 100e-4 * 0.25            # far below a 200-bp seam return
    ok = fit_ok and no_seam
    print("(4) the masked VECM estimates cleanly (%s); largest residual %.1f bps against a 200-bp "
          "reopen gap -- the seam return does not exist (%s) : %s"
          % (fit_ok, biggest * 1e4, no_seam, ok))
    return ok


def check_jump_split_no_longer_books_the_gap():
    import jump_robust as jr
    df, _ = _pair_session()
    masked, _ = mh.mask_frame(df)
    j_un = jr.continuous_jump_information_shares(_mid(df, "SPY"), _mid(df, "ES"), n_lags=10)
    j_ma = jr.continuous_jump_information_shares(_mid(masked, "SPY"), _mid(masked, "ES"), n_lags=10)
    fu, fm = float(j_un["jump_frac_cf"]), float(j_ma["jump_frac_cf"])
    inflated = fu > 0.5                      # the gap dominates the unmasked day's QV
    deflated = fm < fu / 5
    ok = inflated and deflated
    print("(5) common-factor jump fraction: unmasked %.1f%% (the 200-bp reopen gap booked as a "
          "jump, %s) vs masked %.1f%% (%s) : %s" % (100 * fu, inflated, 100 * fm, deflated, ok))
    print("    this is 2020-03-16's 44%%-vs-10.8%% discrepancy, reproduced and closed.")
    return ok


def check_driver_helper_and_escape_hatch():
    import run_analysis as ra
    df, n_halt = _pair_session()
    sessions = [("2020-03-09", "volatile", df)]
    on = ra._mask_halt_rows(sessions, enabled=True)
    off = ra._mask_halt_rows(sessions, enabled=False)
    masked_on = int(pd.to_numeric(on[0][2]["SPY_bidprice_1"], errors="coerce").isna().sum()) == n_halt
    untouched = off[0][2] is df
    # a session with NO halt anywhere must come through bit-identical in values
    clean = df.copy(); clean.attrs = {"halt_windows_SPY": [], "halt_windows_ES": []}
    out = ra._mask_halt_rows([("2024-07-24", "volatile", clean)], enabled=True)
    same = int(pd.to_numeric(out[0][2]["SPY_bidprice_1"], errors="coerce").isna().sum()) == 0
    ok = masked_on and untouched and same
    print("(6) the driver helper masks (%s), --no-halt-mask passes the frame through untouched "
          "(%s), and a session that never halted is unchanged (%s) : %s"
          % (masked_on, untouched, same, ok))
    return ok


def main():
    checks = [check_mask_frame_masks_the_right_rows,
              check_per_leg_masking_preserves_the_sole_venue_window,
              check_design_drops_halt_seam_and_lag_windows,
              check_vecm_still_estimates_and_no_seam_return,
              check_jump_split_no_longer_books_the_gap,
              check_driver_helper_and_escape_hatch]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("halt-masked-estimation checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
