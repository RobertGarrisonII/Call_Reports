#!/usr/bin/env python3
"""Gate: the propagation-horizon ladder (memo item E1, v0.9.75).

The findings memo bracketed the SPY<-ES cross-impact between ~0 at 10ms and
~0.4 at 1s; the ladder exists to locate the horizon inside that bracket. The
gate plants the answer and demands the machinery find it:

  1. delayed DGP -- ES flow impounded into the SPY mid through a uniform kernel
     over 200ms: the profile must RISE monotonically along the ladder, capture
     little of the impact at 10ms, and place the half-impact horizon in a band
     around the kernel's half-mass point; the instantaneous DGP must place it
     at the resolution floor. The CONTRAST is the check -- a broken ladder that
     smears everything to one end fails one side or the other.
  2. half_impact_horizon unit behaviour: log-interpolation between rungs,
     at-floor profiles return the finest rung, never-crossing profiles NaN,
     non-finite rungs dropped
  3. frequency scaling rides the ladder: each rung's n_lags and fleeting-filter
     width match frequency_defaults at that dt (wall-clock-comparable settings,
     not one setting reused across grids)
  4. the runner writes the per-day and summary CSVs and prints the verdict
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import horizon_profile as hp

LADDER = ("10ms", "50ms", "100ms", "250ms", "500ms", "1s")


def check_planted_horizon():
    sess = hp.propagation_demo_sessions(n_days=2, n_bars=30000, delay_ms=200, seed=0)
    _pd, summ, meta = hp.profile_ladder(sess, intervals=LADDER, n_levels=3)
    r = summ["lambda_spy_from_es_ratio"].to_numpy(float)
    hi = meta["half_impact_spy_from_es_s"]
    ok_mono = bool(np.all(np.diff(r) > -0.05))            # rises along the ladder
    ok_floor = r[0] < 0.35                                # 10ms captures little
    ok_band = 0.05 <= hi < 0.5                            # around the kernel half-mass
    sess_i = hp.propagation_demo_sessions(n_days=2, n_bars=30000, delay_ms=200, seed=1,
                                          instantaneous=True)
    _p2, summ_i, meta_i = hp.profile_ladder(sess_i, intervals=LADDER, n_levels=3)
    hi_i = meta_i["half_impact_spy_from_es_s"]
    r_i = summ_i["lambda_spy_from_es_ratio"].to_numpy(float)
    ok_inst = r_i[0] > 0.6 and hi_i <= 0.05               # already there at the floor
    print("(1) delayed DGP (200ms kernel): ratio %s, half-impact %.3fs"
          % (np.round(r, 3).tolist(), hi))
    print("    monotone rise (%s), little at 10ms (%s), horizon in [0.05, 0.5) (%s)"
          % (ok_mono, ok_floor, ok_band))
    print("    instantaneous DGP: ratio(10ms)=%.2f, half-impact %.3fs -> at the floor (%s)"
          % (r_i[0], hi_i, ok_inst))
    return bool(ok_mono and ok_floor and ok_band and ok_inst)


def check_half_impact_unit():
    dt = np.array([0.01, 0.1, 1.0])
    # crossing between 0.1 and 1.0 with log-interpolation: ratio 0.3 -> 0.7 crosses
    # 0.5 halfway in ratio-space => exp(log(0.1) + 0.5*(log(1.0)-log(0.1)))
    h = hp.half_impact_horizon(dt, np.array([0.1, 0.3, 0.7]))
    want = float(np.exp(np.log(0.1) + 0.5 * (np.log(1.0) - np.log(0.1))))
    ok_interp = abs(h - want) < 1e-12
    ok_floor = hp.half_impact_horizon(dt, np.array([0.6, 0.8, 1.0])) == 0.01
    ok_nan = not np.isfinite(hp.half_impact_horizon(dt, np.array([0.1, 0.2, 0.3])))
    h_drop = hp.half_impact_horizon(dt, np.array([np.nan, 0.3, 0.7]))
    ok_drop = np.isfinite(h_drop) and h_drop == hp.half_impact_horizon(
        dt[1:], np.array([0.3, 0.7]))
    print("(2) half-impact: log-interpolation exact (%s), at-floor -> finest rung (%s), "
          "never-crossing -> NaN (%s), non-finite rungs dropped (%s)"
          % (ok_interp, ok_floor, ok_nan, ok_drop))
    return bool(ok_interp and ok_floor and ok_nan and ok_drop)


def check_frequency_scaling():
    sess = hp.propagation_demo_sessions(n_days=1, n_bars=12000, seed=3)
    per_day, _s, _m = hp.profile_ladder(sess, intervals=("10ms", "100ms", "1s"), n_levels=3)
    import cross_asset_pd_liquidity as ca
    ok = True
    for iv, dt in (("10ms", 0.01), ("100ms", 0.1), ("1s", 1.0)):
        fc = ca.frequency_defaults(dt=dt)
        sub = per_day[per_day.interval == iv]
        ok &= len(sub) > 0 and (sub["n_lags"] == fc["n_lags"]).all() \
            and (sub["min_rest_steps"] == fc["min_rest_steps"]).all()
    lags = per_day.groupby("interval")["n_lags"].first().to_dict()
    print("(3) per-rung settings follow frequency_defaults (n_lags: %s) : %s" % (lags, ok))
    return bool(ok)


def check_runner():
    import run_horizon_profile as rhp
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rhp.main(["--source", "demo", "--n-demo", "1", "--n-demo-bars", "12000",
                           "--intervals", "10ms,100ms,1s", "--out-dir", td, "--tag", "gate"])
        files = sorted(os.listdir(td))
        ok_files = (len([f for f in files if "per_day" in f]) == 1
                    and len([f for f in files if "summary" in f]) == 1)
        ok_cols = False
        if ok_files:
            s = pd.read_csv(os.path.join(td, [f for f in files if "summary" in f][0]))
            ok_cols = ({"interval", "lambda_spy_from_es", "lambda_spy_from_es_ratio",
                        "half_life_s"} <= set(s.columns)) and len(s) == 3
        out = buf.getvalue()
        ok_print = "HALF-IMPACT HORIZON" in out and "SPY <- ES" in out
    print("(4) runner: rc=%s, both CSVs written (%s), summary columns (%s), verdict "
          "printed (%s)" % (rc, ok_files, ok_cols, ok_print))
    return rc == 0 and ok_files and ok_cols and ok_print


def main():
    checks = [check_planted_horizon, check_half_impact_unit, check_frequency_scaling,
              check_runner]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("horizon-profile checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
