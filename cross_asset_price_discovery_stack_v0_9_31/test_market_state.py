#!/usr/bin/env python3
"""test_market_state.py -- the two regulatory constraints, and the honesty of their coverage.

`mt_product_status` carries both. Neither is a result; both are confounds:

  Rule 201 (SSR)   After a 10% decline a covered security cannot have short sales EXECUTED or
                   DISPLAYED at or below the national best bid, for that day and the next. The
                   constraint is one-sided by construction -- sell side yes, buy side no -- and the
                   paper's Tables 5 and 7 count SIGNED order flow. A restricted day therefore has
                   mechanically asymmetric flow that looks exactly like directional tandem trading.

  LULD bands      A quote may not be displayed outside the band. Approaching one changes quoted
                  spread, depth and cancellation behaviour because it must, not because anyone
                  learned anything -- and on a volatility day the mid spends real time near a band.

The coverage story matters as much as the values. On 2020-03-09 SPY, `shortsaleindicator` is
populated on every venue row while the three LULD fields are empty on every row: the bands come
from the SIP, not the direct venue feeds. A control that is silently all-NaN is WORSE than no
control, because the regression still runs and the coefficient means nothing. So the coverage is
measured and reported rather than assumed.

Run: python test_market_state.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import market_state as ms
import mstbook_loader as ml

TZ = "America/New_York"


def _status(rows):
    """rows = [(ts, ssr, lo, hi, ind)] -> an mt_product_status-shaped frame."""
    return pd.DataFrame({"f": ["a"] * len(rows),
                         "shortsaleindicator": [r[1] for r in rows],
                         "luldlowerlimit": [r[2] for r in rows],
                         "luldupperlimit": [r[3] for r in rows],
                         "limituplimitdownindicator": [r[4] for r in rows]},
                        index=pd.DatetimeIndex([r[0] for r in rows]).tz_localize(TZ))


def _book(day="2020-03-09", px=280.0, n=60):
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="min", tz=TZ)
    return pd.DataFrame({"SPY_bidprice_1": px - 0.005, "SPY_askprice_1": px + 0.005}, index=idx)


def check_selftest():
    ok = ms._selftest()
    print("(1) market_state self-test : %s\n" % ok)
    return ok


def check_real_2020_03_09_shape():
    """The verbatim shape of the uploaded rows: SSR populated, LULD empty."""
    st = _status([("2020-03-09 09:34:13", "ShortSaleRestrictionNotInEffect", "", "", ""),
                  ("2020-03-09 09:49:13", "ShortSaleRestrictionNotInEffect", "", "", "")])
    cov = ms.coverage(st)
    out = ms.attach_market_state(_book(), st)
    rep = ms.summarize(out)
    ssr_ok = cov["ssr"] == 1.0 and rep["ssr_frac_on"] == 0.0 and not rep["ssr_any"]
    luld_absent = cov["luld_upper"] == 0.0 and rep["luld_known"] == 0.0
    cols = all(c in out.columns for c in ("SPY_ssr", "SPY_luld_lower", "SPY_luld_upper",
                                          "SPY_luld_band_bps", "SPY_dist_upper_bps",
                                          "SPY_dist_lower_bps", "SPY_luld_binding"))
    all_nan = not np.isfinite(out["SPY_luld_band_bps"].to_numpy(float)).any()
    warns = "do NOT use them as a control" in ms.describe(rep)
    ok = ssr_ok and luld_absent and cols and all_nan and warns
    print("(2) the real 2020-03-09 shape: SSR known on 100%% and OFF all session=%s; LULD absent=%s"
          % (ssr_ok, luld_absent))
    print("    all seven columns are still emitted (%s) with the band ones all-NaN (%s), and the "
          "summary refuses to call an all-NaN column a control (%s) : %s"
          % (cols, all_nan, warns, ok))
    return ok


def check_ssr_on_is_flagged_everywhere():
    """A restricted session must be visible in the frame, the summary, and the gate table."""
    st = _status([("2020-03-12 09:30:00", "ShortSaleRestrictionInEffect", "", "", "")])
    out = ms.attach_market_state(_book("2020-03-12"), st)
    rep = ms.summarize(out)
    on_frame = float(out["SPY_ssr"].mean()) == 1.0
    on_summary = rep["ssr_any"] and rep["ssr_frac_on"] == 1.0
    text = ms.describe(rep)
    explains = "confound" in text and "Tables 5 and 7" in text

    import qc_frames as qf
    out.attrs["market_state"] = rep
    tbl = qf.qc_sessions([("2020-03-12", "volatile", out)])
    in_gate = "ssr_frac" in tbl.columns and float(tbl["ssr_frac"].iloc[0]) == 1.0

    ok = on_frame and on_summary and explains and in_gate
    print("(3) an SSR session is flagged on the frame (%s), in the summary (%s), with the confound "
          "named (%s), and in the qc_frames gate table (%s) : %s"
          % (on_frame, on_summary, explains, in_gate, ok))
    return ok


def check_unknown_is_not_zero():
    """The dangerous default: a missing SSR value must be NaN, never 'not restricted'."""
    st = _status([("2020-03-09 09:30:00", "", "", "", "")])
    out = ms.attach_market_state(_book(), st)
    rep = ms.summarize(out)
    nan_not_zero = not np.isfinite(out["SPY_ssr"].to_numpy(float)).any()
    reported = rep["ssr_known"] == 0.0
    text = ms.describe(rep)
    refuses = "NOT REPORTED" in text and "do not treat its absence as 'unrestricted'" in text
    ok = nan_not_zero and reported and refuses
    print("(4) an unknown SSR is NaN, not 0 (%s); coverage says 0%% known (%s); the text refuses to "
          "read absence as 'unrestricted' (%s) : %s" % (nan_not_zero, reported, refuses, ok))
    return ok


def check_band_geometry():
    """When bands ARE present, the derived distances must be right and binding must trigger."""
    st = _status([("2020-03-09 09:30:00", "ShortSaleRestrictionNotInEffect", "266.00", "294.00", "")])
    out = ms.attach_market_state(_book(px=280.0), st)
    mid = 280.0
    width = (294.0 - 266.0) / mid * 1e4
    up = (294.0 - mid) / mid * 1e4
    w_ok = abs(float(out["SPY_luld_band_bps"].iloc[0]) - width) < 1.0
    u_ok = abs(float(out["SPY_dist_upper_bps"].iloc[0]) - up) < 1.0
    not_binding = float(out["SPY_luld_binding"].iloc[0]) == 0.0

    near = ms.attach_market_state(_book(px=293.999), st)
    binding = float(near["SPY_luld_binding"].iloc[0]) == 1.0
    rep = ms.summarize(near)
    known = rep["luld_known"] == 1.0 and np.isfinite(rep["luld_band_bps_median"])
    ok = w_ok and u_ok and not_binding and binding and known
    print("(5) band 266-294 around a 280 mid: width %.0f bps (%s), distance-to-upper %.0f bps (%s), "
          "not binding (%s)" % (out["SPY_luld_band_bps"].iloc[0], w_ok,
                                out["SPY_dist_upper_bps"].iloc[0], u_ok, not_binding))
    print("    an ask at the band IS binding (%s) and coverage reports 100%% known (%s) : %s"
          % (binding, known, ok))
    return ok


def check_no_lookahead():
    """State is carried forward only. A change at 11:00 must not colour 10:00."""
    st = _status([("2020-03-09 09:30:00", "ShortSaleRestrictionNotInEffect", "", "", ""),
                  ("2020-03-09 11:00:00", "ShortSaleRestrictionInEffect", "", "", "")])
    out = ms.attach_market_state(_book(n=180), st)          # 09:30 .. 12:29 by minute
    before = out.loc[out.index < pd.Timestamp("2020-03-09 11:00", tz=TZ), "SPY_ssr"]
    after = out.loc[out.index >= pd.Timestamp("2020-03-09 11:00", tz=TZ), "SPY_ssr"]
    ok = float(before.max()) == 0.0 and float(after.min()) == 1.0
    print("(6) no look-ahead: SSR is 0 before the 11:00 change and 1 from it onward : %s" % ok)
    return ok


def main():
    checks = [check_selftest, check_real_2020_03_09_shape, check_ssr_on_is_flagged_everywhere,
              check_unknown_is_not_zero, check_band_geometry, check_no_lookahead]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("market-state checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
