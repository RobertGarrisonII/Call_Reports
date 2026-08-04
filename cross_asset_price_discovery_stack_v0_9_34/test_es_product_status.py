#!/usr/bin/env python3
"""test_es_product_status.py -- the futures status stream, and four ways it breaks equity assumptions.

Everything in market_halts.py and market_state.py was built against SPY: thirteen venues, an SSR
flag, empty band columns. The CME stream for ES violates all three of those assumptions, and each
violation produced a WRONG NUMBER rather than an error -- which is the dangerous kind.

The rows below are verbatim from
    mstwx-lakequery --mtype mt_product_status --product ESZ4 --date 20241218 --source futures
    mstwx-lakequery --mtype mt_product_status --product ESH0 --date 20200312 --source futures

  (1) CME DOES publish price limits.  563025 / 647725 -> 5630.25 / 6477.25 index points. The claim
      that "the band columns are empty because they come from the SIP" is true of the direct EQUITY
      feeds and false of the futures feed. The futures leg has a usable band control today.

  (2) INT64_MAX is the "no limit this side" sentinel.  9223372036.8547758070 is 2^63-1 scaled.
      Parsed as a price it is finite, so nothing raised: it turned the ES band width into
      1.5e+10 bps -- a plausible-looking column in a regression table that is pure sentinel. On
      2020-03-12 it is the UPPER limit for most of the session while the lower ratchets down: the
      contract is limit-DOWN constrained on one side only, which is the actual economics.

  (3) `haltreason` is a status-reason field, not a halt flag.  ESZ4 carries 18 `GroupSchedule` and
      19 `MarketEvent` rows -- routine session bookkeeping (`NoEvent`, `NoCancel`,
      `ChangeOfTradingSessionResetStatistics`). Read as halts they produced spans of 843 s, 5,914 s,
      43,910 s and 20,055 s on days the future never stopped trading. Halts are now WHITELISTED, and
      that direction is deliberate: a false halt EXCUSES crossing and can hide a replay fault, while
      a missed halt merely flags a session that turns out to be fine.

  (4) CME is ONE venue.  The market-wide quorum (`min_venues=2`) that stops a single equity venue's
      hour-long regulatory status from excusing a crossed consolidated book suppressed EVERY futures
      halt, because there is no second futures venue to agree. Quorum is now capped at the number of
      venues that publish status at all.

The payoff for (3) and (4) together is a real event: 2020-03-12 09:36:45.103 -> 09:36:51.478 ET,
6.38 s of `SuspendedBySurveillance` on ESH0 -- a CME Velocity Logic pause -- INSIDE the 09:35:44
SPY circuit-breaker window. The futures stopped matching while the equity leg was already halted.
That is a cross-asset event this paper is about, and until this test it was invisible three times
over: no ES status fetch, no quorum, and a 30 s duration floor.

Run: python test_es_product_status.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import market_halts as mh
import market_state as ms

TZ = "America/New_York"
SENTINEL = "9223372036.8547758070000000000"
SCALE = 0.01                       # CME integer-hundredths -> index points, as in cfg["futures_scale"]

# ── verbatim rows ────────────────────────────────────────────────────────────────────────────────
# (receipttimestamp ns, haltreason, tradingevent, luldlowerlimit, luldupperlimit)
ESZ4_20241218 = [
    (1734475098688597559, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734475500052317118, "GroupSchedule", "ChangeOfTradingSessionResetStatistics", "", ""),
    (1734476342951267219, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734476370005854639, "GroupSchedule", "NoCancel", "", ""),
    (1734476400002707769, "GroupSchedule", "NoEvent", "", ""),
    (1734476400002978853, "GroupSchedule", "NoEvent", "", ""),
    (1734476522911197299, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734480081111755108, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734528480930013334, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734528720908560200, "", "", "563025.0000000000000000000", "647725.0000000000000000000"),
    (1734532199371553264, "", "", "563025.0000000000000000000", SENTINEL),
    (1734548099911817221, "", "", "563025.0000000000000000000", SENTINEL),
    (1734548520896277595, "", "", "563025.0000000000000000000", SENTINEL),
    (1734553499182595910, "", "", "484375.0000000000000000000", SENTINEL),
    (1734555612437603213, "", "", "484375.0000000000000000000", SENTINEL),
    (1734555654395763743, "", "", "544325.0000000000000000000", "629025.0000000000000000000"),
    (1734556207882021656, "", "", "544325.0000000000000000000", "629025.0000000000000000000"),
    (1734559200002425098, "GroupSchedule", "NoEvent", "", ""),
]

ESH0_20200312 = [
    (1583962230010888785, "", "", "259400.0000000000000000000", "288200.0000000000000000000"),
    (1583962916753564201, "", "", "260100.0000000000000000000", "287500.0000000000000000000"),
    (1583963100012217305, "GroupSchedule", "ChangeOfTradingSessionResetStatistics", "", ""),
    (1583963970015813261, "GroupSchedule", "NoCancel", "", ""),
    (1583964000006094582, "GroupSchedule", "NoEvent", "", ""),
    (1583964000007066600, "GroupSchedule", "NoEvent", "", ""),
    (1583969014422446303, "", "", "260100.0000000000000000000", "287500.0000000000000000000"),
    (1583975592940798453, "MarketEvent", "NoEvent", "", ""),
    (1583975602932916790, "MarketEvent", "NoEvent", "", ""),
    (1583975602933762758, "MarketEvent", "NoEvent", "", ""),
    (1584016096384295135, "MarketEvent", "NoEvent", "", ""),
    (1584016106385610811, "MarketEvent", "NoEvent", "", ""),
    (1584016106386569239, "MarketEvent", "NoEvent", "", ""),
    (1584016106390577808, "MarketEvent", "NoEvent", "", ""),
    (1584016111394328996, "MarketEvent", "NoEvent", "", ""),
    (1584016111394770853, "MarketEvent", "NoEvent", "", ""),
    (1584019500073257863, "MarketEvent", "NoEvent", "", ""),
    (1584019503095850497, "", "", "254650.0000000000000000000", SENTINEL),
    (1584019800004739187, "GroupSchedule", "NoEvent", "", ""),
    (1584019800007186762, "GroupSchedule", "NoEvent", "", ""),
    (1584020205102769650, "SuspendedBySurveillance", "NoEvent", "", ""),          # 09:36:45.103 ET
    (1584020211477875642, "", "", "238200.0000000000000000000", SENTINEL),        # 09:36:51.478 ET
    (1584021044003233589, "GroupSchedule", "NoEvent", "", ""),
    (1584021044006974078, "GroupSchedule", "NoEvent", "", ""),
    (1584021044010713696, "MarketEvent", "NoEvent", "", ""),
    (1584021049033866450, "MarketEvent", "NoEvent", "", ""),
    (1584021049034446649, "MarketEvent", "NoEvent", "", ""),
    (1584021049046041950, "MarketEvent", "NoEvent", "", ""),
    (1584021054056663475, "MarketEvent", "NoEvent", "", ""),
    (1584021054056976854, "MarketEvent", "NoEvent", "", ""),
    (1584021056962071374, "MarketEvent", "NoEvent", "", ""),
    (1584021061968626134, "MarketEvent", "NoEvent", "", ""),
    (1584021061969303704, "MarketEvent", "NoEvent", "", ""),
    (1584041099230099976, "", "", "219000.0000000000000000000", SENTINEL),
    (1584043141549610584, "", "", "219000.0000000000000000000", SENTINEL),
    (1584043247345735400, "", "", "233250.0000000000000000000", "260650.0000000000000000000"),
    (1584043261443167649, "", "", "233250.0000000000000000000", "260650.0000000000000000000"),
    (1584044100001144453, "GroupSchedule", "NoEvent", "", ""),
    (1584044970103016063, "GroupSchedule", "NoCancel", "", ""),
    (1584045000003873251, "GroupSchedule", "NoEvent", "", ""),
    (1584045000009093448, "GroupSchedule", "NoEvent", "", ""),
    (1584046800005268402, "GroupSchedule", "NoEvent", "", ""),
]


def _frame(rows):
    """-> an mt_product_status-shaped frame indexed by receipt time in ET, as the loader builds it."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], unit="ns", tz="UTC") for r in rows]).tz_convert(TZ)
    return pd.DataFrame({"f": ["cme_globex30_cme"] * len(rows),      # ONE venue -- that is the point
                         "haltreason": [r[1] for r in rows],
                         "tradingevent": [r[2] for r in rows],
                         "shortsaleindicator": ["\\N"] * len(rows),  # futures have no Rule 201
                         "luldlowerlimit": [r[3] for r in rows],
                         "luldupperlimit": [r[4] for r in rows],
                         "limituplimitdownindicator": [""] * len(rows)}, index=idx)


def check_cme_publishes_price_limits():
    """(1) The correction: the band columns are NOT universally empty. ES has them."""
    st = _frame(ESZ4_20241218)
    cov = ms.coverage(st)
    ser = ms.state_series(st)
    lo = ser["luld_lower"].dropna().unique() * SCALE
    known = cov["luld_lower"] > 0.5
    scaled = abs(float(lo.max()) - 5630.25) < 1e-9 and abs(float(lo.min()) - 4843.75) < 1e-9
    ok = known and scaled
    print("(1) CME publishes price limits for ESZ4: lower populated on %.0f%% of rows (%s); "
          "563025 -> %.2f index points at scale %.2f (%s) : %s"
          % (100 * cov["luld_lower"], known, float(lo.max()), SCALE, scaled, ok))
    print("    (the equity feeds leave these empty on all four SPY dates checked -- that is a "
          "property of those feeds, not of the schema)")
    return ok


def check_int64_sentinel_is_not_a_price():
    """(2) 9223372036.85 means 'no limit this side'. Unhandled it is a finite, wrong control."""
    st = _frame(ESZ4_20241218)
    ser = ms.state_series(st)
    hi = ser["luld_upper"].to_numpy(float)
    raw = pd.to_numeric([r[4] for r in ESZ4_20241218 if r[4]], errors="coerce")
    present = bool((raw > 1e9).any())
    nan_now = not np.isfinite(hi[[i for i, r in enumerate(ESZ4_20241218) if r[4] == SENTINEL]]).any()
    finite_kept = np.isfinite(hi[0]) and abs(float(hi[0]) * SCALE - 6477.25) < 1e-9

    # The ES book is already in index points (price_scale is applied at reconstruction), so the
    # bands must be too. The grid spans the full Globex day: ESZ4's upper limit is the sentinel
    # right through the 2024-12-18 RTH session and only becomes finite again at 16:00:54, so an
    # RTH-only window would make this check pass vacuously on an all-NaN column.
    grid = pd.date_range("2024-12-17 18:00", "2024-12-18 17:00", freq="min", tz=TZ)
    book = pd.DataFrame({"ES_bidprice_1": 6040.00, "ES_askprice_1": 6040.25}, index=grid)
    out = ms.attach_market_state(book, st, asset="ES", price_scale=SCALE)
    band = pd.to_numeric(out["ES_luld_band_bps"], errors="coerce").to_numpy(float)
    fin = band[np.isfinite(band)]
    # the bug: (9223372036.85 - 5630.25) / 6040.125 * 1e4 ~= 1.5e10 bps
    sane = fin.size > 0 and float(np.nanmax(np.abs(fin))) < 1e5
    # non-vacuous both ways: the two-sided stretches give a real width, the sentinel stretch NaN
    unbounded_is_nan = 0 < float(np.isfinite(band).mean()) < 1
    ok = present and nan_now and finite_kept and sane and unbounded_is_nan
    print("(2) the INT64_MAX sentinel is present in the real rows (%s), maps to NaN rather than a "
          "price (%s), and does not take the finite limits with it (%s)"
          % (present, nan_now, finite_kept))
    print("    widest surviving band %.0f bps over the %.0f%% of the day that HAS a two-sided limit "
          "(was 1.5e+10 bps -- finite, plausible in a table, and pure sentinel) : %s"
          % (float(np.nanmax(np.abs(fin))) if fin.size else float("nan"),
             100 * float(np.isfinite(band).mean()), ok))
    return ok


def check_routine_session_notices_are_not_halts():
    """(3) GroupSchedule / MarketEvent are bookkeeping. ESZ4 never stopped trading on 2024-12-18."""
    st = _frame(ESZ4_20241218)
    n_gs = sum(1 for r in ESZ4_20241218 if r[1] == "GroupSchedule")
    res = mh.windows_from_status(st, tz=TZ, min_venues=2)
    none_now = res["windows"] == [] and res["venue_only"] == []
    # what the old blacklist did: anything non-null is a halt
    spans = []
    rows = sorted((pd.Timestamp(r[0], unit="ns", tz="UTC"), bool(r[1])) for r in ESZ4_20241218)
    start = None
    for ts, h in rows:
        if h and start is None:
            start = ts
        elif (not h) and start is not None:
            spans.append((ts - start).total_seconds()); start = None
    was_bad = bool(spans) and max(spans) > 800
    not_unknown = "GroupSchedule" not in res["unknown_reasons"]
    ok = none_now and was_bad and not_unknown
    print("(3) ESZ4 2024-12-18: %d GroupSchedule row(s) -> %d halt window(s) (%s)"
          % (n_gs, len(res["windows"]), none_now))
    print("    a blacklist would have made them %s s of 'halt' on a day the future never stopped "
          "(%s); they are classified, not merely unrecognized (%s) : %s"
          % (", ".join("%.0f" % s for s in sorted(spans, reverse=True)[:3]), was_bad, not_unknown, ok))
    return ok


def check_single_venue_quorum_admits_the_cme_pause():
    """(4) One venue is the whole futures market, so quorum must cap at what exists."""
    st = _frame(ESH0_20200312)
    res = mh.windows_from_status(st, tz=TZ, min_venues=2)      # the EQUITY default, unchanged
    found = len(res["windows"]) == 1
    if not found:
        print("(4) FAILED: %d window(s) %s" % (len(res["windows"]), res["windows"]))
        return False
    a, b = res["windows"][0]
    dur = (b - a).total_seconds()
    t_ok = a.strftime("%H:%M:%S") == "09:36:45" and b.strftime("%H:%M:%S") == "09:36:51"
    d_ok = abs(dur - 6.375) < 0.01
    why = res["reasons"] == {"SuspendedBySurveillance"}
    nothing_stranded = res["venue_only"] == []
    ok = found and t_ok and d_ok and why and nothing_stranded
    print("(4) ESH0 2020-03-12: %s -> %s (%.2f s, %s), found under the equity default min_venues=2 "
          "because quorum caps at the 1 venue that publishes (%s)"
          % (a.strftime("%H:%M:%S.%f")[:-3], b.strftime("%H:%M:%S.%f")[:-3], dur,
             ", ".join(sorted(res["reasons"])), nothing_stranded))
    print("    a CME Velocity Logic pause is 5-10 s by design, so the old 30 s floor discarded it; "
          "the floor is 1 s and the REASON whitelist does the filtering : %s" % ok)
    return ok


def check_the_pause_lands_inside_the_spy_halt():
    """The reason any of this matters: both legs stopped, 61 s apart, on the same morning."""
    es = mh.windows_from_status(_frame(ESH0_20200312), tz=TZ)["windows"][0]
    spy = mh.halt_windows("2020-03-12")[0]
    inside = spy[0] <= es[0] <= spy[1] and spy[0] <= es[1] <= spy[1]
    lag = (es[0] - spy[0]).total_seconds()
    ok = inside and abs(lag - 61.1) < 1.0
    print("(5) SPY MWCB %s-%s (900 s); ESH0 %s-%s (6.4 s), starting %.1f s into it -- the futures "
          "stopped matching while the equity leg was already halted : %s"
          % (spy[0].strftime("%H:%M:%S"), spy[1].strftime("%H:%M:%S"),
             es[0].strftime("%H:%M:%S"), es[1].strftime("%H:%M:%S"), lag, ok))
    print("    a pair estimate needs BOTH legs live, so the union of the two is what excludes "
          "snapshots; each BOOK is still judged against its own halt (a CME pause cannot excuse a "
          "crossed SPY top).")
    return ok


def check_limit_down_ratchet_is_visible():
    """The economics the sentinel was hiding: one-sided limits ratcheting through the crash."""
    ser = ms.state_series(_frame(ESH0_20200312))
    lo = (ser["luld_lower"].dropna() * SCALE).tolist()
    steps = [round(v, 2) for i, v in enumerate(lo) if i == 0 or v != lo[i - 1]]
    expect = [2594.00, 2601.00, 2546.50, 2382.00, 2190.00, 2332.50]
    ratchet = steps == expect
    hi = ser.loc[ser["luld_lower"].notna(), "luld_upper"]
    one_sided = int(hi.isna().sum()) == 4 and int(hi.notna().sum()) == 5
    ok = ratchet and one_sided
    print("(6) the ESH0 lower limit through the crash: %s (%s)"
          % (" -> ".join("%.2f" % s for s in steps), ratchet))
    print("    with the upper unbounded on %d of %d band rows -- the contract is limit-DOWN "
          "constrained on one side only, which is the actual economics and was previously a "
          "1.5e+10 bps 'width' : %s" % (int(hi.isna().sum()), len(hi), ok))
    return ok


def check_qc_judges_each_book_against_its_own_halt():
    """A halt on one leg must not excuse a crossed top on the other."""
    import mstbook_loader as ml
    grid = pd.date_range("2020-03-12 09:30:00", periods=1200, freq="s", tz=TZ)
    df = pd.DataFrame({"SPY_bidprice_1": 250.00, "SPY_askprice_1": 250.01,
                       "ES_bidprice_1": 2500.00, "ES_askprice_1": 2500.25}, index=grid)
    # SPY crosses for 100 s, entirely inside the 6.4 s ES pause's *day* but nowhere near it
    m = (grid >= pd.Timestamp("2020-03-12 09:40:00", tz=TZ)) & \
        (grid < pd.Timestamp("2020-03-12 09:41:40", tz=TZ))
    df.loc[m, "SPY_bidprice_1"] = 250.05
    df.attrs["halt_windows_ES"] = [("2020-03-12T09:36:45.102769-04:00",
                                    "2020-03-12T09:36:51.477875-04:00")]
    df.attrs["halt_windows_SPY"] = []
    df.attrs["halt_windows"] = df.attrs["halt_windows_ES"]
    qc = ml.session_qc(df, date="2020-03-12")
    flagged = (not qc["ok"]) and any(r.startswith("SPY: top is CROSSED") for r in qc["reasons"])
    # .102769 -> .477875 on a 1 s grid = the six snapshots 09:36:46 .. 09:36:51
    es_halt = qc["ES"]["halt_snapshots"] == 6 and qc["SPY"]["halt_snapshots"] == 0
    ok = flagged and es_halt
    print("(7) 100 s of crossed SPY on a day with a 6.4 s ES pause: SPY is still FAILED (%s), and "
          "the halt snapshots land on the ES leg only (ES=%d, SPY=%d) : %s"
          % (flagged, qc["ES"]["halt_snapshots"], qc["SPY"]["halt_snapshots"], ok))
    return ok


def main():
    checks = [check_cme_publishes_price_limits, check_int64_sentinel_is_not_a_price,
              check_routine_session_notices_are_not_halts,
              check_single_venue_quorum_admits_the_cme_pause,
              check_the_pause_lands_inside_the_spy_halt, check_limit_down_ratchet_is_visible,
              check_qc_judges_each_book_against_its_own_halt]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("ES product-status checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
