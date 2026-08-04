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

  (5) THE FLAG IS NOT THE DURATION.  `haltreason` marks the stop to the millisecond and then clears
      while the market stays down. On 2020-03-18 ESM0 it is set 3 ms after the last trade and
      cleared 5.83 s later -- and then nothing trades for 817.3 s. Reading the clear as a resume
      understated that halt by 140x, and the two readings support OPPOSITE claims about which
      market leads. Halt ends now come from the resumption of trading (`activity=`), with the
      unextended spans kept under `flag_windows`. The equity feeds do not need this: their MWCB
      spans clear at exactly 900.0 s.

The payoff for (3), (4) and (5) together is a real event, on a day already in the volatile panel:

    12:56:11              SPY MWCB Level 1 halt begins
    12:56:11-12:57:39.7   ES trades on, 3,927 contracts in 88.7 s -- the ONLY price venue
    12:57:39.716          ES haltreason set, 3 ms after its last trade
    13:11:11 / 13:11:17   SPY reopens; ES follows 6.0 s later

So the object is 88.7 seconds of solitude, bounded at both ends -- not a 15-minute divergence.
Until these checks it was invisible four times over: no ES status fetch, no venue quorum, a 30 s
duration floor, and a resume read off a flag that had already cleared.

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


# ── the roll file: ESH0 vs ESM0 on 2020-03-16 and 2020-03-18 ────────────────────────────────────
# (receipt ns, seq, exchange ns, product, haltreason, lower, upper) -- verbatim, both contracts.
ROLL = [
    (1584307805348763972, 1821, 1584307776613206983, "ESH0", "", "256750", "283850"),
    (1584307805348763972, 1821, 1584307757763848469, "ESM0", "", "255550", "282650"),
    (1584309570020704105, 5980, 1584309570000000000, "ESH0", "GroupSchedule", "", ""),
    (1584309570020704105, 5980, 1584309570000000000, "ESM0", "GroupSchedule", "", ""),
    (1584365102086021870, 788789, 1584365102085316903, "ESH0", "", "251350", SENTINEL),
    (1584365102086021870, 788789, 1584365102085316903, "ESM0", "", "250150", SENTINEL),
    (1584365454973158575, 796560, 1584365454968025073, "ESH0", "SuspendedBySurveillance", "", ""),
    (1584365454973158575, 796560, 1584365454968025073, "ESM0", "SuspendedBySurveillance", "", ""),
    (1584365462246928960, 796828, 1584365462246236823, "ESH0", "", "235100", SENTINEL),
    (1584365462246928960, 796828, 1584365462246236823, "ESM0", "", "233900", SENTINEL),
]
# 2020-03-18, the second pause of the day (the one that lands in the equity halt)
ROLL_0318 = [
    (1584550659715872667, 68805988, 1584550659714102239, "ESH0", "SuspendedBySurveillance", "", ""),
    (1584550659715872667, 68805988, 1584550659714102239, "ESM0", "SuspendedBySurveillance", "", ""),
    (1584550665549165107, 68806421, 1584550665548547789, "ESH0", "", "220100", SENTINEL),
    (1584550665549165107, 68806421, 1584550665548547789, "ESM0", "", "219100", SENTINEL),
]


def _roll_frame(rows, product):
    r = [x for x in rows if x[3] == product]
    idx = pd.DatetimeIndex([pd.Timestamp(x[0], unit="ns", tz="UTC") for x in r]).tz_convert(TZ)
    return pd.DataFrame({"f": ["cme_globex30_cme"] * len(r), "haltreason": [x[4] for x in r],
                         "shortsaleindicator": ["\\N"] * len(r),
                         "luldlowerlimit": [x[5] for x in r], "luldupperlimit": [x[6] for x in r],
                         "limituplimitdownindicator": [""] * len(r)}, index=idx)


def check_sequence_is_channel_level_not_product_level():
    """Hard evidence for a thing the crossed-book diagnosis had to infer.

    `sequencenumber` is a CHANNEL counter, not a per-product one. ESH0 and ESM0 return rows carrying
    the SAME sequence number at the SAME receipt timestamp with DIFFERENT prices -- one CME packet
    carrying several instruments. A per-product fetch therefore sees a sparse subset of a
    channel-wide counter, so gaps in it are the OTHER products, not lost messages. `debug_crossing`
    CHECK 4 calls those gaps `not-ours` on exactly this reasoning; here it is demonstrated rather
    than argued.

    The same rows kill the other idea: ordering or pruning events by `sequencenumber` across a
    per-product fetch is meaningless, because the counter does not belong to the product."""
    shared = [(a, b) for a, b in zip(ROLL[::2], ROLL[1::2]) if a[1] == b[1] and a[0] == b[0]]
    all_shared = len(shared) == len(ROLL) // 2
    # ...and the payloads under one sequence number are genuinely different instruments
    px = [(a[5], b[5]) for a, b in shared if a[5]]
    differ = all(x != y for x, y in px) and len(px) >= 3
    # the exchange timestamp is PER INSTRUMENT inside the shared packet, and can be far older
    skew = max(abs(a[2] - b[2]) for a, b in shared) / 1e9
    ok = all_shared and differ and skew > 1.0
    print("(8) ESH0 and ESM0 share a sequence namespace: %d/%d rows carry the same sequencenumber "
          "at the same receipt timestamp (%s)" % (len(shared), len(ROLL) // 2, all_shared))
    print("    with different per-contract prices under it (%s), and per-instrument exchange "
          "timestamps up to %.1f s apart inside one packet (%s)" % (differ, skew, skew > 1.0))
    print("    -> a gap in a per-product sequence is the OTHER products, not a lost message, and the")
    print("       receipt clock is the only one that orders the packet itself : %s" % ok)
    return ok


def check_halts_are_group_level_not_contract_level():
    """Velocity Logic pauses the ES GROUP: both contracts stop at the same nanosecond.

    That is a useful robustness property -- the halt window does not depend on getting the front
    month right, so a roll ambiguity cannot silently move it."""
    h = mh.windows_from_status(_roll_frame(ROLL, "ESH0"), tz=TZ)["windows"]
    m = mh.windows_from_status(_roll_frame(ROLL, "ESM0"), tz=TZ)["windows"]
    same = len(h) == 1 and h == m
    dur = (h[0][1] - h[0][0]).total_seconds() if h else float("nan")
    ok = same and abs(dur - 7.27) < 0.01
    print("(9) 2020-03-16: ESH0 and ESM0 pause at the SAME instant (%s), %s-%s, %.2f s : %s"
          % (same, h[0][0].strftime("%H:%M:%S.%f")[:-3] if h else "-",
             h[0][1].strftime("%H:%M:%S.%f")[:-3] if h else "-", dur, ok))
    print("    the halt is a property of the ES group, so it survives a wrong front-month choice.")
    return ok


def check_es_halt_onset_lags_the_equity_halt():
    """The result: three MWCB days, three ES halts, each beginning ~1 minute into the equity halt.

    This check is about the ONSET only, and the durations it prints are FLAG durations -- the span
    over which `haltreason` is populated. Check (13) shows those understate the actual trading stop
    by 140x on the one day where we have the tape to measure it, so nothing here should be read as
    "the futures paused for six seconds". What the flag does give, to the millisecond, is when the
    futures STOPPED: on 2020-03-18 the flag is set 3 ms after ESM0's last trade.

    The lag is the finding. CME halts equity-index futures in coordination with the primary market,
    as its rules require, and +54 to +89 s across three days is that relay -- not a volatility
    threshold being crossed independently. The window before it, in which the futures are the only
    venue still matching, is check (14)."""
    es = {"2020-03-12": mh.windows_from_status(_frame(ESH0_20200312), tz=TZ)["windows"],
          "2020-03-16": mh.windows_from_status(_roll_frame(ROLL, "ESH0"), tz=TZ)["windows"],
          "2020-03-18": mh.windows_from_status(_roll_frame(ROLL_0318, "ESH0"), tz=TZ)["windows"]}
    lags, nested = [], 0
    for d, ws in es.items():
        rep = mh.cross_asset_summary(mh.halt_windows(d), ws)
        nested += rep["n_nested"]
        for o in rep["overlaps"]:
            lags.append(o["lag_s"])
            print("    %s  ES flag %s-%s (%.2fs) begins inside SPY %s-%s, +%.1fs in"
                  % (d, o["es"][0].strftime("%H:%M:%S"), o["es"][1].strftime("%H:%M:%S"),
                     o["es_seconds"], o["spy"][0].strftime("%H:%M:%S"),
                     o["spy"][1].strftime("%H:%M:%S"), o["lag_s"]))
    all_nested = nested == 3 and len(lags) == 3
    in_family = all(40.0 < x < 120.0 for x in lags)
    text = mh.describe_cross_asset(mh.cross_asset_summary(mh.halt_windows("2020-03-12"),
                                                          es["2020-03-12"]))
    says_union = "Exclude the UNION" in text and "own event onset" in text
    ok = all_nested and in_family and says_union
    print("(10) all three ES halts BEGIN inside the equity halt (%s), at +%.1f/%.1f/%.1f s -- a "
          "consistent relay lag, not a coincidence (%s)" % (all_nested, lags[0], lags[1], lags[2],
                                                            in_family))
    print("     (durations shown are FLAG spans; check (13) measures the real stop against the "
          "tape and it is 140x longer)")
    print("    and the summary says what to do about it: exclude the union, treat the pause as its "
          "own onset (%s) : %s" % (says_union, ok))
    return ok


def check_futures_only_pause_is_not_swallowed():
    """2020-03-18 also has a 2.09 s pause at 09:24:58 -- pre-open, with no equity halt anywhere near.

    It must come back as `es_only`, not be quietly dropped for failing to match a SPY window."""
    rows = [(1584537898649350889, 0, 0, "ESH0", "SuspendedBySurveillance", "", ""),
            (1584537900739027991, 0, 0, "ESH0", "", "235250", SENTINEL)]
    ws = mh.windows_from_status(_roll_frame(rows, "ESH0"), tz=TZ)["windows"]
    rep = mh.cross_asset_summary(mh.halt_windows("2020-03-18"), ws)
    kept = len(rep["es_only"]) == 1 and not rep["overlaps"]
    spy_alone = len(rep["spy_only"]) == 1              # the equity halt has no ES partner here
    txt = mh.describe_cross_asset(rep)
    named = "futures-only event" in txt and "ES matching throughout" in txt
    ok = kept and spy_alone and named
    print("(11) the 09:24:58 pre-open pause (2.09 s) has no equity halt to nest in: returned as "
          "es_only rather than dropped (%s), the unpartnered equity halt is reported too (%s), and "
          "both are named in words (%s) : %s" % (kept, spy_alone, named, ok))
    return ok


def check_flag_clear_is_not_a_resume():
    """The CME status flag marks the STOP to the millisecond and says nothing about the resume.

    Measured against ESM0's own cumulative-volume tape on 2020-03-18:

        last trade before      12:57:39.713
        haltreason SET         12:57:39.716    <- 3 ms later; the onset is exact
        haltreason CLEARED     12:57:45.549    <- 5.83 s; reads as a brief pause
        first trade after      13:11:17.008    <- 817.3 s with ZERO contracts traded

    Zero for 13.6 minutes, then 1,221 contracts in the first print. Trusting the clear excluded 6 s
    of a session in which 817 s of the futures leg had no market -- 811 snapshots of a stale book
    entering every pair estimate as though it were live. Worse, the two readings support opposite
    claims: flag-only says the futures traded through almost the whole equity halt, the tape says
    they were down for all but the first 89 seconds of it.

    The equity feeds do NOT have this problem -- their MWCB spans clear at exactly 900.0 s -- so the
    resume signal is a parameter, not a change of rule."""
    rows = [(1584550659715872667, 0, 0, "ESH0", "SuspendedBySurveillance", "", ""),
            (1584550665549165107, 0, 0, "ESH0", "", "220100", SENTINEL)]
    st = _roll_frame(rows, "ESH0")

    # a trade tape with the real hole in it: dense before 12:57:39.713, nothing until 13:11:17.008
    base = pd.Timestamp("2020-03-18 12:47:39.713", tz=TZ)
    before = pd.date_range(base, periods=6000, freq="100ms")            # 10 min at 10/s
    before = before[before <= pd.Timestamp("2020-03-18 12:57:39.713", tz=TZ)]
    after = pd.date_range(pd.Timestamp("2020-03-18 13:11:17.008", tz=TZ), periods=600, freq="100ms")
    act = before.append(after)

    flag = mh.windows_from_status(st, tz=TZ)["windows"][0]
    res = mh.windows_from_status(st, tz=TZ, activity=act)
    real = res["windows"][0]
    flag_s = (flag[1] - flag[0]).total_seconds()
    real_s = (real[1] - real[0]).total_seconds()
    understated = abs(flag_s - 5.83) < 0.01 and abs(real_s - 817.3) < 0.5
    ratio = real_s / flag_s
    kept = len(res["flag_windows"]) == 1 and res["flag_windows"][0] == flag
    ext = res["extended"][0] if res["extended"] else {}
    honest = ext.get("quiet_ratio", 0) > 1000 and ext.get("pre_halt_median_gap_s", 9) < 1.0

    # ...and the equity case must be left ALONE: its flag already spans the real halt
    spy_rows = [("2020-03-09 09:34:13", "xdp_arca_integrated", "MarketWideCircuitBreakerLevel1"),
                ("2020-03-09 09:34:13", "xdp_nyse_integrated", "MarketWideCircuitBreakerLevel1"),
                ("2020-03-09 09:49:13", "xdp_arca_integrated", ""),
                ("2020-03-09 09:49:13", "xdp_nyse_integrated", "")]
    spy_st = pd.DataFrame({"f": [r[1] for r in spy_rows], "haltreason": [r[2] for r in spy_rows]},
                          index=pd.DatetimeIndex([r[0] for r in spy_rows]).tz_localize(TZ))
    spy_act = pd.DatetimeIndex(["2020-03-09 09:34:12.900", "2020-03-09 09:49:13.100"]).tz_localize(TZ)
    spy_w = mh.windows_from_status(spy_st, tz=TZ, activity=spy_act)
    spy_dur = (spy_w["windows"][0][1] - spy_w["windows"][0][0]).total_seconds()
    # the reopening auction prints a moment after the flag clears, so a sub-second nudge is expected
    # and correct; what must NOT happen is the 140x rewrite the futures leg needs.
    untouched = abs(spy_dur - 900.0) < 1.0

    ok = understated and kept and honest and untouched
    print("(13) the CME flag clears after %.2f s but nothing trades for %.1f s -- understated %.0fx"
          % (flag_s, real_s, ratio))
    print("     the unextended span is still reported under flag_windows (%s), and the extension "
          "carries its own evidence: pre-halt median inter-trade gap %.3f s vs %.0f s of silence, "
          "ratio %.0f (%s)" % (kept, ext.get("pre_halt_median_gap_s", float("nan")),
                               ext.get("extra_seconds", float("nan")),
                               ext.get("quiet_ratio", float("nan")), honest))
    print("     the EQUITY case is left essentially alone -- its flag already spans the real halt, "
          "%.1f s vs 900.0 (%s) : %s" % (spy_dur, untouched, ok))
    return ok


def check_the_sole_venue_window():
    """What the corrected windows actually show: 88.7 s in which ES is the only price venue.

    Not a 15-minute divergence -- a short, sharp, bounded window. The equity market halts at
    12:56:11 and ES keeps matching for 88.7 s (3,927 contracts) before halting too; both reopen
    within 6 s of each other. That is the object worth an event study, and it only becomes visible
    once the ES window ends at the resumption of trading rather than at the flag's clear."""
    spy = mh.halt_windows("2020-03-18")[0]
    es = (pd.Timestamp("2020-03-18 12:57:39.713", tz=TZ), pd.Timestamp("2020-03-18 13:11:17.008", tz=TZ))
    rep = mh.cross_asset_summary([spy], [es])
    o = rep["overlaps"][0]
    sole = (es[0] - spy[0]).total_seconds()
    reopen_gap = (es[1] - spy[1]).total_seconds()
    nested = not o["nested"]          # ES now OUTLASTS the equity window, so it is no longer nested
    ok = abs(sole - 88.7) < 0.1 and abs(reopen_gap - 6.0) < 0.1 and abs(o["lag_s"] - 88.7) < 0.1 and nested
    print("(14) SPY halts %s; ES keeps trading %.1f s (sole price venue) then halts %s"
          % (spy[0].strftime("%H:%M:%S"), sole, es[0].strftime("%H:%M:%S.%f")[:-3]))
    print("     both reopen within %.1f s of each other (SPY %s, ES %s) -- so the corrected ES "
          "window OUTLASTS the equity one rather than nesting inside it (%s) : %s"
          % (reopen_gap, spy[1].strftime("%H:%M:%S"), es[1].strftime("%H:%M:%S.%f")[:-3], nested, ok))
    return ok


# ── measured against the ES trade tapes (mt_product_statistics cumulative `volume`) ──────────────
# (date, SPY halt, ES last trade, ES first trade after, flag span s) -- identical for BOTH contracts,
# because Velocity Logic stops the ES group rather than a contract.
TAPE_HALTS = [
    ("2020-03-16", ("09:30:01", "09:45:01"), "09:30:54.949", "09:45:01.012", 7.27),
    ("2020-03-18", ("12:56:11", "13:11:11"), "12:57:39.704", "13:11:17.008", 5.83),
]
# RTH 09:30-16:00 volume, from the same tapes
RTH_VOLUME = {("2020-03-16", "ESH0"): 1986076, ("2020-03-16", "ESM0"): 3027078,
              ("2020-03-18", "ESH0"): 732903,  ("2020-03-18", "ESM0"): 2605122}


def check_flag_understatement_across_days():
    """The 140x is not a one-day artifact. Two dates, two contracts, the same two orders of magnitude.

        date        flag span   actual stop   understated   ES resumes vs the equity reopen
        2020-03-16     7.27 s       846.06 s        116x    +0.012 s
        2020-03-18     5.83 s       817.30 s        140x    +6.0 s

    The onset is exact on both: the flag is set 24 ms and 11 ms after the respective last trades.
    It is only the CLEAR that is meaningless.

    2020-03-16 also cross-validates the equity side. `MWCB_HALTS` puts the SPY halt end at 09:45:01,
    derived from the SPY status tape; ES's first trade back is 09:45:01.012. Two independent feeds,
    12 ms apart, on a boundary that was hand-entered before either was checked."""
    rows, ok = [], True
    for d, (sa, sb), last, first, flag_s in TAPE_HALTS:
        A = pd.Timestamp(f"{d} {last}", tz=TZ); B = pd.Timestamp(f"{d} {first}", tz=TZ)
        spy = mh.halt_windows(d)[0]
        real = (B - A).total_seconds()
        rows.append((d, flag_s, real, real / flag_s, (A - spy[0]).total_seconds(),
                     (B - spy[1]).total_seconds()))
        ok &= real / flag_s > 100 and 40 < (A - spy[0]).total_seconds() < 120
        ok &= spy[0].strftime("%H:%M:%S") == sa and spy[1].strftime("%H:%M:%S") == sb
    print("(15) the flag/tape gap on every day we can measure:")
    for d, f, r, ratio, sole, reopen in rows:
        print("     %s  flag %5.2f s   actual stop %7.2f s   understated %3.0fx   "
              "sole-venue %.1f s   ES reopens %+.3f s" % (d, f, r, ratio, sole, reopen))
    # the cross-validation
    es_resume = pd.Timestamp("2020-03-16 09:45:01.012", tz=TZ)
    spy_end = mh.halt_windows("2020-03-16")[0][1]
    xval = abs((es_resume - spy_end).total_seconds()) < 0.05
    ok &= xval
    print("     2020-03-16: the ES tape resumes %.0f ms after the SPY halt end derived from the "
          "equity status tape -- two independent feeds agreeing on a hand-entered boundary (%s) : %s"
          % (1000 * (es_resume - spy_end).total_seconds(), xval, ok))
    return bool(ok)


def check_roll_is_settled_by_volume():
    """ESM0 is the front month on both dates -- `rollover_days=8` was right -- but 03-16 is split.

        date        ESH0 RTH vol   ESM0 RTH vol   front-month share
        2020-03-16     1,986,076      3,027,078   60.4%
        2020-03-18       732,903      2,605,122   78.0%

    On an ordinary day the front month is essentially all of the volume. During this roll week it is
    60-78%, so a single-contract ES leg misses 22-40% of the futures activity on two sessions that
    are IN the volatile panel. That is not a bug to fix by stitching -- the two contracts are
    different instruments with a 10-12 point carry spread, and splicing them would manufacture a
    price jump at the seam. It is a sample fact the paper has to state."""
    ok, shares = True, {}
    for d in ("2020-03-16", "2020-03-18"):
        h, m = RTH_VOLUME[(d, "ESH0")], RTH_VOLUME[(d, "ESM0")]
        shares[d] = m / (h + m)
        ok &= m > h                                   # ESM0 is front, as rollover_days=8 picks
    import mstbook_loader as ml
    picked = {d: ml.get_front_month_contract("ES", as_of_date=ml._parse_yyyymmdd(d.replace("-", "")))
              for d in shares}
    agrees = all(picked[d] == "ESM0" for d in shares)
    split = 0.55 < shares["2020-03-16"] < 0.65 and 0.75 < shares["2020-03-18"] < 0.80
    ok = bool(ok and agrees and split)
    print("(16) front month by RTH volume: %s"
          % ", ".join("%s ESM0 %.1f%%" % (d, 100 * s) for d, s in sorted(shares.items())))
    print("     the calendar rule picks %s -- it agrees with the tape (%s)"
          % (", ".join("%s=%s" % (d, c) for d, c in sorted(picked.items())), agrees))
    print("     but the roll week is SPLIT 60/40 and 78/22, so the single-contract ES leg misses "
          "22-40%% of futures volume on two volatile-panel sessions (%s)" % split)

    # ...and the extractor now says so. Distance must be measured in BOTH directions: 2020-03-16 is
    # four days PAST the March boundary, and a forward-only measure calls it 87 days from the June
    # roll -- silent on exactly the session that needs the warning.
    near = {d: ml.roll_window_days(ml._parse_yyyymmdd(d.replace("-", "")))
            for d in ("2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18", "2020-03-25")}
    flags = all(near[d] <= 7 for d in ("2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"))
    quiet = near["2020-03-25"] > 7
    ok = bool(ok and flags and quiet)
    print("     nearest roll boundary: %s -- all four MWCB dates are inside the roll week (%s) and "
          "an ordinary session a week later is not (%s) : %s"
          % (", ".join("%s=%dd" % (d, n) for d, n in sorted(near.items())), flags, quiet, ok))
    return ok


def check_calendar_spread_sanity():
    """A cheap cross-check that both contracts' limits are real: their difference is the carry."""
    h = ms.state_series(_roll_frame(ROLL, "ESH0"))["luld_lower"].dropna() * SCALE
    m = ms.state_series(_roll_frame(ROLL, "ESM0"))["luld_lower"].dropna() * SCALE
    sp = (h.to_numpy() - m.to_numpy())
    tight = np.allclose(sp, 12.00, atol=1e-9)
    ok = tight and len(sp) == 3
    print("(12) ESH0 - ESM0 price limits on 2020-03-16 = %s index points across %d paired rows -- a "
          "constant calendar spread, so both contracts' limits are genuine and comparable : %s"
          % (", ".join("%.2f" % v for v in sp), len(sp), ok))
    print("    NOTE: this does NOT settle which is the front month. Status is published for both")
    print("    contracts on every 2020 date checked; only VOLUME can decide the roll.")
    return ok


def main():
    checks = [check_cme_publishes_price_limits, check_int64_sentinel_is_not_a_price,
              check_routine_session_notices_are_not_halts,
              check_single_venue_quorum_admits_the_cme_pause,
              check_the_pause_lands_inside_the_spy_halt, check_limit_down_ratchet_is_visible,
              check_qc_judges_each_book_against_its_own_halt,
              check_sequence_is_channel_level_not_product_level,
              check_halts_are_group_level_not_contract_level,
              check_es_halt_onset_lags_the_equity_halt,
              check_futures_only_pause_is_not_swallowed, check_flag_clear_is_not_a_resume,
              check_the_sole_venue_window, check_flag_understatement_across_days,
              check_roll_is_settled_by_volume, check_calendar_spread_sanity]
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
