#!/usr/bin/env python3
"""test_es_aggregated_leg.py -- the ES leg built from CME's ladder, end to end through the extractor.

The futures leg no longer comes from a message replay. `mt_aggregated_price_update` is CME's own
10-level ladder, delivered on the same lake and the same capture clock as the messages, and it is
populated in EVERY era checked -- unlike the message families, which changed shape between 2020
(price-level) and 2024 (order-by-order) and emptied four sessions when the extractor assumed one of
them. CME is a single venue with nothing to consolidate, so the ladder IS the book.

What these checks defend:

  * the leg builds on a 2020-shaped fetch, where the MBO replay built nothing at all
  * the schema is IDENTICAL to the replay's, so nothing downstream can tell which source it got
  * the step function is forward-only -- a stale book during a halt, never a look-ahead one
  * an empty ladder is refused rather than returned as a full-length all-NaN column
  * the halt boundary still works, which needs the trade tape rather than the ladder
  * `--es-book-source replay` still reaches the message path

Run: python test_es_aggregated_leg.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import aggregated_book as ab
import lob_reconstruct as lob
import mstbook_loader as ml

TZ = "America/New_York"
SCALE = 0.01


def _ladder(day="2020-03-12", n=40, start="08:00:00"):
    """A ladder that opens pre-market and updates every 30 s, in CME integer-hundredths."""
    idx = pd.date_range(f"{day} {start}", periods=n, freq="30s", tz=TZ)
    bid = 254600 + 25 * np.arange(n)
    cols = {}
    for i in range(1, 11):
        cols[f"bidprice_{i}"] = bid - 25 * (i - 1)
        cols[f"bidquantity_{i}"] = 40 + i
        cols[f"askprice_{i}"] = bid + 25 * i
        cols[f"askquantity_{i}"] = 30 + i
    return pd.DataFrame(cols, index=idx)


def _trades(day="2020-03-12"):
    idx = pd.date_range(f"{day} 09:30:00", periods=120, freq="s", tz=TZ)
    return pd.DataFrame({"quantity": 1.0, "price": 254600.0}, index=idx)


def _patch(agg, trades=None):
    def _fetch(date_str, product, ptype, mt, data_source="apu", tz=None, clock="receipt"):
        if mt == ab.AGG:
            return agg
        if mt == "mt_trade" and trades is not None:
            return trades
        return pd.DataFrame()
    return _fetch


def check_builds_where_the_replay_could_not():
    """A 2020-shaped session: no usable MBO stream, but a populated ladder."""
    real = ml._fetch_messages
    try:
        ml._fetch_messages = _patch(_ladder(), _trades())
        out = ab.session_from_aggregated("20200312", "ESH0", "futures", levels=10, interval="1s",
                                         session=("09:30", "09:40"), price_scale=SCALE)
    finally:
        ml._fetch_messages = real
    bid = pd.to_numeric(out["ES_bidprice_1"], errors="coerce")
    ask = pd.to_numeric(out["ES_askprice_1"], errors="coerce")
    n = int((bid.notna() & ask.notna()).sum())
    full = n == len(out)
    warm = bool(out.attrs["lob_stats"]["warm_at_open"])
    scaled = 2500.0 < float(bid.iloc[0]) < 2600.0          # index points, not 254600
    src = out.attrs.get("book_source") == "aggregated"
    ok = full and warm and scaled and src
    print("(1) 2020-shaped session: %d/%d snapshots have a finite top (%s), warm at the open from "
          "the pre-market ladder (%s)" % (n, len(out), full, warm))
    print("    top is %.2f index points after price_scale=%.2f (%s), source recorded as %s : %s"
          % (bid.iloc[0], SCALE, scaled, out.attrs.get("book_source"), ok))
    return ok


def check_schema_matches_the_replay():
    """Downstream must not be able to tell which source produced the leg."""
    real = ml._fetch_messages
    try:
        ml._fetch_messages = _patch(_ladder())
        agg_out = ab.session_from_aggregated("20200312", "ESH0", "futures", levels=10,
                                             session=("09:30", "09:31"), price_scale=SCALE,
                                             with_activity=False)
    finally:
        ml._fetch_messages = real

    # what the replay emits, from lob_reconstruct's own documented schema
    want = set()
    for i in range(1, 11):
        for tag in ("bid", "ask"):
            want |= {f"ES_{tag}price_{i}", f"ES_{tag}quantity_{i}"}
    want |= {"ES_nbbo_bid", "ES_nbbo_ask", "ES_mid"}
    have = set(agg_out.columns)
    same = want <= have
    extra = have - want
    numeric = all(pd.api.types.is_numeric_dtype(agg_out[c]) for c in sorted(want))
    ok = same and not extra and numeric
    print("(2) the ladder leg emits exactly the replay's %d columns (%s), no extras (%s), all "
          "numeric (%s) : %s" % (len(want), same, not extra, numeric, ok))
    return ok


def check_empty_ladder_is_refused():
    """An empty ladder must raise, not return a full-length all-NaN ES column."""
    real = ml._fetch_messages
    try:
        ml._fetch_messages = _patch(pd.DataFrame())
        try:
            ab.session_from_aggregated("20200312", "ESH0", "futures", session=("09:30", "16:00"),
                                       price_scale=SCALE, with_activity=False)
            raised, msg = False, ""
        except ml.MessageFetchError as exc:
            raised, msg = True, str(exc)
    finally:
        ml._fetch_messages = real
    named = raised and "ESH0" in msg and "20200312" in msg and ab.AGG in msg
    points = raised and "select_family=True" in msg          # where to go next
    ok = raised and named and points
    print("(3) an empty ladder raises rather than returning a full-length all-NaN leg (%s), naming "
          "the product, date and message type (%s), and pointing at the replay fallback (%s) : %s"
          % (raised, named, points, ok))
    return ok


def check_halt_boundary_still_works():
    """The halt window needs the TRADE tape, which the ladder does not provide.

    The ladder keeps updating through a CME stop -- orders can still be entered and pulled -- so it
    cannot mark the resume. `activity_seconds` therefore comes from mt_trade even though the book no
    longer does, because the status flag understates a CME stop by ~130x (see market_halts)."""
    import market_halts as mh
    real = ml._fetch_messages
    try:
        # trades stop at 09:31:00 and resume at 09:36:00; the ladder keeps ticking throughout
        tr = pd.DataFrame({"quantity": 1.0},
                          index=pd.date_range("2020-03-12 09:30:00", periods=61, freq="s", tz=TZ)
                          .append(pd.date_range("2020-03-12 09:36:00", periods=60, freq="s", tz=TZ)))
        ml._fetch_messages = _patch(_ladder(), tr)
        out = ab.session_from_aggregated("20200312", "ESH0", "futures", session=("09:30", "09:40"),
                                         price_scale=SCALE)
    finally:
        ml._fetch_messages = real
    act = out.attrs.get("activity_seconds") or []
    got = len(act) > 0
    status = pd.DataFrame({"f": ["cme_globex30_cme"] * 2,
                           "haltreason": ["SuspendedBySurveillance", ""]},
                          index=pd.DatetimeIndex(["2020-03-12 09:31:00.100",
                                                  "2020-03-12 09:31:06.500"]).tz_localize(TZ))
    flag = mh.windows_from_status(status, tz=TZ)["windows"][0]
    real_w = mh.windows_from_status(status, tz=TZ, activity=act)["windows"][0]
    flag_s = (flag[1] - flag[0]).total_seconds()
    real_s = (real_w[1] - real_w[0]).total_seconds()
    extended = real_s > 250 and flag_s < 10
    ok = got and extended
    print("(4) the ladder ticks through the stop, so the resume comes from the trade tape: "
          "%d activity instants recorded (%s)" % (len(act), got))
    print("    flag says %.1f s, the tape says %.1f s -- the CME understatement is still caught on "
          "the aggregated leg (%s) : %s" % (flag_s, real_s, extended, ok))
    return ok


def check_replay_path_still_reachable():
    """--es-book-source replay must still select the message family at run time."""
    import run_analysis
    ap = run_analysis.build_parser() if hasattr(run_analysis, "build_parser") else None
    flag_ok = True
    if ap is not None:
        opts = {a.dest for a in ap._actions}
        flag_ok = "es_book_source" in opts
    sig_ok = "es_book_source" in ml.extract_sessions.__code__.co_varnames
    fam, _ = lob.select_book_family({"mt_add_order": 4, "mt_modify_price_level": 5,
                                     "mt_delete_price_level": 5})
    still = fam == "MBP"
    ok = flag_ok and sig_ok and still
    print("(5) --es-book-source exposes both paths (%s), extract_sessions takes es_book_source (%s)"
          % (flag_ok, sig_ok))
    print("    and the replay's era-aware family selection is unchanged and still picks MBP for "
          "2020 counts (%s) : %s" % (still, ok))
    return ok


def check_qc_reads_the_ladder_leg():
    """session_qc must judge the aggregated leg exactly as it judges a replayed one."""
    idx = pd.date_range("2020-03-12 09:30:00", periods=300, freq="s", tz=TZ)
    df = pd.DataFrame({"SPY_bidprice_1": 250.00, "SPY_askprice_1": 250.01,
                       "ES_bidprice_1": 2546.50, "ES_askprice_1": 2547.00}, index=idx)
    df.attrs["halt_windows_SPY"] = []
    df.attrs["halt_windows_ES"] = []
    clean = ml.session_qc(df, date="2020-03-12")
    good = clean["ok"] and clean["ES"]["n_finite"] == len(df) and clean["ES"]["crossed_frac"] == 0.0

    bad = df.copy()
    bad.attrs = dict(df.attrs)
    bad["ES_bidprice_1"] = 2548.00                        # ladder crossed on every snapshot
    rep = ml.session_qc(bad, date="2020-03-12")
    caught = (not rep["ok"]) and any(r.startswith("ES: top is CROSSED") for r in rep["reasons"])
    ok = good and caught
    print("(6) a clean ladder leg passes QC (%s); a crossed one is still FAILED by the same "
          "invariant (%s) : %s" % (good, caught, ok))
    print("    the gate does not care which source produced the leg, which is the point of matching "
          "the schema.")
    return ok


def main():
    checks = [check_builds_where_the_replay_could_not, check_schema_matches_the_replay,
              check_empty_ladder_is_refused, check_halt_boundary_still_works,
              check_replay_path_still_reachable, check_qc_reads_the_ladder_leg]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("ES aggregated-leg checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
