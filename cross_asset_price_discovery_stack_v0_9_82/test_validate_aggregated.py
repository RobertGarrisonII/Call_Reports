#!/usr/bin/env python3
"""test_validate_aggregated.py -- the ES tape carries its own benchmark; check we read it correctly.

The ESZ4 sample for 2024-12-18 established three things this pins:

  * mt_price_level_update / mt_modify_price_level / mt_delete_price_level are HEADER-ONLY for CME
    futures, so the ES leg is order-by-order by necessity. The stack already assumed this; now it is
    verified rather than asserted.
  * mt_aggregated_price_update IS populated for ES -- CME's own 10-level ladder with quantity and
    order count per level, from the same lake and the same capture clock as the messages the replay
    consumes. That makes it a benchmark with no clock confound, unlike mstbook-query.
  * mt_product_statistics is a RUNNING stream whose early rows carry the PREVIOUS session (the
    17:38 ET row on 2024-12-18 still reports the 2024-12-15 settlement), so a reader that takes the
    first value reports last week's numbers.

Run: python test_validate_aggregated.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import validate_aggregated as va

TZ = "America/New_York"


def _agg_frame():
    """Two rows of the real ESZ4 shape: integer-hundredths prices, \\N nulls beyond the top levels."""
    idx = pd.to_datetime(["2024-12-18 09:30:00.5", "2024-12-18 09:30:01.5"]).tz_localize(TZ)
    d = {"f": ["cme_globex30_cme"] * 2,
         "bidprice_1": [605175, 605200], "bidquantity_1": [9, 11], "bidnumorders_1": [4, 5],
         "bidprice_2": [605150, 605175], "bidquantity_2": [12, 9], "bidnumorders_2": [5, 4],
         "askprice_1": [605225, 605250], "askquantity_1": [2, 7], "asknumorders_1": [1, 3],
         "askprice_2": [605250, 605275], "askquantity_2": [11, 12], "asknumorders_2": [4, 6],
         "bidprice_3": ["\\N", "\\N"], "bidquantity_3": ["\\N", "\\N"]}
    return pd.DataFrame(d, index=idx)


def check_canonical():
    c = va.aggregated_to_canonical(_agg_frame(), "ES", levels=3, price_scale=0.01)
    p1 = np.isclose(c["ES_bidprice_1"].iloc[0], 6051.75) and np.isclose(c["ES_askprice_1"].iloc[0], 6052.25)
    q1 = np.isclose(c["ES_bidquantity_1"].iloc[0], 9)
    mid = np.isclose(c["ES_mid"].iloc[0], 6052.00)
    nulls = not np.isfinite(c["ES_bidprice_3"].iloc[0])          # '\N' -> NaN, not 0 and not a crash
    ok = p1 and q1 and mid and nulls
    print("(1) integer-hundredths -> index points (605175 -> %.2f), quantities kept, mid %.2f, "
          "'\\N' beyond the populated levels -> NaN : %s"
          % (c["ES_bidprice_1"].iloc[0], c["ES_mid"].iloc[0], ok))
    return ok


def check_compare_asof():
    """The benchmark is event-stamped, the reconstruction grid-stamped: carry forward, don't drop."""
    asset = "ES"
    grid = pd.date_range("2024-12-18 09:30:00", periods=4, freq="s", tz=TZ)
    recon = pd.DataFrame({f"{asset}_bidprice_1": [6051.75, 6051.75, 6052.00, 6052.00],
                          f"{asset}_bidquantity_1": [9.0, 9.0, 11.0, 11.0],
                          f"{asset}_askprice_1": [6052.25, 6052.25, 6052.50, 6052.50],
                          f"{asset}_askquantity_1": [2.0, 2.0, 7.0, 7.0]}, index=grid)
    bench = va.aggregated_to_canonical(_agg_frame(), asset, levels=1, price_scale=0.01)
    tbl = va.compare(recon, bench, asset, levels=1)
    # grid 09:30:00 precedes the first benchmark row -> not comparable; 01,02,03 are
    perfect = bool((tbl["price_match"].dropna() == 1.0).all()) and bool((tbl["qty_match"].dropna() == 1.0).all())
    n_ok = bool((tbl["n"] == 3).all())

    # now break one quantity and confirm it is caught rather than absorbed
    bad = recon.copy(); bad.loc[bad.index[2], f"{asset}_bidquantity_1"] = 999.0
    tb = va.compare(bad, bench, asset, levels=1)
    caught = float(tb[tb["side"] == "bid"]["qty_match"].iloc[0]) < 1.0
    prices_still_fine = float(tb[tb["side"] == "bid"]["price_match"].iloc[0]) == 1.0

    ok = perfect and n_ok and caught and prices_still_fine
    print("(2) as-of alignment: a matching replay scores 100%% on price AND quantity over the 3 "
          "comparable grid points=%s; a single wrong quantity is caught (%.2f%%) while prices stay "
          "clean=%s : %s" % (perfect and n_ok,
                             100 * float(tb[tb['side'] == 'bid']['qty_match'].iloc[0]),
                             caught and prices_still_fine, ok))
    return ok


def check_statistics_recency():
    """The early rows carry the PREVIOUS session. Take the last value, never the first."""
    idx = pd.to_datetime(["2024-12-17 17:38:18", "2024-12-18 15:59:00"]).tz_localize(TZ)
    df = pd.DataFrame({"highprice": [607925, 608900],       # stale (prior session), then real
                       "lowprice": [604075, 585500],
                       "volume": [1025012, 2100000],
                       "settlementprice": [608050, "\\N"]}, index=idx)
    st = va.session_statistics(df, price_scale=0.01)
    fresh = np.isclose(st["highprice"], 6089.00) and np.isclose(st["lowprice"], 5855.00)
    vol = np.isclose(st["volume"], 2100000)                  # volume is NOT price-scaled
    # a field whose only non-null value is the stale one still comes through (it is all there is)
    settle = np.isclose(st["settlementprice"], 6080.50)
    ok = fresh and vol and settle
    print("(3) statistics: last non-null wins, so high=%.2f low=%.2f (not the 17:38 ET row's prior-"
          "session figures)=%s; volume unscaled=%s; a field with only a stale value survives=%s : %s"
          % (st["highprice"], st["lowprice"], fresh, vol, settle, ok))
    return ok


def check_futures_has_no_mbp():
    """CME publishes no price-level types for ES -- verified, so the MBO-only path is necessary."""
    import lob_reconstruct as lob
    import mstbook_loader as ml
    # header-only responses come back as empty frames; the replay must still produce a book
    idx = pd.to_datetime(["2024-12-18 09:29:00", "2024-12-18 09:29:01"]).tz_localize(TZ)
    adds = pd.DataFrame({"sequencenumber": [1, 2], "f": ["cme_globex30_cme"] * 2,
                         "side": ["Bid", "Ask"], "price": [605175, 605225],
                         "quantity": [9, 2], "orderreferencenumber": ["1", "2"]}, index=idx)
    msgs = {"mt_add_order": adds, "mt_cancel_order": pd.DataFrame(),
            "mt_modify_order": pd.DataFrame(), "mt_trade": pd.DataFrame(),
            "mt_price_level_update": pd.DataFrame(),      # header-only for futures
            "mt_clear_orders": pd.DataFrame(), "mt_clear_price_levels": pd.DataFrame()}
    b = lob.reconstruct_book(msgs, asset="ES", levels=2, interval="1s",
                             session=("09:30", "09:31"), tz=TZ, date_str="20241218",
                             clock="receipt", price_scale=0.01)
    built = (np.isclose(b["ES_bidprice_1"].iloc[0], 6051.75)
             and np.isclose(b["ES_askprice_1"].iloc[0], 6052.25)
             and b.attrs["mbo_feeds"] == ["cme_globex30_cme"] and not b.attrs["mbp_feeds"])
    # and the clear types are now fetchable at all (they were not in _MSG_QTY_COL before)
    fetchable = all(t in ml._MSG_QTY_COL for t in
                    ("mt_clear_orders", "mt_clear_price_levels", "mt_aggregated_price_update",
                     "mt_product_statistics", "mt_missing_product_messages", "mt_error"))
    # mt_aggregated_price_update names its clock lastreceipttimestamp -- without the alias it is unfetchable
    clock_alias = "lastreceipttimestamp" in ml._CLOCK_COLS["receipt"]
    ok = built and fetchable and clock_alias
    print("(4) ES with every price-level type empty still replays MBO-only (%.2f/%.2f, mbp_feeds=%s)"
          "=%s; the validation/oracle types are fetchable=%s; lastreceipttimestamp alias present=%s "
          ": %s" % (b["ES_bidprice_1"].iloc[0], b["ES_askprice_1"].iloc[0], b.attrs["mbp_feeds"],
                    built, fetchable, clock_alias, ok))
    return ok


def main():
    checks = [check_canonical, check_compare_asof, check_statistics_recency, check_futures_has_no_mbp]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("aggregated-benchmark checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
