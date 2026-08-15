#!/usr/bin/env python3
"""test_es_book_family.py -- why the 2020 ES leg was empty, and the fix that is not another hardcode.

`probe_es_2020.py`, run against all four MWCB dates and both contracts, with 2024-12-18 as control:

    message type                 2020 ES (4 dates, both contracts)   2024 ES
    mt_add_order                 1-4 rows for a whole session*       populated
    mt_cancel_order              EMPTY                               populated
    mt_modify_order              EMPTY                               populated
    mt_price_level_update        EMPTY                               EMPTY
    mt_modify_price_level        populated                           EMPTY
    mt_delete_price_level        populated                           EMPTY
    mt_trade                     populated                           populated
    mt_aggregated_price_update   populated                           populated

    * under a probe limit of five, so those are COMPLETE counts. Stray messages, not a stream.

**The 2020 CME capture is market-by-PRICE; the 2024 capture is market-by-ORDER.** The extractor
asked for order-by-order on every date, so on the four 2020 sessions it fetched a handful of orphan
adds, no cancels, no modifies -- and built nothing. `median ES=nan (ES/SPY=nan)`, four times.

The sharpest part is why this survived a direct check. `mt_price_level_update` -- the one MBP type
the ES fetch list did carry -- is empty in BOTH eras. Testing it against 2024 returned "header only",
which read as "CME publishes no price-level types" and got written into the code as a fact about the
venue. It was a fact about the one MBP type CME never populates.

So the fix is deliberately NOT "2020 uses MBP". That would hardcode a second era observation one
release after the first one broke, and a third capture change would break it again. Every candidate
type is fetched and the ROW COUNTS choose the family, with both an insert source and a removal
source required -- which is what rules out 2020's four orphan adds. Adds without cancels do not make
a thin book; they make a book nothing ever leaves.

Run: python test_es_book_family.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import lob_reconstruct as lob
import mstbook_loader as ml

TZ = "America/New_York"

# verbatim from the probe: row counts under --limit 5 (so "4" means exactly 4 exist)
PROBE_2020 = {"mt_add_order": 4, "mt_cancel_order": 0, "mt_modify_order": 0, "mt_trade": 5,
              "mt_price_level_update": 0, "mt_modify_price_level": 5, "mt_delete_price_level": 5,
              "mt_clear_orders": 0, "mt_clear_price_levels": 0}
PROBE_2024 = {"mt_add_order": 5, "mt_cancel_order": 5, "mt_modify_order": 5, "mt_trade": 5,
              "mt_price_level_update": 0, "mt_modify_price_level": 0, "mt_delete_price_level": 0,
              "mt_clear_orders": 0, "mt_clear_price_levels": 0}


def check_family_from_the_real_counts():
    """The two eras, decided from the probe's own numbers."""
    f20, why20 = lob.select_book_family(PROBE_2020)
    f24, why24 = lob.select_book_family(PROBE_2024)
    ok = f20 == "MBP" and f24 == "MBO"
    print("(1) 2020 counts -> %s   (%s)" % (f20, why20))
    print("    2024 counts -> %s   (%s)" % (f24, why24))
    print("    the capture changed shape between the eras and the row counts say so : %s" % ok)
    return ok


def check_orphan_adds_do_not_pass_as_a_stream():
    """2020's four adds, alone, must NOT be accepted as a book source.

    This is the failure mode that matters more than an empty leg. Adds with no cancels build a book
    in which nothing ever leaves: every level rests to the close, depth grows monotonically and the
    top crosses on most snapshots. That is the exact signature of the original crossed-book bug, and
    it would have been produced by 'fall back to whatever has rows'."""
    only_adds = {"mt_add_order": 4, "mt_cancel_order": 0, "mt_modify_order": 0}
    fam, why = lob.select_book_family(only_adds)
    refused = fam is None and "NO removals" in why
    explains = "rest to the close" in why
    # and with the MBP family present, the adds lose rather than contaminate
    fam2, _ = lob.select_book_family(PROBE_2020)
    ok = refused and explains and fam2 == "MBP"
    print("(2) four adds with no cancels: family = %s, refused (%s) with the reason named (%s)"
          % (fam, refused, explains))
    print("    and alongside a populated MBP family they lose to it rather than joining it (%s) : %s"
          % (fam2 == "MBP", ok))
    return ok


def check_hybrid_equity_book_keeps_both():
    """A consolidated equity book is genuinely MBO+MBP. Choosing one would drop whole venues."""
    spy = {"mt_add_order": 900, "mt_cancel_order": 800, "mt_modify_order": 300,
           "mt_price_level_update": 1000}
    fam, why = lob.select_book_family(spy)
    ok = fam == "BOTH" and "both are kept" in why
    print("(3) a hybrid SPY-shaped count set -> %s : %s" % (fam, ok))
    print("    some venues publish order-by-order and others price-level; taking 'the denser "
          "family' would silently drop the others from the NBBO.")
    return ok


def _mbp_messages(day="2020-03-12"):
    """A minimal 2020-shaped fetch: price-level sets and deletes, plus the orphan adds."""
    base = pd.Timestamp(f"{day} 09:30:00", tz=TZ).value

    def frame(rows, cols):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["receipttimestamp"], unit="ns", utc=True).dt.tz_convert(TZ)
        return df

    def row(off, seq, side, price, qty):
        return dict(receipttimestamp=base + off, exchangetimestamp=base + off, sequencenumber=seq,
                    f="cme_globex30_cme", product="ESH0", side=side, price=price, quantity=qty,
                    previousprice=price, previousquantity=qty, orderreferencenumber="",
                    previousorderreferencenumber="", maintainpriority="false", pricelevel=1)

    msgs = {
        # the book: bid 2546.50 x 40, ask 2547.00 x 30  (CME integer-hundredths)
        "mt_modify_price_level": frame([row(int(1e6), 10, "Bid", 254650, 40),
                                        row(int(2e6), 12, "Ask", 254700, 30)], None),
        "mt_delete_price_level": frame([row(int(5e9), 90, "Ask", 254700, 30)], None),
        # the orphan adds that must NOT be used: an ask INSIDE the bid, never cancelled
        "mt_add_order": frame([row(int(3e6), 14, "Ask", 254600, 25)], None),
        "mt_cancel_order": pd.DataFrame(),
        "mt_modify_order": pd.DataFrame(),
        "mt_trade": pd.DataFrame(),
        "mt_price_level_update": pd.DataFrame(),
        "mt_clear_orders": pd.DataFrame(),
        "mt_clear_price_levels": pd.DataFrame(),
    }
    return msgs


def check_2020_shaped_fetch_builds_a_book():
    """End to end on a 2020-shaped fetch: a book appears, and the orphan add does not cross it."""
    msgs = _mbp_messages()
    real = ml._fetch_messages
    try:
        ml._fetch_messages = lambda d, p, pt, mt, ds="apu", tz=None, clock="receipt": msgs.get(
            mt, pd.DataFrame())
        out = lob.reconstruct_session(
            "20200312", "ESH0", "futures", levels=3, interval="1s", price_scale=0.01,
            session=("09:30", "09:31"), select_family=True,
            message_types=("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade",
                           "mt_price_level_update", "mt_modify_price_level",
                           "mt_delete_price_level", "mt_clear_orders", "mt_clear_price_levels"))
    finally:
        ml._fetch_messages = real

    bid = pd.to_numeric(out.get("ES_bidprice_1"), errors="coerce")
    ask = pd.to_numeric(out.get("ES_askprice_1"), errors="coerce")
    built = int((bid.notna() & ask.notna()).sum()) > 0
    fam = out.attrs.get("book_family") == "MBP"
    scaled = built and abs(float(bid.dropna().iloc[0]) - 2546.50) < 1e-6
    # the orphan ask at 2546.00 would sit INSIDE the spread and cross the book if it were used
    crossed = int((bid > ask + 1e-9).sum())
    clean = crossed == 0 and built and abs(float(ask.dropna().iloc[0]) - 2547.00) < 1e-6
    ok = built and fam and scaled and clean
    print("(4) a 2020-shaped fetch (MBP populated, MBO orphaned) builds a book on %d snapshot(s) "
          "(%s), family recorded as %s" % (int((bid.notna() & ask.notna()).sum()), built,
                                           out.attrs.get("book_family")))
    print("    top is %.2f / %.2f index points after price_scale (%s), and the orphan ask at "
          "2546.00 is NOT in it -- 0 crossed snapshots (%s) : %s"
          % (float(bid.dropna().iloc[0]) if built else float("nan"),
             float(ask.dropna().iloc[0]) if built else float("nan"), scaled, clean, ok))
    return ok


def check_old_fetch_list_would_still_fail_loudly():
    """The old MBO-only fetch on 2020 data must now RAISE rather than return 23,401 NaN rows."""
    msgs = _mbp_messages()
    real = ml._fetch_messages
    try:
        ml._fetch_messages = lambda d, p, pt, mt, ds="apu", tz=None, clock="receipt": msgs.get(
            mt, pd.DataFrame())
        try:
            lob.reconstruct_session("20200312", "ESH0", "futures", levels=3, price_scale=0.01,
                                    session=("09:30", "09:31"),
                                    message_types=("mt_add_order", "mt_cancel_order",
                                                   "mt_modify_order", "mt_trade"))
            raised, msg = False, ""
        except ml.MessageFetchError as exc:
            raised, msg = True, str(exc)
    finally:
        ml._fetch_messages = real
    names = raised and "ESH0" in msg and "20200312" in msg
    verdict = raised and "Family verdict" in msg and "NO removals" in msg
    ok = raised and names and verdict
    print("(5) the OLD order-by-order fetch list on 2020 data raises instead of returning an "
          "all-NaN leg (%s), naming the product and date (%s)" % (raised, names))
    print("    and the message carries the family verdict, so the next reader is told the capture "
          "is price-level rather than having to re-derive it (%s) : %s" % (verdict, ok))
    return ok


def main():
    checks = [check_family_from_the_real_counts, check_orphan_adds_do_not_pass_as_a_stream,
              check_hybrid_equity_book_keeps_both, check_2020_shaped_fetch_builds_a_book,
              check_old_fetch_list_would_still_fail_loudly]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("ES book-family checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
