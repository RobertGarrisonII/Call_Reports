"""test_crossed_regression.py -- synthetic regression guards for the trade-no-ref crossing bug.

Vendor-free. These exercise the exact code paths that crossed the ES book on Liberation Day, plus the
edges the original minimal repro did not reach. Keep alongside smoke_test_crossed.py: that one audits
the real tape, this one guards the logic in CI so a regression is caught without a MIDAS run.

    python test_crossed_regression.py        # exit 0 iff every guard passes

Guards
  single_stranding       refless aggressor-coded print lifts the best ask; a bid then joins above it.
                         pre-fix: stranded ask -> crossed (~95% of snapshots). post-fix: no cross.
  multilevel_sweep       one marketable order walks THREE ask levels via refless prints (two full
                         clears + one PARTIAL); a bid joins in the swept region. checks the surviving
                         level/qty are right and the book never crosses.
  referenced_unaffected  a print WITH a valid ref but an aggressor-coded `side` must still reduce the
                         resting order via reduce() (stored side), untouched by the wrong side field.
  reduce_at_price_units  unit coverage of the price-keyed consumer: ask-only(->1), bid-only(->0),
                         non-displayed(->-1, no-op), both-sides-locked(->1, deterministic), and the
                         partial vs full-clear arithmetic.
"""
import sys
import numpy as np
import pandas as pd
import lob_reconstruct as lob

TZ = "America/New_York"
T0 = pd.Timestamp("2025-04-03 09:30:00", tz=TZ).tz_convert("UTC").value
S = 1_000_000_000
ADD_COLS = ["f", "side", "price", "quantity", "orderreferencenumber"]
TRD_COLS = ["f", "side", "price", "quantity", "orderreferencenumber", "aggressorside"]


def _run(adds, trades, levels=5):
    msgs = {"mt_add_order": lob._msg_frame(adds, ADD_COLS, TZ),
            "mt_trade": lob._msg_frame(trades, TRD_COLS, TZ)}
    return lob.reconstruct_book(msgs, asset="ES", levels=levels, interval="1s",
                               session=("09:30", "09:31"), tz=TZ, date_str="20250403")


def _at(book, secs):
    return book.loc[pd.Timestamp("2025-04-03 09:30:%02d" % secs, tz=TZ)]


def _crossed(book):
    return book.attrs["lob_stats"].get("consolidated_crossed", 0)


# ── end-to-end guards ─────────────────────────────────────────────────────────
def single_stranding():
    adds = [(T0, "cme", "Ask", 100.15, 200, "a2"), (T0, "cme", "Ask", 100.10, 200, "a1"),
            (T0, "cme", "Bid", 100.05, 200, "b1"), (T0 + 3 * S, "cme", "Bid", 100.11, 100, "b2")]
    trd = [(T0 + 2 * S, "cme", "Bid", 100.10, 200, "", "Buy")]      # refless, aggressor-coded
    b = _run(adds, trd)
    r3 = _at(b, 3)
    return (_crossed(b) == 0 and abs(r3["ES_askprice_1"] - 100.15) < 1e-9
            and abs(r3["ES_bidprice_1"] - 100.11) < 1e-9)


def multilevel_sweep():
    # asks 100.10x100, 100.15x100, 100.20x150 ; one buy sweep -> refless prints 100@.10, 100@.15, 60@.20
    # (two full clears + a partial leaving 90 at 100.20); then a bid joins at 100.16 (in the swept band).
    adds = [(T0, "cme", "Ask", 100.20, 150, "a3"), (T0, "cme", "Ask", 100.15, 100, "a2"),
            (T0, "cme", "Ask", 100.10, 100, "a1"), (T0, "cme", "Bid", 100.05, 200, "b1"),
            (T0 + 2 * S, "cme", "Bid", 100.16, 100, "b2")]
    trd = [(T0 + 1 * S, "cme", "Bid", 100.10, 100, "", "Buy"),
           (T0 + 1 * S, "cme", "Bid", 100.15, 100, "", "Buy"),
           (T0 + 1 * S, "cme", "Bid", 100.20, 60, "", "Buy")]
    b = _run(adds, trd)
    r1 = _at(b, 1)            # after the sweep, before the 100.16 bid
    asks = {round(r1["ES_askprice_%d" % i], 2) for i in range(1, 6) if np.isfinite(r1["ES_askprice_%d" % i])}
    swept_gone = (100.10 not in asks) and (100.15 not in asks)
    surviving = abs(r1["ES_askprice_1"] - 100.20) < 1e-9 and abs(r1["ES_askquantity_1"] - 90) < 1e-9
    return _crossed(b) == 0 and swept_gone and surviving


def packet_resurrection():
    # CME PACKET-level sequencenumber: a cancel and a modify of the SAME order share ONE sequencenumber,
    # so the venue sequence does not disambiguate them. The true order is modify-then-cancel (the cancel
    # is the final word -> the order leaves the book). The pre-v0.9.7 within-tie fallback is the arbitrary
    # concatenation order (cancels before modifies); it applies cancel-then-modify, and since modify ==
    # remove+add it RESURRECTS the cancelled ask at a price BELOW the bid, pinning the consolidated top
    # crossed on every later snapshot (the 88%-crossed ES signature). eliminations-last makes the cancel win.
    # Resting (distinct seq): ask x1 @100.10 (seq1); bid b1 @100.05 (seq2); bid b2 @100.08 (seq3, T0+0.5s).
    # Packet seq=10 @ T0+1s: cancel x1  AND  modify x1 -> Ask @100.07 (which, if x1 survives, sits below b2).
    adds = [(T0, "cme", "Ask", 100.10, 100, "x1", 1),
            (T0, "cme", "Bid", 100.05, 100, "b1", 2),
            (T0 + S // 2, "cme", "Bid", 100.08, 100, "b2", 3)]
    can = [(T0 + 1 * S, "cme", "Ask", 100.10, 100, "x1", 10)]                     # cancel x1 (packet seq 10)
    mod = [(T0 + 1 * S, "cme", "Ask", 100.07, 100, "x1", "x1", 100.10, 100, "true", 10)]  # reprice x1 (seq 10)
    msgs = {"mt_add_order": lob._msg_frame(adds, ADD_COLS + ["sequencenumber"], TZ),
            "mt_cancel_order": lob._msg_frame(
                can, ["f", "side", "price", "previousquantity", "orderreferencenumber", "sequencenumber"], TZ),
            "mt_modify_order": lob._msg_frame(
                mod, ["f", "side", "price", "quantity", "orderreferencenumber", "previousorderreferencenumber",
                      "previousprice", "previousquantity", "maintainpriority", "sequencenumber"], TZ)}
    b = lob.reconstruct_book(msgs, asset="ES", levels=5, interval="1s",
                             session=("09:30", "09:31"), tz=TZ, date_str="20250403")
    r5 = _at(b, 5)                                          # well after the packet
    a1, bid1 = r5["ES_askprice_1"], r5["ES_bidprice_1"]
    no_resurrected_ask = not (np.isfinite(a1) and a1 < bid1 - 1e-9)   # cancel won -> no sub-bid ask survives
    best_bid_ok = abs(bid1 - 100.08) < 1e-9
    return _crossed(b) == 0 and no_resurrected_ask and best_bid_ok


def referenced_unaffected():
    # trade carries a valid ref to resting ask a1 but side="Bid" (aggressor). Must reduce a1 (ask) by 50.
    adds = [(T0, "cme", "Ask", 100.10, 200, "a1"), (T0, "cme", "Bid", 100.05, 200, "b1")]
    trd = [(T0 + 2 * S, "cme", "Bid", 100.10, 50, "a1", "Buy")]
    b = _run(adds, trd)
    r3 = _at(b, 3)
    st = b.attrs["lob_stats"]
    return (_crossed(b) == 0 and abs(r3["ES_askprice_1"] - 100.10) < 1e-9
            and abs(r3["ES_askquantity_1"] - 150) < 1e-9 and st.get("trade_no_ref", 0) == 0)


# ── unit coverage of the price-keyed consumer ─────────────────────────────────
def reduce_at_price_units():
    b = lob._Book()
    b.add("cme", "k1", 1, 100.10, 100)                    # ask only
    if b.reduce_at_price("cme", 100.10, 40) != 1: return False        # -> ask
    if abs(b.ask["cme"][100.10] - 60) > 1e-9: return False            # partial: 100-40
    if b.reduce_at_price("cme", 100.10, 60) != 1: return False        # full clear
    if 100.10 in b.ask["cme"]: return False
    b.add("cme", "k2", 0, 100.05, 100)                    # bid only
    if b.reduce_at_price("cme", 100.05, 100) != 0: return False       # -> bid, full
    if 100.05 in b.bid["cme"]: return False
    if b.reduce_at_price("cme", 99.99, 10) != -1: return False        # non-displayed -> no-op
    b.add("cme", "z1", 0, 100.30, 100); b.add("cme", "z2", 1, 100.30, 100)   # both-sides-locked
    if b.reduce_at_price("cme", 100.30, 50) != 1: return False        # deterministic -> ask
    if abs(b.ask["cme"][100.30] - 50) > 1e-9: return False            # ask reduced
    if abs(b.bid["cme"][100.30] - 100) > 1e-9: return False           # bid untouched
    return True


TESTS = [("single_stranding", single_stranding),
         ("multilevel_sweep", multilevel_sweep),
         ("packet_resurrection", packet_resurrection),
         ("referenced_unaffected", referenced_unaffected),
         ("reduce_at_price_units", reduce_at_price_units)]

if __name__ == "__main__":
    ok_all = True
    for name, fn in TESTS:
        try:
            ok = bool(fn())
        except Exception as e:
            ok = False; name = "%s [EXC %s]" % (name, e)
        ok_all &= ok
        print("  %-22s %s" % (name, "PASS" if ok else "FAIL"))
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
    sys.exit(0 if ok_all else 1)
