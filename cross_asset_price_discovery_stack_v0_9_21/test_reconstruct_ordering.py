"""test_reconstruct_ordering.py -- guards the event-ordering fix in lob_reconstruct.

A single venue builds a sane book, then an order R near the touch is Modified (seq 10) and Cancelled
(seq 12) -- but their RECEIPT timestamps are inverted (cancel received 100ms in, modify 200ms in), as
UDP multicast reordering produces. Then the bid rises above R's price. Under legacy clock ordering the
cancel is applied before the modify, the modify RESURRECTS the dead order, and the stale ask pins a
crossed top once the bid rises. Under sequencenumber ordering the modify precedes the cancel, R is
removed, and the book stays clean. Same events, same code, only `order_by` differs.
"""
import sys
import numpy as np
import pandas as pd
import lob_reconstruct as lob

ET = "America/New_York"
BASE = pd.Timestamp("2025-04-03 10:00:00", tz=ET).value          # ns since epoch

# (mtype, seq, receipt_offset_ns, side, price, qty, ref, pref)
EVENTS = [
    ("mt_add_order",    2, int(1e6),  "Bid", 100.00, 100, "b1", ""),
    ("mt_add_order",    4, int(2e6),  "Ask", 100.01, 100, "R",  ""),   # phantom candidate (at the touch)
    ("mt_add_order",    6, int(3e6),  "Ask", 100.05, 100, "a1", ""),
    ("mt_add_order",    8, int(4e6),  "Ask", 100.06, 100, "a2", ""),
    ("mt_modify_order", 10, int(200e6), "Ask", 100.01, 50, "R", "R"),  # LATE receipt (200ms), reduce 100->50
    ("mt_cancel_order", 12, int(100e6), "Ask", 100.01, 50, "R", "R"),  # EARLY receipt (100ms) -> inversion
    ("mt_add_order",   14, int(2e9),  "Bid", 100.04, 100, "b2", ""),   # +2s: bid rises to 100.04 (< 100.05)
]

def build_messages(events, feed="F"):
    by = {}
    for mt, seq, off, side, price, qty, ref, pref in events:
        by.setdefault(mt, []).append(dict(
            receipttimestamp=BASE + off, exchangetimestamp=BASE + off, sequencenumber=seq,
            f=feed, product="X", side=side, price=price, quantity=qty,
            previousprice=price, previousquantity=qty,
            orderreferencenumber=ref, previousorderreferencenumber=pref, maintainpriority="false"))
    out = {}
    for mt, rows in by.items():
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["receipttimestamp"], unit="ns", utc=True).dt.tz_convert(ET)
        out[mt] = df
    return out

def crossed_fraction(book):
    bb = book["X_bidprice_1"].to_numpy(); ba = book["X_askprice_1"].to_numpy()
    both = np.isfinite(bb) & np.isfinite(ba)
    if both.sum() == 0:
        return np.nan, 0
    return float(np.mean(bb[both] > ba[both] + 1e-9)), int(both.sum())

def run(order_by):
    msgs = build_messages(EVENTS)
    return lob.reconstruct_book(msgs, asset="X", levels=3, interval="1s",
                                session=("10:00", "10:04"), tz=ET, date_str="20250403",
                                clock="receipt", order_by=order_by)

def main():
    ok = True
    leg = run("clock")
    seq = run("sequence")
    xf_leg, n_leg = crossed_fraction(leg)
    xf_seq, n_seq = crossed_fraction(seq)
    # final-snapshot top of book
    def final_top(b):
        nb = b.dropna(subset=["X_bidprice_1", "X_askprice_1"])
        if not len(nb): return (np.nan, np.nan)
        r = nb.iloc[-1]; return (float(r["X_bidprice_1"]), float(r["X_askprice_1"]))
    lb, la = final_top(leg); sb, sa = final_top(seq)

    a_ok = xf_leg > 0.0 and lb > la                              # legacy: crosses (resurrected ask pins top)
    b_ok = (xf_seq == 0.0) and sb < sa and n_seq > 0            # sequence: clean throughout
    print("(A) legacy(clock)   crossed_frac=%.2f over %d snaps | final bid=%.2f ask=%.2f -> %s"
          % (xf_leg, n_leg, lb, la, a_ok))
    print("(B) fixed(sequence) crossed_frac=%.2f over %d snaps | final bid=%.2f ask=%.2f -> %s"
          % (xf_seq, n_seq, sb, sa, b_ok))
    ok &= a_ok and b_ok

    # the fix is the ONLY difference: assert the books actually differ (sanity that order_by bit)
    c_ok = abs((la if np.isfinite(la) else 0) - (sa if np.isfinite(sa) else 0)) > 1e-9
    print("(C) the two orderings produce different books (ask %.2f vs %.2f) -> %s" % (la, sa, c_ok))
    ok &= c_ok

    print("reconstruct-ordering checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
