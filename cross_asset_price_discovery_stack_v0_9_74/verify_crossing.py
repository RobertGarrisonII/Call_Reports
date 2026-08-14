#!/usr/bin/env python3
"""verify_crossing.py -- one-session before/after diagnostic for the reconstruction ordering fix.

Fetches a single session's raw messages ONCE, reconstructs the consolidated book TWICE -- under the
legacy clock ordering (order_by='clock') and the fixed exchange-sequence ordering (order_by='sequence')
-- and prints the crossed-book fraction and the trade/cancel reference-miss rates side by side. This is
the number for the paper's data-cleaning appendix: the all-day-crossed artifact under the old ordering,
and its disappearance under the fix. The clock (cross-feed interleaver) is held fixed across the two
runs, so the ONLY thing that varies is the intra-feed ordering.

  python verify_crossing.py --date 20250403 --product SPY
  python verify_crossing.py --date 20250403 --product ESM5 --product-type futures --price-scale 0.01
  python verify_crossing.py --demo                      # synthetic resurrection (no MIDAS feed needed)
"""
import argparse

import numpy as np
import pandas as pd

import lob_reconstruct as lob

_MTS = ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade", "mt_price_level_update")


def _fetch_once(date_str, product, product_type, data_source, tz, clock):
    """Replicate reconstruct_session's fetch loop, but return the message dict so it can be replayed
    twice (once per ordering) without re-pulling the tape."""
    import mstbook_loader as ml
    msgs = {}
    for mt in _MTS:
        try:
            msgs[mt] = ml._fetch_messages(date_str, product, product_type, mt, data_source, tz=tz, clock=clock)
        except Exception as exc:
            print(f"  [warn] fetch {mt} failed: {exc}")
            msgs[mt] = pd.DataFrame()
    return msgs, ml.canonical_root(product, product_type)


def _demo_messages():
    """The resurrection scenario from test_reconstruct_ordering as a fetchable message dict: a Modify
    (seq 10) and Cancel (seq 12) of the same order with inverted receipt/exchange times, then the bid
    rises above the order's price."""
    ET = "America/New_York"; BASE = pd.Timestamp("2025-04-03 10:00:00", tz=ET).value
    ev = [("mt_add_order", 2, 1e6, "Bid", 100.00, 100, "b1", ""),
          ("mt_add_order", 4, 2e6, "Ask", 100.01, 100, "R", ""),
          ("mt_add_order", 6, 3e6, "Ask", 100.05, 100, "a1", ""),
          ("mt_add_order", 8, 4e6, "Ask", 100.06, 100, "a2", ""),
          ("mt_modify_order", 10, 200e6, "Ask", 100.01, 50, "R", "R"),
          ("mt_cancel_order", 12, 100e6, "Ask", 100.01, 50, "R", "R"),
          ("mt_add_order", 14, 2e9, "Bid", 100.04, 100, "b2", "")]
    by = {}
    for mt, seq, off, side, price, qty, ref, pref in ev:
        by.setdefault(mt, []).append(dict(
            receipttimestamp=int(BASE + off), exchangetimestamp=int(BASE + off), sequencenumber=seq,
            f="F", product="X", side=side, price=price, quantity=qty, previousprice=price,
            previousquantity=qty, orderreferencenumber=ref, previousorderreferencenumber=pref,
            maintainpriority="false"))
    out = {}
    for mt, rows in by.items():
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["receipttimestamp"], unit="ns", utc=True).dt.tz_convert(ET)
        out[mt] = df
    return out, "X"


def _metrics(book, asset):
    st = book.attrs.get("lob_stats", {})
    n = len(book)
    bb = book.get(f"{asset}_bidprice_1"); ba = book.get(f"{asset}_askprice_1")
    if bb is not None and ba is not None:
        bb = bb.to_numpy(float); ba = ba.to_numpy(float)
        both = np.isfinite(bb) & np.isfinite(ba)
        xf = float(np.mean(bb[both] > ba[both] + 1e-9)) if both.any() else float("nan")
        nb = int(both.sum())
    else:
        xf, nb = float("nan"), 0
    return dict(snapshots=n, both=nb, crossed_frac=xf,
                consolidated_crossed=int(st.get("consolidated_crossed", 0)),
                locked_or_crossed=int(st.get("consolidated_locked_or_crossed", 0)),
                trades=int(st.get("n_trade", 0)), trade_no_ref=int(st.get("trade_no_ref", 0)),
                cancel_no_order=int(st.get("cancel_no_order", 0)),
                resting_eod=int(book.attrs.get("resting_orders_eod", 0)))


def run(msgs, asset, **kw):
    """Reconstruct twice (consume=False so the same messages serve both orderings) and return metrics."""
    leg = _metrics(lob.reconstruct_book(msgs, asset=asset, consume=False, order_by="clock", **kw), asset)
    seq = _metrics(lob.reconstruct_book(msgs, asset=asset, consume=False, order_by="sequence", **kw), asset)
    return leg, seq


def _report(asset, date_str, sess, interval, clock, leg, seq):
    def pct(a, b):
        return f"{a} ({100.0 * a / b:.2f}%)" if b else f"{a}"
    rate = lambda a, b: 100.0 * a / b if b else float("nan")
    L = [f"\nCROSSED-BOOK VERIFICATION -- {asset} {date_str} (session {sess[0]}-{sess[1]}, {interval} grid, clock={clock})",
         f"  {'metric':28s} {'legacy (clock)':>20s} {'fixed (sequence)':>20s}",
         f"  {'snapshots (both sides)':28s} {leg['both']:>20d} {seq['both']:>20d}",
         f"  {'consolidated CROSSED':28s} {pct(leg['consolidated_crossed'], leg['snapshots']):>20s} {pct(seq['consolidated_crossed'], seq['snapshots']):>20s}",
         f"  {'crossed frac (bid1>ask1)':28s} {leg['crossed_frac']:>19.2%} {seq['crossed_frac']:>19.2%}",
         f"  {'locked-or-crossed':28s} {pct(leg['locked_or_crossed'], leg['snapshots']):>20s} {pct(seq['locked_or_crossed'], seq['snapshots']):>20s}",
         f"  {'trades':28s} {leg['trades']:>20d} {seq['trades']:>20d}",
         f"  {'trade_no_ref':28s} {pct(leg['trade_no_ref'], leg['trades']):>20s} {pct(seq['trade_no_ref'], seq['trades']):>20s}",
         f"  {'cancel_no_order':28s} {leg['cancel_no_order']:>20d} {seq['cancel_no_order']:>20d}",
         f"  {'resting orders EOD':28s} {leg['resting_eod']:>20d} {seq['resting_eod']:>20d}",
         f"\n  => crossing {leg['crossed_frac']:.1%} -> {seq['crossed_frac']:.1%}"
         f"  |  trade_no_ref {rate(leg['trade_no_ref'], leg['trades']):.1f}% -> {rate(seq['trade_no_ref'], seq['trades']):.1f}%"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Before/after crossed-book verification for the ordering fix")
    ap.add_argument("--date", default=None, help="YYYYMMDD")
    ap.add_argument("--product", default="SPY")
    ap.add_argument("--product-type", default="direct", choices=["direct", "futures"])
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--session", default="09:30-16:00", help="HH:MM-HH:MM")
    ap.add_argument("--clock", default="exchange", choices=["receipt", "exchange"],
                    help="cross-feed interleave clock (held fixed across both runs)")
    ap.add_argument("--price-scale", type=float, default=1.0, help="0.01 for CME integer-hundredths (ES)")
    ap.add_argument("--data-source", default="apu")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--demo", action="store_true", help="synthetic resurrection (no MIDAS feed needed)")
    args = ap.parse_args(argv)
    s0, s1 = args.session.split("-")

    if args.demo:
        msgs, asset = _demo_messages()
        kw = dict(levels=3, interval=args.interval, session=("10:00", "10:04"), tz=args.tz,
                  date_str="20250403", clock=args.clock, price_scale=1.0)
        date_str, sess = "20250403", ("10:00", "10:04")
    else:
        if not args.date:
            ap.error("--date YYYYMMDD is required (or use --demo)")
        msgs, asset = _fetch_once(args.date, args.product, args.product_type, args.data_source, args.tz, args.clock)
        kw = dict(levels=args.levels, interval=args.interval, session=(s0, s1), tz=args.tz,
                  date_str=args.date, clock=args.clock, price_scale=args.price_scale)
        date_str, sess = args.date, (s0, s1)

    leg, seq = run(msgs, asset, **kw)
    print(_report(asset, date_str, sess, args.interval, args.clock, leg, seq))
    return {"legacy": leg, "fixed": seq}


if __name__ == "__main__":
    main()
