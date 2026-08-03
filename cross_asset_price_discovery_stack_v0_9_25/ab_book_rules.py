#!/usr/bin/env python3
"""ab_book_rules.py -- attribute a crossed book to a specific replay rule, from ONE fetch.

v0.9.23 changed three things in the replay at once, and on two of the four March-2020 SPY sessions
the crossed fraction moved sharply when they went in:

    2020-03-09   3.9% ->  3.9%     (unchanged)
    2020-03-18   3.9% ->  3.9%     (unchanged)
    2020-03-16   4.0% -> 11.9%
    2020-03-12   3.9% -> 97.3%

Which rule is responsible is an empirical question, and guessing at it is how the DCC work went
wrong twice. This driver fetches the session's messages ONCE and replays them under every
combination of the three rules, printing the crossed fraction for each. The rule whose OFF column
restores the old number is the one that matters.

Note which direction each result implies:

  * If `consume_undisplayed=ON` restores the old figure, the pre-v0.9.23 behaviour was not
    *avoiding* the crossing -- it was MASKING it. Letting a hidden print consume displayed size is
    wrong in principle (there is no displayed order behind such a print), but it deleted stale
    levels as a side effect. Then the 97.3% is the true staleness of that session's replay, which
    was there all along, and the fix belongs upstream in whatever leaves those levels stale.
  * If `use_leaves=OFF` restores it, some feed populates `leavesquantity` from the aggressor rather
    than the resting order (it is verified only on bats_edgx), and the rule needs a per-feed guard.
  * If `apply_clears=OFF` restores it, a reset is being applied that should not be -- most likely
    ranked wrongly against the messages that rebuild the feed.

    python ab_book_rules.py --date 20200312 --product SPY
    python ab_book_rules.py --date 20200312 --product SPY --save-messages msgs_20200312.pkl
    python ab_book_rules.py --date 20200312 --product SPY --load-messages msgs_20200312.pkl

Exit code is always 0 unless the run fails: this is a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys

import numpy as np
import pandas as pd

import lob_reconstruct as lob

_MTS = ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade",
        "mt_price_level_update", "mt_clear_orders", "mt_clear_price_levels")
_RULES = ("apply_clears", "use_leaves", "consume_undisplayed")


def fetch_once(date_str, product, ptype, data_source, tz, clock):
    """One fetch, reused for every replay -- the point of the exercise is that only the RULES vary."""
    import mstbook_loader as ml
    msgs = {}
    for mt in _MTS:
        try:
            msgs[mt] = ml._fetch_messages(date_str, product, ptype, mt, data_source, tz=tz, clock=clock)
            print(f"    {mt:28s} {len(msgs[mt]):>12,d} rows", file=sys.stderr)
        except Exception as exc:
            msgs[mt] = pd.DataFrame()
            print(f"    {mt:28s} fetch failed: {str(exc).splitlines()[0][:120]}", file=sys.stderr)
    return msgs


def replay(msgs, rules, asset, levels, interval, session, tz, date_str, clock, price_scale):
    """Replay a DEEP COPY under one rule set. consume=False and a copy per run: reconstruct_book
    pops frames in consume mode, and a shared frame mutated by the first replay would make every
    later one a different experiment."""
    m = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in msgs.items()}
    a, b = session.split("-")
    df = lob.reconstruct_book(m, asset=asset, levels=levels, interval=interval, session=(a, b),
                              tz=tz, date_str=date_str, clock=clock, price_scale=price_scale,
                              consume=True, rules=rules)
    st = dict(df.attrs.get("lob_stats", {}))
    bid = pd.to_numeric(df.get(f"{asset}_bidprice_1"), errors="coerce").to_numpy(float)
    ask = pd.to_numeric(df.get(f"{asset}_askprice_1"), errors="coerce").to_numpy(float)
    both = np.isfinite(bid) & np.isfinite(ask)
    crossed = float((bid[both] > ask[both] + lob.EPS).mean()) if both.any() else np.nan
    return crossed, int(both.sum()), st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attribute a crossed book to a replay rule")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--product", default="SPY")
    ap.add_argument("--product-type", default="direct", choices=["direct", "futures"])
    ap.add_argument("--price-scale", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--session", default="09:30-16:00")
    ap.add_argument("--data-source", default="apu")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--clock", default="receipt", choices=["receipt", "exchange"])
    ap.add_argument("--save-messages", default="", help="pickle the fetched messages for re-use")
    ap.add_argument("--load-messages", default="", help="replay from a saved pickle (no vendor I/O)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    date_str = a.date.replace("-", "")
    asset = "X"

    if a.load_messages:
        with open(a.load_messages, "rb") as fh:
            msgs = pickle.load(fh)
        print(f"replaying from {a.load_messages}", file=sys.stderr)
    else:
        print(f"fetching {a.product} {date_str} once:", file=sys.stderr)
        msgs = fetch_once(date_str, a.product, a.product_type, a.data_source, a.tz, a.clock)
        if a.save_messages:
            with open(a.save_messages, "wb") as fh:
                pickle.dump(msgs, fh, protocol=4)
            print(f"saved messages -> {a.save_messages}", file=sys.stderr)

    rows = []
    for combo in itertools.product((True, False), repeat=len(_RULES)):
        rules = dict(zip(_RULES, combo))
        crossed, n, st = replay(msgs, rules, asset, a.levels, a.interval, a.session, a.tz,
                                date_str, a.clock, a.price_scale)
        rows.append({**{k: ("on" if v else "off") for k, v in rules.items()},
                     "crossed": crossed, "n_snap": n,
                     "resets": st.get("feed_clears", 0),
                     "cleared_orders": st.get("orders_cleared", 0),
                     "leaves_corr": st.get("trade_leaves_corrected", 0),
                     "leaves_gt_size": st.get("trade_leaves_gt_size", 0),
                     "refless_disp": st.get("trade_no_ref_displayed", 0),
                     "undisplayed": st.get("trade_undisplayed", 0)})
    tbl = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    show = tbl.copy()
    show["crossed"] = show["crossed"].map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.2%}")
    lines = [f"replay-rule A/B: {a.product} {date_str} ({a.product_type}), grid {a.interval} over "
             f"{a.session}, clock={a.clock}",
             "one fetch, %d replays -- only the RULES differ" % len(tbl), "",
             show.to_string(index=False), ""]

    # the current default, and the single-rule flips away from it
    base = tbl[(tbl.apply_clears == "on") & (tbl.use_leaves == "on") & (tbl.consume_undisplayed == "off")]
    if len(base):
        b = float(base["crossed"].iloc[0])
        lines.append("current default (clears=on, leaves=on, consume_undisplayed=off): %.2f%% crossed"
                     % (100 * b))
        deltas = []
        for r in _RULES:
            flip = {"apply_clears": "on", "use_leaves": "on", "consume_undisplayed": "off"}
            flip[r] = "off" if flip[r] == "on" else "on"
            row = tbl[(tbl.apply_clears == flip["apply_clears"]) & (tbl.use_leaves == flip["use_leaves"])
                      & (tbl.consume_undisplayed == flip["consume_undisplayed"])]
            if len(row):
                v = float(row["crossed"].iloc[0])
                deltas.append((abs(v - b), r, v))
        deltas.sort(reverse=True)
        lines.append("")
        lines.append("flipping ONE rule at a time from that default:")
        for d, r, v in deltas:
            lines.append("    %-20s -> %6.2f%% crossed   (moves %.2f pp)" % (r, 100 * v, 100 * d))
        if deltas and deltas[0][0] > 0.01:
            top = deltas[0]
            lines += ["", "%s accounts for most of the movement (%.1f pp)." % (top[1], 100 * top[0])]
            if top[1] == "consume_undisplayed":
                lines += ["Read that carefully: turning it ON lowers the crossed fraction because a "
                          "hidden print then deletes DISPLAYED size at its price. That is wrong in "
                          "principle -- there is no displayed order behind such a print -- so the old "
                          "low number was a MASK, not a correct book. The staleness is real and the "
                          "fix belongs wherever those levels are left stale, not in restoring the "
                          "spurious deletion."]
            elif top[1] == "use_leaves":
                lines += ["leavesquantity is verified only on bats_edgx. If another feed populates it "
                          "from the AGGRESSOR, the rule needs a per-feed guard -- check the "
                          "leaves_gt_size column, which counts prints claiming more left than we hold."]
            elif top[1] == "apply_clears":
                lines += ["A reset is being applied that should not be. Check the cleared_orders "
                          "column and when the resets land relative to the messages that rebuild "
                          "that feed (feed_health.py lists them with timestamps)."]
        elif deltas:
            lines += ["", "No single rule explains the crossing -- it is not attributable to the "
                      "v0.9.23 replay changes. Look upstream: feed_health.py for capture gaps, and "
                      "debug_crossing.py CHECK 5/8 for orphaned removals and pinned levels."]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ab_book_rules: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
