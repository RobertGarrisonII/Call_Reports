#!/usr/bin/env python3
"""test_feed_reset.py -- three message-level facts the replay was ignoring, from the SPY tape.

A sample pull of every mt_* type for SPY on 2024-12-18 showed the reconstruction was consuming four
message types and ignoring what the rest of the tape says. Three of those omissions change the book:

  (1) mt_clear_orders / mt_clear_price_levels -- a FEED RESET. The venue disowns everything it has
      published and rebuilds from the next message. The 2024-12-18 tape carries one at 05:25:44 ET on
      miax_pearl_equities_dom, in the pre-market, with the replay already holding that feed's state
      (the replay starts from the first message of the day and only the SNAPSHOT GRID is 09:30-16:00).
      Unapplied, every order that venue disowned rests in the consolidated ladder for the rest of the
      session -- the venue never cancels them, because as far as it is concerned they are gone. A
      stale pre-market bid pinned that way sits above the current ask for hours.

  (2) mt_trade.leavesquantity -- the venue's own remaining size on the resting order AFTER the
      execution. The tape proves the semantics: order 3189381831070546706 prints leaves 1198 -> 1194
      -> 1192 on trades of 2, 4 and 2. Authoritative where a decrement is only arithmetic, and
      leaves=0 removes a filled order deterministically.

  (3) mt_trade.executionattribute='Hidden' / printable='NonPrintable' -- an execution against
      NON-DISPLAYED liquidity. It carries no orderreferencenumber by construction. Consuming displayed
      size at its price deletes liquidity that is still resting, and counting it as a reference-match
      failure inflates the diagnostic that is supposed to detect one.

Run: python test_feed_reset.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import lob_reconstruct as lob

TZ = "America/New_York"
D = "2024-12-18"


def _f(mtype, rows, cols):
    """Build a message frame with a receipt-clock index, like _fetch_messages returns."""
    df = pd.DataFrame(rows, columns=cols)
    idx = pd.to_datetime([f"{D} {t}" for t in df.pop("t")]).tz_localize(TZ)
    return df.set_index(idx)


def _book(msgs, session=("09:30", "09:31")):
    return lob.reconstruct_book(msgs, asset="SPY", levels=3, interval="1s", session=session,
                                tz=TZ, date_str="20241218", clock="receipt")


# ── (1) a feed reset must wipe that venue, and only that venue ───────────────────────────────────
def check_feed_reset():
    # miax rests a stale pre-market bid at 605.50; the market then trades around 604.
    # arca quotes the real book. Without the reset the stale 605.50 bid sits above arca's 604.90 ask.
    adds = _f("mt_add_order", [
        ["08:00:00", 1, "miax_pearl_equities_dom", "Bid", 605.50, 1000, "M1"],
        ["09:29:00", 1, "xdp_arca_integrated", "Bid", 604.80, 500, "A1"],
        ["09:29:01", 2, "xdp_arca_integrated", "Ask", 604.90, 500, "A2"],
    ], ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    clears = _f("mt_clear_orders", [
        ["08:30:00", 2, "miax_pearl_equities_dom"],
    ], ["t", "sequencenumber", "f"])

    without = _book({"mt_add_order": adds.copy()})
    with_ = _book({"mt_add_order": adds.copy(), "mt_clear_orders": clears.copy()})

    bb_w, ba_w = without["SPY_bidprice_1"].iloc[0], without["SPY_askprice_1"].iloc[0]
    bb_r, ba_r = with_["SPY_bidprice_1"].iloc[0], with_["SPY_askprice_1"].iloc[0]
    crossed_without = bb_w > ba_w + lob.EPS
    clean_with = np.isclose(bb_r, 604.80) and np.isclose(ba_r, 604.90) and bb_r < ba_r
    arca_survives = clean_with                      # the OTHER venue is untouched by miax's reset
    st = with_.attrs["lob_stats"]
    counted = st.get("feed_clears", 0) == 1 and st.get("orders_cleared", 0) == 1

    ok = crossed_without and clean_with and arca_survives and counted
    print("(1) stale pre-market order on a reset feed:")
    print("    reset IGNORED  -> top %.2f / %.2f  crossed=%s  (the phantom pins the book)"
          % (bb_w, ba_w, crossed_without))
    print("    reset APPLIED  -> top %.2f / %.2f  crossed=%s  (other venues untouched)"
          % (bb_r, ba_r, bb_r > ba_r + lob.EPS))
    print("    stats record the reset (feed_clears=1, orders_cleared=1) : %s" % counted)
    print("    -> %s" % ok)
    return ok


def check_reset_then_readd():
    """A reset must not eat the re-adds that follow it, including in the same sequence tie."""
    adds = _f("mt_add_order", [
        ["09:00:00", 5, "bats_edgx", "Bid", 600.00, 100, "B1"],
        ["09:29:00", 9, "bats_edgx", "Bid", 604.80, 300, "B2"],      # same seq as the clear
        ["09:29:00", 9, "bats_edgx", "Ask", 604.90, 300, "B3"],
    ], ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    clears = _f("mt_clear_orders", [["09:29:00", 9, "bats_edgx"]], ["t", "sequencenumber", "f"])
    b = _book({"mt_add_order": adds, "mt_clear_orders": clears})
    bb, ba = b["SPY_bidprice_1"].iloc[0], b["SPY_askprice_1"].iloc[0]
    l2 = b["SPY_bidprice_2"].iloc[0]
    ok = np.isclose(bb, 604.80) and np.isclose(ba, 604.90) and not np.isfinite(l2)
    print("(2) reset tied with the re-adds in one packet: the pre-reset 600.00 is gone, the re-adds "
          "survive (top %.2f/%.2f, no L2 bid) : %s" % (bb, ba, ok))
    return ok


# ── (3) leavesquantity is authoritative ──────────────────────────────────────────────────────────
def check_leaves_quantity():
    adds = _f("mt_add_order", [
        ["09:29:00", 1, "bats_edgx", "Ask", 605.00, 1200, "R1"],
        ["09:29:01", 2, "bats_edgx", "Bid", 604.00, 500, "R2"],
    ], ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    # the tape's own arithmetic: 1200 -> 1198 -> 1194 -> 1192 on 2, 4, 2
    trades = _f("mt_trade", [
        ["09:29:10", 3, "bats_edgx", 605.00, 2, "R1", 1198, "Visible", "Printable"],
        ["09:29:11", 4, "bats_edgx", 605.00, 4, "R1", 1194, "Visible", "Printable"],
        ["09:29:12", 5, "bats_edgx", 605.00, 2, "R1", 1192, "Visible", "Printable"],
    ], ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber", "leavesquantity",
        "executionattribute", "printable"])
    b = _book({"mt_add_order": adds.copy(), "mt_trade": trades.copy()})
    q = b["SPY_askquantity_1"].iloc[0]
    tracks = np.isclose(q, 1192)

    # the case that matters: a FULL fill. leaves=0 must remove the order outright.
    full = _f("mt_trade", [
        ["09:29:10", 3, "bats_edgx", 605.00, 1200, "R1", 0, "Visible", "Printable"],
    ], ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber", "leavesquantity",
        "executionattribute", "printable"])
    b2 = _book({"mt_add_order": adds.copy(), "mt_trade": full})
    gone = not np.isfinite(b2["SPY_askprice_1"].iloc[0])

    # and it CORRECTS a drifted size rather than compounding the error: the resting order really has
    # 1200, but a print claims 900 remain. The venue is right.
    drift = _f("mt_trade", [
        ["09:29:10", 3, "bats_edgx", 605.00, 2, "R1", 900, "Visible", "Printable"],
    ], ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber", "leavesquantity",
        "executionattribute", "printable"])
    b3 = _book({"mt_add_order": adds.copy(), "mt_trade": drift})
    corrected = (np.isclose(b3["SPY_askquantity_1"].iloc[0], 900)
                 and b3.attrs["lob_stats"].get("trade_leaves_corrected", 0) == 1)

    ok = tracks and gone and corrected
    print("(3) leavesquantity: 1200 -(2,4,2)-> %.0f (want 1192)=%s | leaves=0 removes the order=%s | "
          "a drifted size is corrected to the venue's figure, not compounded=%s : %s"
          % (q, tracks, gone, corrected, ok))
    return ok


# ── (4) undisplayed executions consume nothing and are counted apart ─────────────────────────────
def check_undisplayed_trade():
    adds = _f("mt_add_order", [
        ["09:29:00", 1, "xdp_arca_integrated", "Bid", 604.87, 500, "A1"],
        ["09:29:01", 2, "xdp_arca_integrated", "Ask", 605.02, 500, "A2"],
    ], ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    cols = ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber", "leavesquantity",
            "executionattribute", "printable"]
    # a hidden midpoint-ish print AT a displayed price, with no reference -- exactly the tape's shape
    hidden = _f("mt_trade", [["09:29:10", 3, "xdp_arca_integrated", 604.87, 300, "", np.nan,
                              "Hidden", "NonPrintable"]], cols)
    # the same print, but displayed and referencing an order we never saw (the crash-day case)
    displayed = _f("mt_trade", [["09:29:10", 3, "xdp_arca_integrated", 604.87, 300, "GHOST", np.nan,
                                 "Visible", "Printable"]], cols)

    bh = _book({"mt_add_order": adds.copy(), "mt_trade": hidden})
    bd = _book({"mt_add_order": adds.copy(), "mt_trade": displayed})
    qh = bh["SPY_bidquantity_1"].iloc[0]
    qd = bd["SPY_bidquantity_1"].iloc[0]
    sh, sd = bh.attrs["lob_stats"], bd.attrs["lob_stats"]

    kept = np.isclose(qh, 500)                       # hidden print removed nothing
    consumed = np.isclose(qd, 200)                   # displayed refless print consumed 300
    split = (sh.get("trade_undisplayed", 0) == 1 and sh.get("trade_no_ref_displayed", 0) == 0
             and sd.get("trade_no_ref_displayed", 0) == 1 and sd.get("trade_undisplayed", 0) == 0)
    ok = kept and consumed and split
    print("(4) a refless HIDDEN print leaves the displayed 500 intact (got %.0f)=%s; a refless "
          "DISPLAYED print still consumes (500->%.0f)=%s; the two are counted separately=%s : %s"
          % (qh, kept, qd, consumed, split, ok))
    return ok


def check_rules_switchable():
    """Each v0.9.23 rule must be independently switchable, and leavesquantity must never GROW an order.

    The three rules went in together and two March-2020 sessions moved sharply (4.0%->11.9%,
    3.9%->97.3%). Attribution needs the ability to replay the same messages with one rule flipped;
    ab_book_rules.py does that, and this pins the mechanism it relies on."""
    adds = _f("mt_add_order", [
        ["09:00:00", 1, "venueA", "Bid", 300.50, 500, "S1"],
        ["09:29:00", 2, "venueB", "Bid", 280.00, 500, "B1"],
        ["09:29:01", 3, "venueB", "Ask", 280.10, 500, "B2"],
    ], ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    hidden = _f("mt_trade", [["09:29:30", 4, "venueA", 300.50, 500, "", np.nan, "Hidden",
                              "NonPrintable"]],
                ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber",
                 "leavesquantity", "executionattribute", "printable"])

    def _crossed(rules):
        b = lob.reconstruct_book({"mt_add_order": adds.copy(), "mt_trade": hidden.copy()},
                                 asset="SPY", levels=2, interval="1s", session=("09:30", "09:31"),
                                 tz=TZ, date_str="20241218", clock="receipt", rules=rules)
        bid = b["SPY_bidprice_1"].to_numpy(float); ask = b["SPY_askprice_1"].to_numpy(float)
        m = np.isfinite(bid) & np.isfinite(ask)
        return float((bid[m] > ask[m] + lob.EPS).mean()) if m.any() else np.nan

    default = _crossed(None)
    consume_on = _crossed({"consume_undisplayed": True})
    switchable = np.isclose(default, 1.0) and np.isclose(consume_on, 0.0)

    try:
        lob.reconstruct_book({"mt_add_order": adds.copy()}, asset="SPY", levels=2, interval="1s",
                             session=("09:30", "09:31"), tz=TZ, date_str="20241218",
                             rules={"no_such_rule": True})
        typo_caught = False
    except ValueError as exc:
        typo_caught = "no_such_rule" in str(exc)

    # leavesquantity must never GROW a resting order: a trade cannot add liquidity.
    grow_adds = _f("mt_add_order", [["09:29:00", 1, "bats_edgx", "Bid", 280.00, 100, "G1"],
                                    ["09:29:01", 2, "bats_edgx", "Ask", 280.10, 100, "G2"]],
                   ["t", "sequencenumber", "f", "side", "price", "quantity", "orderreferencenumber"])
    grow_tr = _f("mt_trade", [["09:29:30", 3, "bats_edgx", 280.00, 5, "G1", 9999, "Visible",
                               "Printable"]],
                 ["t", "sequencenumber", "f", "price", "quantity", "orderreferencenumber",
                  "leavesquantity", "executionattribute", "printable"])
    gb = lob.reconstruct_book({"mt_add_order": grow_adds, "mt_trade": grow_tr}, asset="SPY",
                              levels=2, interval="1s", session=("09:30", "09:31"), tz=TZ,
                              date_str="20241218", clock="receipt")
    q = gb["SPY_bidquantity_1"].iloc[0]
    no_grow = np.isclose(q, 100) and gb.attrs["lob_stats"].get("trade_leaves_gt_size", 0) == 1
    not_crossed = gb["SPY_bidprice_1"].iloc[0] < gb["SPY_askprice_1"].iloc[0]

    ok = switchable and typo_caught and no_grow and not_crossed
    print("(5) rules are independently switchable (default %.0f%% crossed, consume_undisplayed=on "
          "%.0f%%)=%s; an unknown rule name raises=%s" % (100 * default, 100 * consume_on,
                                                          switchable, typo_caught))
    print("    leavesquantity=9999 against a 100-lot order does NOT inflate the level (stayed %.0f, "
          "counted as leaves_gt_size)=%s, and the book stays uncrossed=%s : %s"
          % (q, no_grow, not_crossed, ok))
    return ok


def main():
    checks = [check_feed_reset, check_reset_then_readd, check_leaves_quantity,
              check_undisplayed_trade, check_rules_switchable]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            import traceback; traceback.print_exc()
            print("%s RAISED: %s: %s" % (fn.__name__, type(exc).__name__, exc))
            res.append(False)
        print()
    ok = all(res)
    print("feed-reset / trade-semantics checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
