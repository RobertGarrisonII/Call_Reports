"""test_debug_crossing.py -- known-answer guard for the crossed-book diagnostic.

A diagnostic that cannot be validated is just more output to stare at. So this builds five sessions
whose fault is KNOWN BY CONSTRUCTION and asserts debug_crossing reaches the right verdict for the
right reason -- not merely that it runs:

  clean        complete, well-formed messages                      -> no findings
  no_cancels   the cancel stream is empty (a failed fetch)          -> DATA, and the book must cross
  bad_side     one feed encodes side as 'B'/'S'                     -> CODE (adds silently dropped)
  code_orphan  cancels precede their own adds (ordering fault)      -> CODE, add PRESENT in messages
  pre_window   cancels for orders resting before 09:30              -> DATA, but flagged BENIGN (early)
  purge_only   a handful of bids lose their in-session cancels and   -> DATA via CHECK 10; the venue's
               are cancelled only in the 16:30 post-close purge         post-close purge hides them from
                                                                        the end-of-stream census (the
                                                                        2017-12-05 signature)

The last two are the pair that matters. Both produce orphaned removals; only one is a code fault, and
the discriminator is whether an add for that reference exists anywhere in the fetched messages. If the
tool cannot separate those two it cannot do its job, so they are asserted explicitly.
"""
import sys
import warnings

import numpy as np
import pandas as pd

import debug_crossing as dbg
import lob_reconstruct as lr

NY = "America/New_York"
DATE = "20230103"
OPEN = pd.Timestamp(f"2023-01-03 09:30:00", tz=NY)
FEEDS = ("bats_edgx", "total_view")


def _frame(rows, tz=NY):
    """rows: list of dicts with a 'ts' key -> a message frame indexed by tz-aware timestamp."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    idx = pd.DatetimeIndex(df.pop("ts"))
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    df.index = idx
    return df.sort_index()


def build(kind):
    """One synthetic session. Base pattern: each feed posts a two-sided quote, then cancels it, over
    and over, so a correctly replayed book is never crossed and never accumulates."""
    adds, cancels, trades = [], [], []
    seq = {f: 0 for f in FEEDS}
    ref_n = 0
    mid = 400.0
    n_cycles = 900                                  # ~ every 4s over 6.5h
    # purge_only: these feed-0 bid cycles (near the sinusoid's peak, so the later drift strands
    # them ABOVE the market) lose their in-session cancel and are cancelled only at 16:30 --
    # after the close, in the venue's purge. Chosen off the c%7 trade cycles to stay clean.
    purge_cycles = {101, 102, 103, 104, 106, 108, 109}
    purge_at_close = []                             # (feed, ref, price) awaiting the 16:30 purge
    if kind == "purge_only":
        # A deep resting base book (far from the market on both sides, resting all day), so the
        # seven stranded bids are a 1.1x blip on the resting count -- as on the real session, where
        # ~900 stuck orders sat in an ~9k-order book -- and CHECK 6 cannot call it accumulation.
        for f in FEEDS:
            for side, px in (("Bid", 300.0), ("Ask", 500.0)):
                for i in range(30):
                    ref_n += 1
                    seq[f] += 1
                    adds.append(dict(ts=OPEN + pd.Timedelta(milliseconds=100 + ref_n), f=f,
                                     side=side, price=px - i * 0.01 if side == "Bid" else px + i * 0.01,
                                     quantity=100.0, orderreferencenumber="R%08d" % ref_n,
                                     sequencenumber=seq[f]))
    for c in range(n_cycles):
        t0 = OPEN + pd.Timedelta(seconds=4 * c + 1)
        mid = 400.0 + 3.0 * np.sin(c / 90.0)        # drift, so stale orders WILL cross if not removed
        cycle_bid = None                            # feed-0 bid of THIS cycle, for the trade
        for fi, f in enumerate(FEEDS):
            for side, px in (("Bid", round(mid - 0.01 - 0.01 * fi, 2)),
                             ("Ask", round(mid + 0.01 + 0.01 * fi, 2))):
                ref_n += 1
                ref = "R%08d" % ref_n
                tok = side
                if kind == "bad_side" and f == "total_view":
                    tok = "B" if side == "Bid" else "S"     # unrecognized encoding -> add() drops it
                seq[f] += 1
                add_seq = seq[f]
                a = dict(ts=t0, f=f, side=tok, price=px, quantity=200.0,
                         orderreferencenumber=ref, sequencenumber=add_seq)
                seq[f] += 1
                x = dict(ts=t0 + pd.Timedelta(seconds=2), f=f, side=tok, price=px,
                         previousquantity=200.0, orderreferencenumber=ref, sequencenumber=seq[f])
                if kind == "code_orphan" and c % 2 == 0:
                    # the add EXISTS but is ordered AFTER its own cancel: an ordering fault. Provenance
                    # must therefore report the add as PRESENT (=> CODE), not missing.
                    a["ts"] = t0 + pd.Timedelta(seconds=3)
                    a["sequencenumber"] = seq[f] + 1
                if kind == "pre_window" and c < 40:
                    # a cancel near the open for an order resting BEFORE the window: the add is
                    # genuinely absent from the fetch, but this is expected, not a fault.
                    cancels.append(x)
                    continue
                if kind == "purge_only":
                    if f == FEEDS[0] and side == "Bid" and c in purge_cycles:
                        adds.append(a)               # rests marketable for hours...
                        purge_at_close.append((f, ref, px))
                        continue                     # ...its cancel arrives only at 16:30
                    if c == n_cycles - 1:
                        adds.append(a)               # the last cycle RESTS: a live two-sided book
                        continue                     # at the close for the census to judge against
                adds.append(a)
                cancels.append(x)
                if fi == 0 and side == "Bid":
                    cycle_bid = (ref, px, add_seq)
        # trade against this cycle's feed-0 bid, sequenced BETWEEN its add and its cancel (intra-feed
        # order is by sequencenumber, so a later seq would arrive after the cancel and orphan itself)
        if c % 7 == 0 and cycle_bid is not None:
            ref, px, add_seq = cycle_bid
            trades.append(dict(ts=t0 + pd.Timedelta(seconds=1), f=FEEDS[0], side="Bid",
                               price=px, quantity=100.0, orderreferencenumber=ref,
                               sequencenumber=add_seq + 0.5))
    for f, ref, px in purge_at_close:                # the venue's post-close purge (16:30 ET)
        seq[f] += 1
        cancels.append(dict(ts=OPEN + pd.Timedelta(hours=7), f=f, side="Bid", price=px,
                            previousquantity=200.0, orderreferencenumber=ref,
                            sequencenumber=seq[f]))
    msgs = {"mt_add_order": _frame(adds),
            "mt_cancel_order": _frame([] if kind == "no_cancels" else cancels),
            "mt_modify_order": pd.DataFrame(),
            "mt_trade": _frame(trades),
            "mt_price_level_update": pd.DataFrame()}
    return msgs


def run(kind):
    msgs = build(kind)
    L, findings = [], []
    findings += dbg.check_inventory(msgs, {}, L)
    findings += dbg.check_feed_matrix(msgs, L)
    findings += dbg.check_fields(msgs, L)
    findings += dbg.check_sequence(msgs, L)
    R = dbg.instrumented_replay(msgs, NY, "exchange", 1.0, ("09:30", "16:00"), DATE, "1s", "sequence", L)
    findings += dbg.report_replay(R, L)
    kinds = {k for k, _s, _m in findings}
    cross = float(np.mean(R["crossed"])) if R is not None else float("nan")
    return findings, kinds, cross, R, "\n".join(L)


def main():
    warnings.simplefilter("ignore")
    ok = True

    # (A) clean session: no findings, and the book genuinely never crosses
    f, kinds, cross, R, _ = run("clean")
    a_ok = (not f) and cross < 0.01
    print("(A) clean       -> findings=%d crossed=%.2f%%  (want 0 findings, ~0%% crossed) : %s"
          % (len(f), 100 * cross, a_ok))
    if not a_ok:
        for k, s, m in f:
            print("      unexpected [%s/%s] %s" % (k, s, m))
    ok &= a_ok

    # (B) empty cancel stream: must be DATA, must be caught BEFORE the replay, and the book must in
    #     fact cross -- proving the fixture reproduces the production symptom.
    f, kinds, cross, R, _ = run("no_cancels")
    pre = dbg.check_inventory(build("no_cancels"), {}, [])
    b_ok = ("DATA" in kinds and "CODE" not in kinds and cross > 0.5
            and any(s == "FATAL" and k == "DATA" for k, s, _m in pre))
    print("(B) no_cancels  -> kinds=%s crossed=%.1f%% caught_pre_replay=%s : %s"
          % (sorted(kinds), 100 * cross, bool(pre), b_ok))
    ok &= b_ok

    # (C) unrecognized side tokens on one feed: CODE, and the message must name the feed
    f, kinds, cross, R, _ = run("bad_side")
    named = any("total_view" in m for k, _s, m in f if k == "CODE")
    c_ok = "CODE" in kinds and named
    print("(C) bad_side    -> kinds=%s names_the_feed=%s : %s" % (sorted(kinds), named, c_ok))
    ok &= c_ok

    # (D) THE discriminator, part 1: cancels before their own adds. The add EXISTS in the messages,
    #     so provenance must attribute this to CODE.
    f, kinds, cross, R, _ = run("code_orphan")
    pres = R["orphan_add_present"]
    absent = R["orphan_add_absent"]
    d_ok = pres > 0 and pres > absent and "CODE" in kinds
    print("(D) code_orphan -> orphans add_present=%s add_absent=%s kinds=%s : %s"
          % (f"{pres:,}", f"{absent:,}", sorted(kinds), d_ok))
    ok &= d_ok

    # (E) THE discriminator, part 2: the add is genuinely absent AND concentrated at the open, so it
    #     must NOT be reported as a code fault, and must be called out as benign pre-window resting.
    f, kinds, cross, R, txt = run("pre_window")
    pres2, abs2 = R["orphan_add_present"], R["orphan_add_absent"]
    benign = "RESTING BEFORE the session window" in txt
    e_ok = abs2 > 0 and abs2 > pres2 and benign and "CODE" not in kinds
    print("(E) pre_window  -> orphans add_present=%s add_absent=%s benign_note=%s kinds=%s : %s"
          % (f"{pres2:,}", f"{abs2:,}", benign, sorted(kinds), e_ok))
    ok &= e_ok

    # (F) D and E must be SEPARATED: same symptom (orphaned removals), opposite verdicts.
    f_ok = ("CODE" in run("code_orphan")[1]) and ("CODE" not in run("pre_window")[1])
    print("(F) same symptom, opposite verdicts (code_orphan=CODE, pre_window=not CODE) : %s" % f_ok)
    ok &= f_ok

    # (G) the 2017-12-05 signature: bids stranded above the market with their cancels arriving only
    #     in the venue's post-close purge. CHECK 5 sees no orphans (every cancel matches), CHECK 8's
    #     end-of-stream census reads 0 (the purge already removed the evidence), and CHECK 7's
    #     'single venue crossed, not accumulating -> CODE' inference fires falsely. CHECK 10 must
    #     take the census AT the close, classify the pins as purge-only, deliver the DATA verdict,
    #     and withdraw the CHECK 7 inference.
    f, kinds, cross, R, txt = run("purge_only")
    ws_post = sum(len(v) for v in (R.get("wrong_side") or {}).values())
    n_pin = len(R.get("pin_life") or [])
    n_purge = sum(1 for p in (R.get("pin_life") or []) if p["cls"] == "purge_only")
    data_msg = any("post-close purge" in m for k, s, m in f if k == "DATA" and s == "SEVERE")
    g_ok = ("CODE" not in kinds and data_msg and cross > 0.5
            and n_pin >= 7 and n_purge >= 7 and ws_post == 0 and "WITHDRAWN" in txt)
    print("(G) purge_only  -> kinds=%s crossed=%.1f%% pins_at_close=%d (purge_only=%d) "
          "post-stream census=%d check7_withdrawn=%s : %s"
          % (sorted(kinds), 100 * cross, n_pin, n_purge, ws_post, "WITHDRAWN" in txt, g_ok))
    if not g_ok:
        for k, s, m in f:
            print("      [%s/%s] %s" % (k, s, m[:140]))
    ok &= g_ok

    print("\ndebug-crossing checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
