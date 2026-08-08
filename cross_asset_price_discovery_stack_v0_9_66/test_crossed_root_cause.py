"""test_crossed_root_cause.py -- the loader-prunes-sequencenumber fault, end to end.

This guards the specific failure that produced a 99.9%-crossed SPY book while every frame-level check
came back clean (side tokens 100% 'Bid'/'Ask', no NaN prices, no bad quantities, one price scale):

    mstbook_loader._read_messages_csv applies `usecols` built from _MSG_NEEDED_COLS, and
    'sequencenumber' was not in that set. The column was therefore dropped during the CSV read --
    before lob_reconstruct ever saw the frame. _event_arrays(order_by='sequence') found no sequence
    anywhere and fell back, SILENTLY, to its degraded clock-only ordering. On a real multicast feed
    that ordering applies a Cancel before the Modify it precedes; modify() is remove-old + add-new, so
    the Modify re-creates the just-cancelled order. The cancel is spent, the phantom is immortal, and
    it pins the venue's own top crossed for the rest of the session.

Nothing in the stack caught it. Every synthetic test builds message frames directly and hands them to
reconstruct_book, so they all carried a sequencenumber and all exercised the healthy path; the only
code that could drop the column was the one layer the tests bypassed. So this file asserts at BOTH
levels:

  (1) the loader preserves sequencenumber through the real CSV read            -- the regression guard
  (2) with it, the book is clean; without it, the same messages cross          -- the mechanism
  (3) debug_crossing NAMES the missing sequence as a CODE fault, and CHECK 8   -- the diagnosis
      reports the resurrected orders that are actually holding the top apart

(3) matters as much as (1). The diagnostic ran on the real session and returned "CODE (structural:
side/scale/column parsing)" plus a DATA finding about missing messages -- both symptoms, neither the
cause -- because it had no check that could see an ordering fault and no check that could see an order
that should have left. A tool that reaches the wrong conclusion confidently is worse than no tool.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

import debug_crossing as dbg
import lob_reconstruct as lr
import mstbook_loader as ml

NY = "America/New_York"
DATE = "20240103"
OPEN = pd.Timestamp("2024-01-03 09:30:00", tz=NY)
FEEDS = ("bats_edgx", "total_view")
SESSION = ("09:30", "16:00")


# ══════════════════════════════════════════════════════════════════════════════
# (1) the loader must not drop the column
# ══════════════════════════════════════════════════════════════════════════════
def test_loader_keeps_sequencenumber():
    """The exact production read path, on a CSV shaped like lakequery's output."""
    hdr = ("receipttimestamp,sourcetimestamp,sequencenumber,f,side,price,quantity,"
           "orderreferencenumber,previousorderreferencenumber,somethingelse")
    row = "1704290000000000000,1704290000000000000,42,bats_edgx,Bid,470.10,100,R1,,junk"
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w") as fh:
            fh.write(hdr + "\n" + row + "\n")
        df = ml._read_messages_csv(path, "mt_add_order")
    finally:
        os.unlink(path)
    kept = "sequencenumber" in df.columns
    pruned = "somethingelse" not in df.columns          # the prune itself must still work
    print("(1) loader read keeps 'sequencenumber'=%s and still prunes unused columns=%s : %s"
          % (kept, pruned, kept and pruned))
    return kept and pruned


# ══════════════════════════════════════════════════════════════════════════════
# a session whose ONLY fault is the clock disagreeing with the venue's own order
# ══════════════════════════════════════════════════════════════════════════════
def build(with_sequence=True, n_cycles=400):
    """Two venues quoting around a drifting mid. Each cycle an order is Modified and then Cancelled --
    but the two arrive out of clock order, as UDP multicast reordering produces. Sequence order is
    correct throughout, so the venue's own order is recoverable; only the clock is misleading.

    Under sequence ordering: modify, then cancel -> the order leaves. Book never crosses.
    Under clock ordering:    cancel, then modify -> the modify RESURRECTS it. Book crosses as mid drifts.
    """
    adds, cancels, mods = [], [], []
    seq = {f: 0 for f in FEEDS}
    ref_n = 0
    for c in range(n_cycles):
        t0 = OPEN + pd.Timedelta(seconds=6 * c + 1)
        mid = 470.0 + 4.0 * np.sin(c / 60.0)               # drift, so a stale order WILL cross
        for fi, f in enumerate(FEEDS):
            for side, px in (("Bid", round(mid - 0.01 - 0.01 * fi, 2)),
                             ("Ask", round(mid + 0.01 + 0.01 * fi, 2))):
                ref_n += 1
                ref = "R%08d" % ref_n
                seq[f] += 1
                adds.append(dict(ts=t0, f=f, side=side, price=px, quantity=200.0,
                                 orderreferencenumber=ref, sequencenumber=seq[f]))
                # modify (seq n+1) then cancel (seq n+2) -- but the CLOCK has them inverted
                seq[f] += 1
                s_mod = seq[f]
                seq[f] += 1
                s_can = seq[f]
                newpx = round(px + (0.01 if side == "Bid" else -0.01), 2)
                mods.append(dict(ts=t0 + pd.Timedelta(milliseconds=300), f=f, side=side,
                                 price=newpx, quantity=200.0, previousprice=px, previousquantity=200.0,
                                 orderreferencenumber=ref, previousorderreferencenumber=ref,
                                 maintainpriority="false", sequencenumber=s_mod))
                cancels.append(dict(ts=t0 + pd.Timedelta(milliseconds=100), f=f, side=side,
                                    price=newpx, previousquantity=200.0,
                                    orderreferencenumber=ref, sequencenumber=s_can))

    def _frame(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(df.pop("ts"))
        if not with_sequence:                              # what the pruning loader handed the replay
            df = df.drop(columns=["sequencenumber"])
        return df.sort_index()

    return {"mt_add_order": _frame(adds), "mt_cancel_order": _frame(cancels),
            "mt_modify_order": _frame(mods), "mt_trade": pd.DataFrame(),
            "mt_price_level_update": pd.DataFrame()}


def _diagnose(msgs):
    L, findings = [], []
    findings += dbg.check_inventory(msgs, {}, L)
    findings += dbg.check_feed_matrix(msgs, L)
    findings += dbg.check_fields(msgs, L)
    findings += dbg.check_sequence(msgs, L)
    findings += dbg.check_clock(msgs, "receipt", NY, SESSION, DATE, L)
    R = dbg.instrumented_replay(msgs, NY, "receipt", 1.0, SESSION, DATE, "1s", "sequence", L)
    findings += dbg.report_replay(R, L)
    findings += dbg.check_ordering_ab(msgs, NY, "receipt", 1.0, SESSION, DATE, "1s", L, "sequence")
    return findings, R, "\n".join(L)


def main():
    warnings.simplefilter("ignore")
    ok = test_loader_keeps_sequencenumber()

    # (2) the mechanism: identical messages, the column is the only difference
    good = build(with_sequence=True)
    bad = build(with_sequence=False)
    f_ok, R_ok, _ = _diagnose(good)
    f_bad, R_bad, txt_bad = _diagnose(bad)
    x_ok = float(np.mean(R_ok["crossed"]))
    x_bad = float(np.mean(R_bad["crossed"]))
    m_ok = x_ok < 0.01 and x_bad > 0.5
    print("(2) same messages: with sequencenumber crossed=%.1f%%, without it crossed=%.1f%% : %s"
          % (100 * x_ok, 100 * x_bad, m_ok))
    ok &= m_ok

    # the healthy session must be reported clean -- no false alarms from the new checks
    clean_ok = not f_ok
    print("(3) healthy session -> %d findings (want 0) : %s" % (len(f_ok), clean_ok))
    for k, s, m in f_ok:
        print("      unexpected [%s/%s] %s" % (k, s, m[:100]))
    ok &= clean_ok

    # (4) the diagnosis must NAME the missing sequence, not just report the crossing
    named = any(k == "CODE" and "sequencenumber" in m for k, _s, m in f_bad)
    fatal = any(k == "CODE" and s == "FATAL" for k, s, _m in f_bad)
    print("(4) faulty session -> a CODE finding names 'sequencenumber'=%s, CODE/FATAL raised=%s : %s"
          % (named, fatal, named and fatal))
    ok &= named and fatal

    # (5) CHECK 8 must find the resurrected orders that are actually pinning the top
    n_res = sum((R_bad.get("resurrect_by_feed") or {}).values())
    n_ws = sum(len(v) for v in (R_bad.get("wrong_side") or {}).values())
    res_ok = n_res > 0 and n_ws > 0 and "RESURRECTED by a modify" in txt_bad
    print("(5) CHECK 8 -> %s resurrections, %s wrong-side orders resting at the close, named in the "
          "report=%s : %s" % (f"{n_res:,}", f"{n_ws:,}", "RESURRECTED by a modify" in txt_bad, res_ok))
    ok &= res_ok

    # (6) the verdict must lead with CODE, and must NOT send the reader off to re-fetch the data
    L2 = []
    dbg.verdict(f_bad, L2)
    v = "\n".join(L2)
    verdict_ok = "VERDICT: CODE" in v
    print("(6) verdict leads with CODE (not DATA/re-fetch) : %s" % verdict_ok)
    ok &= verdict_ok

    # (7) a low-rate orphan tail must not be able to swing the verdict to DATA on its own
    mixed = [("CODE", "FATAL", "ordering"), ("DATA", "MINOR", "orphan tail"), ("DATA", "MINOR", "more tail")]
    L3 = []
    dbg.verdict(mixed, L3)
    minor_ok = "VERDICT: CODE" in "\n".join(L3)
    print("(7) two DATA/MINOR findings cannot outvote one CODE/FATAL : %s" % minor_ok)
    ok &= minor_ok

    print("\ncrossed-root-cause checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
