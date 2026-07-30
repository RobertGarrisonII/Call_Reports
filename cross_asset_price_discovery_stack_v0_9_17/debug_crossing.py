#!/usr/bin/env python3
"""debug_crossing.py -- localize a crossed-book invariant violation to DATA or CODE.

A crossed top (bid1 > ask1) is impossible in a real matching engine, so it is always OUR fault -- but
"our fault" splits into two very different repairs:

  DATA  the messages needed to keep the book correct never reached the replay. Nothing in
        lob_reconstruct can fix this; the fetch/feed has to be repaired or the session dropped.
  CODE  the messages WERE present and the replay still failed to apply them (ordering, reference
        matching, side/qty parsing, modify semantics).

The discriminator this tool is built around is ORPHAN PROVENANCE. When a cancel or trade references a
resting order the book does not have, ask whether an ADD for that same reference exists ANYWHERE in the
fetched messages:
      add present  -> we had the data and still failed to match it        => CODE
      add absent   -> the removal instruction arrived, the creation never => DATA
and split "absent" by time-of-day, because an orphan in the first minutes is usually a legitimate order
that was resting before the session window opened (benign, expected), while orphans spread evenly
across the session are genuinely missing messages.

Two silent-failure paths in the production stack make this diagnosis necessary rather than optional:

  1. reconstruct_session() catches a fetch exception per message type and substitutes an EMPTY frame,
     logging a warning and continuing. If mt_cancel_order fails, the replay sees ZERO cancels, no order
     is ever removed, the book grows without bound, and the top crosses on essentially every snapshot.
     The run still "succeeds" and writes a full-length frame of garbage.
  2. _Book.add() silently DROPS any order whose side is not exactly "Bid"/"Ask" or whose price/size is
     not finite and positive. A venue that encodes side differently, or a renamed quantity column,
     therefore contributes no liquidity at all -- and if that loss is one-sided, the surviving side
     goes stale and the consolidated top crosses.

Both are invisible in the output frame. Both are found by CHECK 1-3 below in seconds.

Usage
    python debug_crossing.py --date 20230103 --product SPY
    python debug_crossing.py --date 20230103 --product ESM3 --product-type future --price-scale 0.01
    python debug_crossing.py --messages-pickle msgs.pkl --asset SPY
    python debug_crossing.py --date 20230103 --product SPY --save-messages msgs.pkl   # fetch once, iterate offline
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import lob_reconstruct as lr

EPS = getattr(lr, "EPS", 1e-9)
BENIGN_OPEN_MINUTES = 5          # orphans this early are usually orders resting from before the window
# Above this share of all cancel/trade events, "the add is missing" stops being explainable by orders
# carried in from a prior session and starts meaning the capture lost messages. Below it the two are
# indistinguishable from the time profile alone (see CHECK 5), so the finding is ranked MINOR.
ORPHAN_RATE_SEVERE = 0.01
SEV_RANK = {"FATAL": 0, "SEVERE": 1, "MINOR": 2}


# ══════════════════════════════════════════════════════════════════════════════
# input
# ══════════════════════════════════════════════════════════════════════════════
def fetch_messages(date_str, product, product_type, data_source, clock, tz, message_types):
    """Fetch via the SAME path production uses -- but surface per-type failures instead of swallowing
    them into an empty frame, since an empty stream is the single most destructive fault here."""
    import mstbook_loader as ml
    msgs, failures = {}, {}
    for mt in message_types:
        try:
            msgs[mt] = ml._fetch_messages(date_str, product, product_type, mt, data_source, tz=tz, clock=clock)
        except Exception as exc:                                    # noqa: BLE001 -- we report, not handle
            msgs[mt] = pd.DataFrame()
            failures[mt] = repr(exc)
    return msgs, failures


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1-3: the frames, before any replay
# ══════════════════════════════════════════════════════════════════════════════
def check_inventory(msgs, failures, L):
    """Per-message-type row counts. An EMPTY removal stream is fatal on its own."""
    L.append("CHECK 1  message inventory")
    findings = []
    for mt in ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade", "mt_price_level_update"):
        df = msgs.get(mt)
        n = 0 if df is None else len(df)
        note = ""
        if mt in failures:
            note = "  <-- FETCH FAILED: %s" % failures[mt][:90]
        elif n == 0:
            note = "  <-- EMPTY"
        L.append("    %-24s %12s%s" % (mt, f"{n:,}", note))
    n_add = len(msgs.get("mt_add_order", []))
    n_cancel = len(msgs.get("mt_cancel_order", []))
    n_modify = len(msgs.get("mt_modify_order", []))
    n_level = len(msgs.get("mt_price_level_update", []))
    if n_add and not (n_cancel or n_modify or n_level):
        findings.append(("DATA", "FATAL", "adds present but NO removal stream (cancel/modify/level) -- "
                                          "nothing can ever leave the book, so it must cross"))
    elif n_add and n_cancel and n_cancel < 0.2 * n_add:
        findings.append(("DATA", "SEVERE", "cancels are only %.1f%% of adds -- a healthy equity feed "
                                           "cancels the large majority of orders; the stream looks truncated"
                         % (100.0 * n_cancel / max(n_add, 1))))
    if failures:
        findings.append(("DATA", "FATAL", "%d message type(s) failed to fetch and were replaced by EMPTY "
                                          "frames: %s" % (len(failures), ", ".join(sorted(failures)))))
    if not n_add and not n_level:
        findings.append(("DATA", "FATAL", "no adds and no level updates -- there is no book to build"))
    return findings


def check_feed_matrix(msgs, L):
    """Per-feed x per-type counts. One venue missing its removal stream crosses the CONSOLIDATED top
    even when every other venue is perfect."""
    L.append("")
    L.append("CHECK 2  per-feed message matrix")
    types = [mt for mt in msgs if len(msgs.get(mt, []))]
    feeds = set()
    counts = defaultdict(Counter)
    for mt in types:
        f = lr._col(msgs[mt], "f", "").astype(str)
        vc = f.value_counts()
        for name, k in vc.items():
            feeds.add(name)
            counts[name][mt] = int(k)
    if not feeds:
        L.append("    (no feed column found)")
        return []
    short = {mt: mt.replace("mt_", "").replace("_order", "")[:6] for mt in types}
    L.append("    %-26s %s" % ("feed", " ".join("%9s" % short[mt] for mt in types)))
    findings = []
    for name in sorted(feeds):
        L.append("    %-26s %s" % (name[:26], " ".join("%9s" % f"{counts[name][mt]:,}" for mt in types)))
        adds = counts[name].get("mt_add_order", 0)
        rem = (counts[name].get("mt_cancel_order", 0) + counts[name].get("mt_modify_order", 0)
               + counts[name].get("mt_price_level_update", 0))
        if adds > 1000 and rem == 0:
            findings.append(("DATA", "FATAL", "feed '%s' has %s adds and ZERO removals -- its orders "
                                              "never leave and will pin the consolidated top" % (name, f"{adds:,}")))
        elif adds > 1000 and rem < 0.2 * adds:
            findings.append(("DATA", "SEVERE", "feed '%s' removals are %.1f%% of its adds -- stream looks partial"
                             % (name, 100.0 * rem / adds)))
    return findings


def check_fields(msgs, L):
    """Side tokens and NaN rates per feed. _Book.add() DROPS anything that fails these, silently."""
    L.append("")
    L.append("CHECK 3  field health (add stream)")
    findings = []
    add = msgs.get("mt_add_order")
    if add is None or not len(add):
        L.append("    (no add stream)")
        return findings
    f = lr._col(add, "f", "").astype(str).to_numpy()
    side = lr._col(add, "side", "").astype(str).str.strip().to_numpy()
    price = pd.to_numeric(lr._col(add, "price"), errors="coerce").to_numpy(float)
    qty = pd.to_numeric(lr._col(add, lr._QTY["mt_add_order"]), errors="coerce").to_numpy(float)
    ref = lr._col(add, "orderreferencenumber", "").astype(str).to_numpy()
    for name in sorted(set(f)):
        m = f == name
        toks = Counter(side[m])
        recognized = toks.get("Bid", 0) + toks.get("Ask", 0)
        frac_ok = recognized / max(m.sum(), 1)
        bad_price = float(np.mean(~np.isfinite(price[m])))
        bad_qty = float(np.mean(~(np.isfinite(qty[m]) & (qty[m] > 0))))
        bad_ref = float(np.mean((ref[m] == "") | (ref[m] == "nan")))
        top = ", ".join("%r:%s" % (k, f"{v:,}") for k, v in toks.most_common(4))
        L.append("    %-26s side_ok=%5.1f%%  nan_price=%5.1f%%  bad_qty=%5.1f%%  no_ref=%5.1f%%"
                 % (name[:26], 100 * frac_ok, 100 * bad_price, 100 * bad_qty, 100 * bad_ref))
        L.append("        side tokens: %s" % top)
        dropped = 1.0 - (1.0 - bad_price) * (1.0 - bad_qty) * frac_ok
        if frac_ok < 0.99:
            findings.append(("CODE", "FATAL" if frac_ok < 0.5 else "SEVERE",
                             "feed '%s': %.1f%% of adds have a side token the parser does not recognize "
                             "(it accepts only exactly 'Bid'/'Ask'); those orders are DROPPED by _Book.add"
                             % (name, 100 * (1 - frac_ok))))
        if bad_price > 0.01 or bad_qty > 0.01:
            findings.append(("CODE", "SEVERE",
                             "feed '%s': %.1f%% non-finite price / %.1f%% non-positive qty on adds -- "
                             "dropped silently; check the column names for this venue"
                             % (name, 100 * bad_price, 100 * bad_qty)))
        if dropped > 0.02:
            L.append("        => ~%.1f%% of this feed's adds never enter the book" % (100 * dropped))
    # price-scale sanity: a venue on a different scale crosses the consolidated top by construction
    med = {}
    for name in sorted(set(f)):
        m = (f == name) & np.isfinite(price)
        if m.sum() > 100:
            med[name] = float(np.median(price[m]))
    if len(med) > 1:
        lo, hi = min(med.values()), max(med.values())
        if hi > 5 * lo:
            findings.append(("DATA", "FATAL",
                             "median add price differs by %.0fx across feeds (%s) -- a venue is on a "
                             "different price scale; the consolidated top is crossed by construction"
                             % (hi / lo, ", ".join("%s=%.4g" % (k, v) for k, v in med.items()))))
    return findings


def check_sequence(msgs, L):
    """Per-feed sequence continuity AND -- more importantly -- whether the sequence exists at all.

    ``sequencenumber`` is not a nice-to-have here: it is the venue's authoritative within-feed event
    order, and ``lob_reconstruct._event_arrays`` uses it to order each feed, interleaving feeds by the
    clock only. Without it the replay drops into its documented DEGRADED mode -- clock-only ordering --
    where adjacent events invert, a Modify can re-add an order a Cancel already removed, and the
    resulting phantom level crosses the top on nearly every snapshot. ``test_reconstruct_ordering.py``
    measures exactly that on a fixture: 100% crossed under clock ordering, 0% under sequence ordering.

    So an absent sequence column is not context for the report, it is a CODE finding in its own right,
    and it must outrank the downstream symptoms it causes -- otherwise the tool reports the crossing
    (true) and the orphans (true) and leaves the reader to guess at the cause.
    """
    L.append("")
    L.append("CHECK 4  per-feed sequence continuity")
    findings = []
    nonempty = {mt: df for mt, df in msgs.items() if df is not None and len(df)}
    have = sorted(mt for mt, df in nonempty.items() if "sequencenumber" in df.columns)
    missing = sorted(mt for mt, df in nonempty.items() if "sequencenumber" not in df.columns)
    seq_by_feed = defaultdict(list)
    inv_by_feed = {}
    for mt, df in nonempty.items():
        s = pd.to_numeric(lr._col(df, "sequencenumber"), errors="coerce").to_numpy(float)
        f = lr._col(df, "f", "").astype(str).to_numpy()
        for name in set(f):
            v = s[(f == name) & np.isfinite(s)]
            if v.size:
                seq_by_feed[name].append(v)
    if not seq_by_feed:
        L.append("    NO 'sequencenumber' COLUMN ON ANY FRAME (%s)" % ", ".join(sorted(nonempty)))
        L.append("    => lob_reconstruct._event_arrays(order_by='sequence') cannot sequence anything and")
        L.append("       silently falls back to clock-only intra-feed ordering -- its DEGRADED mode, and")
        L.append("       the exact ordering that test_reconstruct_ordering.py measures at 100% crossed.")
        findings.append(("CODE", "FATAL",
                         "no message frame carries 'sequencenumber', so the replay orders each feed by "
                         "the clock alone -- the degraded mode the sequence path exists to replace. This "
                         "resurrects cancelled orders via modify and crosses the book on nearly every "
                         "snapshot regardless of data quality. Check that the fetch SELECTS the column: "
                         "mstbook_loader._MSG_NEEDED_COLS prunes every column not listed in it, and a "
                         "column pruned there is gone before the replay ever sees the frame"))
        return findings
    if missing:
        L.append("    'sequencenumber' present on: %s" % ", ".join(have))
        L.append("    'sequencenumber' MISSING on: %s" % ", ".join(missing))
        findings.append(("CODE", "SEVERE",
                         "message type(s) %s carry no 'sequencenumber' while others do -- _event_arrays "
                         "sorts unsequenced events LAST within their feed, so every one of those events "
                         "is applied at end of session instead of at its true position"
                         % ", ".join(missing)))
    # How much the clock actually disagrees with the venue's own order. This is the size of the fault
    # the sequence path repairs: every inversion is a pair the degraded ordering would apply backwards.
    for mt, df in nonempty.items():
        if "sequencenumber" not in df.columns:
            continue
        s = pd.to_numeric(df["sequencenumber"], errors="coerce").to_numpy(float)
        f = lr._col(df, "f", "").astype(str).to_numpy()
        t = df.index.tz_convert("UTC").as_unit("ns").astype("int64").to_numpy()
        for name in set(f):
            m = (f == name) & np.isfinite(s)
            if m.sum() < 2:
                continue
            o = np.argsort(s[m], kind="stable")
            tt = t[m][o]
            bad, tot = inv_by_feed.get(name, (0, 0))
            inv_by_feed[name] = (bad + int(np.count_nonzero(np.diff(tt) < 0)), tot + int(tt.size - 1))
    worst_inv = 0.0
    for name, parts in sorted(seq_by_feed.items()):
        v = np.unique(np.concatenate(parts))
        span = v[-1] - v[0] + 1
        present = v.size
        gap = max(0.0, span - present)
        frac = gap / span if span > 0 else 0.0
        bad, tot = inv_by_feed.get(name, (0, 0))
        inv = bad / tot if tot else 0.0
        worst_inv = max(worst_inv, inv)
        L.append("    %-26s span=%12s present=%12s missing=%6.2f%%  clock_inversions=%5.2f%%"
                 % (name[:26], f"{int(span):,}", f"{present:,}", 100 * frac, 100 * inv))
        if frac > 0.02:
            findings.append(("DATA", "SEVERE" if frac < 0.2 else "FATAL",
                             "feed '%s' is missing %.1f%% of its sequence range -- the capture dropped "
                             "messages, so the book cannot be correct" % (name, 100 * frac)))
    if worst_inv > 0.0:
        L.append("    clock_inversions = adjacent events (in the venue's own sequence order) whose clock")
        L.append("    RUNS BACKWARDS. Every one is a pair that clock-only ordering applies in reverse.")
    return findings


def check_clock(msgs, clock, tz, session, date_str, L):
    """Health of the timestamp that orders the replay and places events on the grid.

    Two failure modes here are invisible downstream and both cross the book from the very first
    snapshot, so they mimic a parsing fault:

      NULL clock   ``pd.to_datetime(...).astype('int64')`` turns NaT into INT64_MIN, which does not
                   just misplace the event -- it sorts it before every other event in the session.
      DEAD clock   a feed whose chosen clock is constant (or zero) has its entire day collapsed onto
                   one instant, so the opening snapshot already holds that feed's end-of-day net state.

    Either one puts a whole feed's worth of net state at t=0 with a *stable* resting-order count, which
    is precisely the signature CHECK 6 attributes to "structural parsing". Measure it instead of
    inferring it.
    """
    L.append("")
    L.append("CHECK 4b timestamp health (clock=%s)" % clock)
    findings = []
    d = pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:])
    g0 = pd.Timestamp(f"{d.date()} {session[0]}", tz=tz).tz_convert("UTC").value
    g1 = pd.Timestamp(f"{d.date()} {session[1]}", tz=tz).tz_convert("UTC").value
    cands = lr._CLOCK_COLS.get(clock, lr._CLOCK_COLS["receipt"])
    per_feed = defaultdict(lambda: [0, 0, 0, np.inf, -np.inf])   # n, n_null, n_outside, tmin, tmax
    src = {}
    for mt, df in msgs.items():
        if df is None or not len(df):
            continue
        tcol = next((c for c in cands if c in df.columns), None)
        src[mt] = tcol or "(frame index)"
        if tcol is not None:
            t = pd.to_datetime(df[tcol], unit="ns", utc=True)
        else:
            idx = df.index
            if getattr(idx, "tz", None) is None:
                idx = pd.to_datetime(idx).tz_localize(tz)
            t = pd.Series(idx.tz_convert("UTC"), index=df.index)
        null = t.isna().to_numpy()
        tv = t.astype("int64").to_numpy()
        f = lr._col(df, "f", "").astype(str).to_numpy()
        for name in set(f):
            m = f == name
            rec = per_feed[name]
            rec[0] += int(m.sum())
            rec[1] += int(np.count_nonzero(null[m]))
            good = m & ~null
            if good.any():
                v = tv[good]
                rec[2] += int(np.count_nonzero((v < g0) | (v > g1)))
                rec[3] = min(rec[3], float(v.min()))
                rec[4] = max(rec[4], float(v.max()))
    if not per_feed:
        L.append("    (no messages)")
        return findings
    if len(set(src.values())) > 1:
        L.append("    NOTE: the clock is read from different sources per message type: %s"
                 % ", ".join("%s=%s" % (k, v) for k, v in sorted(src.items())))
    L.append("    %-26s %10s %8s %9s  %s" % ("feed", "events", "null", "off-sess", "clock span"))
    for name, (n, nnull, noff, tmin, tmax) in sorted(per_feed.items()):
        if not np.isfinite(tmin):
            span_txt = "ALL NULL"
        else:
            span_txt = "%s -> %s" % (pd.Timestamp(int(tmin), tz="UTC").tz_convert(tz).strftime("%H:%M:%S"),
                                     pd.Timestamp(int(tmax), tz="UTC").tz_convert(tz).strftime("%H:%M:%S"))
        L.append("    %-26s %10s %8s %8.1f%%  %s"
                 % (name[:26], f"{n:,}", f"{nnull:,}", 100.0 * noff / max(n, 1), span_txt))
        if nnull:
            findings.append(("CODE", "FATAL",
                             "feed '%s': %s events have a NULL %s clock. NaT casts to INT64_MIN, so every "
                             "one of them sorts AHEAD of the whole session and is applied before the open "
                             "-- the book is wrong from the first snapshot no matter how good the data is"
                             % (name, f"{nnull:,}", clock)))
        elif np.isfinite(tmin) and n > 1000 and (tmax - tmin) < 60e9:
            findings.append(("DATA", "FATAL",
                             "feed '%s': %s events span only %.1fs of %s clock -- the venue is not "
                             "populating this timestamp, so the feed's entire day collapses onto one "
                             "instant and its net end-of-day state appears at the open"
                             % (name, f"{n:,}", (tmax - tmin) / 1e9, clock)))
    return findings


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5-8: instrumented replay
# ══════════════════════════════════════════════════════════════════════════════
def instrumented_replay(msgs, tz, clock, price_scale, session, date_str, interval, order_by, L):
    """Replay with lob_reconstruct's OWN primitives (_event_arrays + _Book), instrumented to record
    orphan provenance, per-feed resting-order counts, and per-feed vs consolidated crossing over time.
    Using the production primitives means a bug found here is a bug in the real path, not in a copy."""
    ev = lr._event_arrays(dict(msgs), tz, clock=clock, price_scale=price_scale, consume=False,
                          order_by=order_by)
    if ev is None:
        L.append("    (no events)")
        return None
    d = pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:])
    g0 = pd.Timestamp(f"{d.date()} {session[0]}", tz=tz)
    g1 = pd.Timestamp(f"{d.date()} {session[1]}", tz=tz)
    grid = pd.date_range(g0, g1, freq=interval)
    grid_ns = grid.tz_convert("UTC").as_unit("ns").astype("int64").to_numpy()

    # Every order that is ever CREATED, keyed by (feed, ref) -- the SAME key _Book uses. Keying by ref
    # alone would count an add on a DIFFERENT venue as provenance for this venue's orphan and
    # misattribute a data fault to code; references are only unique within a feed.
    is_add = ev.code == lr._ADD
    span = int(ev.ref.max()) + 1 if ev.ref.size else 1
    key_all = ev.feed.astype(np.int64) * span + ev.ref.astype(np.int64)
    created = set(np.unique(key_all[is_add]).tolist())          # (feed, ref) composite keys
    created_anyfeed = set(np.unique(ev.ref[is_add]).tolist())   # ref only, to spot cross-feed collisions

    book = lr._Book()
    orphan_add_present, orphan_add_absent, orphan_absent_ts = 0, 0, []
    orphan_other_feed = 0
    orphan_by_feed = Counter()
    resting_trace, cross_trace, feedcross_trace, ts_trace = [], [], [], []
    # --- provenance for the ORDERS THAT PIN THE TOP (the mirror of the orphan analysis) --------------
    # CHECK 5 asks "a removal arrived -- where is its order?". That can only ever explain a removal that
    # did nothing. It cannot explain a CROSSED book, because crossing needs the opposite fault: an order
    # that should have LEFT and is still resting. Nothing here measured that, which is why a session can
    # come back "CODE, structural" with no statement of what is actually holding the top apart.
    born = {}                       # composite key -> ts of the add/modify that created the resting order
    killed = set()                  # composite keys a cancel/trade has already successfully removed
    resurrect_by_feed = Counter()   # modify whose previous ref was ALREADY removed by a cancel/trade
    resurrect_keys = {}             # composite key -> ts of the resurrecting modify
    pin_price = Counter()           # (feed, side, price) -> snapshots this price held a venue crossed
    gi, ng = 0, len(grid_ns)

    def _snap(ts_ns):
        bagg, aagg = book.consolidated("Bid", 100, True), book.consolidated("Ask", 100, True)
        cb = max(bagg) if bagg else np.nan
        ca = min(aagg) if aagg else np.nan
        crossed = bool(np.isfinite(cb) and np.isfinite(ca) and cb > ca + EPS)
        nfeed = 0
        for fd in set(list(book.bid.keys()) + list(book.ask.keys())):
            b = book.bid.get(fd) or {}
            a = book.ask.get(fd) or {}
            bb = max((p for p, s in b.items() if s > EPS), default=None)
            aa = min((p for p, s in a.items() if s > EPS), default=None)
            if bb is not None and aa is not None and bb > aa + EPS:
                nfeed += 1
                # the two prices actually holding THIS venue's own book apart, this snapshot
                pin_price[(int(fd), 0, float(bb))] += 1
                pin_price[(int(fd), 1, float(aa))] += 1
        resting_trace.append(len(book.orders))
        cross_trace.append(crossed)
        feedcross_trace.append(nfeed)
        ts_trace.append(ts_ns)

    for j in range(len(ev.ts)):
        ts = ev.ts[j]
        while gi < ng and grid_ns[gi] < ts:
            _snap(grid_ns[gi]); gi += 1
        code, feed = ev.code[j], ev.feed[j]
        if code == lr._ADD:
            book.add(feed, ev.ref[j], ev.side[j], ev.price[j], ev.size[j])
            k = int(feed) * span + int(ev.ref[j])
            born[k] = ts
            killed.discard(k)                                   # a genuine re-use of the reference
        elif code == lr._CANCEL:
            if not book.remove(feed, ev.ref[j]):
                orphan_by_feed[feed] += 1
                if int(feed) * span + int(ev.ref[j]) in created:
                    orphan_add_present += 1
                else:
                    orphan_add_absent += 1
                    orphan_absent_ts.append(ts)
                    if ev.ref[j] in created_anyfeed:
                        orphan_other_feed += 1
            else:
                killed.add(int(feed) * span + int(ev.ref[j]))
        elif code == lr._MODIFY:
            # THE resurrection test. modify() is remove-old + add-new, so a modify whose previous
            # reference is no longer in the book *because a cancel/trade already removed it* does not
            # modify anything -- it RE-CREATES a dead order. That phantom never receives another
            # removal (its cancel was spent) and pins the top for the rest of the session. It is the
            # documented consequence of applying a feed's events out of the venue's own order, and it
            # leaves NO orphan behind, so CHECK 5 is structurally blind to it.
            pk = int(feed) * span + int(ev.pref[j])
            if (feed, ev.pref[j]) not in book.orders and pk in killed:
                resurrect_by_feed[feed] += 1
                resurrect_keys[int(feed) * span + int(ev.ref[j])] = ts
            book.modify(feed, ev.pref[j], ev.ref[j], ev.side[j], ev.price[j], ev.size[j],
                        ev.pprice[j], ev.psize[j], ev.maintain[j])
            k = int(feed) * span + int(ev.ref[j])
            born.setdefault(k, ts)
            killed.discard(k)
        elif code == lr._LEVEL_SET:
            book.set_level(feed, ev.side[j], ev.price[j], ev.size[j])
        elif code == lr._LEVEL_DEL:
            book.set_level(feed, ev.side[j], ev.price[j], 0.0)
        elif code == lr._TRADE:
            if not book.reduce(feed, ev.ref[j], ev.size[j]):
                if feed not in book.mbp_feeds:
                    orphan_by_feed[feed] += 1
                    if int(feed) * span + int(ev.ref[j]) in created:
                        orphan_add_present += 1
                    else:
                        orphan_add_absent += 1
                        orphan_absent_ts.append(ts)
                        if ev.ref[j] in created_anyfeed:
                            orphan_other_feed += 1
                    book.reduce_at_price(feed, ev.price[j], ev.size[j])
            elif (feed, ev.ref[j]) not in book.orders:          # the trade consumed the order outright
                killed.add(int(feed) * span + int(ev.ref[j]))
    while gi < ng:
        _snap(grid_ns[gi]); gi += 1

    # End-of-session census of orders sitting on the WRONG side of their own venue's top. Each one is
    # an order the venue removed and our replay kept. Reported with the time it entered the book, so a
    # phantom that has been pinning the top since 09:31 is distinguishable from ordinary end-of-day rest.
    resting_keys = {int(fd) * span + int(rf) for (fd, rf) in book.orders}
    wrong_side = defaultdict(list)
    for (fd, rf), (sd, px, sz) in book.orders.items():
        b = book.bid.get(fd) or {}
        a = book.ask.get(fd) or {}
        bb = max((p for p, s in b.items() if s > EPS), default=None)
        aa = min((p for p, s in a.items() if s > EPS), default=None)
        if bb is None or aa is None or not (bb > aa + EPS):
            continue                                            # this venue's book is not crossed
        if (sd == 0 and px > aa + EPS) or (sd == 1 and px < bb - EPS):
            k = int(fd) * span + int(rf)
            wrong_side[int(fd)].append((float(px), int(sd), float(sz), born.get(k),
                                        k in resurrect_keys))
    return dict(feed_names=ev.feed_names, book=book,
                orphan_add_present=orphan_add_present, orphan_add_absent=orphan_add_absent,
                orphan_absent_ts=np.array(orphan_absent_ts, dtype="int64"),
                orphan_by_feed=orphan_by_feed, orphan_other_feed=orphan_other_feed,
                resting=np.array(resting_trace), crossed=np.array(cross_trace),
                feedcross=np.array(feedcross_trace), grid=grid, g0=g0,
                n_removals=int(np.count_nonzero(np.isin(ev.code, (lr._CANCEL, lr._TRADE)))),
                resurrect_by_feed=resurrect_by_feed,
                n_resurrect_resting=len(set(resurrect_keys) & resting_keys),
                resurrect_keys=resurrect_keys, pin_price=pin_price, wrong_side=wrong_side,
                stats=dict(book.stats), tz=tz)


def report_replay(R, L):
    findings = []
    if R is None:
        return findings
    fname = R["feed_names"]
    n = len(R["crossed"])
    cross_rate = float(np.mean(R["crossed"])) if n else float("nan")

    L.append("")
    L.append("CHECK 5  reference resolution (orphan provenance -- THE data/code discriminator)")
    tot = R["orphan_add_present"] + R["orphan_add_absent"]
    if tot == 0:
        L.append("    no orphaned cancel/trade references -- every removal found its order")
    else:
        pres = 100.0 * R["orphan_add_present"] / tot
        L.append("    orphaned removals: %s" % f"{tot:,}")
        L.append("      add EXISTS in the fetched messages : %10s (%5.1f%%)  -> CODE (match failed)"
                 % (f"{R['orphan_add_present']:,}", pres))
        L.append("      add ABSENT everywhere              : %10s (%5.1f%%)  -> DATA (message missing)"
                 % (f"{R['orphan_add_absent']:,}", 100 - pres))
        nrem = R.get("n_removals") or 0
        rate = R["orphan_add_absent"] / nrem if nrem else 0.0
        L.append("      orphan RATE: %s of %s cancel/trade events (%.3f%%)"
                 % (f"{tot:,}", f"{nrem:,}", 100.0 * tot / max(nrem, 1)))
        if R["orphan_add_absent"]:
            t = R["orphan_absent_ts"]
            mins = (t - R["g0"].value) / 6e10
            early = float(np.mean(mins <= BENIGN_OPEN_MINUTES))
            L.append("      of the ABSENT: %.1f%% land in the first %d minutes"
                     % (100 * early, BENIGN_OPEN_MINUTES))
            if early > 0.8:
                L.append("        => consistent with orders RESTING BEFORE the session window: benign, "
                         "expected when the fetch starts at 09:30")
            elif rate < ORPHAN_RATE_SEVERE:
                # Spread-across-the-session is NOT sufficient on its own. An order carried in from a
                # previous session (GTC / good-till-cancel) has no add anywhere in a one-day fetch and
                # is cancelled at whatever hour its owner chooses -- uniformly across the day, by
                # construction. That is the same time profile as dropped messages, so the only thing
                # separating them is the RATE: carry-over is a tail (well under 1% of removals),
                # a capture that is genuinely losing messages is not.
                # It also cannot be the cause of a crossed book either way: an orphaned removal removes
                # nothing, and crossing requires an order that stayed. Rank it accordingly.
                L.append("        => %.3f%% of removals; at this rate the uniform time profile is equally "
                         "consistent with orders CARRIED IN from a prior session (GTC), which have no add "
                         "in a single-day fetch and are cancelled at any hour. Note an orphaned removal "
                         "removes nothing, so it cannot itself cross the book." % (100 * rate))
                findings.append(("DATA", "MINOR",
                                 "%.3f%% of cancel/trade events reference an order with no add in the "
                                 "fetch. Spread across the session, but at a rate consistent with "
                                 "prior-session carry-over rather than a lossy capture -- and orphaned "
                                 "removals cannot cross a book in any case, so this does not explain the "
                                 "crossing" % (100 * rate)))
            else:
                findings.append(("DATA", "SEVERE",
                                 "%.1f%% of orphaned removals reference orders whose add never appears, "
                                 "they are spread across the session (not just the open), and they are "
                                 "%.2f%% of all removals -- too many for prior-session carry-over, so "
                                 "messages are genuinely missing from the fetch"
                                 % (100.0 * R["orphan_add_absent"] / tot, 100 * rate)))
        if R.get("orphan_other_feed"):
            L.append("      of the ABSENT: %s reference an order added on a DIFFERENT feed -- feed "
                     "attribution differs between the add and the removal" % f"{R['orphan_other_feed']:,}")
            findings.append(("DATA", "SEVERE",
                             "%s removals reference an order whose add is recorded under a different "
                             "feed -- the venue label is inconsistent between message types, so the book "
                             "(keyed by feed+ref) can never match them" % f"{R['orphan_other_feed']:,}"))
        if pres > 5.0:
            findings.append(("CODE", "FATAL" if pres > 25 else "SEVERE",
                             "%.1f%% of orphaned removals reference an order whose ADD IS PRESENT in the "
                             "fetched messages -- the data was there and the replay failed to match it "
                             "(ordering or reference-matching bug)" % pres))
        if R["orphan_by_feed"]:
            L.append("      by feed: %s" % ", ".join(
                "%s=%s" % (fname[k] if k < len(fname) else k, f"{v:,}")
                for k, v in R["orphan_by_feed"].most_common(6)))

    L.append("")
    L.append("CHECK 6  replay dynamics")
    rest = R["resting"]
    growth = 1.0
    head0 = max(60, n // 100)
    if rest.size > 10:
        q = [float(np.mean(rest[i * rest.size // 6:(i + 1) * rest.size // 6])) for i in range(6)]
        L.append("    resting orders by session sixth: %s" % "  ".join("%.0f" % v for v in q))
        # PEAK vs OPENING level, not last-sixth vs first-sixth: a leaking book that fills early and
        # then plateaus (because the message supply ends) has a last/first ratio near 1 and would be
        # misread as stable, which is exactly how an accumulation fault disguises itself as structural.
        growth = (float(np.max(rest)) + 1.0) / (float(np.mean(rest[:head0])) + 1.0)
        L.append("    peak/opening ratio = %.1fx" % growth)
        if growth > 5.0:
            findings.append(("DATA", "SEVERE",
                             "resting-order count grows %.0fx across the session -- orders enter and never "
                             "leave, which is the accumulation signature of a missing/failed removal path"
                             % growth))
    if n:
        sixth = [float(np.mean(R["crossed"][i * n // 6:(i + 1) * n // 6])) for i in range(6)]
        L.append("    crossed rate by session sixth  : %s" % "  ".join("%5.1f%%" % (100 * v) for v in sixth))
        L.append("    crossed overall                : %.2f%%" % (100 * cross_rate))
        # Structural faults (side/scale/column parsing) cross from the very FIRST snapshots with a
        # STABLE book. Accumulation faults also reach a high rate early when the leak is fast, so
        # "crossed early" alone cannot separate them -- the resting-order growth is what does.
        head = head0
        early = float(np.mean(R["crossed"][:head]))
        L.append("    crossed in first %d snapshots  : %.1f%%   (book growth %.1fx)" % (head, 100 * early, growth))
        if early > 0.5 and growth < 2.0:
            findings.append(("CODE", "SEVERE",
                             "crossed on %.0f%% of the first %d snapshots while the resting-order count is "
                             "stable (%.1fx) -- the book is wrong from the start, which is structural "
                             "(side/scale/column parsing), not accumulation" % (100 * early, head, growth)))
        elif cross_rate > 0.05 and growth > 2.0:
            L.append("    => crossing rises with a GROWING book: accumulation (orders not being removed),")
            L.append("       not a structural parse fault")

    L.append("")
    L.append("CHECK 7  per-venue vs consolidated crossing")
    fc = R["feedcross"]
    any_feed = float(np.mean(fc > 0)) if fc.size else float("nan")
    L.append("    snapshots where >=1 SINGLE venue's own book is crossed : %6.2f%%" % (100 * any_feed))
    L.append("    snapshots where the CONSOLIDATED top is crossed        : %6.2f%%" % (100 * cross_rate))
    if cross_rate > 0.01 and any_feed < 0.1 * cross_rate:
        findings.append(("DATA", "SEVERE",
                         "the consolidated top crosses but individual venue books do not -- this is a "
                         "CROSS-VENUE artifact (one venue stale, mis-scaled, or missing its removals), "
                         "not a per-venue replay bug"))
    elif any_feed > 0.5 * cross_rate and cross_rate > 0.01:
        if growth > 2.0:
            L.append("    => single-venue books cross too, but the book is ACCUMULATING (%.1fx) -- that is"
                     % growth)
            L.append("       fully explained by the missing removals above, not a separate replay bug")
        else:
            findings.append(("CODE", "SEVERE",
                             "individual venue books are themselves crossed while the book is NOT "
                             "accumulating -- a single venue's own replay is wrong, which no cross-venue "
                             "or leak explanation can account for"))

    findings += report_pins(R, growth, L)
    return findings


def report_pins(R, growth, L):
    """CHECK 8 -- name the orders holding the top apart, and say how they got there.

    Everything above this point describes the crossing; none of it says WHICH orders are crossed or why
    they are still in the book. That gap is why a session can be diagnosed as "structural (side/scale/
    column parsing)" when the side tokens are 100% clean and the scale is uniform -- the report has no
    other vocabulary for "the book is wrong from the start".

    A crossed book needs an order that should have left and did not. There are only a few ways to get
    one, and they are separable:
      resurrected   a Modify re-added an order a Cancel had already removed. Its removal is spent, so
                    the phantom is immortal. Caused by applying a feed's events out of the venue's own
                    order -- i.e. by CHECK 4.
      never removed the removal for it is simply absent from the fetch (DATA), which shows up as the
                    book accumulating -- already covered by CHECK 6's growth ratio.
    """
    findings = []
    tz = R.get("tz", "America/New_York")
    fname = R["feed_names"]

    def _fn(c):
        return fname[c] if c < len(fname) else str(c)

    def _hhmm(ts):
        return "  ?  " if ts is None else pd.Timestamp(int(ts), tz="UTC").tz_convert(tz).strftime("%H:%M:%S")

    L.append("")
    L.append("CHECK 8  what is pinning the top (order-level provenance)")
    st = R.get("stats") or {}
    L.append("    replay counters: %s" % ", ".join(
        "%s=%s" % (k, f"{v:,}") for k, v in sorted(st.items())
        if k in ("dup_add", "cancel_no_order", "trade_no_ref", "trade_no_ref_undisplayed",
                 "trade_on_mbp_skipped", "modify_reprioritized", "mbp_level_events")) or "(none)")

    res = R.get("resurrect_by_feed") or Counter()
    n_res = sum(res.values())
    n_res_resting = R.get("n_resurrect_resting", 0)
    L.append("    modifies that re-created an order a cancel had ALREADY removed : %s" % f"{n_res:,}")
    if n_res:
        L.append("      by feed: %s" % ", ".join("%s=%s" % (_fn(k), f"{v:,}") for k, v in res.most_common(6)))
        L.append("      of those, still resting at the close (immortal -- their cancel is spent): %s"
                 % f"{n_res_resting:,}")
        findings.append(("CODE", "FATAL",
                         "%s modifies re-added an order that a cancel/trade had already removed (%s of "
                         "them are still resting at the close). A modify is remove-old + add-new, so a "
                         "modify applied AFTER the cancel of the order it references resurrects that "
                         "order permanently -- its removal is already spent. This is the out-of-order "
                         "intra-feed replay in CHECK 4, and it leaves no orphan, which is why CHECK 5 "
                         "cannot see it" % (f"{n_res:,}", f"{n_res_resting:,}")))

    ws = R.get("wrong_side") or {}
    n_ws = sum(len(v) for v in ws.values())
    L.append("    orders resting on the WRONG side of their own venue's top at the close : %s" % f"{n_ws:,}")
    for fd, rows in sorted(ws.items(), key=lambda kv: -len(kv[1]))[:6]:
        rows = sorted(rows, key=lambda r: (r[3] is None, r[3]))
        n_from_res = sum(1 for r in rows if r[4])
        L.append("      %-24s %5d orders  oldest entered %s  resurrected=%d"
                 % (_fn(fd)[:24], len(rows), _hhmm(rows[0][3]), n_from_res))
        for px, sd, sz, ts, was_res in rows[:3]:
            L.append("          %s %10.4f x%-10.0f entered %s%s"
                     % ("Bid" if sd == 0 else "Ask", px, sz, _hhmm(ts),
                        "   <-- RESURRECTED by a modify" if was_res else ""))
    if n_ws and growth < 2.0 and not n_res:
        findings.append(("CODE", "SEVERE",
                         "%s orders are resting on the wrong side of their own venue's top at the close "
                         "while the book is NOT accumulating (%.1fx) -- these orders were removed by the "
                         "venue and kept by the replay. With no accumulation there is no missing-removal "
                         "explanation, so the removals arrived and were misapplied" % (f"{n_ws:,}", growth)))

    pin = R.get("pin_price") or Counter()
    if pin:
        L.append("    longest-held pinned prices (venue, side, price, snapshots held):")
        for (fd, sd, px), k in pin.most_common(6):
            L.append("      %-24s %s %10.4f  %s snapshots" % (_fn(fd)[:24], "Bid" if sd == 0 else "Ask",
                                                              px, f"{k:,}"))
    return findings


def _crossing_rate(msgs, tz, clock, price_scale, session, date_str, interval, order_by):
    """Minimal replay -- consolidated crossing rate only, no instrumentation. Used for the A/B."""
    ev = lr._event_arrays(dict(msgs), tz, clock=clock, price_scale=price_scale, consume=False,
                          order_by=order_by)
    if ev is None:
        return float("nan"), 0
    d = pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:])
    grid = pd.date_range(pd.Timestamp(f"{d.date()} {session[0]}", tz=tz),
                         pd.Timestamp(f"{d.date()} {session[1]}", tz=tz), freq=interval)
    grid_ns = grid.tz_convert("UTC").as_unit("ns").astype("int64").to_numpy()
    book = lr._Book()
    n_cross, gi, ng = 0, 0, len(grid_ns)

    def _snap():
        b, a = book.consolidated("Bid", 100, True), book.consolidated("Ask", 100, True)
        return bool(b and a and max(b) > min(a) + EPS)

    for j in range(len(ev.ts)):
        while gi < ng and grid_ns[gi] < ev.ts[j]:
            n_cross += _snap(); gi += 1
        code, feed = ev.code[j], ev.feed[j]
        if code == lr._ADD:
            book.add(feed, ev.ref[j], ev.side[j], ev.price[j], ev.size[j])
        elif code == lr._CANCEL:
            book.remove(feed, ev.ref[j])
        elif code == lr._MODIFY:
            book.modify(feed, ev.pref[j], ev.ref[j], ev.side[j], ev.price[j], ev.size[j],
                        ev.pprice[j], ev.psize[j], ev.maintain[j])
        elif code == lr._LEVEL_SET:
            book.set_level(feed, ev.side[j], ev.price[j], ev.size[j])
        elif code == lr._LEVEL_DEL:
            book.set_level(feed, ev.side[j], ev.price[j], 0.0)
        elif code == lr._TRADE:
            if not book.reduce(feed, ev.ref[j], ev.size[j]) and feed not in book.mbp_feeds:
                book.reduce_at_price(feed, ev.price[j], ev.size[j])
    while gi < ng:
        n_cross += _snap(); gi += 1
    return (n_cross / ng if ng else float("nan")), ng


def check_ordering_ab(msgs, tz, clock, price_scale, session, date_str, interval, L, order_by="sequence"):
    """CHECK 9 -- settle the ordering question by experiment instead of by argument.

    Replays the SAME messages twice, changing only the intra-feed event order, and reports the crossing
    rate of each. If the run AS CONFIGURED crosses and the other ordering does not, the messages were
    sufficient and the fault is ordering (CODE). If both cross equally, ordering is exonerated and the
    fault is elsewhere. Only meaningful when a sequencenumber exists -- without one both runs ARE the
    same run, which is the state CHECK 4 reports.

    A finding is raised only against the CONFIGURED ordering. A session that is replaying correctly and
    would merely have crossed under the legacy ordering is not faulty -- that gap is the fix working,
    and reporting it as a fault would flag every healthy session on a jittery feed.
    """
    L.append("")
    L.append("CHECK 9  ordering A/B (same messages, intra-feed order is the only difference)")
    findings = []
    if not any(df is not None and len(df) and "sequencenumber" in df.columns for df in msgs.values()):
        L.append("    SKIPPED: no sequencenumber, so 'sequence' and 'clock' ordering are the same run.")
        L.append("    Re-fetch with the column selected (CHECK 4) and this A/B becomes decisive.")
        return findings
    seq_rate, n = _crossing_rate(msgs, tz, clock, price_scale, session, date_str, interval, "sequence")
    clk_rate, _ = _crossing_rate(msgs, tz, clock, price_scale, session, date_str, interval, "clock")
    cfg = "sequence" if order_by == "sequence" else "clock"
    L.append("    crossed under CLOCK    ordering (legacy)   : %6.2f%%  of %s snapshots"
             % (100 * clk_rate, f"{n:,}"))
    L.append("    crossed under SEQUENCE ordering (venue's)  : %6.2f%%" % (100 * seq_rate))
    L.append("    (this run is configured order_by=%s)" % order_by)
    cur, alt, alt_name = ((seq_rate, clk_rate, "clock") if cfg == "sequence"
                          else (clk_rate, seq_rate, "sequence"))
    if cur > 0.05 and alt < 0.5 * cur:
        L.append("    => switching to %s ordering removes %.0f%% of the crossing."
                 % (alt_name, 100 * (1 - alt / cur)))
        findings.append(("CODE", "FATAL" if alt < 0.01 else "SEVERE",
                         "replaying the identical messages in %s order drops the crossing rate from "
                         "%.1f%% to %.1f%% -- the messages were sufficient and this run is applying "
                         "them out of order" % (alt_name, 100 * cur, 100 * alt)))
    elif cur > 0.05:
        L.append("    => the alternative ordering does NOT fix it; the fault is not intra-feed ordering.")
    else:
        L.append("    => this run does not cross. The %.1f%% under %s ordering is the margin the "
                 "sequence path is buying on this feed." % (100 * alt, alt_name))
    return findings


# ══════════════════════════════════════════════════════════════════════════════
def verdict(findings, L):
    L.append("")
    L.append("=" * 78)
    if not findings:
        L.append("VERDICT: no fault localized. The book reconstructs cleanly on this session.")
        L.append("=" * 78)
        return 0
    findings = sorted(findings, key=lambda x: (SEV_RANK.get(x[1], 9), x[0]))
    kinds = Counter(k for k, _s, _m in findings)
    # MINOR findings are reported but do not vote. A single tail-rate observation should not be able to
    # outvote a FATAL, and it should never be able to swing the verdict to DATA -- which sends the reader
    # to re-fetch a session whose messages are fine.
    voting = Counter(k for k, s, _m in findings if s != "MINOR")
    lead = ("DATA" if voting["DATA"] > voting["CODE"]
            else ("CODE" if voting["CODE"] > voting["DATA"] else "MIXED"))
    L.append("VERDICT: %s  (%d DATA finding(s), %d CODE finding(s))" % (lead, kinds["DATA"], kinds["CODE"]))
    L.append("=" * 78)
    for kind, sev, msg in findings:
        L.append("  [%s/%s] %s" % (kind, sev, msg))
    L.append("")
    if kinds["DATA"]:
        L.append("  DATA findings cannot be fixed in lob_reconstruct. Re-fetch the failing message types,")
        L.append("  or drop the session. Note that reconstruct_session() substitutes an EMPTY frame on a")
        L.append("  fetch exception and continues, so a failed fetch produces a full-length frame of")
        L.append("  garbage rather than an error -- check the run log for 'fetch ... failed'.")
    if kinds["CODE"]:
        L.append("  CODE findings are repairable in the replay: side/quantity parsing for the named feed,")
        L.append("  or the intra-feed ordering that governs reference matching.")
    return 2 if lead == "DATA" else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description="localize a crossed-book violation to DATA or CODE")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--date", help="YYYYMMDD; fetches via mstbook_loader (the production path)")
    src.add_argument("--messages-pickle", help="a pickled {message_type: DataFrame} dict")
    ap.add_argument("--product", default="SPY")
    ap.add_argument("--product-type", default="direct")
    ap.add_argument("--asset", default=None, help="canonical asset name (defaults to --product)")
    ap.add_argument("--data-source", default="apu")
    ap.add_argument("--clock", default="exchange", choices=["exchange", "receipt"])
    ap.add_argument("--price-scale", type=float, default=1.0, help="0.01 for CME integer-hundredths")
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--session-start", default="09:30")
    ap.add_argument("--session-end", default="16:00")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--order-by", default="sequence")
    ap.add_argument("--ab-ordering", action="store_true",
                    help="CHECK 9: replay twice (sequence vs clock ordering) and compare crossing. "
                         "Two extra full replays -- slow on a full session, but decisive")
    ap.add_argument("--save-messages", default=None, help="pickle the fetched messages for offline iteration")
    ap.add_argument("--no-replay", action="store_true", help="frame checks only (CHECK 1-4), no replay")
    ap.add_argument("--out", default=None, help="also write the report to this path")
    a = ap.parse_args(argv)

    mtypes = ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade", "mt_price_level_update")
    if a.messages_pickle:
        with open(a.messages_pickle, "rb") as fh:
            msgs = pickle.load(fh)
        failures = {}
        date_str = a.date or _infer_date(msgs, a.tz)
    else:
        msgs, failures = fetch_messages(a.date, a.product, a.product_type, a.data_source,
                                        a.clock, a.tz, mtypes)
        date_str = a.date
        if a.save_messages:
            with open(a.save_messages, "wb") as fh:
                pickle.dump(msgs, fh)

    L = ["=" * 78,
         "crossed-book diagnosis   product=%s  date=%s  clock=%s  scale=%g"
         % (a.product, date_str, a.clock, a.price_scale),
         "=" * 78, ""]
    findings = []
    session = (a.session_start, a.session_end)
    findings += check_inventory(msgs, failures, L)
    findings += check_feed_matrix(msgs, L)
    findings += check_fields(msgs, L)
    findings += check_sequence(msgs, L)
    findings += check_clock(msgs, a.clock, a.tz, session, date_str, L)
    if not a.no_replay:
        R = instrumented_replay(msgs, a.tz, a.clock, a.price_scale,
                                session, date_str, a.interval, a.order_by, L)
        findings += report_replay(R, L)
        if a.ab_ordering:
            findings += check_ordering_ab(msgs, a.tz, a.clock, a.price_scale, session, date_str,
                                          a.interval, L, order_by=a.order_by)
    rc = verdict(findings, L)
    text = "\n".join(L)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return rc


def _infer_date(msgs, tz):
    for df in msgs.values():
        if df is not None and len(df):
            idx = df.index
            if getattr(idx, "tz", None) is None:
                idx = pd.to_datetime(idx).tz_localize(tz)
            return idx[0].tz_convert(tz).strftime("%Y%m%d")
    raise SystemExit("cannot infer a date from empty messages; pass --date")


if __name__ == "__main__":
    sys.exit(main())
