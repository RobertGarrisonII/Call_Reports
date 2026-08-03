"""
lob_reconstruct.py
==================
Event-time reconstruction of a CONSOLIDATED, multi-venue equity limit order book (SPY) and NBBO
from MayStreet message data (mt_add_order / mt_cancel_order / mt_modify_order / mt_trade via
mstwx-lakequery), emitted in the cross_asset_price_discovery_stack canonical schema.

Why (see the methodology report): for sub-second cross-asset price discovery the vendor book
snapshot (mstbook-query) inherits conflation / time-grid / level-aggregation choices that bias
Hasbrouck / Gonzalo-Granger / lead-lag estimates toward whichever series is sampled finer or
carries less noise. Reconstructing the book from messages on a single capture-receipt clock makes
the construction auditable and lets you control the NBBO definition. Use mstbook-query as a
benchmark (``validate_against_snapshot``), not as the primary book for the leadership claim.

Design decisions encoded here
-----------------------------
* **Per-venue replay keyed on (feed, order_reference_number).** SPY spans several feeds
  (bats_edgx, xdp_arca_integrated, total_view), each with its own sequence and order-ref namespace,
  so each venue gets an independent book; reference numbers are never shared across feeds.
* **Message semantics.** add -> insert; cancel(previousquantity) -> delete the order; trade ->
  decrement the referenced resting order (fallback: reduce the price level on that feed); modify ->
  remove the old order (located via previousorderreferencenumber / previousprice / previousquantity)
  and insert the new one (the order-ref can change, e.g. total_view re-IDs on modify). For the book
  *state* (NBBO + depth) modify is remove-old + add-new; ``maintainpriority`` matters only for queue
  position, which a state snapshot does not track (recorded in stats, not used for the ladder).
* **Single clock.** Every message is ordered by the capture-receipt timestamp (ns), the one
  reference frame shared across venues; the consolidated book is sampled as-of each grid point.
* **Two top-of-book objects (per the report).** A strict Reg NMS round-lot NBBO
  (``SPY_nbbo_bid``/``SPY_nbbo_ask``: each venue must show >= round_lot at its best, then best across
  venues) AND an odd-lot-inclusive consolidated 10-level ladder (the canonical
  ``SPY_{bid|ask}{price|quantity}_{i}``). Consolidated locked/crossed states are retained, not
  "corrected" -- they are legitimate across venues at sub-second scale.

The mstwx-lakequery subprocess needs the MayStreet binary; ``reconstruct_book`` /
``validate_against_snapshot`` operate on already-fetched frames and are unit-tested on a synthetic
multi-venue stream. ``reconstruct_session`` is the live entry point.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict, namedtuple
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
EPS = 1e-9
# Strings that mean "this field was null in the source", after .astype(str). Which one you get depends
# on the dtype the CSV reader chose: object/float columns stringify a missing value to "nan", while the
# pandas nullable "string" dtype (what mstbook_loader._read_messages_csv now requests for the ref/side
# columns) stringifies pd.NA to "<NA>". Testing only for "nan" therefore silently stopped recognizing
# nulls the moment the reader switched dtypes -- so match all of them.
_NA_TOKENS = ("", "nan", "<NA>", "None", "NaN", "null", "NULL")

# Message types whose FETCH may fail without invalidating the replay. The MBP (price-level) types are
# a supplement to the MBO stream -- they are empty for CME futures by construction and only add depth
# for the venues that publish them, so losing them degrades the ladder. Losing any MBO type
# (add/cancel/modify/trade) does not degrade the book, it fabricates one: see reconstruct_session.
# The clear types are optional for a different reason: they were not fetched at all until v0.9.23, so
# a venue that does not publish them must not now fail the session.
_OPTIONAL_MSG_TYPES = frozenset({"mt_price_level_update", "mt_modify_price_level",
                                 "mt_delete_price_level", "mt_clear_orders", "mt_clear_price_levels"})

# message-type -> the column carrying the quantity that the event acts on
_QTY = {"mt_add_order": "quantity", "mt_cancel_order": "previousquantity",
        "mt_modify_order": "quantity", "mt_trade": "quantity",
        "mt_price_level_update": "quantity", "mt_modify_price_level": "quantity",
        "mt_delete_price_level": "quantity",
        # the clear types carry no quantity: they wipe a whole feed's state
        "mt_clear_orders": "quantity", "mt_clear_price_levels": "quantity"}
# which timestamp column drives event ordering + the snapshot grid. Under source capture next to the
# CME engine (CyrusOne Aurora I) each feed is hardware GPS-stamped at its origin colo, so "receipt"
# is one uniform UTC methodology across venues -> the default. "exchange" is per-venue engine time
# (heterogeneous publication semantics) and is a robustness lens only.
# mt_aggregated_price_update names its clocks last{receipt,exchange}timestamp -- it is a SNAPSHOT of
# the whole ladder after a burst, not a single event, so the column says "as of". Without these
# aliases that type cannot be fetched at all (no recognized clock column -> ValueError).
_CLOCK_COLS = {"receipt": ("receipttimestamp", "lastreceipttimestamp"),
               "exchange": ("exchangetimestamp", "exchange_timestamp", "sourcetimestamp",
                            "lastexchangetimestamp")}
# MBO (order-by-order) events vs MBP (price-level) events. Order-based venues (bats_edgx,
# xdp_arca_integrated, total_view, ...) publish add/cancel/modify/trade and are replayed per order;
# price-level venues (iex_deep, and the CME MBP modify/delete) publish the NEW aggregate size at a
# price and are applied directly to the level. Both feed the same per-venue consolidation layer -- a
# level is a level however it was built. This is the fix for MBP-only venues (notably IEX) being
# absent from the "consolidated" book under an MBO-only engine.
#
# mt_clear_orders / mt_clear_price_levels are FEED RESETS: the venue is telling you to discard
# everything it has told you so far and rebuild from the next message. They are issued on a line
# failover, a gap recovery, or a session-state transition, and they carry no price or quantity --
# the whole point is that the previous state is void. Not applying one is unrecoverable: the venue
# never cancels the orders it just disowned, so every one of them rests in the consolidated ladder
# for the remainder of the session while the venue re-adds its book under fresh reference numbers.
# A stale pre-market bid pinned that way sits above the current ask for hours, which is exactly the
# "resting orders that should have left are pinning the top" signature. The 2024-12-18 SPY tape
# carries one at 05:25:44 ET on miax_pearl_equities_dom, mid-pre-market, with the replay already
# holding that feed's state -- and until v0.9.23 these types were never even fetched.
_ADD, _CANCEL, _MODIFY, _TRADE, _LEVEL_SET, _LEVEL_DEL, _CLEAR = 0, 1, 2, 3, 4, 5, 6
_CODE = {"mt_add_order": _ADD, "mt_cancel_order": _CANCEL, "mt_modify_order": _MODIFY, "mt_trade": _TRADE,
         "mt_price_level_update": _LEVEL_SET, "mt_modify_price_level": _LEVEL_SET,
         "mt_delete_price_level": _LEVEL_DEL,
         "mt_clear_orders": _CLEAR, "mt_clear_price_levels": _CLEAR}


# ════════════════════════════════════════════════════════════════════════════
# Per-venue book + consolidation
# ════════════════════════════════════════════════════════════════════════════
class _Book:
    """Per-feed order books. orders[(feed, ref)] = [side, price, size]; bid/ask[feed][price] = size."""

    def __init__(self):
        self.orders: dict = {}
        self.bid: dict = defaultdict(lambda: defaultdict(float))
        self.ask: dict = defaultdict(lambda: defaultdict(float))
        self.mbo_feeds: set = set()                  # feeds populated by order replay
        self.mbp_feeds: set = set()                  # feeds populated by price-level updates
        self.stats = Counter()

    def _lv(self, feed, side):
        return self.bid[feed] if side == 0 else self.ask[feed]      # side: 0=Bid, 1=Ask (int codes)

    def add(self, feed, ref, side, price, size):
        if not (np.isfinite(price) and np.isfinite(size) and size > 0) or side not in (0, 1):
            return
        self.mbo_feeds.add(feed)
        key = (feed, ref)
        if key in self.orders:                      # duplicate add ref -> replace (treat as re-add)
            self.remove(feed, ref); self.stats["dup_add"] += 1
        self.orders[key] = [side, price, size]
        self._lv(feed, side)[price] += size

    def set_level(self, feed, side, price, qty):
        """MBP: assign the NEW aggregate size at (feed, side, price); qty<=0 deletes the level.
        Distinct from add()/remove(), which accumulate per order. MBP and MBO venues have different
        feed keys, so the two paths never touch the same price map."""
        if side not in (0, 1) or not np.isfinite(price):
            return
        self.mbp_feeds.add(feed)
        lv = self._lv(feed, side)
        if not np.isfinite(qty) or qty <= EPS:
            lv.pop(price, None)
        else:
            lv[price] = qty                          # assign, not +=
        self.stats["mbp_level_events"] += 1

    def remove(self, feed, ref):
        o = self.orders.pop((feed, ref), None)
        if o is None:
            return False
        side, price, size = o
        lv = self._lv(feed, side)
        lv[price] -= size
        if lv[price] <= EPS:
            lv.pop(price, None)
        return True

    def reduce(self, feed, ref, qty):               # trade against a referenced resting order
        o = self.orders.get((feed, ref))
        if o is None:
            return False
        side, price, size = o
        lv = self._lv(feed, side)
        if size - qty <= EPS:
            lv[price] -= size
            if lv[price] <= EPS:
                lv.pop(price, None)
            self.orders.pop((feed, ref), None)
        else:
            lv[price] -= qty
            o[2] = size - qty
        return True

    def set_remaining(self, feed, ref, leaves):
        """Trade against a referenced resting order, using the venue's OWN post-trade remaining size.

        ``mt_trade.leavesquantity`` is what is left on the resting order after this execution, so it
        is authoritative where a decrement is merely arithmetic: it is immune to a missed earlier
        partial fill, to a duplicate print, and to a size we mis-parsed on the add. ``leaves = 0``
        removes the order outright, which is the case that matters -- a fully filled order that is
        not removed rests forever and pins the top. Verified against the tape: order
        3189381831070546706 prints leaves 1198 -> 1194 -> 1192 on trades of 2, 4 and 2.

        Returns False when the reference is unknown (caller falls back), True otherwise."""
        o = self.orders.get((feed, ref))
        if o is None:
            return False
        side, price, size = o
        lv = self._lv(feed, side)
        if not np.isfinite(leaves) or leaves <= EPS:
            lv[price] -= size
            if lv[price] <= EPS:
                lv.pop(price, None)
            self.orders.pop((feed, ref), None)
        else:
            if leaves > size + EPS:
                # A trade cannot GROW the order it executed against. If the venue's figure exceeds
                # what we hold, one of two things is true and neither justifies inflating a level:
                # our size is already wrong, or leavesquantity does not mean the resting order's
                # remainder on this feed (it is verified on bats_edgx; other venues may populate it
                # from the aggressor). Keep our size, count it, and let the diagnostic report the
                # rate -- a level silently inflated on the bid side is a crossed book.
                self.stats["trade_leaves_gt_size"] += 1
                return True
            lv[price] += (leaves - size)              # assign the venue's figure, keep the level in step
            if lv[price] <= EPS:
                lv.pop(price, None)
            o[2] = leaves
            if abs(leaves - size) > EPS:
                self.stats["trade_leaves_corrected"] += 1
        return True

    def clear_feed(self, feed):
        """Feed reset (mt_clear_orders / mt_clear_price_levels): discard EVERYTHING this venue has
        published. Orders are dropped by key and both price maps for the feed are emptied, so nothing
        it disowned can rest in the consolidated ladder afterwards. Other venues are untouched --
        a reset is per feed, and treating it as global would blank a book that is perfectly good."""
        n = 0
        for key in [k for k in self.orders if k[0] == feed]:
            del self.orders[key]
            n += 1
        n_lv = len(self.bid.get(feed, ())) + len(self.ask.get(feed, ()))
        self.bid.pop(feed, None)
        self.ask.pop(feed, None)
        self.stats["feed_clears"] += 1
        self.stats["orders_cleared"] += n
        self.stats["levels_cleared"] += n_lv
        return n, n_lv

    def reduce_level(self, feed, side, price, qty):  # SUPERSEDED on the trade path by reduce_at_price
        # Kept for callers that genuinely know the book side. NOT used for trade consumption any more:
        # a trade's `side` field is aggressor-coded on these feeds, so reducing that side strands the
        # consumed (opposite-side) order and crosses the book -- see reduce_at_price.
        lv = self._lv(feed, side)
        if price in lv:
            lv[price] -= qty
            if lv[price] <= EPS:
                lv.pop(price, None)

    def reduce_at_price(self, feed, price, qty):
        """Trade with no matching resting ref: consume `qty` of displayed size at `price` on whichever
        side actually holds it. A print executes against resting liquidity AT the print price, and in a
        non-crossed book a price sits on at most one side, so the PRICE -- not the trade message's
        `side` field (aggressor-coded on these feeds, and the cause of the stranding) -- identifies the
        consumed side unambiguously. Returns the side reduced (0=Bid, 1=Ask), or -1 if `price` is on
        neither displayed side (a hidden / midpoint / non-displayed odd-lot print, which removes
        nothing and so cannot strand anything)."""
        if not np.isfinite(price):
            return -1
        bid = self._lv(feed, 0); ask = self._lv(feed, 1)
        on_bid = bid.get(price, 0.0) > EPS
        on_ask = ask.get(price, 0.0) > EPS
        if on_bid == on_ask:                          # neither side has it (hidden/odd-lot -> no-op), or
            if not on_bid:                            # both do (this feed is already locked/crossed --
                return -1                             # a pre-existing pathology, not a refless print);
            side, lv = 1, ask                         # in that rare case consume the ask by convention.
        else:
            side, lv = (0, bid) if on_bid else (1, ask)
        if qty >= lv[price] - EPS:
            lv.pop(price, None)
        else:
            lv[price] -= qty
        return side

    def modify(self, feed, prev_ref, ref, side, price, size, prev_price, prev_size, maintain):
        if not self.remove(feed, prev_ref) and prev_ref != ref:
            self.remove(feed, ref)                  # in case it was already under the new ref
        self.add(feed, ref, side, price, size)
        if not maintain:
            self.stats["modify_reprioritized"] += 1

    def consolidated(self, side, round_lot, odd_lot_inclusive):
        """price -> size summed across venues (optionally only venue levels with size >= round_lot)."""
        agg = defaultdict(float)
        per = self.bid if side == "Bid" else self.ask
        for lv in per.values():
            for price, size in lv.items():
                if size <= EPS:
                    continue
                if not odd_lot_inclusive and size < round_lot:
                    continue
                agg[price] += size
        return agg

    def best_round_lot(self, side, round_lot):
        """Reg NMS round-lot best: each venue's best price showing >= round_lot, then best across venues."""
        per = self.bid if side == "Bid" else self.ask
        best = None
        for lv in per.values():
            cand = [p for p, s in lv.items() if s >= round_lot]
            if not cand:
                continue
            vb = max(cand) if side == "Bid" else min(cand)
            best = vb if best is None else (max(best, vb) if side == "Bid" else min(best, vb))
        return best


# ════════════════════════════════════════════════════════════════════════════
# Event normalization
# ════════════════════════════════════════════════════════════════════════════
def _col(df, name, default=None):
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


_Events = namedtuple("_Events", "ts code feed side price size pprice psize ref pref maintain "
                                "leaves undisplayed feed_names")

# An execution against NON-DISPLAYED liquidity carries no orderreferencenumber, because there is no
# displayed resting order to reference. The tape marks it: executionattribute='Hidden' (and/or
# printable='NonPrintable'). Those prints are legitimately reference-less -- they are not evidence of
# a broken replay, and consuming displayed size at their price REMOVES liquidity that is still
# resting. On 2024-12-18 SPY they are a large share of the odd-lot tape.
_UNDISPLAYED_EXEC = ("hidden",)
_UNDISPLAYED_PRINT = ("nonprintable",)


def _event_arrays(messages: dict, tz: str, clock: str = "exchange", price_scale: float = 1.0,
                  consume: bool = False, order_by: str = "sequence") -> Optional["_Events"]:
    """Normalize the message frames into ONE clock-ordered set of **columnar numpy arrays** (not a
    Python list of tuples -- the per-object overhead of tens of millions of 11-tuples with string
    feed/ref/side is the dominant memory sink on crash days). ``feed``, ``side`` and the order
    references are factorized to **integer codes**: feed -> int32 (with ``feed_names`` to map back),
    side -> int8 (0=Bid, 1=Ask, -1=other), and ``ref``/``pref`` -> a shared int64 code space (so a
    reference and a later previous-reference to the same order collide on the same code). Integer keys
    are ~6x smaller than object strings in both the events and the resting-order book, and are robust
    to references that overflow int64 or are alphanumeric (which a naive ``to_numeric`` would corrupt).
    ``clock`` picks the ordering timestamp; ``price_scale`` multiplies price/previousprice (CME
    integer-hundredths -> index points). With ``consume=True`` each input frame is popped and freed as
    its arrays are extracted, so the heavy concatenate+sort+replay phase does not co-reside with the
    full set of message DataFrames (the caller must not reuse the dict)."""
    cands = _CLOCK_COLS.get(clock, _CLOCK_COLS["receipt"])
    ts_p, code_p, feed_p, side_p, price_p, size_p = [], [], [], [], [], []
    pprice_p, psize_p, ref_p, pref_p, maint_p, seq_p = [], [], [], [], [], []
    leaves_p, undisp_p = [], []
    for mtype in list(messages.keys()):
        df = messages.pop(mtype) if consume else messages.get(mtype)
        if df is None or len(df) == 0:
            continue
        n = len(df)
        tcol = next((c for c in cands if c in df.columns), None)
        if tcol is not None:                            # explicit clock column on the frame
            _t = pd.to_datetime(df[tcol], unit="ns", utc=True)
            # A NULL clock value becomes NaT, and NaT.astype("int64") is INT64_MIN -- which does not
            # merely lose the event's position, it sorts it BEFORE every other event in the session.
            # A feed whose exchange timestamp is null therefore has its whole day collapsed onto the
            # first grid point, so the opening snapshot already holds the day's net state and the book
            # is crossed from the first sample. Carry those rows on the frame's own clock index
            # instead (the fetch indexed it), and say so rather than emitting a 1970 event.
            if _t.isna().any():
                _n_bad = int(_t.isna().sum())
                _idx = df.index
                if getattr(_idx, "tz", None) is None:
                    _idx = pd.to_datetime(_idx).tz_localize(tz)
                _t = _t.fillna(pd.Series(_idx.tz_convert("UTC"), index=df.index))
                log.warning("%s: %d/%d rows have a NULL %s; falling back to the frame index for those "
                            "rows (left as NaT they would sort to the start of the session and cross "
                            "the book from the first snapshot)", mtype, _n_bad, n, tcol)
            ts = _t.astype("int64").to_numpy()
        else:                                           # fall back to the index (already the chosen clock)
            idx = df.index
            if getattr(idx, "tz", None) is None:
                idx = pd.to_datetime(idx).tz_localize(tz)
            ts = idx.tz_convert("UTC").as_unit("ns").astype("int64").to_numpy()
        seq = pd.to_numeric(_col(df, "sequencenumber"), errors="coerce").to_numpy()
        feed = _col(df, "f", "").astype(str).to_numpy()
        side = _col(df, "side", "").astype(str).str.strip().to_numpy()
        price = pd.to_numeric(_col(df, "price"), errors="coerce").to_numpy(float)
        size = pd.to_numeric(_col(df, _QTY[mtype]), errors="coerce").to_numpy(float)
        pprice = pd.to_numeric(_col(df, "previousprice"), errors="coerce").to_numpy(float)
        if price_scale != 1.0:
            price = price * price_scale
            pprice = pprice * price_scale
        psize = pd.to_numeric(_col(df, "previousquantity"), errors="coerce").to_numpy(float)
        if _CODE[mtype] == _TRADE:
            # the venue's own post-trade remaining size on the resting order (authoritative), and the
            # markers that say this print had no displayed order to reference in the first place
            leaves = pd.to_numeric(_col(df, "leavesquantity"), errors="coerce").to_numpy(float)
            _ea = _col(df, "executionattribute", "").astype(str).str.strip().str.lower().to_numpy()
            _pr = _col(df, "printable", "").astype(str).str.strip().str.lower().to_numpy()
            undisp = np.isin(_ea, _UNDISPLAYED_EXEC) | np.isin(_pr, _UNDISPLAYED_PRINT)
        else:
            leaves = np.full(n, np.nan)
            undisp = np.zeros(n, bool)
        ref = _col(df, "orderreferencenumber", "").astype(str).to_numpy()
        pref = _col(df, "previousorderreferencenumber", "").astype(str).to_numpy()
        pref = np.where(np.isin(pref, _NA_TOKENS), ref, pref)        # missing prev-ref -> the ref (legacy)
        maintain = _col(df, "maintainpriority", "").astype(str).str.strip().str.lower().to_numpy()
        admindel = _col(df, "admindelete", "").astype(str).str.strip().str.lower().to_numpy()
        if _CODE[mtype] == _LEVEL_SET:                  # MBP: an admin delete clears the level
            size = np.where(admindel == "true", 0.0, size)
        ts_p.append(ts); seq_p.append(seq); code_p.append(np.full(n, _CODE[mtype], np.int8))
        feed_p.append(feed); side_p.append(side); price_p.append(price); size_p.append(size)
        pprice_p.append(pprice); psize_p.append(psize)
        ref_p.append(ref); pref_p.append(pref); maint_p.append(maintain != "false")
        leaves_p.append(leaves); undisp_p.append(undisp)
        del df                                          # in consume mode this was the last reference
    if not ts_p:
        return None
    ts = np.concatenate(ts_p); seq_s = np.concatenate(seq_p); code = np.concatenate(code_p)
    feed_s = np.concatenate(feed_p); side_s = np.concatenate(side_p)
    price = np.concatenate(price_p); size = np.concatenate(size_p)
    pprice = np.concatenate(pprice_p); psize = np.concatenate(psize_p)
    ref_s = np.concatenate(ref_p); pref_s = np.concatenate(pref_p)
    maintain = np.concatenate(maint_p)
    leaves = np.concatenate(leaves_p); undisp = np.concatenate(undisp_p)
    del ts_p, seq_p, code_p, feed_p, side_p, price_p, size_p, pprice_p, psize_p, ref_p, pref_p, maint_p
    del leaves_p, undisp_p

    feed_code, feed_names = pd.factorize(feed_s, sort=False)        # feed -> int32 + names for attrs
    feed_code = feed_code.astype(np.int32); del feed_s
    side = np.where(side_s == "Bid", 0, np.where(side_s == "Ask", 1, -1)).astype(np.int8); del side_s
    ref_code, ref_uniq = pd.factorize(ref_s, sort=False)           # ref/pref share one code space
    pref_code = pd.Index(ref_uniq).get_indexer(pref_s)             # prev-refs absent from the ref set -> -1
    ref = ref_code.astype(np.int64); pref = pref_code.astype(np.int64)
    del ref_s, pref_s, ref_code, pref_code, ref_uniq

    # Event order = a k-way merge of per-feed streams, each in EXCHANGE-SEQUENCE order, interleaved
    # across feeds by the clock. UDP multicast arrives out of packet order, so ordering a feed by ANY
    # timestamp inverts ~10% of adjacent events: a Cancel can precede the Modify it follows, the Modify
    # then RESURRECTS the deleted order, and that phantom level crosses the consolidated top on every
    # snapshot. sequencenumber is the venue's authoritative within-feed order, so it governs intra-feed;
    # we bump the clock to be non-decreasing along that order (per-feed cummax) and stable-sort on it --
    # exactly heapq.merge of the per-feed streams keyed by clock, but O(n log n) numpy over ~1e8 events.
    # Cross-feed order is immaterial (the book keys orders by (feed, ref) -> distinct feeds commute); the
    # clock only places events on the grid, and the per-venue exchange clock removes the differential
    # capture-latency bias a single receipt clock bakes in (~0.2 ms venue-to-venue on SPY).
    # Within-TIE order = eliminations LAST (v0.9.7). The venue order key is not always a total order:
    # CME's sequencenumber is PACKET-level, so one value can carry an add/modify AND the cancel/trade
    # that removes the order it references; the degraded (no-sequencenumber) state ties EVERY event in a
    # feed. In both cases the within-tie fallback was the arbitrary concatenation order -- adds, then
    # cancels, then modifies, then trades -- which can apply a removal BEFORE the modify that follows it.
    # Since modify() == remove()+add(), that modify then RE-ADDS the just-removed order, resurrecting a
    # level that pins the consolidated top below the bid on every later snapshot (the 88%-crossed ES
    # signature; trade_no_ref=0 distinguishes it from the refless-trade bug). Ranking liquidity removals
    # last makes the cancel/trade the final word within its packet/instant, so a resurrecting modify
    # cannot survive it. No-op wherever sequencenumber is strictly increasing per feed (equities: SPY
    # 0% -> 0%) -- it only bites on genuine ties, which is exactly the CME packet case.
    # Within-tie rank: 0 = feed reset, 1 = add/modify/level-set, 2 = removal.
    # A reset ranks FIRST, not last, even though it removes more than any cancel does. The venue clears
    # and then rebuilds, so a clear tied with adds in the same packet must precede them -- ranking it
    # with the removals would wipe the very re-adds it exists to make room for.
    elim_rank = np.where(np.isin(code, (_CANCEL, _TRADE, _LEVEL_DEL)), 2, 1).astype(np.int8)
    elim_rank[code == _CLEAR] = 0
    if order_by == "sequence":
        if not np.isfinite(seq_s).any():
            # This branch is a DEGRADED MODE, not a supported configuration: it is the legacy clock-only
            # ordering that the sequence path exists to replace, and on a real multicast feed it crosses
            # the book on essentially every snapshot. It used to be entered silently, which is how a
            # column-pruning change in the loader (sequencenumber missing from _MSG_NEEDED_COLS) disabled
            # the ordering fix across every production session without a single line in the run log.
            # Say so at WARNING so the degrade is visible in the log next to the crossed-book warning.
            log.warning("ORDERING DEGRADED: no usable 'sequencenumber' on any message frame, so intra-feed "
                        "order falls back to the clock alone. Adjacent events invert under UDP reordering / "
                        "coarse engine stamps, a Modify can resurrect an already-Cancelled order, and the "
                        "resulting phantom level crosses the top on most snapshots. Confirm the fetch is "
                        "selecting the sequencenumber column (mstbook_loader._MSG_NEEDED_COLS).")
            # The per-feed sequence cummax is meaningless without a
            # sequence, and it is precisely what dragged a late same-instant event ahead of an earlier one;
            # order by the clock DIRECTLY, removals last within an instant. (Cross-feed order is immaterial:
            # the book keys orders by (feed, ref), so distinct feeds commute.)
            order = np.lexsort((elim_rank, ts))
        else:
            # PARTIAL coverage is its own hazard: a frame that lacks the column gets seq=inf and is
            # therefore ordered AFTER every sequenced event in the same feed. If, say, mt_trade is the
            # frame missing it, every trade is applied at end of day and the orders they consumed rest
            # untouched all session -- crossing the book exactly like a missing removal stream would.
            _cov = float(np.isfinite(seq_s).mean())
            if _cov < 0.999:
                log.warning("ORDERING PARTIAL: only %.1f%% of events carry a 'sequencenumber'; the rest are "
                            "ordered LAST within their feed, not at their true position. If an entire "
                            "message type is missing the column its events all land at end of session.",
                            100.0 * _cov)
            seqf = np.where(np.isfinite(seq_s), seq_s, np.inf)    # missing sequence -> last within its feed
            presort = np.lexsort((elim_rank, seqf, feed_code))    # feed-major, seq-ascending, removals last within a tied packet
            ck = ts[presort].astype(np.int64).copy(); fc = feed_code[presort]
            # A FEED RESET IS PLACED BY ITS CLOCK, NOT BY ITS SEQUENCE. The whole meaning of a reset
            # is that the venue's previous state -- including its sequence namespace -- is void, so a
            # clear's sequence number is not comparable to the stream around it and must not be
            # threaded into the per-feed cummax. Doing so put a 22:25:47 clear at a mid-day sequence
            # position on 2020-03-12; the replay's grid pointer only moves forward, so on reaching an
            # event stamped 22:25 it flushed EVERY remaining grid point at once and froze the whole
            # consolidated book at 09:40:31 -- inside that day's 09:35:37-09:50:37 halt. 97.3% of the
            # session then showed the frozen, correctly-crossed halt book. (100% - 97.3% = 632
            # snapshots = 09:40:31 exactly.)
            _cl = (code == _CLEAR)[presort]
            for _f in np.unique(fc):                              # clock made monotone in sequence, per feed
                _m = fc == _f
                _sub, _subcl = ck[_m], _cl[_m]
                _acc = np.where(_subcl, np.iinfo(np.int64).min, _sub)   # a reset never raises the running max
                _acc = np.maximum.accumulate(_acc)
                ck[_m] = np.where(_subcl, _sub, _acc)             # ... and keeps its own clock
            order = presort[np.argsort(ck, kind="stable")]        # stable: clock ties keep (seq, removal) order
    else:                                                        # legacy: clock-only (pre-fix; reproduces the bug)
        order = np.argsort(ts, kind="stable")
    return _Events(ts[order], code[order], feed_code[order], side[order], price[order], size[order],
                   pprice[order], psize[order], ref[order], pref[order], maintain[order],
                   leaves[order], undisp[order], list(feed_names))


# ════════════════════════════════════════════════════════════════════════════
# Reconstruction
# ════════════════════════════════════════════════════════════════════════════
def reconstruct_book(messages: dict, asset: str = "SPY", levels: int = 10, interval: str = "1s",
                     round_lot: int = 100, odd_lot_inclusive: bool = True,
                     session=("09:30", "16:00"), tz: str = "America/New_York",
                     date_str: Optional[str] = None, clock: str = "exchange",
                     price_scale: float = 1.0, consume: bool = False, order_by: str = "sequence",
                     rules: Optional[dict] = None) -> pd.DataFrame:
    """Replay the messages into per-venue books and sample the CONSOLIDATED book as-of each grid
    point. ``messages`` is a dict {message_type: DataFrame} (raw mstwx columns, tz-aware index).

    Returns a canonical frame: {ASSET}_{bid|ask}{price|quantity}_{i} for i=1..levels (consolidated,
    odd-lot-inclusive unless odd_lot_inclusive=False), plus {ASSET}_nbbo_bid/{ASSET}_nbbo_ask (strict
    round-lot Reg NMS NBBO) and {ASSET}_mid (consolidated level-1 midpoint). Reconstruction stats are
    in ``df.attrs['lob_stats']``. ``clock`` ("receipt"|"exchange") chooses the ordering/grid clock;
    receipt is the GPS source-capture default. With source capture co-located at the venues the two
    are within engine/GPS jitter, so an exchange-clock run is a robustness check, not a correction.
    ``price_scale`` multiplies every price (sizes untouched): CME equity-index futures print integer
    hundredths of an index point (e.g. ESM5 543775 = 5437.75), so pass 0.01 to reconstruct ES in
    index-point units; the stack's log/return/bps math is invariant to a constant scale."""
    ev = _event_arrays(messages, tz, clock=clock, price_scale=price_scale, consume=consume, order_by=order_by)
    if ev is None:
        return pd.DataFrame()
    # NOTE: intra-feed order is governed by sequencenumber (see _event_arrays), so the clock's only
    # role is cross-feed interleaving onto the grid. 'exchange' is the default there because it removes
    # the per-venue differential capture-latency bias a single 'receipt' clock bakes in; both clocks
    # leave events within their grid cell at the resolutions used (>=10ms vs ~0.2ms venue skew).
    if date_str is None:                            # infer the session date from the first (earliest) event
        date_str = pd.Timestamp(int(ev.ts[0]), tz="UTC").tz_convert(tz).strftime("%Y%m%d")
    d = pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:])
    g0 = pd.Timestamp(f"{d.date()} {session[0]}", tz=tz)
    g1 = pd.Timestamp(f"{d.date()} {session[1]}", tz=tz)
    grid = pd.date_range(g0, g1, freq=interval)
    grid_ns = grid.tz_convert("UTC").as_unit("ns").astype("int64").to_numpy()

    # Each of the three v0.9.23 replay rules can be switched off INDEPENDENTLY. They were added
    # together, and on two of the four March-2020 sessions the crossed fraction moved sharply when
    # they went in (4.0% -> 11.9%, 3.9% -> 97.3%) -- which of them is responsible is an empirical
    # question, and one that is answerable only by replaying the same messages under each setting.
    # See ab_book_rules.py, which does exactly that from a single fetch.
    #
    #   apply_clears        replay mt_clear_orders / mt_clear_price_levels as feed resets
    #   use_leaves          take the resting size from mt_trade.leavesquantity, not a decrement
    #   consume_undisplayed  let a refless Hidden/NonPrintable print consume DISPLAYED size
    #                       (pre-v0.9.23 behaviour: wrong in principle -- there is no displayed
    #                        order behind such a print -- but it deleted stale levels as a side
    #                        effect, which could have been masking crossing rather than avoiding it)
    _R = {"apply_clears": True, "use_leaves": True, "consume_undisplayed": False}
    if rules:
        _bad = set(rules) - set(_R)
        if _bad:
            raise ValueError(f"unknown replay rule(s): {sorted(_bad)}; valid: {sorted(_R)}")
        _R.update(rules)

    book = _Book()
    book.stats["rule_apply_clears"] = int(_R["apply_clears"])
    book.stats["rule_use_leaves"] = int(_R["use_leaves"])
    book.stats["rule_consume_undisplayed"] = int(_R["consume_undisplayed"])
    for _c, _nm in ((_ADD, "n_add"), (_CANCEL, "n_cancel"), (_MODIFY, "n_modify"), (_TRADE, "n_trade"),
                    (_LEVEL_SET, "n_level_set"), (_LEVEL_DEL, "n_level_del")):
        book.stats[_nm] = int(np.count_nonzero(ev.code == _c))   # message-type counts (for no-ref RATES)
    rows, gi, ng = [], 0, len(grid_ns)

    def _snap():
        bagg = book.consolidated("Bid", round_lot, odd_lot_inclusive)
        aagg = book.consolidated("Ask", round_lot, odd_lot_inclusive)
        bids = sorted(bagg.items(), key=lambda x: -x[0])[:levels]
        asks = sorted(aagg.items(), key=lambda x: x[0])[:levels]
        row = {}
        for i in range(levels):
            row[f"{asset}_bidprice_{i+1}"] = bids[i][0] if i < len(bids) else np.nan
            row[f"{asset}_bidquantity_{i+1}"] = bids[i][1] if i < len(bids) else 0.0
            row[f"{asset}_askprice_{i+1}"] = asks[i][0] if i < len(asks) else np.nan
            row[f"{asset}_askquantity_{i+1}"] = asks[i][1] if i < len(asks) else 0.0
        nb, na = book.best_round_lot("Bid", round_lot), book.best_round_lot("Ask", round_lot)
        row[f"{asset}_nbbo_bid"] = nb if nb is not None else np.nan
        row[f"{asset}_nbbo_ask"] = na if na is not None else np.nan
        cb = bids[0][0] if bids else np.nan
        ca = asks[0][0] if asks else np.nan
        row[f"{asset}_mid"] = (cb + ca) / 2.0 if (bids and asks) else np.nan
        if bids and asks and bids[0][0] >= asks[0][0] - EPS:
            book.stats["consolidated_locked_or_crossed"] += 1
            if bids[0][0] > asks[0][0] + EPS:                   # STRICTLY crossed: impossible for a real
                book.stats["consolidated_crossed"] += 1         # matching engine -> stale orders pinning the top
        return row

    tsa, codea, feeda, sidea = ev.ts, ev.code, ev.feed, ev.side
    pricea, sizea, ppricea, psizea = ev.price, ev.size, ev.pprice, ev.psize
    refa, prefa, mainta = ev.ref, ev.pref, ev.maintain
    leavesa, undispa = ev.leaves, ev.undisplayed
    # The grid pointer only moves FORWARD, so it silently trusts that ``tsa`` is non-decreasing. It
    # is not guaranteed to be: sequence ordering deliberately places an event at its feed's running
    # clock maximum rather than its own stamp, and any stray late timestamp that reaches the loop
    # would flush EVERY remaining grid point in one step and freeze the book for the rest of the
    # session -- a full-length frame, no error, the exact failure class this stack keeps hitting.
    # Advance on the running maximum instead, which is identical when the input IS sorted and
    # bounded when it is not. Inversions are counted rather than assumed away.
    ts_hi = np.int64(-(2 ** 63))
    for j in range(len(tsa)):
        ts = tsa[j]
        if ts < ts_hi:
            book.stats["clock_inversions"] += 1
            ts = ts_hi                              # never let one late stamp jump the grid
        else:
            ts_hi = ts
        while gi < ng and grid_ns[gi] < ts:         # as-of: snapshot reflects all events with ts <= grid
            rows.append(_snap()); gi += 1
        code = codea[j]; feed = feeda[j]
        if code == _ADD:
            book.add(feed, refa[j], sidea[j], pricea[j], sizea[j])
        elif code == _CANCEL:
            if not book.remove(feed, refa[j]):
                book.stats["cancel_no_order"] += 1
        elif code == _MODIFY:
            book.modify(feed, prefa[j], refa[j], sidea[j], pricea[j], sizea[j], ppricea[j], psizea[j], mainta[j])
        elif code == _LEVEL_SET:                     # MBP venue (e.g. iex_deep): set aggregate level size
            book.set_level(feed, sidea[j], pricea[j], sizea[j])
        elif code == _LEVEL_DEL:                     # MBP explicit level delete
            book.set_level(feed, sidea[j], pricea[j], 0.0)
        elif code == _CLEAR:                         # feed reset: this venue disowns everything it sent
            if not _R["apply_clears"]:
                book.stats["feed_clears_ignored"] += 1
                book.stats["events"] += 1
                continue
            n_o, n_l = book.clear_feed(feed)
            if n_o or n_l:                           # a reset onto an empty book is routine session init
                log.info("feed reset (%s) at %s: dropped %d resting order(s) and %d price level(s) "
                         "for that venue -- they would otherwise rest for the remainder of the "
                         "session", ev.feed_names[feed],
                         pd.Timestamp(int(ts), tz="UTC").tz_convert(tz), n_o, n_l)
        elif code == _TRADE:
            # Prefer the venue's OWN post-trade remaining size (leavesquantity) over a decrement:
            # it is exact, self-correcting, and leaves=0 removes a fully filled order deterministically.
            lq = leavesa[j] if _R["use_leaves"] else np.nan
            hit = (book.set_remaining(feed, refa[j], lq) if np.isfinite(lq)
                   else book.reduce(feed, refa[j], sizea[j]))
            if not hit:
                if feed in book.mbp_feeds:           # MBP feed: a level update will reflect the new size
                    book.stats["trade_on_mbp_skipped"] += 1
                elif undispa[j] and not _R["consume_undisplayed"]:
                    # An execution against NON-DISPLAYED liquidity (executionattribute='Hidden' /
                    # printable='NonPrintable'). It has no orderreferencenumber BY CONSTRUCTION -- there
                    # was no displayed order -- so this is not evidence of a broken replay, and it must
                    # NOT consume displayed size: that size is still resting, and removing it deletes
                    # liquidity the venue never traded. Counted separately so the crossed-book
                    # diagnostic reports the rate that actually means something.
                    book.stats["trade_no_ref"] += 1
                    book.stats["trade_undisplayed"] += 1
                else:
                    # A displayed print referencing a resting order our reconstruction never saw --
                    # the crash-day case. Consume by PRICE rather than the `side` field: on these feeds
                    # `side` is the RESTING order's side where it is populated at all, and is blank on
                    # exactly the prints that reach this branch. Price identifies the side unambiguously
                    # in a non-crossed book.
                    book.stats["trade_no_ref"] += 1
                    book.stats["trade_no_ref_displayed"] += 1
                    if book.reduce_at_price(feed, pricea[j], sizea[j]) < 0:
                        book.stats["trade_no_ref_undisplayed"] += 1   # print at a non-displayed price: no-op
        book.stats["events"] += 1
    while gi < ng:                                  # flush remaining grid with the final state
        rows.append(_snap()); gi += 1

    fname = ev.feed_names
    def _names(codes):                              # int feed codes -> sorted feed-name strings (for attrs)
        return sorted(fname[c] for c in codes)
    dual = book.mbo_feeds & book.mbp_feeds
    if dual:                                         # a feed seen on both paths would be double-counted
        log.warning("feeds present as BOTH order-based and price-level (%s) -- possible double count; "
                    "expected each venue to publish one or the other", _names(dual))
    nx = book.stats.get("consolidated_crossed", 0)   # hard invariant: a matching engine cannot cross
    if nx and ng and nx > 0.005 * ng:
        # Report the DISPLAYED refless rate, not the raw one. Executions against non-displayed
        # liquidity have no order reference by construction, so folding them in inflates the figure
        # (SPY runs 10-15% undisplayed on the odd-lot tape) and makes a healthy replay look broken.
        n_tr = book.stats.get("n_trade", 0)
        log.warning("INVARIANT VIOLATED: %s consolidated top is CROSSED (best_bid > best_ask) on %.1f%% of "
                    "%d snapshots -- a real book never crosses, so resting orders that should have left are "
                    "pinning the top. Refless DISPLAYED prints (the ones that indicate a reference-matching "
                    "fault): %d of %d trades; a further %d were undisplayed (Hidden/NonPrintable), which is "
                    "expected and consumes nothing. Feed resets applied: %d (orders %d, levels %d). "
                    "leavesquantity: %d corrections, %d rejected as larger than the resting size. "
                    "Rules: clears=%d leaves=%d consume_undisplayed=%d -- ab_book_rules.py replays "
                    "one fetch under each combination if you need to attribute the crossing.",
                    asset, 100.0 * nx / ng, ng, book.stats.get("trade_no_ref_displayed", 0), n_tr,
                    book.stats.get("trade_undisplayed", 0), book.stats.get("feed_clears", 0),
                    book.stats.get("orders_cleared", 0), book.stats.get("levels_cleared", 0),
                    book.stats.get("trade_leaves_corrected", 0), book.stats.get("trade_leaves_gt_size", 0),
                    book.stats.get("rule_apply_clears", 1), book.stats.get("rule_use_leaves", 1),
                    book.stats.get("rule_consume_undisplayed", 0))

    out = pd.DataFrame(rows, index=grid)
    out.index.name = "time"
    out.attrs["lob_stats"] = dict(book.stats)
    out.attrs["resting_orders_eod"] = len(book.orders)
    out.attrs["mbo_feeds"] = _names(book.mbo_feeds)
    out.attrs["mbp_feeds"] = _names(book.mbp_feeds)
    out.attrs["clock"] = clock
    return out


def reconstruct_session(date_str: str, product: str = "SPY", product_type: str = "direct",
                        levels: int = 10, interval: str = "1s", round_lot: int = 100,
                        odd_lot_inclusive: bool = True, session=("09:30", "16:00"),
                        tz: str = "America/New_York", data_source: str = "apu", clock: str = "exchange",
                        price_scale: float = 1.0,
                        message_types=("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade",
                                       "mt_price_level_update", "mt_clear_orders",
                                       "mt_clear_price_levels"),
                        progress_cb=None, strict: bool = True, rules: Optional[dict] = None):
    """Live entry point: fetch the message types via mstbook_loader and reconstruct the book on one
    GPS-disciplined capture clock. Multi-venue equities (SPY) come back as the consolidated NBBO/ladder
    (hybrid MBO+MBP); a single-venue future (ES) is a CME order-by-order (MBO) replay -- pass the MBO
    message types and ``price_scale=0.01`` (CME integer-hundredths -> index points). This is the SOLE
    extraction path for both legs, so SPY and ES share one clock (the vendor snapshot tool sits on a
    different clock, which would corrupt the cross-asset lead-lag). ``clock`` selects which timestamp
    drives both the fetch index and the event ordering. ``progress_cb(msg)``, if given, is called before
    each message fetch (a liveness heartbeat for the per-session progress display).

    ``strict`` (default) makes a FAILED fetch of a book-critical message type an error instead of an
    empty frame. The two are not interchangeable: an empty mt_add_order stream is a claim about the
    day, a failed one is a claim about the query, and the book that comes back from the second is not
    thin -- it is fabricated. Dropping the adds leaves cancels and trades referencing orders that were
    never inserted, so nothing ever leaves the ladder and the top crosses on most snapshots; dropping
    the trades leaves executed size resting forever, with the same signature. Both used to produce a
    full-length frame with plausible column names and one WARNING line that did not even name the
    date, which is how a 100%-crossed session reached the dataset. ``mt_price_level_update`` and the
    other MBP types are OPTIONAL (empty for futures by construction), so their failure only warns.
    Pass ``strict=False`` for a forensic replay of a known-partial day."""
    import mstbook_loader as ml
    msgs, failed = {}, {}
    for mt in message_types:
        if progress_cb is not None:
            progress_cb(f"{product} {mt}")
        try:
            msgs[mt] = ml._fetch_messages(date_str, product, product_type, mt, data_source, tz=tz, clock=clock)
        except Exception as exc:
            failed[mt] = str(exc).splitlines()[0][:200]
            log.warning("%s %s: fetch %s FAILED: %s", date_str, product, mt, failed[mt])
            msgs[mt] = pd.DataFrame()
    critical = sorted(mt for mt in failed if mt not in _OPTIONAL_MSG_TYPES)
    if critical and strict:
        raise ml.MessageFetchError(
            "%s %s: fetch failed for %s -- replaying the book without %s would invent a book that "
            "never existed (missing adds leave dangling references; missing trades/cancels leave "
            "executed size resting forever), so the session is refused rather than returned crossed. "
            "Details: %s" % (date_str, product, ", ".join(critical),
                             "them" if len(critical) > 1 else "it",
                             "; ".join(f"{k}: {v}" for k, v in sorted(failed.items()))))
    out = reconstruct_book(msgs, asset=ml.canonical_root(product, product_type), levels=levels,
                           interval=interval, round_lot=round_lot, odd_lot_inclusive=odd_lot_inclusive,
                           session=session, tz=tz, date_str=date_str, clock=clock, price_scale=price_scale,
                           consume=True, rules=rules)
    if failed:
        out.attrs["fetch_failed"] = dict(failed)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Validation against the vendor snapshot
# ════════════════════════════════════════════════════════════════════════════
def validate_against_snapshot(recon: pd.DataFrame, snapshot: pd.DataFrame, asset: str = "SPY",
                              levels: int = 10, tol: float = 1e-6) -> dict:
    """Benchmark a reconstructed book against the vendor snapshot on the shared grid: level-1
    bid/ask/mid match rates and per-level price agreement. Run this on a calm day, a March-2020
    circuit-breaker day, and an April-2025 day, and report it as a robustness table."""
    idx = recon.index.intersection(snapshot.index)
    if len(idx) == 0:
        return {"n": 0, "note": "no overlapping timestamps"}
    r, s = recon.loc[idx], snapshot.loc[idx]

    def _match(col):
        if col not in r.columns or col not in s.columns:
            return np.nan
        a, b = r[col].to_numpy(float), s[col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if not both.any():
            return np.nan
        return float((np.abs(a[both] - b[both]) <= tol * np.maximum(1.0, np.abs(b[both]))).mean())

    res = {"n": int(len(idx)),
           "bid1_match": _match(f"{asset}_bidprice_1"),
           "ask1_match": _match(f"{asset}_askprice_1"),
           "mid_match": _match(f"{asset}_mid")}
    rm = (r.get(f"{asset}_bidprice_1") + r.get(f"{asset}_askprice_1")) / 2.0
    sm = (s.get(f"{asset}_bidprice_1") + s.get(f"{asset}_askprice_1")) / 2.0
    diff = (rm - sm).to_numpy(float)
    diff = diff[np.isfinite(diff)]
    res["mid_mean_abs_diff"] = float(np.mean(np.abs(diff))) if diff.size else np.nan
    res["level_price_match"] = float(np.nanmean([_match(f"{asset}_bidprice_{i}") for i in range(1, levels + 1)]
                                                + [_match(f"{asset}_askprice_{i}") for i in range(1, levels + 1)]))
    return res


# ════════════════════════════════════════════════════════════════════════════
# Self-test (synthetic multi-venue message stream; binary not required)
# ════════════════════════════════════════════════════════════════════════════
def _msg_frame(rows, cols, tz):
    """Build a message DataFrame with a tz-aware ns index from (ts_ns, ...) row tuples."""
    df = pd.DataFrame(rows, columns=["receipttimestamp"] + cols)
    idx = pd.to_datetime(df["receipttimestamp"], unit="ns", utc=True).dt.tz_convert(tz)
    return df.drop(columns=["receipttimestamp"]).set_index(idx)


def _selftest() -> bool:
    tz = "America/New_York"
    t0 = pd.Timestamp("2025-04-23 09:30:00", tz=tz).tz_convert("UTC").value  # ns
    s = 1_000_000_000  # 1s in ns
    # Three venues. Build a known consolidated book at the first grid point (09:30:00).
    # bats_edgx: bid 536.60 x150 (round), 536.50 x40 (odd); ask 536.70 x200
    # arca:      bid 536.65 x300 (round, BEST bid); ask 536.72 x100
    # nasdaq:    bid 536.60 x50 (odd);  ask 536.68 x300 (round, BEST ask)
    adds = [
        (t0, "bats_edgx", "Bid", 536.60, 150, "b1"), (t0, "bats_edgx", "Bid", 536.50, 40, "b2"),
        (t0, "bats_edgx", "Ask", 536.70, 200, "b3"),
        (t0, "xdp_arca_integrated", "Bid", 536.65, 300, "a1"),
        (t0, "xdp_arca_integrated", "Ask", 536.72, 100, "a2"),
        (t0, "total_view", "Bid", 536.60, 50, "n1"), (t0, "total_view", "Ask", 536.68, 300, "n2"),
    ]
    add_df = _msg_frame(adds, ["f", "side", "price", "quantity", "orderreferencenumber"], tz)
    # at t0+2s: a modify that RE-IDs (total_view style: ask n2 300@536.68 -> 100@536.69, new ref n3),
    # a partial-less cancel (arca bid a1 removed), and a trade hitting bats ask b3 for 50.
    mod = [(t0 + 2 * s, "total_view", "Ask", 536.69, 100, 536.68, 300, "n3", "n2", "false")]
    mod_df = _msg_frame(mod, ["f", "side", "price", "quantity", "previousprice", "previousquantity",
                              "orderreferencenumber", "previousorderreferencenumber", "maintainpriority"], tz)
    can = [(t0 + 2 * s, "xdp_arca_integrated", "Bid", 536.65, 300, "a1")]
    can_df = _msg_frame(can, ["f", "side", "price", "previousquantity", "orderreferencenumber"], tz)
    trd = [(t0 + 2 * s, "bats_edgx", "Bid", 536.70, 50, "b3", "Sell")]   # trade vs resting ask b3
    trd_df = _msg_frame(trd, ["f", "side", "price", "quantity", "orderreferencenumber", "aggressorside"], tz)
    # iex_deep is an MBP-ONLY venue (no add/cancel/modify) -> must enter the consolidated book via
    # price-level updates. It sets the BEST bid 536.66 x500 at t0, then deletes it (qty 0) at t0+2s.
    plu = [(t0, "iex_deep", "Bid", 536.66, 500),
           (t0 + 2 * s, "iex_deep", "Bid", 536.66, 0)]
    plu_df = _msg_frame(plu, ["f", "side", "price", "quantity"], tz)

    msgs = {"mt_add_order": add_df, "mt_cancel_order": can_df, "mt_modify_order": mod_df,
            "mt_trade": trd_df, "mt_price_level_update": plu_df}
    book = reconstruct_book(msgs, asset="SPY", levels=10, interval="1s",
                            session=("09:30", "09:31"), tz=tz, date_str="20250423")

    r0 = book.iloc[0]                                # state as-of 09:30:00 (orders + iex MBP level)
    # consolidated bids: 536.66 x500 (iex, MBP), 536.65 x300 (arca), 536.60 x200 (bats+nasdaq), 536.50 x40
    # consolidated asks: 536.68 x300 (nasdaq), 536.70 x200 (bats), 536.72 x100 (arca)
    bid_ok = (abs(r0["SPY_bidprice_1"] - 536.66) < 1e-9 and abs(r0["SPY_bidquantity_1"] - 500) < 1e-9
              and abs(r0["SPY_bidprice_2"] - 536.65) < 1e-9 and abs(r0["SPY_bidquantity_2"] - 300) < 1e-9
              and abs(r0["SPY_bidprice_3"] - 536.60) < 1e-9 and abs(r0["SPY_bidquantity_3"] - 200) < 1e-9)
    ask_ok = (abs(r0["SPY_askprice_1"] - 536.68) < 1e-9 and abs(r0["SPY_askquantity_1"] - 300) < 1e-9
              and abs(r0["SPY_askprice_2"] - 536.70) < 1e-9 and abs(r0["SPY_askquantity_2"] - 200) < 1e-9)
    # round-lot NBBO: best bid >=100 -> 536.66 (iex MBP, 500); best ask >=100 -> 536.68 (nasdaq, 300).
    nbbo_ok = (abs(r0["SPY_nbbo_bid"] - 536.66) < 1e-9 and abs(r0["SPY_nbbo_ask"] - 536.68) < 1e-9
               and abs(r0["SPY_mid"] - (536.66 + 536.68) / 2) < 1e-9)
    mbp_ok = ("iex_deep" in book.attrs["mbp_feeds"]
              and set(book.attrs["mbo_feeds"]) == {"bats_edgx", "xdp_arca_integrated", "total_view"}
              and not (set(book.attrs["mbo_feeds"]) & set(book.attrs["mbp_feeds"])))
    print("reconstruct: consolidated book as-of first grid point (hybrid MBO + iex MBP)")
    print("    bids L1=%.2fx%d (iex MBP) L2=%.2fx%d L3=%.2fx%d : %s"
          % (r0["SPY_bidprice_1"], r0["SPY_bidquantity_1"], r0["SPY_bidprice_2"], r0["SPY_bidquantity_2"],
             r0["SPY_bidprice_3"], r0["SPY_bidquantity_3"], bid_ok))
    print("    asks L1=%.2fx%d L2=%.2fx%d : %s" % (r0["SPY_askprice_1"], r0["SPY_askquantity_1"],
                                                   r0["SPY_askprice_2"], r0["SPY_askquantity_2"], ask_ok))
    print("    round-lot NBBO %.2f / %.2f, mid %.3f : %s"
          % (r0["SPY_nbbo_bid"], r0["SPY_nbbo_ask"], r0["SPY_mid"], nbbo_ok))
    print("    iex_deep in consolidated book via MBP (mbp_feeds=%s, no dual): %s"
          % (book.attrs["mbp_feeds"], mbp_ok))

    # state as-of 09:30:02 (modify re-id + cancel + trade applied)
    r2 = book.loc[book.index[book.index.get_indexer([pd.Timestamp("2025-04-23 09:30:02", tz=tz)])[0]]]
    # arca bid a1 cancelled -> best bid now 536.60 x200; nasdaq ask re-id'd to 536.69x100, and trade
    # took 50 off bats ask b3 (200->150). consolidated asks: 536.69x100 (nasdaq), 536.70x150 (bats), 536.72x100
    evt_ok = (abs(r2["SPY_bidprice_1"] - 536.60) < 1e-9 and abs(r2["SPY_bidquantity_1"] - 200) < 1e-9
              and abs(r2["SPY_askprice_1"] - 536.69) < 1e-9 and abs(r2["SPY_askquantity_1"] - 100) < 1e-9
              and abs(r2["SPY_askprice_2"] - 536.70) < 1e-9 and abs(r2["SPY_askquantity_2"] - 150) < 1e-9)
    print("modify(re-id)+cancel+trade applied as-of 09:30:02")
    print("    best bid %.2fx%d (arca cancelled), best ask %.2fx%d (re-id), L2 ask %.2fx%d (trade -50): %s"
          % (r2["SPY_bidprice_1"], r2["SPY_bidquantity_1"], r2["SPY_askprice_1"], r2["SPY_askquantity_1"],
             r2["SPY_askprice_2"], r2["SPY_askquantity_2"], evt_ok))

    # round-lot ladder variant excludes the 40-share odd level
    rl = reconstruct_book(msgs, asset="SPY", levels=10, interval="1s", odd_lot_inclusive=False,
                          session=("09:30", "09:31"), tz=tz, date_str="20250423").iloc[0]
    rl_ok = not np.any(np.isclose(np.array([rl[f"SPY_bidquantity_{i}"] for i in range(1, 11)]), 40.0))

    # feeds the stack: canonical schema -> weighted_mid / decay_weighted_cost finite
    import liquidity_curve_metrics as lcm
    wm = lcm.weighted_mid(book, "SPY"); auc = lcm.decay_weighted_cost(book, "SPY", 10)
    stack_ok = (np.isfinite(wm).mean() > 0.5 and np.isfinite(auc).mean() > 0.5
                and "SPY_askprice_1" in book.columns)

    # validation helper sanity: a book validated against itself matches 100%
    # clock switch: with source capture co-located at the venues, exchangetimestamp == receipt, so an
    # exchange-clock run must reproduce the receipt-clock book exactly (the invariance we report).
    msgs_x = {}
    for mt, df in msgs.items():
        dx = df.copy()
        dx["exchangetimestamp"] = df.index.tz_convert("UTC").as_unit("ns").astype("int64")
        msgs_x[mt] = dx
    book_x = reconstruct_book(msgs_x, asset="SPY", levels=10, interval="1s",
                              session=("09:30", "09:31"), tz=tz, date_str="20250423", clock="exchange")
    clock_ok = (book_x.attrs.get("clock") == "exchange"
                and book.drop(columns=[], errors="ignore").equals(book_x[book.columns]))
    print("clock switch (exchange==receipt under co-located capture -> identical book):", clock_ok)

    v = validate_against_snapshot(book, book, asset="SPY", levels=10)
    val_ok = v["bid1_match"] == 1.0 and v["ask1_match"] == 1.0 and v["mid_match"] == 1.0

    print("round-lot ladder excludes 40-share odd level:", rl_ok,
          "| feeds stack (wmid/auc finite):", stack_ok, "| validate self==100%:", val_ok)
    print("stats:", book.attrs["lob_stats"])

    # ---- ES single-venue CME MBP replay (integer-hundredths -> index points via price_scale) ----
    et0 = pd.Timestamp("2025-04-23 09:30:00", tz=tz).tz_convert("UTC").value
    es_plu = [(et0, "cme", "Bid", 543750, 5), (et0, "cme", "Ask", 543775, 8),
              (et0, "cme", "Bid", 543725, 12), (et0, "cme", "Ask", 543800, 6),
              (et0 + s, "cme", "Bid", 543750, 0)]                 # delete top bid -> best bid 5437.25
    es_df = _msg_frame(es_plu, ["f", "side", "price", "quantity"], tz)
    es_book = reconstruct_book({"mt_price_level_update": es_df}, asset="ES", levels=10, interval="1s",
                               session=("09:30", "09:31"), tz=tz, date_str="20250423", price_scale=0.01)
    e0 = es_book.iloc[0]
    es_ok = (abs(e0["ES_bidprice_1"] - 5437.50) < 1e-9 and abs(e0["ES_bidquantity_1"] - 5) < 1e-9
             and abs(e0["ES_askprice_1"] - 5437.75) < 1e-9 and abs(e0["ES_bidprice_2"] - 5437.25) < 1e-9
             and "cme" in es_book.attrs["mbp_feeds"] and not es_book.attrs["mbo_feeds"])
    e1 = es_book.loc[es_book.index[es_book.index.get_indexer(
        [pd.Timestamp("2025-04-23 09:30:01", tz=tz)])[0]]]
    es_del_ok = abs(e1["ES_bidprice_1"] - 5437.25) < 1e-9       # top bid deleted -> 5437.25 is best
    print("ES single-venue CME MBP replay (price_scale 0.01 -> index points)")
    print("    L1 %.2f/%.2f, L2 bid %.2f; after top-bid delete best bid %.2f (mbp_feeds=%s, no MBO): %s"
          % (e0["ES_bidprice_1"], e0["ES_askprice_1"], e0["ES_bidprice_2"], e1["ES_bidprice_1"],
             es_book.attrs["mbp_feeds"], es_ok and es_del_ok))

    # ---- #1/#2: integer ref keys (via factorization) handle references that OVERFLOW int64 and
    #      ALPHANUMERIC references -- a naive to_numeric(int64) would alias/corrupt them; and the
    #      consume=True memory path must yield the byte-identical book to consume=False ----
    bt0 = pd.Timestamp("2025-04-23 09:30:00", tz=tz).tz_convert("UTC").value
    big, big2 = str(2**63 + 7), str(2**63 + 8)         # > int64 max; distinct, would collide under float64
    badd = [(bt0, "bats_edgx", "Bid", 100.00, 10, big),
            (bt0, "bats_edgx", "Bid", 100.00, 20, big2),         # same price, two huge distinct refs
            (bt0, "bats_edgx", "Ask", 100.05, 30, "ORD-A1B2")]   # alphanumeric ref
    bcan = [(bt0 + s, "bats_edgx", "Bid", 100.00, 10, big)]      # cancel ONLY the first huge-ref order
    bmsgs = {"mt_add_order": _msg_frame(badd, ["f", "side", "price", "quantity", "orderreferencenumber"], tz),
             "mt_cancel_order": _msg_frame(bcan, ["f", "side", "price", "previousquantity", "orderreferencenumber"], tz)}
    bb = reconstruct_book(bmsgs, asset="SPY", levels=5, interval="1s", session=("09:30", "09:31"),
                          tz=tz, date_str="20250423", consume=False)
    b0 = bb.iloc[0]                                     # before cancel: bid 100.00 x30 (10+20), ask x30
    b1 = bb.loc[bb.index[bb.index.get_indexer([pd.Timestamp("2025-04-23 09:30:01", tz=tz)])[0]]]
    bigref_ok = (abs(b0["SPY_bidquantity_1"] - 30) < 1e-9        # after cancelling ONLY big(10): x20 remains
                 and abs(b1["SPY_bidquantity_1"] - 20) < 1e-9    # big2 untouched -> no ref aliasing
                 and abs(b1["SPY_askquantity_1"] - 30) < 1e-9)
    bb_c = reconstruct_book(dict(bmsgs), asset="SPY", levels=5, interval="1s", session=("09:30", "09:31"),
                            tz=tz, date_str="20250423", consume=True)   # shallow copy -> bmsgs survives
    consume_ok = bb.equals(bb_c[bb.columns])
    print("integer-ref robustness (#1) + consume invariance (#2)")
    print("    huge(>2^63) & alphanumeric refs matched exactly (no float aliasing): %s | consume==keep: %s"
          % (bigref_ok, consume_ok))

    # hard invariant: a correctly reconstructed book is NEVER crossed (best_bid < best_ask). This is the
    # guard for the all-day-crossed ES bug (stale orders pinning the top) -- it must hold on every book.
    crossed_ok = all(b.attrs.get("lob_stats", {}).get("consolidated_crossed", 0) == 0
                     for b in (book, rl, book_x, es_book, bb, bb_c))
    print("invariant: no reconstructed book is ever crossed (best_bid < best_ask):", crossed_ok)

    ok = all([bid_ok, ask_ok, nbbo_ok, evt_ok, rl_ok, stack_ok, val_ok, clock_ok, mbp_ok,
              es_ok, es_del_ok, bigref_ok, consume_ok, crossed_ok])
    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    _selftest()
