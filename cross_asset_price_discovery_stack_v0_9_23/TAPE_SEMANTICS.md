# What the SPY tape says, and what the replay was ignoring

A sample pull of **every** `mt_*` type for SPY on 2024-12-18 (`mstwx-lakequery --print-headers
--limit 20`) settled several questions that had been inferred from symptoms. The reconstruction was
consuming five message types and taking its semantics for two of them from assumptions the tape
contradicts. Three of the omissions change the book.

Read the timestamps carefully: `1734512400` is **04:00 ET**, not 09:30. The pull starts at the
pre-market open, and the replay processes every message from the first one of the day — only the
snapshot **grid** is restricted to 09:30–16:00. Pre-market state is therefore carried into the
session, which is what makes finding 1 matter.

---

## 1. `mt_clear_orders` / `mt_clear_price_levels` — feed resets, never fetched

```
mt_clear_orders,1734505200018350611,3396,...,xdp_nyse_integrated,...        02:00:00 ET
mt_clear_orders,1734517544663671426,5,...,miax_pearl_equities_dom,...       05:25:44 ET
mt_clear_orders,1734574666870010018,247476821,...,memoir_depth,...          21:17:46 ET
```

A clear is the venue saying *discard everything I have told you and rebuild from my next message*.
It is issued on a line failover, a gap recovery, or a session-state transition, and it carries no
price or quantity — the point is that the prior state is void.

**Not applying one is unrecoverable.** The venue never cancels the orders it just disowned, because
as far as it is concerned they no longer exist. Every one of them rests in the consolidated ladder
for the remainder of the session while the venue rebuilds its book under fresh reference numbers.
A stale pre-market bid pinned that way sits above the current ask for hours — precisely "resting
orders that should have left are pinning the top."

The 02:00–02:04 clears land on an empty book and are routine session init. The **05:25:44
miax_pearl_equities_dom** one does not: it is 85 minutes into the pre-market, with the replay
already holding that feed's state.

`test_feed_reset.py` demonstrates the mechanism end to end: with the reset ignored, a stale 605.50
bid pins the top against a 604.90 ask and the book is **100% crossed**; with it applied the same
messages give a clean 604.80 / 604.90 and other venues are untouched.

Fixed: both types are fetched and replayed. Within a sequence tie a reset ranks **first**, not last
— the venue clears and then rebuilds, so a clear tied with adds in the same packet must precede
them, or it would wipe the very re-adds it exists to make room for.

## 2. `mt_trade.leavesquantity` — the venue's own post-trade size, pruned away

```
605,2,...,3189381831070546706,1198,Ask     # trade 2  -> 1198 remain
605,4,...,3189381831070546706,1194,Ask     # trade 4  -> 1194
605,2,...,3189381831070546706,1192,Ask     # trade 2  -> 1192
```

The arithmetic is exact, so the semantics are not in doubt: `leavesquantity` is what is left on the
**resting** order after the execution. It is authoritative where a decrement is merely arithmetic —
immune to a missed earlier partial, a duplicate print, or a size mis-parsed on the add. The case
that matters is `leaves = 0`: a fully filled order that is not removed rests forever and pins the
top.

It was not in `_MSG_NEEDED_COLS`, so `usecols` dropped it at CSV read — **the same failure mode as
the original `sequencenumber` bug**, in the same function. Now read, and the replay assigns the
venue's figure rather than decrementing (a corrected drift is counted in
`lob_stats["trade_leaves_corrected"]`).

This also settles the `side` question. The tape has a **separate** `aggressorside` column (blank for
equities — the stack already falls back to the tick rule), while `side` accompanies
`orderreferencenumber` and `leavesquantity` and describes the **resting** order. The three trades
above are `side=Ask` at 605.00 with the bid at 604.87, so the resting order is genuinely an ask.

## 3. Refless trades are mostly *hidden liquidity*, and were being treated as faults

```
605.02,7,...,RegularTrade,,,,Printable,,,,...,EarlyOpening,...   # no orderreferencenumber
605.02,1,...,OddLotTrade,NonPrintable,,,...,Hidden,...           # no orderreferencenumber
604.54,4,...,OddLotTrade,Printable,3189381831070637292,,,Hidden  # ref, no side/leaves
```

An execution against non-displayed liquidity has **no order reference by construction** — there is
no displayed resting order to reference. The tape marks it: `executionattribute='Hidden'` and/or
`printable='NonPrintable'`.

Two consequences, both bad:

* The diagnostic counted these in `trade_no_ref`, so the reported "11–17% of trades have no
  reference" on the MWCB days conflates an expected property of the SPY tape with a
  reference-matching fault. The number that indicates a fault is the **displayed** refless rate.
* Worse, the fallback consumed displayed size at the print's price. A hidden execution at a price
  that is also displayed then deletes liquidity that is still resting.

Fixed: `executionattribute` is read (it was pruned too), undisplayed prints consume nothing, and
the two rates are counted and reported separately.

## 4. `mt_missing_product_messages` and `mt_error` — the oracle nobody was asking

```
mtype,receipttimestamp,sequencenumber,dt,f,marketparticipant,product,
      expectedproductsequencenumber,currentproductsequencenumber
(no rows for 2024-12-18)
```

This is the venue's own gap report: expected vs received sequence number, i.e. the multicast packets
that never arrived. Missing packets mean missing **adds**, so the cancels and trades referencing
them are orphans and the levels they should have removed rest forever — a crossed book caused by
data loss, which no code change can fix.

Empty on 2024-12-18 (a clean capture). `feed_health.py` runs it, `mt_error`, and the reset inventory
for any product-day and returns a verdict: **capture incomplete → DATA**, or **capture complete →
the replay's fault, fixable in code**. This is the decisive test for the residual ~3.9% crossing on
the four MWCB days, and it costs one query per day:

```bash
python feed_health.py --date 20200309 --product SPY
```

STAGE 3 now runs it automatically on any session the gate flags, before `debug_crossing`.

## 5. The gate was diagnosing a different book than the one it saved

`debug_crossing` was invoked with `--clock exchange` while extraction runs on `receipt`. Two
problems: the root cause then describes a book that was never written to disk, and the tape shows
why exchange is the weaker choice for this feed — many consecutive `bats_edgx` messages carry the
**same** `exchangetimestamp` (`1734512400000047000`) while their receipt timestamps differ by
milliseconds, so exchange-clock ordering is degenerate across those bursts. Now `--clock receipt`,
matching the extraction.

---

## Confirmed correct, no change needed

* **`aggressorside` is blank for equities** and populated for CME futures — the stack's documented
  tick-rule fallback for equities is right.
* **Per-feed sequence namespaces.** `total_view` 282783, `bats_edgx` 5163, `xdp_arca` 7634 at the
  same instant. Ordering keyed on `(feed, sequencenumber)` is correct; a global sort would be
  meaningless.
* **`total_view` re-IDs on modify** (`orderreferencenumber=4616` from `previousorderreferencenumber=24`),
  `bats_edgx` keeps the reference. Both handled.
* **MBP price-level updates key on price, not `level`.** `iex_deep` sends `level` empty and
  `numberoforders=0`, so anything keyed on `level` would fail; the stack keys on price. `quantity=0`
  as a level delete is handled.
* **`mt_bbo_quote` / `mt_nbbo_quote` are empty for SPY**, and `mt_aggregated_price_update` returns
  `\N` for the early rows. The stack builds its own NBBO from the message stream and depends on
  none of them — the right call, and now demonstrably so.
* **`mt_trade_break` / `mt_trade_correction`** are empty for this date; both are already consumed by
  the trade scrubber.

## Still unfetched, deliberately

`mt_retail_price_improvement`, `mt_product_statistics`, `mt_index_update`. None affect the book.
`mt_product_statistics` carries the official open/close/previous-close per venue and would be a
cheap independent check on the reconstructed opening print — worth adding if a referee asks how the
replay's 09:30 state was validated.
