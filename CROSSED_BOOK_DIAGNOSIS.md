# SPY 2024-01-03 crossed book — root cause

**Verdict: CODE.** Not in `lob_reconstruct`'s replay logic, and not in the data.
`mstbook_loader` was deleting the column the replay orders events by, so every
production session ran in the replay's documented *degraded* ordering mode.

The tool's `VERDICT: CODE` was directionally right. Neither of its two CODE
findings names the cause, and its DATA finding is almost certainly a false
positive that would have sent you to re-fetch a session whose messages are fine.

---

## The defect

`mstbook_loader._read_messages_csv` builds `usecols` from `_MSG_NEEDED_COLS` and
drops every column not listed there. `sequencenumber` was not in the set — the
string does not appear anywhere in `mstbook_loader.py`. Demonstrated against the
real read path, with the column present in the CSV:

```
input header : receipttimestamp,sourcetimestamp,sequencenumber,f,side,price,quantity,orderreferencenumber
columns kept : ['receipttimestamp', 'sourcetimestamp', 'f', 'side', 'price', 'quantity', 'orderreferencenumber']
sequencenumber survives? False
```

So the column is gone before `lob_reconstruct` ever sees a frame. In
`_event_arrays`, `order_by="sequence"` then finds no sequence anywhere and falls
through to:

```python
order = np.lexsort((elim_rank, ts))     # clock-only ordering
```

which is the legacy ordering the sequence path exists to replace. It took that
branch **silently** — no warning, no stat, no attribute on the output frame.

That is why `CHECK 4` printed `(no sequencenumber column)`. That line was not
context. It was the answer, and the tool assigned no finding to it.

## Why this crosses the book

Straight from `lob_reconstruct`'s own comment at the ordering code, and from the
regression test that already ships in the stack:

> UDP multicast arrives out of packet order, so ordering a feed by ANY timestamp
> inverts ~10% of adjacent events: a Cancel can precede the Modify it follows,
> the Modify then RESURRECTS the deleted order, and that phantom level crosses
> the consolidated top on every snapshot.

`modify()` is remove-old + add-new. A Modify applied *after* the Cancel of the
order it references re-creates that order — and the cancel is now spent, so
nothing will ever remove it again. The phantom is immortal and pins that
venue's own top crossed for the rest of the session.

`test_reconstruct_ordering.py` already measures exactly this, and it passes today:

```
(A) legacy(clock)   crossed_frac=1.00 over 240 snaps
(B) fixed(sequence) crossed_frac=0.00 over 240 snaps
```

Same events, same code, only `order_by` differs. Your session was running
configuration (A) without knowing it.

## Why the whole test suite stayed green

Every crossing test builds message frames directly and hands them to
`reconstruct_book`. They all include `sequencenumber`, so they all exercise the
healthy path. The one layer that could drop the column — `_read_messages_csv` —
was the one layer no test crossed. The fix path was fully tested and completely
disconnected from production.

## How this matches your output, point by point

| Observation | Explanation under this root cause |
|---|---|
| `CHECK 4  (no sequencenumber column)` | the defect itself |
| crossed from the **first** snapshots (91.9% of first 234) | phantoms appear in the opening burst, immediately |
| resting-order count **stable** (1.1x) | a few thousand phantoms don't move a ~31k book — so it reads as "structural", not "accumulation" |
| **individual venue** books crossed (99.91%) | the book is keyed `(feed, ref)`; a resurrection is confined to one feed |
| side tokens 100% clean, no NaN price, one scale | correct — there is no parsing fault to find |
| orphans concentrated in `xdp_arca_integrated`, `total_view`, `bzx` | the highest modify-traffic feeds (`total_view` alone has 1.98M modifies) |

The report's `[CODE/SEVERE]` conclusion — "structural (side/scale/column
parsing)" — is the only vocabulary it had left after `CHECK 3` came back clean.
It had no check that could see an ordering fault.

## The DATA finding is likely a false positive

`[DATA/SEVERE] 98.4% of orphaned removals reference orders whose add never
appears... messages are genuinely missing from the fetch`

Two problems:

1. **The rate is never considered.** 48,565 orphans against 22.3M cancel+trade
   events is **0.22%**. The finding fires on the *share of orphans that are
   absent* (98.4%), so three orphaned removals in a whole day would trigger it
   just as hard.
2. **The time-of-day test doesn't separate what it claims to.** "Spread across
   the session, not just the open" is presented as proof of a lossy capture. But
   an order carried in from a prior session (GTC) has no add anywhere in a
   one-day fetch and is cancelled *at whatever hour its owner chooses* — uniform
   across the day by construction. At 0.22% the two are indistinguishable from
   the time profile alone.

And decisively: **an orphaned removal removes nothing.** Crossing requires an
order that *stayed*. Orphaned removals cannot cross a book, so this finding
could not have explained the symptom even if it were real.

---

## Changes

### 1. `mstbook_loader.py` — the fix (root cause)

Added `sequencenumber` to `_MSG_NEEDED_COLS`.

### 2. `lob_reconstruct.py` — stop failing silently

- The degraded ordering branch now logs a **WARNING**. A silent degrade is how a
  loader change disabled the ordering fix across every session without one line
  in a run log.
- New warning for **partial** coverage: a message type missing the column gets
  `seq=inf` and is ordered *last within its feed*. If `mt_trade` were the frame
  missing it, every trade would be applied at end of day.
- **NaT clock guard.** `pd.to_datetime(...).astype("int64")` turns a null
  timestamp into `INT64_MIN`, which doesn't just misplace the event — it sorts it
  ahead of the entire session. Those rows now fall back to the frame index.
- **`<NA>` guard.** `_read_messages_csv` reads the ref/side columns as pandas
  `string` dtype, whose missing values stringify to `"<NA>"`, not `"nan"`. The
  existing null test (`pref == "" | pref == "nan"`) stopped matching when that
  dtype was adopted. Now matched via `_NA_TOKENS`.

### 3. `debug_crossing.py` — make the tool able to see this class of fault

- **CHECK 4** now raises `CODE/FATAL` naming `sequencenumber` and pointing at
  `_MSG_NEEDED_COLS`, instead of printing a one-line aside. Also reports
  per-feed **clock inversions**: adjacent events, in the venue's own sequence
  order, whose clock runs backwards — i.e. how many pairs clock-only ordering
  applies in reverse.
- **CHECK 4b (new)** timestamp health: null clocks and dead/constant clocks, per
  feed. Both collapse a feed's day onto the first grid point and produce
  "crossed from the open with a stable book" — the exact signature CHECK 6
  attributes to parsing.
- **CHECK 8 (new)** — *the missing mirror of CHECK 5.* CHECK 5 only asks "a
  removal arrived, where is its order?", which can only ever explain a removal
  that did nothing. It is structurally incapable of explaining a crossed book.
  CHECK 8 asks the question that matters: **which orders are holding the top
  apart, and how did they get there?** It counts modifies that re-created an
  already-cancelled order (the resurrection signature, which leaves no orphan and
  is therefore invisible to CHECK 5), lists the orders resting on the wrong side
  of their own venue's top with the time each entered, and reports the longest-held
  pinned prices. It also surfaces `book.stats` (`dup_add`, `trade_no_ref`, …),
  which were being collected and never printed.
- **CHECK 9 (new, `--ab-ordering`)** replays the identical messages under both
  orderings and reports the crossing rate of each — settling the ordering
  question by experiment. Two extra full replays, so it is opt-in.
- **CHECK 5 recalibrated:** reports the orphan rate against total removals, and
  ranks a low-rate uniform tail as `MINOR` with the carry-over explanation
  stated. `MINOR` findings are reported but do not vote in the verdict, so a
  tail observation can no longer swing a session to `DATA`.

### 4. `test_crossed_root_cause.py` (new)

Guards the fault at both levels: that the loader preserves the column through the
real CSV read, and that the diagnostic *names* it. On a two-venue fixture whose
only defect is the missing column:

```
(1) loader read keeps 'sequencenumber'=True and still prunes unused columns=True : True
(2) same messages: with sequencenumber crossed=0.0%, without it crossed=100.0%   : True
(3) healthy session -> 0 findings (want 0)                                       : True
(4) faulty session -> a CODE finding names 'sequencenumber'=True, FATAL=True     : True
(5) CHECK 8 -> 1,600 resurrections, 1,562 wrong-side orders at the close         : True
(6) verdict leads with CODE (not DATA/re-fetch)                                  : True
(7) two DATA/MINOR findings cannot outvote one CODE/FATAL                        : True
```

---

## Verification

No behavioral change to any existing test — the full 31-file suite gives
identical results before and after; the only delta is the new file. The 16
non-passing tests fail identically in both trees on a missing `scipy`, and none
of them import `lob_reconstruct` or `mstbook_loader`.

Passing on the patched tree: `test_reconstruct_ordering`,
`test_crossed_regression`, `test_debug_crossing`, `test_verify_crossing`,
`test_crossing_qc`, `test_crossed_root_cause`, and `lob_reconstruct._selftest()`.

## What I could not verify

I have no access to your MayStreet data, so the final link — that this defect
accounts for *your* 99.92% on SPY 2024-01-03 — is inference validated on a
fixture, not measured on your session. The loader defect and the mechanism are
proven; the magnitude on your data is not.

Your saved `msgs.pkl` cannot settle it either: it was written *after* the prune,
so it has no `sequencenumber` column. Confirming this needs one re-fetch.

```bash
# 1. re-fetch (the column now survives) and confirm CHECK 4 goes quiet
python debug_crossing.py --date 20240103 --product SPY --save-messages msgs_seq.pkl

# 2. the decisive A/B — same messages, ordering is the only difference
python debug_crossing.py --messages-pickle msgs_seq.pkl --asset SPY --ab-ordering
```

Expected if this is the whole story: CHECK 4 clean, CHECK 9 showing a high
crossing rate under clock ordering and near-zero under sequence, CHECK 8
reporting no resurrections, and the crossed rate collapsing from 99.92%.

If crossing survives with the sequence in place, CHECK 8 will now name the
specific orders pinning the top and when they entered, which is the next thread
to pull rather than a fresh guess.

One thing to check on the re-fetch regardless: your run used `--clock exchange`
(the `debug_crossing` default). `_CLOCK_COLS["exchange"]` resolves to
`sourcetimestamp` on MayStreet, whose population quality varies by feed —
CHECK 4b will now tell you whether any feed's is null or dead. Note the
`lob_reconstruct` docstring calls receipt "the defensible default" while
`reconstruct_book`'s signature defaults to `clock="exchange"`; worth settling
which you intend.
