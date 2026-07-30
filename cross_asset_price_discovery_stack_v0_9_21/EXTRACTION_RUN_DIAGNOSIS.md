# Diagnosis of the first real `--source extract` run (2026-07-30, 24 sessions)

Log: `replication_20260730_122746`, stack `v0_9_19`, Python 3.11.8, numpy 1.26.4, pandas 2.3.2,
scipy 1.17.1, `max_workers=4`. Started 12:28:50, died 15:18. **2h50m, zero output.**

STAGE 0 and STAGE 1 were clean — all six correctness tests passed, so every correction from
v0.9.15–v0.9.20 was present and working in that run. Everything below is in the extraction path,
which STAGE 1 did not cover.

---

## What the log actually says

### A. The run died on the last session and threw away the other 22

```
RuntimeError: mstwx-lakequery failed (rc=1)
  ... _run_mstwx_lakequery_to_file -> _fetch_messages -> trade_flow -> attach_flow
  ... -> _extract_one_session -> joblib _parallel_sessions -> extract_sessions
```

22 of the 24 days had already completed (`2025-04-03` and `2026-06-05` had not). joblib's default
is fail-fast: the first worker exception is re-raised out of the generator and every completed
result is discarded. Nothing had been written to disk yet, so 2h50m of vendor I/O was lost to one
transient vendor exit.

Note the asymmetry that made this possible. `reconstruct_session` wrapped each message-type fetch
in `try/except` and swallowed it. `attach_flow` — the trade-tape half of the same session — did
not. So the *same* vendor error was survivable in one code path and fatal in the other.

**Fixed:** vendor rc≠0 is retried with backoff (`MST_LAKEQUERY_RETRIES`, default 3); a session that
still fails is dropped, named, and cannot take the batch down (`_guarded_session`); good sessions
are cached to disk as they land (`--extract-cache`), so a re-run costs only the missing days.

### B. Two swallowed fetch failures produced books that never existed

```
fetch mt_add_order failed for ESU3: mstwx-lakequery failed (rc=1)
fetch mt_trade    failed for SPY:  mstwx-lakequery failed (rc=1)
```

Neither line names a date, and four sessions were running concurrently, so they are not even
attributable after the fact. Both were turned into an empty DataFrame and the replay continued.

The consequences are in the log, a few lines apart:

| session | symptom | reading |
|---|---|---|
| ~2024-07-24 | `SPY CROSSED on 100.0% of 23401 snapshots`, `trade_no_ref=0 of 0 trades` | zero trades fetched — the swallowed SPY `mt_trade` failure. With no executions, nothing removes filled size, so every resting order stays and the top is crossed on every single snapshot. |
| ~2026-01-19 | `ES CROSSED on 43.8%`, `trade_no_ref=119383 of 119383` | 100% of trades reference an order that was never added — the signature of a lost `mt_add_order` stream. |
| 2026-01-19 | `median SPY=nan` | the SPY leg has no usable top of book at all. |
| 2020-03-09/12/16/18 | `median ES=nan` on all four | **no ES leg on any MWCB day** — the paper's centerpiece event study had nothing to estimate a cross-asset lead-lag from. |

An *empty* `mt_add_order` stream is a statement about the day. A *failed* one is a statement about
the query. Treating them identically is what let a 100%-crossed frame into the dataset with a
single unattributed warning.

**Fixed:** a failed fetch of a critical (MBO) message type raises `MessageFetchError` naming the
date, product and type; the session is refused rather than returned crossed. MBP price-level types
stay optional (empty for futures by construction) and are recorded in `df.attrs["fetch_failed"]`.

### C. Nothing checked the frame before it was saved

All four `median ES=nan` sessions were written out as normal 23,401-row frames. Shape and column
names carry no information about whether a book is real.

**Fixed:** `session_qc` runs on the frame about to be saved (is each leg present; does its top
cross more than `crossed_tol`), `--qc-action warn|drop|raise` decides what to do, and
`extract_report.txt` records requested vs usable sessions next to the results.

### D. STAGE 3 would have caught B and C — but never ran, and cost as much again as STAGE 2

STAGE 3 gated by calling `debug_crossing.py` per session, which re-pulls that session's raw
messages: a second multi-hour pass over the same sample. Worse, it judged a *freshly fetched* book
rather than the frame on disk, so a session could pass the gate and still be estimated from a
broken frame.

**Fixed:** `qc_frames.py` gates the saved pickle in seconds, and `debug_crossing` runs only on the
sessions it flags — root cause, which is what it is for.

---

## Still open — needs the data, not the code

### 1. Residual SPY crossing of ~3.9–4.0% on the four MWCB days

```
SPY CROSSED on 3.9% of 23401 snapshots (trade_no_ref=350275 of 2463095 trades)   # 14.2% refless
SPY CROSSED on 4.0% of 23401 snapshots (trade_no_ref=353106 of 2038280 trades)   # 17.3%
SPY CROSSED on 3.9% of 23401 snapshots (trade_no_ref=359832 of 2735389 trades)   # 13.2%
SPY CROSSED on 3.9% of 23401 snapshots (trade_no_ref=404473 of 3611404 trades)   # 11.2%
```

This is **not** the `sequencenumber` bug — that one crossed 99.9% of snapshots and the regression
test for it passes. It is a much smaller, very stable residual, on the four March-2020 sessions,
with 11–17% of trades unable to find the resting order they executed against.

Two candidate explanations, and they call for different responses:

* **Real.** A consolidated multi-venue top legitimately crosses at sub-second scale (venue A's bid
  above venue B's ask before the quote propagates), and March 2020 is exactly when that is most
  common. If so ~4% is a finding for the data appendix, not a bug — and the strict round-lot NBBO
  (`SPY_nbbo_bid`/`SPY_nbbo_ask`) should cross far less than the odd-lot-inclusive ladder.
* **A replay fault.** 11–17% refless trades is high. If those trades are executing against orders
  added before 09:30 (the session window starts at the open, but resting liquidity does not), the
  book is missing its opening state and the un-removed size pins the top.

The two are distinguishable with one command per day, no code change:

```bash
python debug_crossing.py --date 20200309 --product SPY --clock exchange --ab-ordering
```

CHECK 5 separates orphaned removals whose `add` is present-but-unmatched (CODE) from those whose
add is simply before the window (DATA/benign); CHECK 8 reports resurrections, wrong-side resting
orders and pinned prices. That is the evidence needed to decide. **Do this before running the full
replication again** — 3.9% is small enough to look like noise in a table and large enough to move a
lead-lag estimate.

The new gate's default tolerance (0.5%) flags these four days, which is the intended behaviour: it
forces the question rather than answering it.

### 2. No ES leg on any of the four MWCB days

`ESH0` (03-09, 03-12) and `ESM0` (03-16, 03-18) both returned nothing usable. The front-month
rollover is right (March 2020 ES expired 2020-03-20, so 03-16 and 03-18 correctly roll to June), so
this is either a symbology mismatch for 2020 CME contracts in your lake, or those dates genuinely
are not there. Check with a direct query before the next run:

```bash
mstwx-lakequery --date 20200309 -s futures -p ESH0 -m mt_add_order --print-headers --format csv | head
```

If the lake wants a different symbol form for 2020, `get_front_month_contract` needs the mapping.
Until this resolves, the MWCB panel has one leg.

### 3. The date universe is not Appendix A.1

The run used 2022–2026 volatile/baseline dates; the script's baked-in defaults are the paper's
2014–2017 sample (plus the same four March-2020 MWCB days). If the new universe is intentional —
a re-sample onto the modern period — Appendix A.1 and every "N days" statement in the paper need to
change with it, and the volatile/baseline matching rule (same day-of-week, ~one year prior) should
be re-stated for the new dates. If it was a convenience sample for a plumbing test, note that none
of these numbers are the paper's.

---

## What to run next

```bash
# 1. root-cause the residual crossing on one MWCB day (minutes, one session)
python debug_crossing.py --date 20200309 --product SPY --clock exchange --ab-ordering

# 2. confirm the 2020 ES contracts exist under the symbol the loader asks for
mstwx-lakequery --date 20200309 -s futures -p ESH0 -m mt_add_order --print-headers --format csv | head

# 3. re-run the replication. The cache means a second failure costs only the missing days,
#    and the gate now stops on a bad frame instead of estimating on it.
./run_paper_replication.sh --source extract --extract-cache /scratch/sessions
```

Everything the previous run got through the hard way — 22 completed sessions — would now be on disk
in `/scratch/sessions` and reused.
