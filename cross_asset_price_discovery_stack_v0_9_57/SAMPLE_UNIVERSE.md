# The re-sampled date universe (2022–2026), and what the paper has to say about it

The published sample is Appendix Table A.1: ten volatile days drawn from 2014–2017, ten matched
baseline days, and the four March-2020 MWCB halt days. The stack defaults to a **re-sampled**
universe on 2022–2026 so the results speak to the current market. The construction rule is
unchanged — only the window moved.

`./run_paper_replication.sh --paper-sample` restores the published universe verbatim, so the
2014–2017 numbers remain reproducible from the same driver.

**Revised 2026-08-05 (v0.9.57).** This is the universe the first full analysis run (the
2026-08-05 halt-masked run) actually used, now made the default. It repairs both defects the
previous default carried — see "History" below — and `validate_sample.py` passes it clean.

## The sample

| # | volatile | dow | baseline (matched) | dow | gap |
|---|---|---|---|---|---|
| 1 | 2024-12-18 | Wed | 2023-12-20 | Wed | 364d |
| 2 | 2026-06-05 | Fri | 2025-06-13 | Fri | 357d |
| 3 | 2025-10-10 | Fri | 2024-10-18 | Fri | 357d |
| 4 | 2024-09-03 | Tue | 2023-09-05 | Tue | 364d |
| 5 | 2025-04-03 | Thu | 2024-04-04 | Thu | 364d |
| 6 | 2024-08-05 | Mon | 2023-08-07 | Mon | 364d |
| 7 | 2024-07-24 | Wed | 2023-07-19 | Wed | 371d |
| 8 | 2025-01-27 | Mon | 2024-01-29 | Mon | 364d |
| 9 | 2023-03-09 | Thu | 2022-03-24 | Thu | 350d |
| 10 | 2025-08-01 | Fri | 2024-08-09 | Fri | 357d |

MWCB: 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18 — fixed by history, unchanged.

Every pair is a same-weekday match at 350–371 days. Event classification for the appendix
(E = scheduled macro announcement, U = unexpected): 2024-12-18 FOMC (E), 2024-08-05 yen-carry
unwind (U), 2025-04-03 post-"Liberation Day" tariffs (U), 2025-01-27 DeepSeek/AI-capex repricing
(U), 2025-08-01 payrolls miss + tariff round (E/U), 2023-03-09 SVB (U), 2024-09-03 tech-led
selloff (U). Classify the remaining three from the range table when A.1 is regenerated.

## History: two defects fixed by the 2026-08-05 revision

### 1. 2026-01-19 was Martin Luther King Jr. Day — NYSE closed (now removed)

It could not be a volatile day: there was no equity session. The first extraction paid to learn
this — `median SPY=nan ES=6913.75`, because **CME Globex runs an abbreviated holiday session**,
so the futures leg looked healthy and only the equity leg was missing. `validate_sample.py` now
rejects closed days before anything is extracted; the day is out of the default.

### 2. Pair 8 broke the same-weekday rule (now repaired)

The old pair was 2025-01-07 (Tue) vs 2024-01-29 (Mon) — the only weekday mismatch in the table,
and one a referee checking the calendar would read as an error. The revision keeps 2024-01-29 and
pairs it with **2025-01-27 (Mon, 364 d)** — the DeepSeek selloff, a larger-range day than the one
it replaces. 2025-01-13, the MLK pair's orphaned baseline, is also out; the new tenth pair is
2025-08-01 / 2024-08-09.

## What has to change in the paper

* **Appendix Table A.1** — replaced wholesale: the new dates, their event classification, and the
  intraday range that put each volatile day in the top ten.
* **Sample-period statements** — every "2014–2017" in the text, abstract and table notes.
* **The selection window** — state the period the top-ten ranking was taken over (2022–2026),
  since "largest intraday range" is meaningless without it.
* **The baseline rule** — restate it for the new dates. The table above is now rule-consistent
  throughout, so no exception footnote is needed.
* **Data-availability footnote** — the March-2020 MWCB days are ~6 years before the rest of the
  sample. That gap deserves a sentence: the MWCB panel is a different market-structure era from
  the 2022–2026 panel, and any comparison across them carries that.
* **Roll-window sessions** — six of 24 sessions sit inside an ES roll window (2020-03-12 +0,
  2020-03-16 +4, 2020-03-18 +6, 2023-03-09 +0, 2024-12-18 +6, 2025-06-13 +1). The 2020 shares
  are measured (72.8/60.4/78.0%); the 2022–26 ones are measured automatically at extraction since
  v0.9.54 — report the measured shares in the sample appendix.

## Checking a universe before spending hours on it

```bash
python validate_sample.py \
  --volatile 2024-12-18,2026-06-05,2025-10-10,2024-09-03,2025-04-03,2024-08-05,2024-07-24,2025-01-27,2023-03-09,2025-08-01 \
  --baseline 2023-12-20,2025-06-13,2024-10-18,2023-09-05,2024-04-04,2023-08-07,2023-07-19,2024-01-29,2022-03-24,2024-08-09 \
  --mwcb 2020-03-09,2020-03-12,2020-03-16,2020-03-18
```

Errors (exit 1): weekends, exchange holidays, one-off closures (Sandy, the Bush/Ford/Carter
funerals), duplicates, future dates. Warnings: 13:00 ET half days — the data are real but a
09:30–16:00 grid is two-thirds empty, so the session is not comparable to a full one — and any
volatile↔baseline pair that breaks the matching rule. STAGE 0 of the replication driver runs this
automatically for `--source extract`; `--allow-bad-dates` overrides it deliberately.
