# The re-sampled date universe (2022–2026), and what the paper has to say about it

The published sample is Appendix Table A.1: ten volatile days drawn from 2014–2017, ten matched
baseline days, and the four March-2020 MWCB halt days. The stack now defaults to a **re-sampled**
universe on 2022–2026 so the results speak to the current market. The construction rule is
unchanged — only the window moved.

`./run_paper_replication.sh --paper-sample` restores the published universe verbatim, so the
2014–2017 numbers remain reproducible from the same driver.

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
| 8 | 2025-01-07 | Tue | 2024-01-29 | **Mon** | **344d** |
| 9 | 2023-03-09 | Thu | 2022-03-24 | Thu | 350d |
| 10 | **2026-01-19** | Mon | 2025-01-13 | Mon | 371d |

MWCB: 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18 — fixed by history, unchanged.

## Two things to fix before the next extraction

### 1. 2026-01-19 is Martin Luther King Jr. Day — NYSE closed

It cannot be a volatile day: there was no equity session. The first real run extracted it anyway
and the log shows exactly what came back — `median SPY=nan ES=6913.75`. SPY was empty because the
equity market was shut; **ES had data because CME Globex runs an abbreviated holiday session**, so
the futures leg looked healthy and only the equity leg was missing. A cross-asset study cannot use
a day with one leg, and the failure is quiet: a full-length 23,401-row frame of NaNs on one side.

Its baseline partner, 2025-01-13, is a normal Monday and is fine — it just needs a new volatile
day to match.

Replace it with the next-largest intraday-range day in the window that is a full trading session,
then pick its baseline as the same weekday roughly one year prior. `validate_sample.py` now checks
this before anything is extracted, so a closed day cannot cost three hours again.

### 2. Pair 8 breaks the matching rule

2025-01-07 is a Tuesday; 2024-01-29 is a Monday, 344 days earlier. Every other pair is a same-weekday
match at 350–371 days. Day-of-week matters here — Monday and Friday sessions have different
announcement and expiry structure — so this is either an oversight or a deliberate exception.

The rule-consistent match for 2025-01-07 (Tue) is **2024-01-09** (Tue, 364 days prior). If there is
a reason to keep 2024-01-29 instead, the appendix needs one sentence saying so; otherwise it reads
as an error to a referee who checks the calendar.

## What has to change in the paper

* **Appendix Table A.1** — replaced wholesale: the new dates, their event classification (E =
  scheduled macro announcement, U = unexpected), and the intraday range that put each volatile day
  in the top ten.
* **Sample-period statements** — every "2014–2017" in the text, abstract and table notes.
* **The selection window** — say what period the top-ten ranking was taken over (2022–2026?), since
  "largest intraday range" is meaningless without it.
* **The baseline rule** — restate it for the new dates, and note pair 8 if it stays as it is.
* **Data-availability footnote** — the March-2020 MWCB days are now ~6 years before the rest of the
  sample rather than ~4 after it. That gap deserves a sentence: the MWCB panel is a different
  market-structure era from the 2022–2026 panel, and any comparison across them carries that.
* **The 2020 ES leg** — currently unresolved (see `EXTRACTION_RUN_DIAGNOSIS.md` §2): ESH0/ESM0
  returned nothing for all four MWCB days. Until that is fixed the MWCB panel has no futures leg,
  which is fatal for a cross-asset claim, not cosmetic.

## Checking a universe before spending hours on it

```bash
python validate_sample.py \
  --volatile 2024-12-18,2026-06-05,... --baseline 2023-12-20,2025-06-13,... \
  --mwcb 2020-03-09,2020-03-12,2020-03-16,2020-03-18
```

Errors (exit 1): weekends, exchange holidays, one-off closures (Sandy, the Bush/Ford/Carter
funerals), duplicates, future dates. Warnings: 13:00 ET half days — the data are real but a
09:30–16:00 grid is two-thirds empty, so the session is not comparable to a full one — and any
volatile↔baseline pair that breaks the matching rule. STAGE 0 of the replication driver runs this
automatically for `--source extract`; `--allow-bad-dates` overrides it deliberately.
