#!/usr/bin/env python3
"""Generate the corrected paper-table set from run 20260817_015626.

Reads the analysis bundle directly and emits TABLES_20260817.{md,html} plus
per-table booktabs .tex, with the RUN_FINDINGS_20260817 corrections applied
in place (labels, units, regime taxonomy, inference flags)."""
import csv, io, os, re, html as H

B = "/tmp/claude-0/-home-user-Call-Reports/84163315-e416-5674-8f42-75a7b0ba8d9a/scratchpad/bundle"
OUT = "/home/user/Call_Reports/paper_tables_20260817"
os.makedirs(OUT + "/tex", exist_ok=True)

def rd(path):
    with open(os.path.join(B, path)) as f:
        return list(csv.reader(f))

def f2(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)

TABLES = []  # dicts: id,title,cols,rows,notes,align

def add(tid, title, cols, rows, notes, panel_breaks=None):
    TABLES.append(dict(id=tid, title=title, cols=cols, rows=rows, notes=notes,
                       breaks=panel_breaks or {}))

# ---------------------------------------------------------------- T1 sample/QC
qc = open(os.path.join(B, "qc/qc_frames_10ms.txt")).read().splitlines()
sess = []  # (date, regime, stale, ok, halt>0, ec>0, ssr100)
for ln in qc:
    m = re.match(r"(20\d{2}-\d{2}-\d{2})\s+(benchmark|volatile|mwcb)\s+(\d+)\s+(True|False)\s+(\d+)\s+(\d+)", ln)
    if not m:
        continue
    nums = re.findall(r"0\.\d{6}", ln)          # ES_ladder_mono then ES_ladder_stale
    stale = float(nums[-1]) if nums else float("nan")
    ssr = bool(re.search(r"\s100%\s", ln))
    sess.append((m.group(1), m.group(2), stale, m.group(4) == "True",
                 int(m.group(5)) > 0, int(m.group(6)) > 0, ssr))
assert len(sess) == 66, len(sess)
# the QC table folds MWCB into volatile (RUN_FINDINGS M-8); restore the 4-day breakout
MWCB_DAYS = {"2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"}
sess = [(d, "mwcb" if d in MWCB_DAYS else r, s, ok, h, e, ssr) for d, r, s, ok, h, e, ssr in sess]
regimes = {"benchmark": 0, "volatile": 0, "mwcb": 0}
fails = {"benchmark": 0, "volatile": 0, "mwcb": 0}
for d, r, s, ok, h, e, ssr in sess:
    regimes[r] += 1
    if not ok:
        fails[r] += 1
add("T1", "Sample composition and 10 ms estimability",
    ["Regime", "Sessions", "10 ms QC-fail (ES ladder stale)", "Share of regime"],
    [["Benchmark", regimes["benchmark"], fails["benchmark"], f"{fails['benchmark']/regimes['benchmark']:.0%}"],
     ["Volatile", regimes["volatile"], fails["volatile"], f"{fails['volatile']/regimes['volatile']:.0%}"],
     ["MWCB", regimes["mwcb"], fails["mwcb"], f"{fails['mwcb']/regimes['mwcb']:.0%}"],
     ["All", 66, sum(fails.values()), f"{sum(fails.values())/66:.0%}"]],
    ["QC-fail = ES top of book unchanged on more than 99% of 10 ms snapshots (qc_frames_10ms.txt ok=False); "
     "the run estimated on all 66 sessions against this gate (RUN_FINDINGS D-2). The staleness loads on the "
     "benchmark side of every regime contrast; the v0.9.87 measured bias curve finds attenuation under the "
     "carry-forward mechanism, so 10 ms volatile-minus-benchmark gaps are conservative (D-1). The memo tallies "
     "(6/19 and 11/14) were wrong; the correct split is 16 benchmark / 9 volatile / 0 MWCB (I.2). "
     "Full per-session roster in Appendix A1."])

# ---------------------------------------------------------------- T2 Table 5
t5 = rd("nulls/table5_corrected_null.csv")
rows = []
name = {"A baseline": "A — Baseline (benchmark)", "B volatile": "B — Volatile",
        "C MWCB": "C — MWCB", "C MWCB exSSR": "C — MWCB, ex-SSR member"}
for r in t5[1:5]:
    rows.append([name[r[0]], f2(r[3]), f2(r[6]), f2(r[7]), "(" + f2(r[8], 1) + ")", f2(r[9])])
add("T2", "Table 5 — Crossed-book corner concentration: observed/null ratios and panel log odds ratio",
    ["Panel", "PCMOF obs/null", "NCMOF obs/null", "log OR", "(Woolf z — invalid)", "Corner asymmetry"],
    rows,
    ["PCMOF/NCMOF: positively/negatively co-moving order-flow corners; ratios are observed corner mass over the "
     "independence null. The Woolf z column is retained only for the record and must not be quoted: it is a "
     "pooled-bar statistic whose ordering tracks panel size, not effect size (the largest log OR, MWCB 1.470, has "
     "nearly the smallest z), and on a planted serially dependent DGP it overstates the day-clustered z by roughly "
     "21x (RUN_FINDINGS M-2); the run-level analogue of the same disease deflates a pooled t of 386.9 to a "
     "day-clustered 5.56 (M-1, T15). Day-clustered inference (per-day count matrices, day bootstrap, within-pair "
     "sign-flip, MWCB jackknife) ships in v0.9.87 and attaches to the next run. Corner asymmetry approximately "
     "-0.02 in every panel: no SSR fingerprint at the pooled level (D-6). MWCB is broken out here; T4-T8 fold or "
     "break it out as marked (M-8)."])

# ---------------------------------------------------------------- T3 Table 7
t7raw = [r for r in rd("nulls/table7_frequency_matched_null.csv") if r and r[0]]
hdr = t7raw[-4]
rows = [[r[0], f2(r[1], 2), f2(r[2], 2), f2(r[3], 2), f2(r[4], 2), f2(r[5], 2)] for r in t7raw[-3:]]
add("T3", "Table 7 — NCMOF excess over the frequency-matched null, by aggregation",
    ["Aggregation", "Orders per bar (ETF)", "Orders per bar (FUT)", "NCMOF null %", "NCMOF observed %", "Observed / null"],
    rows,
    ["The published aggregation story reproduces on this sample: the negative-corner excess collapses from 33.5x at "
     "1 s bars to 1.32x at 10 ms to 0.97x in action time. No uncertainty ships with this exhibit in this run "
     "(RUN_FINDINGS F-6); day-clustered inference attaches with the v0.9.87 machinery on the next run."])

# ---------------------------------------------------------------- T4 tier1
rows = []
for grid, path in (("1 s", "1s/flow/flow_corr_tier1_1s_b60s.csv"),
                   ("10 ms", "10ms/flow/flow_corr_tier1_10ms_b60s.csv")):
    t = rd(path)
    for r in t[1:]:
        lbl = {"benchmark": "Benchmark", "volatile": "Volatile", "mwcb": "MWCB",
               "volatile - benchmark": "Volatile − benchmark"}[r[0]]
        rows.append([grid if r[0] == "benchmark" else "", lbl, f2(r[1]), f2(r[2]), f2(r[3]) if r[3] else "—",
                     r[5].rstrip(".0") or r[5], f2(r[7], 4) if r[7] else "—"])
add("T4", "Tandem flow coupling (Tier 1) — OFI-innovation correlation by regime and grid",
    ["Grid", "Regime", "Mean corr", "Mean z", "SE(z)", "Days", "Permutation p"],
    rows,
    ["The run's most robust substantive result: coupling orders monotonically with stress on both grids "
     "(benchmark < volatile < MWCB), and the volatile-benchmark difference is significant at both (p = 0.0008 at "
     "1 s, p = 0.0327 at 10 ms). MWCB broken out (4 days). Staleness biases the 10 ms benchmark cell downward "
     "(attenuation), making the reported gap conservative (T1 note; RUN_FINDINGS F-1, D-1)."],
    panel_breaks={4: "10 ms"})

# ---------------------------------------------------------------- T5 asymmetry
rows = []
for grid, path in (("1 s", "1s/flow/flow_corr_asymmetry_1s_.csv"),
                   ("10 ms", "10ms/flow/flow_corr_asymmetry_10ms_.csv")):
    t = rd(path)
    for r in t[1:]:
        lbl = {"benchmark": "Benchmark", "volatile": "Volatile", "mwcb": "MWCB", "all days": "All days"}[r[0]]
        rows.append([grid if r[0] == "benchmark" else "", lbl, f2(r[1]), f2(r[2]), f2(r[3], 4), f2(r[4], 4),
                     f2(r[5], 4), r[6].rstrip(".0")])
add("T5", "Directional asymmetry of flow coupling — down-move vs up-move bars",
    ["Grid", "Regime", "Down corr (med)", "Up corr (med)", "Mean dz (down−up)", "SE", "Sign-flip p", "Days"],
    rows,
    ["A 1 s result only. At 1 s the benchmark regime couples tighter on down-moves (+0.047, p = 0.0045) and the "
     "volatile regime reverses to tighter on up-moves (−0.090, p < 5e-5). At 10 ms every gap is an order of "
     "magnitude smaller and non-significant. The gap-of-gaps is untested in the shipped tables, and the SSR / "
     "direction-conditioning checks remain outstanding (RUN_FINDINGS F-2). Quote the reversal as an "
     "aggregation-dependent finding or not at all."],
    panel_breaks={4: "10 ms"})

# ---------------------------------------------------------------- T6 mediation
rows = []
for grid, path in (("1 s", "1s/flow/flow_corr_mediation_1s_b60s_p3.csv"),
                   ("10 ms", "10ms/flow/flow_corr_mediation_10ms_b60s_p3.csv")):
    t = rd(path)
    for i, r in enumerate(t[1:]):
        rows.append([grid if i == 0 else "", r[0], r[1], r[2], r[3]])
add("T6", "Tier 2 mediation — does tandem flow mediate the RV → return-correlation link?",
    ["Grid", "Regressor", "Eq A: Δz_flow", "Eq B: Δz_ret (total)", "Eq B: Δz_ret (direct)"],
    rows,
    ["Day-cluster bootstrap SEs in parentheses; */**/*** = 10/5/1%. The mediator link (Δz_flow → Δz_ret: 67.0 at "
     "1 s, 75.3 at 10 ms) is the strongest coefficient in the run and holds on both grids. The suppression "
     "decomposition is a 1 s result: there the indirect level CI [−41.9, −3.14] excludes zero; the share row "
     "(6.05 = 605%) is a ratio to a near-zero total and must not be quoted. At 10 ms the indirect CI includes "
     "zero. Bootstrap resolution is unequal across grids (499 vs 199 draws; RUN_FINDINGS F-3, D-7); the next-run "
     "config re-runs both at n_boot = 2000 with BCa intervals."],
    panel_breaks={8: "10 ms"})

# ---------------------------------------------------------------- T7 lead-lag
ll = rd("10ms/analysis/tables/lead_lag__lead_lag_test.csv")[1]
cj = rd("10ms/analysis/tables/jumps__cojump_lead_test.csv")[1]
add("T7", "Lead-lag at 10 ms — HRY timing estimate and co-jump ordering",
    ["Test", "Point estimate", "Days ES-leading", "Day-level sign-flip p"],
    [["HRY cross-correlation θ (ES leads by)", f"{float(ll[0])*1000:.1f} ms (median {float(ll[1])*1000:.1f} ms)",
      f"{ll[3]} / {ll[2]}", f2(ll[4], 4)],
     ["Co-jump lead share (ES first)", f2(cj[1]), f"{cj[3]} / {cj[2]}", "5.0e-05"]],
    ["Both tests are day-level (immune to the pooled-inference disease of M-1). ES leads SPY by about one 20 ms "
     "grid step. Integrity notes from RUN_FINDINGS F-4: (i) the co-jump ES-lead share correlates POSITIVELY with "
     "ES update counts (Spearman +0.57) — weakest on stale sessions, so not a staleness artifact; (ii) the two "
     "extreme raw-θ days (2017-06-26, 2017-10-05) are the two stalest sessions and should be excluded from θ "
     "magnitude claims; (iii) the pre-averaged θ column reverses sign on 62/66 days and ships untested — the "
     "raw-vs-pre-averaged contradiction is UNADJUDICATED on this record and the ES-lead claim should cite the "
     "co-jump test first. At 1 s neither test resolves (p = 0.52/0.96): a resolution artifact."])

# ---------------------------------------------------------------- T8 IS tests
def reg_row(grid, label, path, kind):
    r = rd(path)[1]
    if kind == "wp":
        return [grid, label, "—", "—", f2(r[3]), f2(r[4], 4)]
    return [grid, label, f2(r[1]), f2(r[2]), f2(r[3]), f2(r[4], 4)]
rows = [
 reg_row("1 s", "CS_ES, free permutation", "1s/analysis/tables/information_shares__regime_test.csv", "f"),
 reg_row("", "IS_mid_ES, free permutation", "1s/analysis/tables/information_shares__regime_test_IS.csv", "f"),
 reg_row("", "CS_ES, within-pair (era-robust)", "1s/analysis/tables/information_shares__regime_test_within_pair.csv", "wp"),
 reg_row("", "CS_ES, κ-weighted", "1s/analysis/tables/information_shares__regime_test_kappa_weighted.csv", "f"),
 reg_row("10 ms", "CS_ES, free permutation", "10ms/analysis/tables/information_shares__regime_test.csv", "f"),
 reg_row("", "IS_mid_ES, free permutation", "10ms/analysis/tables/information_shares__regime_test_IS.csv", "f"),
 reg_row("", "CS_ES, within-pair (era-robust)", "10ms/analysis/tables/information_shares__regime_test_within_pair.csv", "wp"),
 reg_row("", "CS_ES, κ-weighted", "10ms/analysis/tables/information_shares__regime_test_kappa_weighted.csv", "f"),
]
add("T8", "Information-share regime tests — volatile vs benchmark",
    ["Grid", "Test", "Volatile mean", "Benchmark mean", "Difference", "p"],
    rows,
    ["Within-pair rows report the mean pair difference over 25 pairs. The only sub-0.05 cell is the 1 s "
     "free-permutation IS test (p = 0.032) — and its era-robust within-pair counterpart DOES NOT EXIST in this "
     "run (it was run only on CS, where it reverses sign at 1 s, p = 0.976). The claim \"ES gains share in "
     "volatile regimes\" is therefore unsupported by an era-robust test on this record (RUN_FINDINGS F-5). "
     "v0.9.87 adds the within-pair and ex-straddle twins on IS_mid_ES; the next run carries the verdict. "
     "IS is the primary metric; CS is fragile to lags and beta (M-7). Volatile here folds MWCB in (M-8)."],
    panel_breaks={4: "10 ms"})

# ---------------------------------------------------------------- T9 Table 9
t9 = rd("1s/table9/table9_both_ways_informational_w100_p5.csv")
rows = []
for r in t9[3:]:
    rows.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[9], r[10]])
add("T9", "Table 9 — IRF of return-correlation change on liquidity/flow shocks (1 s), Pearson vs HY vs DCC",
    ["Shock", "Pearson bench", "Pearson vol", "HY bench", "HY vol", "DCC bench", "DCC vol",
     "Δ(HY−P) bench", "Δ(HY−P) vol"],
    rows,
    ["Impact orthogonalized responses of Δcorrelation, x100; panel VAR(5) with day fixed effects, within-day lags; "
     "day-cluster bootstrap SEs, Romano-Wolf joint stars (***/**/* = 1/5/10% FWER). The Δ column is the share of "
     "each published Pearson response attributable to Epps-type measurement artifact. Standing caveats "
     "(RUN_FINDINGS): the 10 ms leg of this table died at ~292.6 GiB and does not exist (I.1); no lag-band "
     "certification ran, so no cell is certified lag-robust (I.3); DCC is a boundary estimate (a+b = 0.9999, M-6) "
     "— read its near-zero responses accordingly; the RealBar column is omitted here pending the lag diagnostic "
     "(available in the bundle). The v0.9.87 local-projection twin (T9_LP=1) replaces lag-order-dependent IRFs "
     "next run (M-3)."])

# ---------------------------------------------------------------- T10 FEVD
add("T10", "Forecast-error variance of SPY returns attributed to ES order flow — diagonal FEVD vs generalized (GFEVD), by regime (1 s)",
    ["Decomposition", "Regime", "SPY ret ← OFI_SPY", "SPY ret ← OFI_ES"],
    [["FEVD (diagonal — invalid here)", "Benchmark", "0.721", "0.279"],
     ["", "Volatile", "0.495", "0.505"],
     ["GFEVD (Pesaran–Shin — quote this)", "Benchmark", "0.543", "0.457"],
     ["", "Volatile", "0.500", "0.500"]],
    ["LABELING CORRECTION applied: the shipped irf__fevd.csv / irf__gfevd.csv are the BENCHMARK-regime tables "
     "with no regime label; the volatile decompositions existed only inline in report.md (RUN_FINDINGS M-5, "
     "E-9). Measured OFI-innovation correlation is 0.705 (benchmark) / 0.788 (volatile), so the diagonal FEVD's "
     "orthogonality assumption fails and the GFEVD is the quotable object — with the caveat that under tandem "
     "flow, common flow is counted toward both shocks. Roughly 46-50% of SPY-return forecast-error variance "
     "attributes to ES flow."])

# ---------------------------------------------------------------- T11 copula
cr = rd("1s/copula/copula_regimes_1s_t15m.csv")
rows = []
for r in cr[1:]:
    lbl = {"benchmark": "Benchmark", "volatile": "Volatile", "mwcb": "MWCB", "all days": "All days"}[r[0]]
    rows.append([lbl, r[1], f2(r[2]), f2(r[3]), f2(r[4], 4), f2(r[6], 4), r[8]])
cl = rd("1s/copula/copula_liquidity_1s_t15m.csv")
rows2 = []
for r in cl[1:]:
    lbl = {"benchmark": "Benchmark", "volatile": "Volatile", "mwcb": "MWCB", "all days": "All days"}[r[0]]
    rows2.append([lbl, f2(r[1]), f2(r[2]), f2(r[3], 4), f2(r[4], 4), f2(r[5], 4), r[6].rstrip(".0")])
add("T11a", "Copula tail dependence by regime (1 s, 15 m windows)",
    ["Regime", "Best family (days)", "Median λ_L", "Median λ_U", "Tail asym (L−U)", "Sign-flip p", "Days"],
    rows,
    ["Parametric caveat (RUN_FINDINGS): the t family FORCES λ_L = λ_U, and t wins 19/25 benchmark days — the "
     "benchmark row's symmetry is partly assumed, not measured. The v0.9.87 nonparametric Schmidt-Stadtmüller "
     "estimator (λ at q = 0.95, no family constraint) attaches next run and adjudicates. The upper-tail LR "
     "column (median p = 0.0 in all groups) is omitted pending the E-13 column-definition fix."])
add("T11b", "Lower-tail dependence conditional on book state — thin vs deep liquidity (1 s)",
    ["Regime", "Median λ_L thin", "Median λ_L deep", "Δ (thin−deep)", "SE", "Sign-flip p", "Days"],
    rows2,
    ["In the volatile regime, tail dependence is significantly higher when the book is thin (+0.076, p = 5e-5) — "
     "the liquidity channel of tail coupling. Benchmark and MWCB do not resolve."])

# ---------------------------------------------------------------- T12 MS regimes
ms = rd("1s/flow/flow_corr_ms_regimes_1s_b60s.csv")
rows = []
for r in ms[1:]:
    lbl = {"benchmark": "Benchmark", "volatile": "Volatile", "mwcb": "MWCB", "all days": "All days"}[r[0]]
    rows.append([lbl, r[1], r[2], f2(r[3], 2), r[4], f2(r[5], 4), f2(r[6], 2), f2(r[7], 3), r[8]])
add("T12", "Markov-switching flow-coupling regimes within the day (1 s)",
    ["Regime", "Days by chosen K", "Corr levels (modal K)", "Top-regime share", "Median duration lo/hi (bars)",
     "Duration asym p", "Entry ret (bps)", "Entry-direction p", "Multi-regime days"],
    rows,
    ["64 of 66 days reject a single flow-coupling regime (median LR p = 0.01 — a 99-draw resolution FLOOR, not an "
     "estimate; RUN_FINDINGS D-7). Benchmark days spend long spells in the low-coupling state punctuated by "
     "short high-coupling bursts (156 vs 36 bars, asymmetry p = 0.0007); volatile days are more evenly split. "
     "Regime entries lean toward down-moves overall (entry-direction p = 0.020, all days)."])

# ---------------------------------------------------------------- T13 horizon
hz = rd("horizon/horizon_profile_summary_10ms_6rungs.csv")
rows = []
for r in hz[1:]:
    dt = float(r[1])
    hl_bars = float(r[6])
    rows.append([r[0], f2(r[2]), f2(r[3]), f2(r[4]), f2(r[5]), f"{hl_bars:.1f}", f"{hl_bars*dt:.2f}",
                 f2(r[7], 3)])
add("T13", "Horizon profile — cross-asset transmission across sampling rungs (66 days)",
    ["Interval", "λ SPY←ES", "λ ES←SPY", "IS_mid_ES", "CS_ES", "Half-life (bars)", "Half-life (s, corrected)",
     "ES lead share"],
    rows,
    ["UNIT CORRECTION applied: the shipped half_life_s column is in BAR units; true seconds = bars x bar length "
     "(RUN_FINDINGS E-3 — a x100 label error at 10 ms). Corrected, the error-correction half-life runs 0.05 s at "
     "10 ms to 64 s at 1 s bars. The asymmetric cross-impact (λ SPY←ES an order of magnitude above λ ES←SPY at "
     "every rung) and the ES lead share visible only below 100 ms are the profile's substantive content."])

# ---------------------------------------------------------------- T14 ECM-SDE
def med_hl(path, scale):
    t = rd(path)
    i = t[0].index("half_life_s")
    v = sorted(float(r[i]) for r in t[1:] if r[i])
    return v[len(v)//2] * scale
e10 = rd("10ms/analysis/tables/ecm_sde.csv")[1]
e1 = rd("1s/analysis/tables/ecm_sde.csv")[1]
add("T14", "ECM-SDE error-correction estimates (Kalman, curve-implied)",
    ["Grid", "a1 SPY (t)", "a1 ES (t)", "Median IS_ES (mid)", "Median IS_ES (microprice)",
     "Median half-life (s, corrected)"],
    [["1 s", f"{f2(e1[0],5)} ({f2(e1[1],2)})", f"{f2(e1[2],5)} ({f2(e1[3],2)})", f2(e1[6]), f2(e1[9]),
      f"{med_hl('1s/analysis/tables/ecm_sde__curve.csv', 1.0):.1f}"],
     ["10 ms", f"{f2(e10[0],5)} ({f2(e10[1],2)})", f"{f2(e10[2],5)} ({f2(e10[3],2)})", f2(e10[6]), f2(e10[9]),
      f"{med_hl('10ms/analysis/tables/ecm_sde__curve.csv', 0.01):.2f}"]],
    ["Signs are the textbook error-correction pattern (SPY adjusts up toward, ES down toward, the common curve). "
     "Half-life carries the same bar-unit correction as T13 (x0.01 at 10 ms; RUN_FINDINGS E-3). Microprice-based "
     "IS is above mid-based on both grids. t-statistics are as shipped (per-day Kalman aggregation)."])

# ---------------------------------------------------------------- T15 inference
ic = rd("1s/analysis/tables/legacy__inference_iid_vs_clustered.csv")[1]
ci = rd("10ms/analysis/tables/cross_impact__regime_summary.csv")
add("T15", "Inference exhibit — pooled-iid vs day-clustered t on the same regression (1 s)",
    ["Regression", "Coefficient", "Pooled-iid t", "Day-clustered t", "Wild-bootstrap p", "N obs", "N days"],
    [["SPY_ret ~ ES_OFI", "1.05e-08", f2(ic[2], 1), f2(ic[3], 2), f2(ic[4], 3), ic[5], ic[6]]],
    ["The 69.5x deflation that governs the reading of every pooled statistic in the run (RUN_FINDINGS M-1), and "
     "the reason T2's Woolf z column is struck. All starred results quoted in this document use day-level or "
     "day-clustered inference."])
rows = [[{"benchmark": "Benchmark", "volatile": "Volatile (incl. MWCB)"}[r[0]], r[1], f2(r[2], 4), f2(r[3], 4), r[4]]
        for r in ci[1:]]
add("T16", "Cross-impact regime summary (10 ms)",
    ["Regime", "Type", "Mean", "SD", "Day-windows"],
    rows,
    ["Cross-impact triples from benchmark (0.012) to volatile (0.035) while own-impact barely moves. Taxonomy "
     "note: this family folds MWCB into volatile (41 days; M-8)."])

# ---------------------------------------------------------------- A1 roster
special = {"2017-06-26": "ladder-monotonicity defect (8 snaps); session0 exemplar",
           "2020-12-11": "activity rule extracted ESZ0 (75.8%); QC cols score calendar front (D-5)",
           "2021-11-26": "early close 13:00; post-close rows masked",
           "2020-03-16": "SSR 100%", "2022-02-10": "SSR 100%"}
rows = []
for d, r, s, ok, h, e, ssr in sess:
    flags = []
    if h: flags.append("MWCB halt rows masked")
    if ssr and d not in special: flags.append("SSR 100%")
    if d in special: flags.append(special[d])
    rows.append([d, {"benchmark": "bench", "volatile": "vol", "mwcb": "MWCB"}[r],
                 f"{s:.4f}", "fail" if not ok else "ok", "; ".join(flags)])
add("A1", "Appendix — per-session 10 ms QC roster",
    ["Date", "Regime", "ES top-of-book stale fraction", "10 ms QC", "Notes"],
    rows,
    ["QC fail threshold: ES top of book unchanged on >99% of snapshots. The 1 s frames pass QC on all 66 "
     "sessions; every 10 ms exhibit in this document estimated on all 66 (D-2) — the clean-subsample and "
     "dose-response (es_stale_frac) checks attach next run."])

# ================================================================ renderers
def render_md():
    L = ["# Paper Tables — Run 20260817_015626 (corrected per RUN_FINDINGS_20260817)", ""]
    L.append("**Run:** `replication_20260817_015626` · **Stack:** v0.9.86 · **Sample:** 66 sessions "
             "(25 benchmark / 37 volatile / 4 MWCB), 2017-06-26 – 2026-06-05 · **Grids:** 1 s and 10 ms · "
             "**Prepared:** 2026-08-22, from the shipped analysis bundle, with the corrections of "
             "RUN_FINDINGS_20260817 applied in place.")
    L.append("")
    L.append("**Corrections applied** (details in each table's notes): "
             "(1) Table 5 Woolf z struck as invalid (M-1/M-2); "
             "(2) FEVD/GFEVD tables regime-labeled (M-5/E-9); "
             "(3) half-life bar-unit fix, x0.01 at 10 ms (E-3); "
             "(4) stale-session tally corrected to 16 benchmark / 9 volatile (I.2); "
             "(5) 1 s-only results (asymmetry reversal, suppression mediation) flagged as such (F-2/F-3); "
             "(6) mediation share statistic suppressed (near-zero denominator, F-3); "
             "(7) MWCB taxonomy marked per table (M-8); "
             "(8) copula upper-tail LR column omitted pending E-13; "
             "(9) Rigobon column excluded — unidentified by its own diagnostic (M-4); "
             "(10) pre-averaged θ contradiction flagged as unadjudicated (F-4/E-2).")
    L.append("")
    for t in TABLES:
        L.append(f"## {t['id']}. {t['title']}")
        L.append("")
        L.append("| " + " | ".join(t["cols"]) + " |")
        L.append("|" + "|".join([" --- "] * len(t["cols"])) + "|")
        for r in t["rows"]:
            L.append("| " + " | ".join(str(c) for c in r) + " |")
        L.append("")
        for n in t["notes"]:
            L.append("*" + n + "*")
            L.append("")
    return "\n".join(L)

CSS = """html { -webkit-print-color-adjust: exact; }
body { font-family: "XCharter","Charter","Liberation Serif","Times New Roman",serif;
  font-size: 10pt; line-height: 1.4; color: #000; margin: 0; }
.page { max-width: 7.3in; margin: 0 auto; padding: 0.6in 0.3in; }
h1 { font-size: 15pt; line-height: 1.25; margin: 0 0 6pt 0; }
h2 { font-size: 11pt; margin: 18pt 0 5pt 0; page-break-after: avoid; }
p { margin: 0 0 6pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 6pt 0; font-size: 8.6pt; }
th { border-top: 1.2pt solid #000; border-bottom: 0.6pt solid #000; padding: 2.5pt 4pt;
     text-align: left; font-weight: 700; }
td { padding: 2pt 4pt; text-align: left; }
tr.last td { border-bottom: 1.2pt solid #000; }
tr.pbreak td { border-top: 0.6pt solid #666; }
p.note { font-size: 8.6pt; font-style: italic; margin: 0 0 4pt 0; }
code { font-family: "Liberation Mono",monospace; font-size: 8.5pt; }
@media print { .page { padding: 0.1in 0; } h2 { page-break-after: avoid; } table { page-break-inside: avoid; } }
.a1 table { page-break-inside: auto; }
"""

def render_html():
    L = ['<!doctype html>', '<html lang="en-US">', '<head>', '<meta charset="utf-8">',
         '<title>Paper Tables — Run of August 17, 2026 (Corrected)</title>',
         '<style>', CSS, '</style>', '</head>', '<body>', '<div class="page">']
    L.append('<h1>Paper Tables — Full-Sample Two-Grid Run of August 17, 2026<br>'
             '<span style="font-size:11pt;font-weight:400">Corrected per the Findings of Record '
             '(RUN_FINDINGS_20260817)</span></h1>')
    L.append('<p><strong>Run:</strong> <code>replication_20260817_015626</code> · <strong>Stack:</strong> '
             'v0.9.86 · <strong>Sample:</strong> 66 sessions (25 benchmark / 37 volatile / 4 MWCB), '
             '2017-06-26 – 2026-06-05 · <strong>Grids:</strong> 1 s and 10 ms · <strong>Prepared:</strong> '
             'August 22, 2026, from the shipped analysis bundle, with the corrections of the Findings of '
             'Record applied in place.</p>')
    L.append('<p><strong>Corrections applied</strong> (each table\'s notes give details): (1) Table 5 Woolf '
             'z struck as invalid (M-1/M-2); (2) FEVD/GFEVD tables regime-labeled (M-5/E-9); (3) half-life '
             'bar-unit fix, ×0.01 at 10 ms (E-3); (4) stale-session tally corrected to 16 benchmark / 9 '
             'volatile (I.2); (5) 1 s-only results flagged as such (F-2/F-3); (6) mediation share statistic '
             'suppressed (F-3); (7) MWCB taxonomy marked per table (M-8); (8) copula upper-tail LR column '
             'omitted pending E-13; (9) Rigobon column excluded, unidentified by its own diagnostic (M-4); '
             '(10) pre-averaged θ contradiction flagged unadjudicated (F-4/E-2).</p>')
    for t in TABLES:
        cls = ' class="a1"' if t["id"] == "A1" else ""
        L.append(f'<div{cls}>')
        L.append(f'<h2>{H.escape(t["id"])}. {H.escape(t["title"])}</h2>')
        L.append('<table>')
        L.append('<tr>' + ''.join(f'<th>{H.escape(c)}</th>' for c in t["cols"]) + '</tr>')
        for i, r in enumerate(t["rows"]):
            klass = []
            if i == len(t["rows"]) - 1: klass.append("last")
            if i in t["breaks"]: klass.append("pbreak")
            k = f' class="{" ".join(klass)}"' if klass else ""
            L.append(f'<tr{k}>' + ''.join(f'<td>{H.escape(str(c))}</td>' for c in r) + '</tr>')
        L.append('</table>')
        for n in t["notes"]:
            L.append(f'<p class="note">{H.escape(n)}</p>')
        L.append('</div>')
    L += ['</div>', '</body>', '</html>']
    return "\n".join(L)

def tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("θ", r"$\theta$"), ("λ", r"$\lambda$"), ("κ", r"$\kappa$"),
                 ("Δ", r"$\Delta$"), ("−", "--"), ("←", r"$\leftarrow$"), ("→", r"$\rightarrow$"),
                 ("≈", r"$\approx$"), ("×", r"$\times$"), ("·", r"$\cdot$")]:
        s = s.replace(a, b)
    s = re.sub(r"\*\*\*", r"\\sym{***}", s)
    s = re.sub(r"\*\*", r"\\sym{**}", s)
    return s

def render_tex(t):
    L = [f"% {t['id']} — generated from replication_20260817_015626, corrections per RUN_FINDINGS_20260817",
         r"\begin{table}[htbp]\centering",
         rf"\caption{{{tex_escape(t['title'])}}}",
         rf"\label{{tab:{t['id'].lower()}}}",
         r"\small",
         r"\begin{tabular}{" + "l" * len(t["cols"]) + "}",
         r"\toprule",
         " & ".join(tex_escape(c) for c in t["cols"]) + r" \\",
         r"\midrule"]
    for i, r in enumerate(t["rows"]):
        if i in t["breaks"]:
            L.append(r"\midrule")
        L.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\vspace{2pt}\footnotesize\emph{Notes:} "
          + " ".join(tex_escape(n) for n in t["notes"]) + r"\end{minipage}",
          r"\end{table}", ""]
    return "\n".join(L)

with open(f"{OUT}/TABLES_20260817.md", "w") as f:
    f.write(render_md())
with open(f"{OUT}/TABLES_20260817.html", "w") as f:
    f.write(render_html())
for t in TABLES:
    with open(f"{OUT}/tex/{t['id'].lower()}.tex", "w") as f:
        f.write(render_tex(t))
print("tables:", len(TABLES), "->", OUT)
for t in TABLES:
    print(" ", t["id"], "-", t["title"][:70], f"({len(t['rows'])} rows)")
