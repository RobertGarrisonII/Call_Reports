"""
paper_tables.py
==============
The reporting layer. It turns every calculation in the stack into publication tables, in
two families:

  * REPRODUCTION  -- faithful versions of the working paper's tables (Garrison, Jain &
    Paddrik 2026): Table 5 (the order-imbalance contingency + PCMOF/NCMOF), Table 9 / 11 /
    13 (the correlation-IRF SVAR), and the Section 6 March-2020 reopening regression.
  * REVAMP        -- the same objects rebuilt with the methodological upgrades: spreads and
    a continuous cross-flow chi in place of categorical PCMOF/NCMOF, day-cluster bootstrap
    + Romano-Wolf stars on the IRF, the spread-conditioned information-share curve as the
    structural replacement for Table 9, the MWCB difference-in-differences / triple-diff
    that replaces the OLS, Rigobon order-free identification next to the Cholesky orderings,
    the Markov-switching VECM regime table, the bivariate-Hawkes contagion table, the
    Hasbrouck sigma_s + dollar-welfare table, and the tick-size-corrected information share.

Each builder returns a `Table` (a string-cell DataFrame + title + notes) that renders to
console, GitHub-Markdown, or booktabs LaTeX with no third-party dependency. `build_all_tables`
runs them on a session list and `write_report` emits a combined .md and .tex. Run the module
to regenerate both reports from synthetic data.
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import tandem_order_flow as tof
import correlation_svar as csv
import cross_flow as cf
import stress_index as si
import mwcb_event_study as mw
import rigobon_id as rg
import hawkes_cross as hk
import markov_switching_vecm as ms
import pricing_error as pe
import tick_correction as tk
import inference as inf
import market_shocks as ms_shocks

EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
#  Rendering core
# ══════════════════════════════════════════════════════════════════════════════
class Table:
    """A rendered table: a string-cell DataFrame plus a title and notes."""
    def __init__(self, title, df, notes="", numeric=None):
        self.title = title
        self.df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        self.notes = notes
        self.numeric = numeric

    def to_string(self):
        s = f"{self.title}\n" + "-" * len(self.title) + "\n" + self.df.to_string()
        if self.notes:
            s += "\nNote: " + self.notes
        return s

    def to_markdown(self):
        out = [f"### {self.title}", "", _df_to_markdown(self.df)]
        if self.notes:
            out += ["", f"*{self.notes}*"]
        return "\n".join(out)

    def to_latex(self, label=None):
        return _df_to_latex(self.df, self.title, self.notes, label)


def _stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_cell(coef, se=None, p=None, prec=3):
    """Format a regression cell as 'coef[stars] (se)'."""
    if coef is None or (isinstance(coef, float) and not np.isfinite(coef)):
        return "--"
    s = f"{coef:.{prec}f}{_stars(p)}"
    if se is not None and np.isfinite(se):
        s += f" ({se:.{prec}f})"
    return s


def _fmt_df(num, prec=3):
    """Format a numeric DataFrame to string cells."""
    return num.map(lambda v: ("--" if v is None or (isinstance(v, float) and not np.isfinite(v))
                              else (f"{v:.{prec}f}" if isinstance(v, (int, float, np.floating)) else str(v))))


def _flat_cols(df):
    """Render a MultiIndex column header as 'outer / inner' rather than a Python tuple.
    Paired tables (e.g. Pearson vs Hayashi-Yoshida) carry a two-level header, and the plain
    str() of a tuple is not publication output."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" / ".join(str(x) for x in c if str(x) != "") for c in df.columns]
    return df


def _df_to_markdown(df):
    df = _flat_cols(df).astype(str)
    idx_name = df.index.name or ""
    header = "| " + " | ".join([idx_name] + [str(c) for c in df.columns]) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = ["| " + " | ".join([str(ix)] + [df.loc[ix, c] for c in df.columns]) + " |" for ix in df.index]
    return "\n".join([header, sep] + rows)


def _latex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def _df_to_latex(df, title, notes, label):
    df = _flat_cols(df).astype(str)
    ncol = len(df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + _latex_escape(title) + "}"]
    if label:
        lines.append(r"\label{" + (label if str(label).startswith("tab:") else "tab:" + str(label)) + "}")
    lines.append(r"\begin{tabular}{l" + "r" * ncol + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join([_latex_escape(df.index.name or "")] + [_latex_escape(c) for c in df.columns]) + r" \\")
    lines.append(r"\midrule")
    for ix in df.index:
        lines.append(" & ".join([_latex_escape(ix)] + [_latex_escape(df.loc[ix, c]) for c in df.columns]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\vspace{2pt}\footnotesize " + _latex_escape(notes))
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Builders -- reproduction
# ══════════════════════════════════════════════════════════════════════════════
def table_imbalance_contingency(sessions, counts_fn, regime=None, corr_window=80):
    """Table 5: 3x3 order-imbalance contingency (futures x ETF), with the PCMOF/NCMOF
    same-/opposite-direction masses and conditional d-correlation (x1000) in the notes."""
    res = tof.table5_from_sessions(sessions, counts_fn, corr_window=corr_window)
    if not res:
        return Table("Table 5 (repro): order-imbalance contingency", pd.DataFrame(), "no data")
    reg = regime if regime in res else list(res)[0]
    freq = res[reg]["frequency"]
    disp = _fmt_df(freq.astype(float), 2)
    disp.index.name = "Futures \\ ETF"
    p, n = float(res[reg]["pcmof_ncmof"]["PCMOF"]), float(res[reg]["pcmof_ncmof"]["NCMOF"])
    note = f"Regime '{reg}'. Cells are % of bars. PCMOF (same-direction) = {p:.2f}%, NCMOF (opposite) = {n:.2f}%."
    if "dcorr_x1000" in res[reg]:
        dc = res[reg]["dcorr_x1000"]
        try:
            note += (f" Mean d-corr x1000: PCMOF cells = "
                     f"{(dc.loc['Buy','Buy'] + dc.loc['Sell','Sell']) / 2:.2f}, "
                     f"NCMOF cells = {(dc.loc['Buy','Sell'] + dc.loc['Sell','Buy']) / 2:.2f}.")
        except Exception:
            pass
    return Table(f"Table 5 (repro): order-imbalance contingency [{reg}]", disp, note, numeric=freq)


def table_correlation_irf(sessions, spec="informational", ident="cholesky", cumulative=False,
                          extra_fn=None, n_boot=0, n_lags=4, horizon=6, corr_window=80,
                          min_obs=200, title="Table 9 (repro): IRF of return correlation",
                          seed=0, n_jobs=None):
    """Table 9/11/13: impact (or cumulative) response of the return correlation to a 1-SD
    shock to each regressor, by regime, x100. With n_boot>0 the bootstrap SEs and Romano-Wolf
    joint stars come straight from csv.correlation_irf_inference (FWER over all cells).
    ``n_jobs`` sets the bootstrap worker count (None = all cores; threading)."""
    base = ("Impact" if not cumulative else f"Cumulative ({horizon}-step)") + \
           f" orthogonalized response of d-correlation, x100; identification = {ident}; VAR({n_lags})."
    if n_boot and n_boot > 0:
        res = csv.correlation_irf_inference(sessions, spec=spec, n_lags=n_lags, horizon=horizon,
                                            ident=ident, cumulative=cumulative, corr_window=corr_window,
                                            min_obs=min_obs, extra_fn=extra_fn, n_boot=n_boot,
                                            alpha=0.10, seed=seed, n_jobs=n_jobs)
        if res["point"].empty:
            return Table(title, pd.DataFrame(), "no regime had enough observations")
        disp = res["table"].copy(); disp.index.name = "shock"
        note = base + " Day-cluster bootstrap SE in parentheses; Romano-Wolf joint stars (***/**/* = 1/5/10% FWER)."
        return Table(title, disp, note, numeric=res["point"])
    point = csv.correlation_irf(sessions, spec=spec, n_lags=n_lags, horizon=horizon, ident=ident,
                                cumulative=cumulative, corr_window=corr_window, min_obs=min_obs, extra_fn=extra_fn)
    if point.empty:
        return Table(title, pd.DataFrame(), "no regime had enough observations")
    disp = _fmt_df(point, 3); disp.index.name = "shock"
    return Table(title, disp, base, numeric=point)


def table_correlation_irf_both_ways(sessions, spec="informational", ident="cholesky",
                                    cumulative=False, extra_fn=None, n_boot=499, n_lags=6,
                                    horizon=6, corr_window=100, min_obs=200, seed=0, n_jobs=None,
                                    criterion=None, pmax=12, with_dcc=False, with_bar=True,
                                    bar_seconds=60, panel="fe", mean_group=False,
                                    methods=(("Pearson", "rolling"), ("HY", "hy")),
                                    title="Table 9 (repro): IRF of return correlation, Pearson vs Hayashi-Yoshida"):
    """Table 9 estimated twice -- once on the paper's grid-sampled Pearson d-correlation and once
    on the Epps-robust Hayashi-Yoshida d-correlation over the same window -- reported side by side
    with the difference.

    The pairing is a specification test, not a robustness formality. Grid Pearson is attenuated by
    asynchronous quote updating; the attenuation is a function of trading intensity; and trading
    intensity (volume, message traffic, liquidity demand) sits on the right-hand side of Eq. (5).
    A response can therefore be measurement error moving with its own regressor, and the naive
    estimator cannot tell that apart from a market mechanism. Hayashi-Yoshida imposes no grid, so:

        coefficient stable across both columns   -> mechanism, safe to interpret
        coefficient shrinks toward zero under HY -> partly the Epps channel
        coefficient only significant under HY    -> the naive estimator was masking it

    The `Delta` block is (HY - Pearson): an estimate, per cell, of how much of the published
    response is attributable to the measurement artifact. Its magnitude relative to the Pearson
    column is the number to quote.

    Both columns carry day-cluster bootstrap SEs and Romano-Wolf FWER stars (from
    csv.correlation_irf_inference), so significance is comparable across the pair."""
    # Resolve the lag ONCE, here, so BOTH estimator blocks are fitted at the same order --
    # otherwise a Pearson/HY difference could be a lag difference, which is the one thing this
    # table exists to rule out.
    if with_dcc and not any(m == "dcc" for _l, m in methods):
        # A THIRD dependent variable, and the only one of the three without a fixed-width window.
        # Pearson and HY differ in how they handle asynchronicity but both difference a W-bar box,
        # so neither removes the lag artifact above -- DCC is what the caution in the note points at.
        methods = tuple(methods) + (("DCC", "dcc"),)
    if with_bar and not any(m == "bar" for _l, m in methods):
        # The FOURTH dependent variable, and the recommended one (v0.9.63): the change in the
        # Fisher-z of each bar's NON-OVERLAPPING realized correlation. Bars share no data, so the
        # fixed-width-window MA artifact cannot exist by construction -- measured, not filtered
        # (DCC's advantage without the GARCH model dependence).
        methods = tuple(methods) + (("RealBar", "bar"),)
    lag_note = ""
    if criterion is not None:
        # panel/bar_seconds forwarded (v0.9.64): the criterion must score the design that is
        # actually fitted -- selecting under stacked seam-crossing lags and then estimating
        # under within-day FE lags answers a different question at a different order.
        p_sel = csv.select_svar_lag(sessions, spec=spec, corr_window=corr_window,
                                    extra_fn=extra_fn, criterion=criterion, pmax=pmax,
                                    bar_seconds=bar_seconds, panel=panel)[0]
        if p_sel is not None:
            lag_note = f" Lag order p={int(p_sel)} chosen by {str(criterion).upper()} over p<={pmax} on the pooled SVAR frame."
            # A criterion CAN return 0 -- it does so on a wide rolling window, where the induced
            # spike no longer covers the cost of the lags before it. A VAR(0) has no dynamics and
            # no impulse response to compute, and the IRF code raises on the empty coefficient
            # list rather than saying so. Floor at 1 and put the fact in the note: p*=0 is
            # information (the criterion found nothing), not a value to fit.
            n_lags = max(1, int(p_sel))
            if int(p_sel) == 0:
                lag_note += (" The criterion selected p=0 -- no dynamics at all -- which cannot be "
                             "fitted as an IRF; p=1 is used and the response should be read as "
                             "impact-only. A p=0 selection on a wide correlation window usually "
                             "means the criterion could not see the window's own MA spike, not "
                             "that the system is dynamically empty.")
            # The lag order is a researcher degree of freedom that ends up in this note, so the note
            # is where a reader has to be told when it is not a finding. d(W-bar rolling correlation)
            # carries an MA term at exactly lag W; on data with a CONSTANT correlation the criterion
            # returns p*=W every time (test_svar_lag_artifact.py). A table whose lag equals its own
            # window is reporting its estimator, and this says so in the caption rather than in a log
            # nobody keeps.
            _d = csv.lag_diagnosis(p_sel, corr_window=corr_window, pmax=pmax, corr_method="rolling")
            if not _d["ok"]:
                lag_note += " CAUTION: " + _d["text"]
    kw = dict(spec=spec, n_lags=n_lags, horizon=horizon, ident=ident, cumulative=cumulative,
              corr_window=corr_window, min_obs=min_obs, extra_fn=extra_fn, seed=seed, n_jobs=n_jobs,
              bar_seconds=bar_seconds, panel=panel)
    base = ("Impact" if not cumulative else f"Cumulative ({horizon}-step)") + \
           f" orthogonalized response of d-correlation, x100; identification = {ident}; VAR({n_lags})." + lag_note
    out, nums = {}, {}
    for label, method in methods:
        if n_boot and n_boot > 0:
            res = csv.correlation_irf_inference(sessions, corr_method=method, n_boot=n_boot,
                                                alpha=0.10, **kw)
            pt = res["point"]
            if pt.empty:
                continue
            nums[label] = pt
            disp = res["table"]
        else:
            pt = csv.correlation_irf(sessions, corr_method=method,
                                     **{k: v for k, v in kw.items() if k not in ("seed", "n_jobs")})
            if pt.empty:
                continue
            nums[label] = pt
            disp = _fmt_df(pt, 3)
        for c in disp.columns:
            out[(label, str(c))] = disp[c]
    if not nums:
        return Table(title, pd.DataFrame(), "no regime had enough observations")
    if "Pearson" in nums and "HY" in nums:                       # the artifact estimate
        d = (nums["HY"] - nums["Pearson"]).reindex(index=nums["Pearson"].index)
        for c in d.columns:
            out[("Delta (HY-Pearson)", str(c))] = _fmt_df(d, 3)[c]
    disp = pd.DataFrame(out)
    disp.columns = pd.MultiIndex.from_tuples(disp.columns, names=["estimator", "regime"])
    disp.index.name = "shock"
    note = (base + " Left block: the paper's grid-sampled Pearson d-correlation. Middle: the "
            "Hayashi-Yoshida d-correlation over the same window, which is consistent under "
            "asynchronous quote updating and therefore free of the Epps attenuation that moves "
            "with trading intensity -- itself a regressor here. Right: the difference, i.e. the "
            "share of each published response attributable to that measurement artifact.")
    if n_boot and n_boot > 0:
        note += " Day-cluster bootstrap SE in parentheses; Romano-Wolf joint stars (***/**/* = 1/5/10% FWER)."
    if with_bar:
        note += (" RealBar: change in Fisher-z of the %ds-bar NON-OVERLAPPING realized correlation "
                 "-- no fixed-width rolling window, so the window-length MA artifact (a spike at "
                 "exactly lag W) cannot arise. Differencing per-bar ESTIMATES still leaves an "
                 "MA(1) from estimation noise (adjacent differences share one noisy level), so a "
                 "criterion may spend one extra lag on it -- bounded and lag-1, unlike the W-lag "
                 "artifact. Per-bar realized variance replaces the rolling RV regressor in this "
                 "column for the same reason." % int(bar_seconds))
    if panel != "stack":
        note += (" Estimation is a fixed-effects panel VAR: lags built strictly within-day (no "
                 "overnight seam-crossing) with day fixed effects; --panel stack reproduces the "
                 "pre-v0.9.63 stacked estimation.")
    tbl = Table(title, disp, note, numeric=nums.get("Pearson"))
    if mean_group:
        mg = csv.correlation_irf_mean_group(sessions, spec=spec, n_lags=n_lags, horizon=horizon,
                                            ident=ident, cumulative=cumulative,
                                            corr_method="bar" if with_bar else "rolling",
                                            corr_window=corr_window, extra_fn=extra_fn,
                                            bar_seconds=bar_seconds)
        tbl.mean_group = mg
    return tbl


def table_mwcb_reopening_ols(treated_sessions, release_by_date, asset="SPY",
                             pre_min=8, post_min=8, bin_sec=30):
    """Section 6 (repro): the original reopening regression -- realized illiquidity/volatility
    on a post-reopen indicator, pooled across MWCB days, with the share of variation explained
    (the paper's '~1-2% of volatility' style claim)."""
    rows = []
    for date, df in treated_sessions:
        il = np.asarray(mw.illiquidity(df, asset, "cost_to_fill"), float)
        rel = pd.Timestamp(release_by_date[date])
        secs = (df.index - rel).total_seconds().to_numpy()
        keep = (secs >= -pre_min * 60) & (secs <= post_min * 60) & np.isfinite(il)
        rows.append(pd.DataFrame({"y": il[keep], "post": (secs[keep] > 0).astype(float)}))
    p = pd.concat(rows, ignore_index=True)
    X = np.column_stack([np.ones(len(p)), p["post"].to_numpy()])
    b, sse = inf.ols_naive_se(p["y"].to_numpy(), X)
    yhat = X @ b
    r2 = 1.0 - np.sum((p["y"].to_numpy() - yhat) ** 2) / np.sum((p["y"].to_numpy() - p["y"].mean()) ** 2)
    num = pd.DataFrame({"value": [p.loc[p.post == 0, "y"].mean(), p.loc[p.post == 1, "y"].mean(),
                                  b[1], sse[1], 100 * r2]},
                       index=["mean pre-reopen", "mean post-reopen", "post coefficient",
                              "  std error", "R-squared (%)"])
    note = (f"Pooled OLS of {asset} cost-to-fill (bps) on a post-reopen dummy across "
            f"{len(treated_sessions)} MWCB days. The post coefficient is the unadjusted reopening "
            f"effect; R-squared is the share of variation it explains (the paper's headline magnitude).")
    return Table("Section 6 (repro): reopening regression (OLS)", _fmt_df(num, 3), note, numeric=num)


# ══════════════════════════════════════════════════════════════════════════════
#  Builders -- revamp
# ══════════════════════════════════════════════════════════════════════════════
def table_crossflow_irf(sessions, **kw):
    """Revamped Table 9: spreads + a continuous cross-flow chi (PCMOF/NCMOF as its positive
    and negative parts) replace the categorical imbalance regressors, with bootstrap stars."""
    extra = lambda df: cf.cross_flow_features(df)[["PCMOF", "NCMOF"]]
    return table_correlation_irf(sessions, spec="weighted", extra_fn=extra,
                                 title="Table 9 (revamp): spreads + continuous cross-flow chi", **kw)


def table_spread_conditioned_is(sessions, spread_kind="weighted"):
    """The structural replacement for Table 9: the information share as a function of the
    relative-spread state S = log(spread_ES) - log(spread_SPY). A negative IS_ES gradient in
    S means ES leads precisely when its book is relatively tighter."""
    fit = csv.spread_conditioned_is(sessions, spread_kind=spread_kind)
    curve = fit.get("curve")
    if not isinstance(curve, pd.DataFrame) or curve.empty:
        return Table("Spread-conditioned information share IS(S)", pd.DataFrame(), "fit produced no curve")
    scol = "S" if "S" in curve.columns else curve.columns[0]
    iscols = [c for c in curve.columns if c.upper().startswith("IS") or c.upper().startswith("CS")] or list(curve.columns[1:])
    qs = np.quantile(curve[scol].to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.95])
    pick = [int(np.argmin(np.abs(curve[scol].to_numpy() - q))) for q in qs]
    sub = curve.iloc[pick][[scol] + iscols].reset_index(drop=True)
    sub.index = ["q05", "q25", "q50", "q75", "q95"]
    grad = ""
    a1 = fit.get("a1")
    if a1 is not None:
        try:
            grad = f" Loading gradient dα/dS = {float(np.ravel(a1)[0]):+.4f} (day-clustered)."
        except Exception:
            pass
    return Table("IS(S): information share vs relative-spread state (revamp)", _fmt_df(sub, 4),
                 "Information share read off the spread state S across its distribution." + grad, numeric=sub)


def table_mwcb_did(treated, control, release_by_date, pre_min=8, post_min=8):
    """Section 6 (revamp): the reopening-asymmetry causal design -- 2x2 + regression DiD on
    illiquidity, the RD-in-time jump at the reopen, and the triple-difference cross-market
    spillover (the contagion term)."""
    panel = mw.build_panel(treated, control, release_by_date, "SPY", pre_min=pre_min, post_min=post_min)
    d = mw.did(panel)
    rd = mw.rd_in_time(panel, bandwidth_min=min(5, post_min))
    sp = mw.cross_market_spillover(treated, control, release_by_date, source="ES", target="SPY",
                                   pre_min=pre_min, post_min=post_min)
    # The CRVE t below uses a NORMAL reference at G ~ 8 date clusters (4 of them treated) --
    # the exact few-cluster regime METHODS_VIGNETTE prescribes the wild cluster bootstrap for
    # (Cameron-Gelbach-Miller; Webb six-point weights at G <= 12). Same point estimates,
    # honest p-values. Ibragimov-Muller is the alternative read at these counts.
    def _wcb(frame, X_cols, j):
        try:
            y = frame["illiq"].to_numpy(float)
            X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in X_cols])
            r = inf.wild_cluster_bootstrap(y, X, frame["date"].to_numpy(), test_idx=j, n_boot=999, seed=0)
            return float(r["p"])
        except Exception:
            return np.nan
    p2 = panel.assign(tp=panel["treated"] * panel["post"])
    p_did = _wcb(p2, ["treated", "post", "tp"], 3)
    h = min(5, post_min) * 60.0
    rp = panel[(panel["treated"] == 1) & (panel["t"].abs() <= h)].copy()
    rp["tau"] = rp["t"] / 60.0
    rp["post_f"] = (rp["tau"] > 0).astype(float)
    rp["ptau"] = rp["post_f"] * rp["tau"]
    p_rd = _wcb(rp, ["post_f", "tau", "ptau"], 1)
    rows = OrderedDict()
    rows["DiD (treated x post)"] = (d["did_coef"], d["did_se"], d["did_t"], p_did)
    rows["RD jump at reopen"] = (rd["jump"], rd["jump_se"], rd["jump_t"], p_rd)
    rows["Spillover ES->SPY (post x treated)"] = (sp["spill_post_treated"]["coef"],
                                                  sp["spill_post_treated"]["se"],
                                                  sp["spill_post_treated"]["t"], np.nan)
    disp = pd.DataFrame({"estimate": [fmt_cell(c, s) for c, s, _t, _p in rows.values()],
                         "t-stat (CRVE)": [f"{t:.2f}" for _c, _s, t, _p in rows.values()],
                         "p (wild cluster)": ["--" if not np.isfinite(p) else f"{p:.3f}"
                                              for _c, _s, _t, p in rows.values()]},
                        index=list(rows))
    disp.index.name = "effect"
    note = ("SPY cost-to-fill (bps). DiD treated x post is the causal reopening effect; the RD jump "
            "is the discontinuity at the reopen; the spillover triple-difference is the extra ES->SPY "
            "illiquidity transmission post-reopen on MWCB days (liquidity contagion). The CRVE t uses "
            "a normal reference that over-rejects at few clusters; QUOTE the wild-cluster-bootstrap p "
            "(restricted WCR, Webb weights at G<=12, clusters = dates). The spillover design pools "
            "differenced panels, so its WCR p is not computed here -- read its CRVE t with the same "
            "few-cluster caution, or the Ibragimov-Muller route.")
    return Table("Section 6 (revamp): reopening DiD + cross-market spillover", disp, note)


def table_rigobon(sessions, n_lags=1):
    """Order-free identification of the contemporaneous cross-market matrix via
    heteroskedasticity (Rigobon), shown next to the two Cholesky orderings the paper assumes
    away. Regimes = the SESSION labels (ex-ante date classification) whenever the panel carries
    at least two; the pooled-volatility split is only a fallback, and is labeled as such."""
    R, lab_rows = [], []
    for s in sessions:
        df = s[-1]; reg = str(s[1]).lower() if len(s) == 3 else "all"
        ms_ = np.asarray(ca._mid(df, "SPY"), float); me = np.asarray(ca._mid(df, "ES"), float)
        r = np.column_stack([np.diff(np.log(ms_)) * 1e4, np.diff(np.log(me)) * 1e4])
        R.append(r); lab_rows.append(np.full(len(r), reg, dtype=object))
    Rall = np.vstack(R); labs = np.concatenate(lab_rows)
    fin = np.all(np.isfinite(Rall), axis=1)
    Rall, labs = Rall[fin], labs[fin]
    A, Sig, U = csv.var_ols(Rall, n_lags)
    labs = labs[n_lags:]                             # var_ols drops the first n_lags rows
    # Rigobon's regimes must be EXOGENOUS to the shocks -- splitting on the residuals' own
    # rolling volatility conditions on the endogenous variable and biases the identified matrix
    # (the module's own docs forbid it). The session labels are an ex-ante DATE classification,
    # so they are the valid instrument whenever the panel has two of them.
    uniq = sorted(set(labs))
    if len(uniq) >= 2:
        calm = next((u for u in ("benchmark", "calm", "baseline") if u in uniq), uniq[0])
        stress = next((u for u in ("volatile", "stress", "mwcb") if u in uniq and u != calm), uniq[-1])
        regimes, regime_src = labs, f"session labels ({calm} vs {stress}; exogenous, ex-ante)"
    else:
        vol = pd.Series(np.sqrt((U ** 2).sum(1))).rolling(50, min_periods=1).mean().to_numpy()
        regimes = np.where(vol >= np.quantile(vol, 0.66), "stress", "calm")
        calm, stress = "calm", "stress"
        regime_src = ("ENDOGENOUS pooled-volatility split (single-label panel fallback) -- the "
                      "regimes condition on the residuals themselves; treat as descriptive only")
    out = rg.rigobon_identify(U, regimes, calm=calm, stress=stress, names=("SPY", "ES"))
    diag = rg.identification_diagnostic(rg.regime_residual_cov(U, regimes), names=("SPY", "ES"))
    verdict = "yes" if diag["identified"] else "NO -- do not quote the Rigobon column"
    rC = out["contemp_rigobon"]; cf_ = out["contemp_cholesky_fwd"]; cr = out["contemp_cholesky_rev"]
    idx = ["contemp SPY<-ES", "contemp ES<-SPY",
           "var-ratio spread (max/min - 1; ~0 = common-scale, unidentified)",
           "relative eigenvalue gap (verdict thresholds this > 0.10)",
           "identified (relative het present)"]
    col_r = [rC.loc["SPY", "ES"], rC.loc["ES", "SPY"],
             diag["var_ratio_spread"], diag["rel_eig_gap"], verdict]
    col_f = [cf_.loc["SPY", "ES"], cf_.loc["ES", "SPY"], np.nan, np.nan, ""]
    col_v = [cr.loc["SPY", "ES"], cr.loc["ES", "SPY"], np.nan, np.nan, ""]
    # With 3+ exogenous labels (e.g. benchmark / volatile / mwcb) the constant-A het model is
    # OVER-identified: the same rotation must diagonalize every regime's covariance, and the
    # off-diagonal mass it leaves is a testable restriction rather than a free parameter --
    # ~0 supports the identification, large means the contemporaneous matrix itself moves
    # across regimes and the Rigobon point estimates blend different structures.
    if out.get("overid") is not None:
        idx.append("over-ID off-diag mass across 3+ regimes (~0 = constant-A holds)")
        col_r.append(out["overid"]); col_f.append(np.nan); col_v.append(np.nan)
    num = pd.DataFrame({"Rigobon (order-free)": col_r,
                        "Cholesky SPY->ES": col_f,
                        "Cholesky ES->SPY": col_v}, index=idx)
    note = (f"Contemporaneous structural coefficients. Regimes = {regime_src}. Het-ID pins the "
            f"rotation down only when regimes differ in RELATIVE heteroskedasticity; when the "
            f"verdict row says NO the Rigobon numbers are numerical noise -- quote the Cholesky "
            f"bracket instead. The Rigobon column needs no ordering; the two Cholesky columns "
            f"bracket the ordering assumption. Labeling MWCB days as their own third regime "
            f"(instead of folding them into 'volatile') adds the over-identification row.")
    return Table("Rigobon order-free identification vs Cholesky orderings (revamp)", _fmt_df(num, 4), note, numeric=num)


def _longest_finite_run(Y):
    """Rows of the longest CONTIGUOUS all-finite run. For level-based estimators (VECM, IS)
    this is the only valid NaN handling: dropping masked rows individually would splice the
    pre-halt price to the reopen price, and a contiguous run has no seams by construction."""
    fin = np.all(np.isfinite(np.asarray(Y, float)), axis=1)
    best_i = best_j = i = 0
    n = len(fin)
    while i < n:
        if fin[i]:
            j = i
            while j < n and fin[j]:
                j += 1
            if j - i > best_j - best_i:
                best_i, best_j = i, j
            i = j
        else:
            i += 1
    return Y[best_i:best_j]


def table_regime_is(sessions, p=1, max_iter=80):
    """Markov-switching VECM: endogenous calm/stress regimes with regime-specific
    error-correction speeds and Gonzalo-Granger information shares. Fitted PER DAY --
    stacking day-level log-price LEVELS puts the overnight gap inside the sample as one
    giant 'return' at every seam, and the Hamilton regime chain does not run overnight
    either. Cross-day means with an n_days column are reported instead."""
    per, ll_sw, ll_1, n_used = [], 0.0, 0.0, 0
    for s in sessions:
        df = s[-1]
        Y = np.column_stack([np.log(np.asarray(ca._mid(df, "SPY"), float)),
                             np.log(np.asarray(ca._mid(df, "ES"), float))])
        Y = _longest_finite_run(Y)
        if len(Y) < 50 * (p + 2):
            continue
        try:
            fit = ms.fit_ms_vecm(Y, K=2, p=p, beta=1.0, max_iter=max_iter)
            summ = ms.regime_summary(fit, names=("SPY", "ES")).drop(columns=["regime"]).set_index("label")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        per.append(summ)
        ll_sw += float(fit["loglik"]); ll_1 += float(ms.single_regime_loglik(Y, p, 1.0)); n_used += 1
    if not per:
        return Table("Markov-switching VECM regime table (revamp)", pd.DataFrame(),
                     "no session had a long enough contiguous unmasked run")
    cat = pd.concat(per)
    summ = cat.groupby(level=0).mean()
    summ["n_days"] = cat.groupby(level=0).size().astype(float)
    summ.index.name = "regime"
    note = (f"Two-state Markov-switching VECM (EM), fitted PER DAY on each session's longest "
            f"contiguous unmasked run and averaged across {n_used} days (pre-v0.9.64 this stacked "
            f"day-level log-price levels, making every overnight gap a within-sample return). "
            f"ec_* = error-correction speeds (weaker in stress = arbitrage breaking down); GG_* = "
            f"Gonzalo-Granger information shares; summed switching log-lik {ll_sw:.0f} vs "
            f"single-regime {ll_1:.0f}.")
    return Table("Markov-switching VECM regime table (revamp)", _fmt_df(summ, 4), note, numeric=summ)


def table_hawkes(session_calm, session_stress):
    """Bivariate Hawkes on liquidity events (depth withdrawals): the branching ratio
    (reflexive fragility) and ES<->SPY cross-excitation, calm vs stress."""
    def ev(df):
        return [hk.liquidity_events(df, "SPY"), hk.liquidity_events(df, "ES")]
    Tc = float((session_calm.index[-1] - session_calm.index[0]).total_seconds())
    Ts = float((session_stress.index[-1] - session_stress.index[0]).total_seconds())
    out = hk.stress_conditional_fit(ev(session_calm), Tc, ev(session_stress), Ts, names=("SPY", "ES"))
    Gc = out["cross_excitation_calm"]; Gs = out["cross_excitation_stress"]
    num = pd.DataFrame({
        "calm": [Gc.loc["SPY", "SPY"], Gc.loc["SPY", "ES"], Gc.loc["ES", "SPY"], Gc.loc["ES", "ES"],
                 out["branching_ratio_calm"]],
        "stress": [Gs.loc["SPY", "SPY"], Gs.loc["SPY", "ES"], Gs.loc["ES", "SPY"], Gs.loc["ES", "ES"],
                   out["branching_ratio_stress"]],
    }, index=["SPY<-SPY", "SPY<-ES", "ES<-SPY", "ES<-ES", "branching ratio"])
    note = ("Branching matrix G_ij = expected offspring events of type i triggered by one type-j event; "
            "the spectral radius is the branching ratio (reflexivity, <1 = stationary). A stress rise in "
            "the off-diagonals and the branching ratio is liquidity contagion / fragility.")
    return Table("Bivariate-Hawkes liquidity-event contagion (revamp)", _fmt_df(num, 4), note, numeric=num)


def table_pricing_error(session_calm, session_stress, notional=5e9):
    """Hasbrouck sigma_s (transitory pricing error, bps) per market, calm vs stress, the
    dislocation's incremental dollar welfare cost, and the cross-market basis dislocation."""
    rows = OrderedDict()
    for a in ("SPY", "ES"):
        dc = pe.market_quality(session_calm, a); ds = pe.market_quality(session_stress, a)
        cost = pe.dislocation_cost(dc, ds, notional)
        rows[f"sigma_s {a} (bps)"] = (dc["sigma_s"], ds["sigma_s"])
        if a == "SPY":
            welfare = cost
    rel_c = pe.relative_pricing_error(session_calm); rel_s = pe.relative_pricing_error(session_stress)
    num = pd.DataFrame({"calm": [rows["sigma_s SPY (bps)"][0], rows["sigma_s ES (bps)"][0], rel_c["sigma_s_basis"]],
                        "stress": [rows["sigma_s SPY (bps)"][1], rows["sigma_s ES (bps)"][1], rel_s["sigma_s_basis"]]},
                       index=["sigma_s SPY (bps)", "sigma_s ES (bps)", "basis dislocation (bps)"])
    note = (f"Transitory pricing error sigma_s (Beveridge-Nelson). SPY incremental dislocation cost on "
            f"${notional/1e9:.0f}B notional = ${welfare['incremental_cost']/1e6:.2f}M "
            f"(delta sigma_s = {welfare['delta_sigma_s_bps']:.3f} bps). Basis dislocation = transitory SPY-ES "
            f"mispricing; its stress rise is cross-market contagion.")
    return Table("Hasbrouck pricing error + dollar welfare cost (revamp)", _fmt_df(num, 4), note, numeric=num)


def table_tick_correction(sessions, ticks=(0.01, 0.25), prices=(550.0, 5500.0)):
    """Tick-size-aware information share: raw vs common-grid vs rounding-corrected, plus each
    market's tick-noise share -- referee-proofing the futures-dominance result. Estimated PER
    DAY (an information share is a day-level object; stacking day-level log-price levels puts
    the overnight gap inside the random-walk decomposition) and averaged, leader by day vote."""
    tabs, survives = [], []
    leaders = {"raw": [], "common_grid": [], "rounding_corrected": []}
    for s in sessions:
        df = s[-1]
        Y = np.column_stack([np.log(np.asarray(ca._mid(df, "SPY"), float)),
                             np.log(np.asarray(ca._mid(df, "ES"), float))])
        Y = _longest_finite_run(Y)
        if len(Y) < 200:
            continue
        try:
            res = tk.tick_corrected_information_share(Y, ticks, prices, beta=1.0)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        tabs.append(res["table"])
        for k in leaders:
            leaders[k].append(res[f"leader_{k}"])
        survives.append(bool(res["lead_survives"]))
    if not tabs:
        return Table("Tick-size-corrected information share (revamp)", pd.DataFrame(),
                     "no session had a long enough contiguous unmasked run")
    num = pd.concat(tabs).groupby(level=0).mean().reindex(tabs[0].index)
    num.index.name = "market"
    maj = {k: max(set(v), key=v.count) for k, v in leaders.items()}
    note = (f"Hasbrouck information share (mid), estimated PER DAY on the longest contiguous "
            f"unmasked run and averaged across {len(tabs)} days (pre-v0.9.64 this stacked "
            f"day-level log-price levels across sessions). Leader by day-majority: "
            f"raw={maj['raw']}, common-grid={maj['common_grid']}, "
            f"rounding-corrected={maj['rounding_corrected']}; lead survives on "
            f"{sum(survives)}/{len(survives)} days. ES is the coarser fractional grid.")
    return Table("Tick-size-corrected information share (revamp)", _fmt_df(num, 4), note, numeric=num)


def table_stress_selection(vol):
    """Day-selection summary: regime classification counts and the stress-onset selection
    (level-based, ex-ante), the foundation under the conditional regressions."""
    sel = si.select_days(vol)
    counts = sel["label"].value_counts()
    num = pd.DataFrame({"days": [int(counts.get("calm", 0)), int(counts.get("elevated", 0)),
                                 int(counts.get("stress", 0)), int(sel["onset"].sum())]},
                       index=["calm", "elevated", "stress", "selected (onset)"])
    note = ("Regimes from the long-run (level) volatility state, not a one-day change; 'selected' = "
            "stress-onset days chosen for the event study. Conditioning is on the continuous state, not bins.")
    return Table("Stress-regime selection summary (revamp)", num.astype(object), note, numeric=num)


def table_pcmof_clock_comparison(sessions, n_buckets=40, bucket_window=20):
    """Bucketed (VPIN-style volume clock) vs timed (per-bar) PCMOF / NCMOF, pooled across
    sessions. Shows the same-/opposite-direction masses, net co-direction, and the
    cross-market OFI correlation under each clock -- the latter being the Epps-correction
    number (the volume clock synchronizes the two legs in information time)."""
    dfs = [s[-1] for s in sessions]
    tt = {"pc": [], "nc": [], "corr": [], "n": []}
    vv = {"pc": [], "nc": [], "corr": [], "n": []}
    for df in dfs:
        g_s = cf.signed_ofi(df, "SPY"); g_e = cf.signed_ofi(df, "ES")
        chi = cf.cross_flow(g_s, g_e); pc, nc = cf.pcmof_ncmof(chi)
        m = np.isfinite(g_s) & np.isfinite(g_e)
        tt["pc"].append(float(np.mean(pc))); tt["nc"].append(float(np.mean(nc)))
        tt["corr"].append(float(np.corrcoef(g_s[m], g_e[m])[0, 1])); tt["n"].append(int(m.sum()))
        b = cf.bucketed_cross_flow(df, n_buckets=n_buckets, window=bucket_window)
        w = b["weight"].to_numpy(); sw = w.sum() or 1.0
        vv["pc"].append(float((b["PCMOF"].to_numpy() * w).sum() / sw))
        vv["nc"].append(float((b["NCMOF"].to_numpy() * w).sum() / sw))
        vv["corr"].append(float(np.corrcoef(b["O_SPY"], b["O_ES"])[0, 1])); vv["n"].append(int(len(b)))
    mean = lambda x: float(np.nanmean(x))
    num = pd.DataFrame({
        "time clock": [mean(tt["pc"]), mean(tt["nc"]), mean(tt["pc"]) - mean(tt["nc"]),
                       mean(tt["corr"]), mean(tt["n"])],
        "volume clock": [mean(vv["pc"]), mean(vv["nc"]), mean(vv["pc"]) - mean(vv["nc"]),
                         mean(vv["corr"]), mean(vv["n"])],
    }, index=["mean PCMOF", "mean NCMOF", "net co-direction (P-N)", "cross-market OFI corr", "n observations"])
    note = (f"Pooled over {len(dfs)} sessions. Volume clock = equal-notional buckets on combined SPY+ES "
            f"notional ({n_buckets} buckets/session), weighted by min(activity); time clock = per-bar. "
            f"The cross-market OFI correlation gap is the Epps/synchronization correction (clearer on "
            f"asynchronous data than on this synchronous synthetic set). Book-native |OFI|*mid notional "
            f"proxy; pass real trade notional via notional_fn for a true VPIN clock.")
    return Table("Bucketed (volume-clock) vs timed PCMOF/NCMOF (revamp)", _fmt_df(num, 4), note, numeric=num)


def table_market_shocks(spy_es_only=False):
    """The annotated market-structure & macro shock registry -- the exogenous treatment
    universe (circuit breakers, mass LULD, SIP/exchange outages, flash crashes, vol-regime
    breaks, monetary surprises, sovereign downgrades, trade-policy and geopolitical shocks).
    `timing` encodes how treatment is identified: halt reopen / scheduled 14:00 / next-open."""
    df = ms_shocks.shocks_frame()
    if spy_es_only:
        df = df[df["spy_es"]]
    disp = pd.DataFrame({
        "class": df["event_class"].values,
        "category": df["category"].values,
        "timing": df["timing"].values,
        "S&P %": [f"{x:+.1f}" if np.isfinite(x) else "--" for x in df["spx_pct"].values],
        "SPY/ES": np.where(df["spy_es"].values, "yes", "no"),
        "annotation": df["annotation"].values,
    }, index=[d.date().isoformat() for d in df["date"]])
    disp.index.name = "date"
    note = ("One contagion theme, many triggers (a halt is one mechanism). Identification by `timing`: "
            "halt_reopen -> intraday RD at the reopen; scheduled -> intraday event study at 14:00 ET "
            "(FOMC high-frequency surprise); overnight/weekend/multi_day -> close-to-open gap on the next "
            "session. DATES/TIMES ARE APPROXIMATE -- verify against SEC MWCB & Reg SCI notices, FOMC "
            "statements, ratings-agency releases, and exchange/FINRA halt records before publication.")
    return Table("Market shock registry: structure + macro/geopolitical (annotated)", disp, note, numeric=df)


def table_treatment_control_design(vol):
    """How to pick controls: level-matched (balanced on the ex-ante volatility regime) vs the
    ln(sigma_t/sigma_{t-1})<0 rule (selects on a vol outcome -> imbalanced, biased)."""
    cmp = ms_shocks.compare_control_rules(vol, upper_q=0.60, treat_frac=0.25, n_controls=2, seed=1)
    disp = _fmt_df(cmp["table"], 4); disp.index.name = "group"
    note = (f"Ex-ante level = lagged EWMA of daily vol (never a contemporaneous or one-day-change vol). "
            f"Standardized mean difference from treatment: matched controls are balanced "
            f"(|SMD|={abs(cmp['smd_matched']):.2f}); the ln-change<0 rule is not "
            f"(|SMD|={abs(cmp['smd_naive']):.2f}), so its DiD is biased by the vol level before any event "
            f"channel operates. Treatment days should be exogenous registry events; controls matched on the "
            f"level within a calendar window, excluding a buffer around every event (see market_shocks).")
    return Table("Treatment/control selection: level-matched vs ln(sigma) rule (revamp)", disp, note, numeric=cmp["table"])


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ══════════════════════════════════════════════════════════════════════════════
def build_all_tables(sessions, counts_fn, vol, mwcb_treated, mwcb_control, release_by_date,
                     calm_session, stress_session, n_boot=40, verbose=True, n_jobs=None):
    """Build every table on the provided synthetic/real inputs. Each builder is isolated so
    one failure does not abort the run; failures are recorded as a note. Returns an ordered
    dict {name: Table}."""
    jobs = [
        ("t5_repro", lambda: table_imbalance_contingency(sessions, counts_fn)),
        ("t9_repro", lambda: table_correlation_irf(sessions, spec="informational",
                                                   title="Table 9 (repro): IRF of return correlation [imbalance]")),
        # Table 9 on BOTH dependent variables. The Pearson d-correlation the paper uses is
        # attenuated by asynchronous quote updating, and the attenuation moves with trading
        # intensity -- which is a regressor in the same system. The Delta block estimates how
        # much of each published response is that measurement artifact.
        ("t9_both_ways", lambda: table_correlation_irf_both_ways(sessions, spec="informational",
                                                                 n_boot=n_boot)),
        ("t11_repro_cum", lambda: table_correlation_irf(sessions, spec="informational", cumulative=True,
                                                        title="Table 11 (repro): cumulative IRF of return correlation")),
        ("s6_repro_ols", lambda: table_mwcb_reopening_ols(mwcb_treated, release_by_date)),
        ("market_shocks", lambda: table_market_shocks()),
        ("treatment_control_design", lambda: table_treatment_control_design(vol)),
        ("t9_revamp", lambda: table_crossflow_irf(sessions, n_boot=n_boot, n_jobs=n_jobs)),
        ("pcmof_clock_compare", lambda: table_pcmof_clock_comparison(sessions)),
        ("is_state_revamp", lambda: table_spread_conditioned_is(sessions)),
        ("s6_revamp_did", lambda: table_mwcb_did(mwcb_treated, mwcb_control, release_by_date)),
        ("rigobon_revamp", lambda: table_rigobon(sessions)),
        ("regime_is_revamp", lambda: table_regime_is(sessions)),
        ("hawkes_revamp", lambda: table_hawkes(calm_session, stress_session)),
        ("pricing_error_revamp", lambda: table_pricing_error(calm_session, stress_session)),
        ("tick_revamp", lambda: table_tick_correction(sessions)),
        ("stress_selection_revamp", lambda: table_stress_selection(vol)),
    ]
    out = OrderedDict()
    for name, fn in jobs:
        try:
            out[name] = fn()
            status = f"{out[name].df.shape[0]}x{out[name].df.shape[1]}"
        except Exception as e:
            out[name] = Table(name, pd.DataFrame(), f"BUILD ERROR: {type(e).__name__}: {e}")
            status = "ERROR: " + repr(e)
        if verbose:
            print(f"  [{name:24s}] {status}")
    return out


def write_report(tables, md_path, tex_path):
    """Write all tables to a combined Markdown report and a combined LaTeX file.

    Creates the destination directory if it does not exist. Without this the reporting layer
    could not be smoke-tested on an ordinary checkout: the self-test's default output directory
    was an absolute path from one particular machine, so `python paper_tables.py` ended in
    FileNotFoundError everywhere else."""
    for _p in (md_path, tex_path):
        _d = os.path.dirname(os.path.abspath(_p))
        if _d:
            os.makedirs(_d, exist_ok=True)
    md = ["# Cross-Asset Price-Discovery / Liquidity-Contagion: Table Report", "",
          "Reproduction and revamped tables for Garrison, Jain & Paddrik (2026).", ""]
    tex = [r"\documentclass{article}", r"\usepackage{booktabs}", r"\usepackage[margin=1in]{geometry}",
           r"\begin{document}", ""]
    for name, t in tables.items():
        md.append(t.to_markdown()); md.append("")
        tex.append(t.to_latex(label=name)); tex.append("")
    tex.append(r"\end{document}")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    with open(tex_path, "w") as f:
        f.write("\n".join(tex))
    return md_path, tex_path


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic data (self-test / demo)
# ══════════════════════════════════════════════════════════════════════════════
def _synth_session(date, regime, T=2500, seed=0):
    """A cointegrated two-asset book where ES leads (SPY partially adjusts to the common
    efficient path). Stress widens spreads and withdraws depth. 10 levels each side."""
    rng = np.random.default_rng(seed)
    stress = regime == "stress"
    vol = (2.5 if stress else 1.0) * 1e-4
    eff = np.cumsum(rng.normal(0, vol, T))                         # common efficient % path
    spy = np.zeros(T)
    for t in range(1, T):
        spy[t] = spy[t - 1] + 0.5 * (eff[t] - spy[t - 1]) + rng.normal(0, 0.3e-4)   # SPY lags ES
    mid_spy = 550.0 * np.exp(spy + 0.0); mid_es = 5500.0 * np.exp(eff)
    depth_scale = (0.45 if stress else 1.0)
    cols = {}
    for a, mid, tick, base_depth in (("SPY", mid_spy, 0.01, 320.0), ("ES", mid_es, 0.25, 220.0)):
        for l in range(1, 11):
            q = np.maximum(base_depth * depth_scale * np.exp(-0.3 * (l - 1)) * (1 + rng.normal(0, 0.05, T)), 1.0)
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, T))
    idx = pd.date_range(f"{date} 10:00", periods=T, freq="s", tz="America/New_York")
    return pd.DataFrame(cols, index=idx)


def _counts_fn(df):
    """Per-bar new-order (buy, sell) counts for SPY then ES (tied to each book's OFI sign),
    plus per-bar returns -- the six arrays Table 5 needs."""
    rng = np.random.default_rng(int(df.index[0].value % 2_000_000) + len(df))
    res = {}
    for a in ("SPY", "ES"):
        ofi = np.nan_to_num(np.asarray(ca.order_flow_imbalance(df, a, 10), float))
        mid = np.asarray(ca._mid(df, a), float)
        ret = np.concatenate([[np.nan], np.diff(np.log(mid)) * 1e4])
        total = rng.poisson(8, len(ofi)) + 1
        pbuy = np.clip(0.5 + 0.35 * np.tanh(ofi), 0.02, 0.98)
        buy = rng.binomial(total, pbuy).astype(float)
        res[a] = (buy, (total - buy).astype(float), ret)
    return (res["SPY"][0], res["SPY"][1], res["ES"][0], res["ES"][1], res["SPY"][2], res["ES"][2])


def _synth_release_book(date, release, treated, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(release - pd.Timedelta(minutes=10), release + pd.Timedelta(minutes=10),
                        freq="s", tz="America/New_York")
    secs = (idx - release).total_seconds().to_numpy()
    ramp = np.clip(secs / (4 * 60.0), 0.0, 1.0)
    cols = {}
    for a, base_px, tick, base_depth, wd in (("SPY", 500.0, 0.01, 320.0, 0.35), ("ES", 5000.0, 0.25, 220.0, 0.20)):
        mid = base_px * np.exp(np.cumsum(rng.normal(0, 1.2e-4, len(idx))))
        factor = np.where((secs > 0) & treated, wd + (1.0 - wd) * ramp, 1.0)
        depth = base_depth * factor * (1 + rng.normal(0, 0.03, len(idx)))
        for l in range(1, 11):
            q = np.maximum(depth * np.exp(-0.3 * (l - 1)), 1.0)
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q
    return pd.DataFrame(cols, index=idx)


def _synth_vol(days=252, seed=0):
    rng = np.random.default_rng(seed)
    base = np.exp(np.cumsum(rng.normal(0, 0.05, days)) * 0.3) * 0.01
    base[150:170] *= np.linspace(1, 4, 20)                         # a crisis plateau
    base[170:185] *= np.linspace(4, 1.5, 15)
    return pd.Series(base, index=pd.date_range("2019-09-01", periods=days, freq="B"))


def _selftest():
    print("paper_tables self-test")
    sessions = []
    for i in range(8):
        regime = "calm" if i < 4 else "stress"
        sessions.append((f"2024-08-{i+1:02d}", regime, _synth_session(f"2024-08-{i+1:02d}", regime, T=2500, seed=i)))
    vol = _synth_vol()
    rel = pd.Timestamp("2020-03-09 13:00", tz="America/New_York")
    treated = [(f"T{i}", _synth_release_book(f"2020-03-{9+i:02d}", rel, True, 100 + i)) for i in range(4)]
    control = [(f"C{i}", _synth_release_book(f"2020-02-{10+i:02d}", rel, False, 200 + i)) for i in range(4)]
    release_by_date = {d: rel for d, _ in treated + control}
    calm_df = sessions[0][2]; stress_df = sessions[4][2]

    print("\nbuilding tables:")
    tables = build_all_tables(sessions, _counts_fn, vol, treated, control, release_by_date,
                              calm_df, stress_df, n_boot=30)
    # Default to a path that exists in any checkout; override with OUT_DIR=... for a
    # shared artifact location. An absolute machine-specific default made the module's
    # own self-test fail everywhere but one host.
    out_dir = os.environ.get("OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    md, tex = write_report(tables, os.path.join(out_dir, "paper_tables_report.md"),
                           os.path.join(out_dir, "paper_tables_report.tex"))

    n_ok = sum(1 for t in tables.values() if not t.df.empty)
    n_err = sum(1 for t in tables.values() if "BUILD ERROR" in (t.notes or ""))
    print("\nsample rendered table (markdown):\n")
    print(tables["tick_revamp"].to_markdown())
    print("\n(%d/%d tables non-empty; %d build errors) wrote %s and %s"
          % (n_ok, len(tables), n_err, os.path.basename(md), os.path.basename(tex)))
    print("checks:", n_err == 0 and n_ok == len(tables)
          and os.path.getsize(md) > 1000 and os.path.getsize(tex) > 1000)


if __name__ == "__main__":
    _selftest()
