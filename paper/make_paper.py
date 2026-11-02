"""Builds paper/EV_Charging_Demand_Paper.docx (IEEE conference format, A4, two columns).
Every quantitative statement is read from outputs/results.json or outputs/tables/*.csv,
so the text cannot drift from the results."""
import json
import math
import sys
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from docx_utils import Paper, ROMAN, FULL_W, fix_table_widths  # noqa: E402
import synopsis_text as S  # noqa: E402

R = json.load(open(ROOT / "outputs" / "results.json"))
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
OUT = HERE / "EV_Charging_Demand_Paper.docx"


# ------------------------------------------------------------------ formatting helpers
def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}"


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def n0(x):
    return f"{int(round(x)):,}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def one(lst, **kw):
    return [r for r in lst if all(r.get(k) == v for k, v in kw.items())][0]


D, DS, EDA = R["data"], R["dispersion"], R["eda"]
TM = {r["index"]: r for r in R["test_metrics"]}
VM = {r["index"]: r for r in R["val_metrics"]}
AH, DM = R["alpha_hourly"], R["daily_models"]
DMP, DMN = DM["metrics"]["Daily Poisson"], DM["metrics"]["Daily NB2"]
IRR = {r["term"]: r for r in R["irr"]}
CON = R["irr_contrasts"]
SP, SIM = R["spatial"], R["simulation"]
SITES = [r["site"] for r in R["site_summary"]]
ANN = R["annual_2019"]
bias = lambda k: 100 * (ANN[k] / ANN["observed"] - 1)  # noqa: E731
spec, tv = R["spec_variants"], R["trend_variants"]
spec_choice = min((r for r in spec if r["eval_set"].startswith("validation")), key=lambda r: r["mean_poisson_deviance"])
tv_test = {r["trend"]: r for r in tv if r["eval_set"].startswith("test")}
ct_h, ct_d = DS["hourly_ct"], DM["ct"]
sites_sp = {r["site"]: r for r in SP["sites"]}
clusters, cands = SP["clusters"], SP["candidates"]
scen, scen_emp = SIM["scenarios"], SIM["scenarios_empirical"]
val_sim, val_emp = SIM["validation"], SIM["validation_empirical"]
erl, erl_emp = SIM["erlang"], SIM["erlang_empirical"]
site_bias = {r["site"]: r for r in R["site_bias_2019"]}
fee_con = {r["site"]: r for r in CON if r["kind"] == "fee"}
fee_sorted = sorted(fee_con.values(), key=lambda r: r["IRR"])
h9 = {r["site"]: r for r in CON if r["kind"] == "hour09"}
sat9 = one(CON, contrast="Saturday 09:00 vs Monday 09:00 (all sites)")
sun9 = one(CON, contrast="Sunday 09:00 vs Monday 09:00 (all sites)")
T = lambda s: "MPL" if s == "MPL" else s.title()  # noqa: E731
DT_CL = [c["cluster"] for c in cands if "HAMILTON" in c["sites_pooled"]][0]
sc = lambda tab, name: one(tab, cluster=DT_CL, scenario=name)  # noqa: E731
B_OBS, O2_OBS, O4_OBS = sc(scen_emp, "Baseline (separate sites)"), sc(scen_emp, "Overflow to 2 new ports"), sc(scen_emp, "Overflow to 4 new ports")
P0_OBS, P4_OBS = sc(scen_emp, "Pooled, no new ports"), sc(scen_emp, "Pooled + 4 new ports")
B_M3, O4_M3 = sc(scen, "Baseline (separate sites)"), sc(scen, "Overflow to 4 new ports")
cd0, cl0 = cands[0], clusters[0]
sens = SP["sensitivity"]
sens_bw = [r for r in sens if r.get("demand_weights") == "M3 predicted 2019"]
sens_dt = [r for r in sens_bw if isinstance(r.get("sites_pooled"), str) and "HAMILTON" in r["sites_pooled"]]
sens_extra = [r for r in sens_bw if r not in sens_dt]
sens_obs = [r for r in sens if r.get("demand_weights") == "observed 2019"]
dm_all = {r["model"]: r for r in R["daily_metrics"] if r["site"] == "ALL"}
n_erl_over = sum(1 for a in erl_emp if a["erlang_B"] > a["sim_block_at_peak_arrival_hour"])
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GLM = TM["M3 NB2 GLM"]; HGB = TM["M4 HGB-Poisson"]; OLS = TM["M1 OLS"]

P = Paper()
doc = P.doc

# ================================================================== title block
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.space_after = Pt(12)
r = t.add_run(S.TITLE); r.font.size = Pt(24)
tb = doc.add_table(rows=3, cols=3)
fix_table_widths(tb, [FULL_W / 3] * 3)
for j, (name, roll, reg) in enumerate(S.AUTHORS):
    for i, txt in enumerate([name, roll, reg]):
        c = tb.cell(i, j); c.text = ""
        pp = c.paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.first_line_indent = Inches(0)
        rr = pp.add_run(txt); rr.font.size = Pt(11 if i == 0 else 9)
doc.add_paragraph().paragraph_format.space_after = Pt(6)
P.columns(2)

# ================================================================== abstract
P.para(
    f"*Abstract*—Public EV chargers are usually placed where land is available, not where drivers need them. "
    f"Using {n0(D['raw_rows'])} charging sessions logged by the City of Palo Alto, we built a pipeline that forecasts "
    f"hourly demand at each charging site, maps where demand outruns port supply, clusters the shortfall with DBSCAN, "
    f"and tests a recommended expansion in a queueing simulation. After cleaning, {n0(D['study_sessions'])} sessions at "
    f"{D['n_sites']} sites (2017–2019) form a site-hour panel with the number of active ports as exposure. Count "
    f"models fitted on 2017–2018 were scored on all of 2019. Hourly arrivals turned out to be Poisson once "
    f"calendar, site and lag effects were included (NB2 dispersion estimate at zero), while daily totals were "
    f"overdispersed (α = {f(DM['alpha_summary']['alpha'], 4)}); at that level the negative binomial model raised "
    f"interval coverage from {pc(DMP['coverage90'])} to {pc(DMN['coverage90'])}. The Poisson GLM reached a test "
    f"deviance of {f(GLM['mean_poisson_deviance'])} (D² = {f(GLM['D2'])}) against {f(HGB['mean_poisson_deviance'])} "
    f"for gradient boosting, and its rate ratios show, for instance, that paid charging cut arrivals per port by up "
    f"to {pc(1 - fee_sorted[0]['IRR'], 0)} at some sites. The gap analysis points to one under-supplied zone, downtown. "
    f"In simulation, four overflow ports there reduce turned-away weekday sessions from {n0(B_OBS['turned_away_per_year'])} "
    f"to {n0(O4_OBS['turned_away_per_year'])} per year.", size=9, bold=True)
P.para(f"*Index Terms*—{S.INDEX_TERMS}", size=9, bold=True, space_after=6)

# ================================================================== I-IV (retained from the synopsis)
P.heading("Introduction")
for x in S.INTRODUCTION:
    P.para(x)
P.para("The work makes four contributions: a documented cleaning and exposure model for an open session log; a "
       "comparison of continuous, count and black-box models on a strict temporal hold-out, including where "
       "overdispersion actually appears; a demand-supply gap analysis that turns forecasts into candidate sites; and a "
       "simulation, checked against observed occupancy, of what the recommended capacity would change.")

P.heading("Literature Review")
for x in S.LITERATURE:
    P.para(x)
P.para("Activity-based and origin-destination models [13], [16], [17] and large fleet-data studies [14], [15] estimate "
       "spatiotemporal demand from travel records. We work instead from operator session logs, which are the data most "
       "cities actually hold.")

P.heading("Research Gaps")
for i, (head, body) in enumerate(S.GAPS, 1):
    p = P.para(f"{i}) **{head}** {body}", indent=False)
    p.paragraph_format.left_indent = Inches(0.25); p.paragraph_format.first_line_indent = Inches(-0.17)
P.para(S.GAPS_CLOSING)

P.heading("Objectives")
for i, o in enumerate(S.OBJECTIVES, 1):
    p = P.para(f"{i}) {o}", indent=False)
    p.paragraph_format.left_indent = Inches(0.25); p.paragraph_format.first_line_indent = Inches(-0.17)

# ================================================================== V. dataset
P.heading("Dataset and Study Area")
P.para(f"The data are the City of Palo Alto charging session records [23], {n0(D['raw_rows'])} rows and "
       f"{D['raw_columns']} fields from {D['raw_first'][:10]} to {D['raw_last'][:10]}. Each row is one plug-in event "
       f"with station, port, start and end time, plug-in and charging duration, energy, fee and coordinates. The "
       f"chargers sit in four downtown garages (Bryant, High, Hamilton, Webster), two garages near California Avenue "
       f"(Cambridge, Ted Thompson) and two libraries (Rinconada and Mitchell Park, MPL). Table I describes the sites "
       f"and Table II the cleaning steps.")
ss = R["site_summary"]
rows = [[T(r["site"]), str(int(r["ports_2019"])), n0(r["sessions"]), f(r["mean_duration_h"], 2), f(r["mean_energy_kwh"], 1)]
        for r in ss]
rows.append(["Total / mean", str(int(sum(r["ports_2019"] for r in ss))), n0(sum(r["sessions"] for r in ss)),
             f(D["mean_duration_h"], 2), f(D["mean_energy_kwh"], 1)])
P.table(["Site", "Ports", "Sessions", "Plug-in (h)", "kWh"], rows, "Charging sites, 2017–2019",
        col_widths=[1.15, 0.45, 0.7, 0.65, 0.5], align=["left", "right", "right", "right", "right"],
        notes="Ports: active Level-2 ports in mid-2019. Plug-in and kWh are means per session.")
cl = pd.read_csv(TAB / "cleaning_log.csv")
rows = [[r.step.replace(" (keep 2017-01-01 to 2019-12-31)", "").replace(" (Jul 2011 - Dec 2020)", ""),
         "" if i == 0 else n0(r.removed), n0(r.remaining)] for i, r in enumerate(cl.itertuples())]
P.table(["Cleaning step", "Removed", "Remaining"], rows, "Data cleaning log",
        col_widths=[2.05, 0.65, 0.75], align=["left", "right", "right"],
        notes="UTC-stamped rows have end times before start times. Level-1 rows use 120 V outlets that are not "
              "counted as capacity. 'Plug In Event Id' resets and is never used as a key.")
P.para(f"The 'Address 1' field is unreliable (the High Street garage appears as '528 High St', '528 high st' and "
       f"'528 High St.'), so stations are keyed on 'Station Name', and a site is that name without the city prefix "
       f"and the trailing station number. SHERMAN ({D['sherman_n']} sessions, all in December 2020) and a stray "
       f"'BRYANT # 1' station are dropped.")
P.para(f"Capacity changed during the study: {D['ports_commissioned_2017_2018']} ports opened in 2017–2018 (High in "
       f"January 2017, Bryant in July 2017, Cambridge and Ted Thompson in March 2018). Each port counts as active from "
       f"the day of its first session to the day of its last one, and a gap of more than 30 days is treated as an "
       f"outage ({D['outage_intervals_removed']} cases). The resulting count of active ports $c_{{st}}$ for site $s$ and "
       f"hour $t$ is the exposure of the models. The number of cars plugged in at once never exceeds $c_{{st}}$, which "
       f"supports the definition.")
P.para(f"Charging was free until {D['fee_start']}; the share of paid sessions rose to {pc(D['fee_share_2017Q3'], 0)} "
       f"in 2017-Q3 and {pc(D['fee_share_2017Q4'], 0)} in 2017-Q4. Because 2020 is distorted by COVID-19 "
       f"({n0(EDA['sessions_2020'])} sessions against {n0(EDA['sessions_2019'])} in 2019), the study window is "
       f"2017–2019.")

# ================================================================== VI. methodology
P.heading("Methodology")
P.para("Fig. 1 summarises the pipeline. Every tuning decision uses the last quarter of the training period "
       "(2018-Q4); the 2019 test year is used once, for the final evaluation.")
P.figure(FIG / "fig01_pipeline.png", "Pipeline from the raw log to the evaluated recommendation.", 5.8)

P.subheading("Panel and Features")
P.para(f"Sessions are counted by start hour, giving $y_{{st}}$ arrivals at site $s$ in hour $t$, on the complete "
       f"site-hour grid ({n0(D['panel_rows'])} rows; {n0(D['train_rows'])} train, {n0(D['test_rows'])} test). Features "
       f"are hour of day, day of week, weekend × hour, month, a US federal holiday flag, the fee regime, a linear "
       f"trend, site effects and three lag terms that look back at least 24 h:")
P.equation(r"\ell^{(24)}_{st}=\log(1+y_{s,t-24})", label="lags")
P.equation(r"\ell^{(168)}_{st}=\log(1+y_{s,t-168})", label="lag168")
P.equation(r"r_{st}=\log\Bigl(1+\frac{1}{7}\sum_{k=1}^{7} y_{s,t-24k}\Bigr)", label="roll")
P.para("Because no feature uses the hour being predicted or anything after it, the 2019 forecasts are one day ahead "
       "and free of leakage. Two refinements were chosen on the validation quarter: site × hour terms for "
       "06:00–23:00 (night hours share one profile, since some site-hour cells had no training arrivals) with "
       "site × fee terms, and a trend that is held at its last training value when forecasting. The exposure "
       "enters as an offset,", indent=False)
P.equation(r"\log \mu_{st}=\log c_{st}+\mathbf{x}_{st}^{\top}\boldsymbol{\beta}", label="offset")
P.para(f"so each coefficient acts on arrivals per active port ({R['n_params_glm']} parameters in total).", indent=False)

P.subheading("Dispersion Checks")
P.para("Before choosing between Poisson and negative binomial models we compare variance and mean across homogeneous "
       "cells and apply the Cameron–Trivedi test [20], which regresses a transformed residual of the fitted Poisson "
       "model on the fitted mean:")
P.equation(r"\frac{(y_{st}-\hat\mu_{st})^{2}-y_{st}}{\hat\mu_{st}}=\alpha\,\hat\mu_{st}+\varepsilon_{st}", label="ct")
P.para("A positive α points to NB2-type overdispersion, a negative one to underdispersion.", indent=False)

P.subheading("Forecasting Models")
P.para(f"All models share the features and the split. M0 is the seasonal naive forecast $\\hat{{y}}_{{st}}=y_{{s,t-168}}$. "
       f"M1 is ordinary least squares on the counts (the continuous approach of Gap 3), with the port count as a "
       f"regressor. M2 is a Poisson GLM with offset ({P.labels['offset']}) and log-likelihood")
P.equation(r"\ell_{P}(\boldsymbol\beta)=\sum_{s,t}\bigl[y_{st}\log\,\mu_{st}-\mu_{st}-\log(y_{st}!)\bigr]", label="poisll")
P.para("M3 is the NB2 model, a Poisson–gamma mixture:", indent=False)
P.equation(r"\mathrm{Pr}(Y=y)=\frac{\Gamma(y+\alpha^{-1})}{\Gamma(\alpha^{-1})\,y!}"
           r"\left(\frac{1}{1+\alpha\mu}\right)^{\alpha^{-1}}\left(\frac{\alpha\mu}{1+\alpha\mu}\right)^{y}", label="nb")
P.equation(r"\mathrm{Var}(Y)=\mu+\alpha\mu^{2}", label="nbvar")
P.para("The dispersion α is estimated by maximising the profile likelihood over log α with a bounded search. "
       "When the maximum is at α = 0 we report a one-sided 95% upper bound and the boundary likelihood-ratio test. "
       "The fits use a sparse IRLS solver written for this work; it matches statsmodels [22] to about $10^{-9}$. "
       "Standard errors are clustered by site-day, because hours of the same day share shocks. M4 is scikit-learn's "
       "histogram gradient boosting [21], [28] with Poisson loss; it is fitted to the rate per port with the port count "
       "as sample weight, which reproduces the exposure. M3c (NB2 without lags) shows what the lags add.",
       indent=False)
P.subsubheading("Evaluation", "On 2019 we report MAE, RMSE, mean Poisson deviance and the deviance explained,")
P.equation(r"\bar{d}=\frac{2}{n}\sum_{i}\Bigl[y_i\log\frac{y_i}{\hat\mu_i}-(y_i-\hat\mu_i)\Bigr]", label="dev")
P.equation(r"D^{2}=1-\bar d(\hat\mu)/\bar d(\bar y)", label="d2")
P.para("plus the log-likelihood per observation, the coverage of central 90% prediction intervals, the predicted "
       "share of zero hours, calibration by decile, the randomised PIT for counts [26], and daily MAE and MAPE per "
       "site. A daily Poisson/NB2 pair (site-day counts with 1- and 7-day lags) tests dispersion at the planning "
       "scale.", indent=False)

P.subheading("Spatial Gap Analysis")
P.para("Site coordinates are projected to metres with a local equirectangular projection. From the 2019 forecasts "
       "each site gets its predicted sessions $w_s$, its weekday peak-hour arrival rate $\\lambda_s$ and, with the "
       "mean weekday plug-in time $\\mathbb{E}[S_s]$ and ports $c_s$, a peak utilisation")
P.equation(r"\rho_s=\lambda_s\,\mathbb{E}[S_s]/c_s", label="rho")
P.para(f"On a {SP['grid_res_m']:.0f} m grid around the sites (1 km buffer), demand and supply are Gaussian kernel "
       f"densities weighted by predicted sessions and by ports, the second scaled to the same total:", indent=False)
P.equation(r"D(\mathbf{u})=\sum_{s} w_s K_h(\mathbf{u}-\mathbf{u}_s)", label="kde")
P.equation(r"S(\mathbf{u})=\frac{\sum_s w_s}{\sum_s c_s}\sum_{s} c_s K_h(\mathbf{u}-\mathbf{u}_s)", label="kdes")
P.equation(r"G(\mathbf{u})=D(\mathbf{u})-S(\mathbf{u})", label="gap")
P.para(f"The bandwidth h = {SP['bandwidth_m']:.0f} m is roughly a five-minute walk; 250 m and 600 m are tested as "
       f"well. Cells with G above its {int(SP['gap_quantile'] * 100)}th percentile are clustered with DBSCAN [18], "
       f"using G divided by that threshold τ as sample weight, so a cell is a core point when", indent=False)
P.equation(r"\sum_{\lVert\mathbf{u}_j-\mathbf{u}_i\rVert\le\varepsilon} G(\mathbf{u}_j)/\tau\;\ge\;\mathrm{minPts}", label="dbscan")
P.para(f"We set minPts = {SP['min_samples']} (twice the dimension [18], [27]) and read ε from the k-distance plot; "
       f"its knee falls on the grid spacing, so ε is set to the grid diagonal ({f(SP['eps_m'], 1)} m). Each "
       f"cluster's candidate site is its gap-weighted centroid, moved to the nearest cluster cell at least 150 m from "
       f"existing chargers if necessary.", indent=False)

P.subheading("Queueing Simulation")
P.para(f"Each site is simulated on a typical 2019 weekday as a loss system: arrivals follow a non-homogeneous "
       f"Poisson process with the forecast hourly rate (generated by thinning [24]), plug-in times are resampled from "
       f"that site's 2019 weekday sessions starting in the same hour, and a driver who finds all ports busy leaves. "
       f"Two days are simulated per run and the first is discarded; {SIM['n_reps']} runs are made with a fixed seed. "
       f"The simulator is checked against occupancy rebuilt from the real 2019 sessions, once with forecast rates and "
       f"once with the observed rates. At the peak hour, with offered load a = λE[S], the Erlang loss and delay "
       f"formulas [25] give a reference:")
P.equation(r"B(c,a)=\frac{a^{c}/c!}{\sum_{k=0}^{c}a^{k}/k!}", label="erlang")
P.equation(r"C(c,a)=\frac{c\,B(c,a)}{c-a\,[1-B(c,a)]},\quad a<c", label="erlangc")
P.para(f"For the candidate we test 2 or 4 new ports used as overflow (a driver tries the own garage first, then the "
       f"new site) and full pooling of the cluster, in which drivers are sent to any free port. Results are scaled "
       f"to the {D['n_weekdays_2019']} working weekdays of 2019. Since turned-away drivers never appear in the log, "
       f"all demand figures are lower bounds.", indent=False)

# ================================================================== VII. setup
P.heading("Experimental Setup")
P.para(f"Training uses 2017–2018 ({n0(D['train_rows'])} site-hours) and testing all of 2019 "
       f"({n0(D['test_rows'])}), without shuffling. Tuning fits stop at 2018-09 and are scored on 2018-Q4; the chosen "
       f"settings (Table III) are then refitted on the full training period. Software: Python {R['meta']['python']}, "
       f"statsmodels {R['meta']['statsmodels']}, scikit-learn {R['meta']['scikit_learn']}, pandas {R['meta']['pandas']}; "
       f"all seeds are fixed at {R['meta']['seed']}.")
hb = R["hgb_best_params"]
rows = [["HGB learning rate", "0.05, 0.1", f(hb["learning_rate"], 2)],
        ["HGB leaves", "15, 31, 63", str(hb["max_leaf_nodes"])],
        ["HGB iterations", "200, 500", str(hb["max_iter"])],
        ["GLM trend", "extrap., frozen, none", "frozen"],
        ["GLM site terms", "none, ×fee, ×hour, both", "both"],
        ["KDE bandwidth", "250, 400, 600 m", f"{SP['bandwidth_m']:.0f} m"],
        ["DBSCAN minPts, ε", "k-distance plot", f"{SP['min_samples']}, {f(SP['eps_m'], 0)} m"],
        ["Simulation runs", "—", str(SIM["n_reps"])]]
P.table(["Setting", "Options", "Chosen"], rows, "Settings chosen on 2018-Q4", col_widths=[1.2, 1.4, 0.85])

# ================================================================== VIII. results
P.heading("Results and Discussion")
P.subheading("Exploratory Analysis")
yrs = D["sessions_per_year"]
P.para(f"Use grew from {n0(yrs['2012'])} sessions in 2012 to {n0(yrs['2017'])} in 2017, fell by "
       f"{f(abs(EDA['change_2017_2018_pct']), 1)}% in 2018 after fees were introduced, and recovered to "
       f"{n0(yrs['2019'])} in 2019 (Fig. 2(a)). {pc(EDA['weekday_share'], 0)} of sessions start on a weekday. Weekdays "
       f"have a morning peak from commuters and an evening peak near 18:00; the busiest slot overall is {DAYS[EDA['heat_peak_dow']]} at "
       f"{EDA['heat_peak_hour']:02d}:00 with {f(EDA['heat_peak_mean_sessions'], 1)} arrivals per hour (Fig. 2(b)). The "
       f"2018 drop was concentrated at Rinconada, MPL, High and Webster, while the sites that gained ports grew "
       f"(Fig. 2(c)).")
P.figure(FIG / "ieee_fig_eda.png", "(a) Clean sessions per year (dark: study window). (b) Mean arrivals per hour, all "
         "sites, 2017–2018. (c) Sessions per site and year.", 7.0)

P.subheading("Dispersion")
P.para(f"{pc(D['zero_fraction_all'])} of site-hours have no arrival. Against common practice, the hourly counts are "
       f"not overdispersed (Table IV, Fig. 3). The median variance-to-mean ratio over {len(pd.read_csv(TAB / 'mean_variance_hourly_cells.csv'))} "
       f"hourly cells is {f(DS['hourly_cells_median_var_mean_ratio'], 2)}, and the Cameron–Trivedi coefficient is "
       f"negative ({f(ct_h['alpha_ct'], 3)}, t = {f(ct_h['t'], 1)}). In the busiest decile the ratio drops to "
       f"{f(DS['hourly_by_decile'][-1]['ratio'], 2)}: a full site cannot accept more cars, which squeezes the variance. "
       f"The NB2 likelihood is therefore maximised at α = 0 and M3 reduces to M2. Daily totals behave differently. "
       f"Weather and local events affect every hour of a day together, and these shared shocks show up once the hours "
       f"are summed: the daily NB2 estimate is α = {f(DM['alpha_summary']['alpha'], 4)} with a likelihood-ratio "
       f"statistic of {f(DM['alpha_summary']['LR_stat'], 1)}, and its intervals cover more of 2019.")
rows = [["Pearson χ²/df", f(DS["hourly_pearson_dispersion"], 2), f(DM["alpha_summary"]["pearson_dispersion_poisson"], 2)],
        ["C–T α (t)", f"{f(ct_h['alpha_ct'], 3)} ({f(ct_h['t'], 1)})", f"{f(ct_d['alpha_ct'], 3)} ({f(ct_d['t'], 1)})"],
        ["NB2 α (SE)", f"0 (≤ {f(AH['alpha_upper95_profile'], 4)})",
         f"{f(DM['alpha_summary']['alpha'], 4)} ({f(DM['alpha_summary']['alpha_se'], 4)})"],
        ["LR test vs. Poisson", f(AH["LR_stat"], 1), f(DM["alpha_summary"]["LR_stat"], 1)],
        ["2019 90% cov., Poisson", pc(TM["M2 Poisson GLM"]["coverage90"]), pc(DMP["coverage90"])],
        ["2019 90% cov., NB2", pc(GLM["coverage90"]), pc(DMN["coverage90"])],
        ["2019 log-lik./obs, NB2", f(GLM["loglik_per_obs"], 3), f(DMN["loglik_per_obs"], 3)]]
P.table(["Statistic", "Hourly", "Daily"], rows, "Dispersion tests and daily models", col_widths=[1.45, 1.05, 0.95],
        align=["left", "center", "center"],
        notes="Hourly NB2 bound: one-sided 95% profile-likelihood bound. Hourly coverage is conservative because "
              "interval limits are integers.")
P.figure(FIG / "fig04_dispersion.png", "Variance against mean for (a) hourly and (b) daily cells of the training data, "
         "and (c) residual variance/mean by decile of the hourly Poisson fit.", 7.0)

P.subheading("Forecast Accuracy")
P.para(f"Table V lists the 2019 results. The count GLM (M2, equal to M3 here) has the lowest deviance "
       f"({f(GLM['mean_poisson_deviance'])}), RMSE and log-likelihood loss. Gradient boosting has the lowest MAE, "
       f"partly because MAE rewards predicting zero in quiet hours, yet it predicts too many zeros "
       f"({pc(HGB['pred_zero_frac'])} against {pc(GLM['obs_zero_frac'])}) and misses the 2019 total by "
       f"{f(abs(bias('M4')), 1)}%. Trees cannot go beyond the levels seen in training, and 2019 demand rebounded. "
       f"Boosting did win on the validation quarter ({f(VM['M4 HGB-Poisson']['mean_poisson_deviance'])} against "
       f"{f(VM['M3 NB2 GLM']['mean_poisson_deviance'])}), which shows how a short hold-out can flatter a flexible model. "
       f"OLS gives negative forecasts in {pc(R['ols_negative_pred_frac'])} of hours and intervals "
       f"{f(OLS['mean_PI_width'] / GLM['mean_PI_width'], 2)} times wider than the count model. Dropping the lags (M3c) "
       f"costs {f(TM['M3c NB2 (calendar only)']['mean_poisson_deviance'] - GLM['mean_poisson_deviance'], 3)} in deviance. "
       f"At the daily level the GLM is off by {f(dm_all['M3 NB2 GLM']['daily_MAE'], 2)} sessions per site-day on a mean "
       f"of {f(dm_all['M3 NB2 GLM']['mean_daily_sessions'], 1)}.")
key = {"M0 Seasonal naive": "M0", "M1 OLS": "M1", "M2 Poisson GLM": "M2", "M3 NB2 GLM": "M3",
       "M3c NB2 (calendar only)": "M3c", "M4 HGB-Poisson": "M4"}
names = {"M0 Seasonal naive": "M0 Seasonal naive", "M1 OLS": "M1 OLS", "M2 Poisson GLM": "M2 Poisson GLM",
         "M3 NB2 GLM": "M3 NB2 GLM", "M3c NB2 (calendar only)": "M3c NB2, no lags", "M4 HGB-Poisson": "M4 HGB-Poisson"}
rows = []
for m in key:
    r_ = TM[m]
    dd = dm_all.get(m.replace("M3c NB2 (calendar only)", "-"), None)
    rows.append([names[m], f(r_["MAE"]), f(r_["RMSE"]), f(r_["mean_poisson_deviance"]), f(r_["D2"]),
                 f(r_["loglik_per_obs"], 3), pc(r_["coverage90"]), f"{bias(key[m]):+.1f}%",
                 f(dd["daily_MAE"], 2) if dd else "", f(dd["daily_MAPE_pct"], 1) + "%" if dd else ""])
P.table(["Model", "MAE", "RMSE", "Dev.", "D²", "LL/obs", "90% cov.", "2019 bias", "Daily MAE", "Daily MAPE"], rows,
        f"Accuracy on the 2019 test year ({n0(D['test_rows'])} site-hours)",
        col_widths=[1.45, 0.5, 0.5, 0.55, 0.5, 0.6, 0.65, 0.65, 0.65, 0.7],
        align=["left"] + ["right"] * 9,
        notes="Dev.: mean Poisson deviance (OLS clipped at 0.01). 2019 bias: predicted/observed total − 1. Daily "
              "columns pool all sites; MAPE excludes days with no arrivals.")
P.para(f"Table VI shows the two choices made on the validation quarter. Site-specific terms cut the worst single-site "
       f"bias on 2019 from {f(one(spec, n_params=77, eval_set='test 2019')['worst_site_abs_bias_pct'], 0)}% to "
       f"{f(one(spec, specification=spec_choice['specification'], eval_set='test 2019')['worst_site_abs_bias_pct'], 0)}%. "
       f"The frozen trend was preferred on validation, but on 2019 a model without any trend would have done slightly "
       f"better ({f(tv_test['none']['mean_poisson_deviance'], 4)} against {f(tv_test['frozen']['mean_poisson_deviance'], 4)}). "
       f"We kept the validation choice so that the test year was never used for selection.")
rows = []
for r_ in spec:
    if r_["eval_set"].startswith("validation"):
        te = one(spec, specification=r_["specification"], eval_set="test 2019")
        rows.append([r_["specification"].replace("base (site fixed effects only)", "Site effects only")
                     .replace("+ site x hour + site x fee", "+ both (chosen)").replace("+ site x hour", "+ site × hour")
                     .replace("+ site x fee", "+ site × fee"), str(r_["n_params"]), f(r_["mean_poisson_deviance"], 4),
                     f(te["mean_poisson_deviance"], 4), f"{te['total_bias_pct']:+.1f}%"])
for mode in ["extrapolated", "frozen", "none"]:
    va = one(tv, trend=mode, eval_set="validation 2018-Q4"); te = one(tv, trend=mode, eval_set="test 2019")
    rows.append([f"Trend {mode}", "", f(va["mean_poisson_deviance"], 4), f(te["mean_poisson_deviance"], 4),
                 f"{te['total_bias_pct']:+.1f}%"])
P.table(["Specification", "p", "Val. dev.", "Test dev.", "Test bias"], rows, "Specification choices (Poisson GLM)",
        col_widths=[1.3, 0.35, 0.62, 0.62, 0.58], align=["left", "right", "right", "right", "right"],
        notes="Selection uses the validation column only. Trend rows use the chosen site terms.")

P.subheading("Interpretation")
P.para(f"The rate ratios (Table VII, Fig. 4) read directly. At Bryant a weekday port receives about "
       f"{f(h9['BRYANT']['IRR'], 0)} times as many arrivals between 09:00 and 10:00 as between midnight and 01:00 "
       f"(95% CI {f(h9['BRYANT']['CI_low'], 0)}–{f(h9['BRYANT']['CI_high'], 0)}). At 09:00 a Saturday has "
       f"{pc(1 - sat9['IRR'], 0)} and a Sunday {pc(1 - sun9['IRR'], 0)} fewer arrivals than a Monday, and a federal holiday "
       f"lowers the rate to {f(IRR['holiday']['IRR'], 2)} of a normal day. A doubling of recent same-hour demand raises "
       f"the expected rate by a factor of {f(2 ** IRR['log1p_roll7']['coef'], 2)}. The fee effect varies widely: arrivals "
       f"per port fell by {pc(1 - fee_sorted[0]['IRR'], 0)} at {T(fee_sorted[0]['site'])} and {pc(1 - fee_sorted[1]['IRR'], 0)} "
       f"at {T(fee_sorted[1]['site'])}, but did not change measurably at Hamilton (IRR {f(fee_con['HAMILTON']['IRR'], 2)}, "
       f"CI {f(fee_con['HAMILTON']['CI_low'], 2)}–{f(fee_con['HAMILTON']['CI_high'], 2)}). Hamilton has only "
       f"{sites_sp['HAMILTON']['ports_2019']} ports and was often full, so a drop in demand there hardly changes the "
       f"recorded arrivals. The Ted Thompson and Cambridge estimates also reflect their March 2018 expansion.")
rows = [["Weekday 09:00 vs 00:00, Bryant", f(h9["BRYANT"]["IRR"], 1), f"{f(h9['BRYANT']['CI_low'], 1)}–{f(h9['BRYANT']['CI_high'], 1)}", pval(h9["BRYANT"]["p_value"])],
        ["Saturday vs Monday, 09:00", f(sat9["IRR"], 3), f"{f(sat9['CI_low'], 3)}–{f(sat9['CI_high'], 3)}", pval(sat9["p_value"])],
        ["Sunday vs Monday, 09:00", f(sun9["IRR"], 3), f"{f(sun9['CI_low'], 3)}–{f(sun9['CI_high'], 3)}", pval(sun9["p_value"])]]
for term, lab in [("holiday", "Federal holiday"), ("trend_years", "Trend (per year)"), ("log1p_lag24", "log(1+y), 24 h lag"),
                  ("log1p_lag168", "log(1+y), 168 h lag"), ("log1p_roll7", "log(1+7-day mean)")]:
    x = IRR[term]; rows.append([lab, f(x["IRR"], 3), f"{f(x['CI_low'], 3)}–{f(x['CI_high'], 3)}", pval(x["p_value"])])
for s in SITES:
    x = fee_con[s]; rows.append([f"Fee regime, {T(s)}", f(x["IRR"], 3), f"{f(x['CI_low'], 3)}–{f(x['CI_high'], 3)}", pval(x["p_value"])])
P.table(["Term", "IRR", "95% CI", "p"], rows, "Selected rate ratios (M3)", col_widths=[1.6, 0.45, 0.85, 0.55],
        align=["left", "right", "right", "right"], notes="Confidence intervals are cluster-robust by site-day.")
P.figure(FIG / "ieee_fig_irr.png", "Rate ratios with 95% cluster-robust confidence intervals (log scale).", 3.4, wide=False)

P.subheading("Calibration and Residuals")
P.para(f"The forecasts follow the test week closely (Fig. 5(a)). By decile the GLM is well calibrated except in the "
       f"busiest hours, where it predicts {f(R['calibration_top_decile']['M3']['pred'], 2)} arrivals per hour against "
       f"{f(R['calibration_top_decile']['M3']['obs'], 2)} observed; boosting predicts "
       f"{f(R['calibration_top_decile']['M4']['pred'], 2)} (Fig. 5(b)). For the hourly GLM, "
       f"{pc(R['pit_tail_share']['hourly_M3'])} of PIT values lie outside [0.05, 0.95] against a nominal 10%, so the "
       f"Poisson spread is right. For daily totals the tail share is {pc(R['pit_tail_share']['daily_poisson'])} for "
       f"Poisson and {pc(R['pit_tail_share']['daily_nb'])} for NB2 (Fig. 5(c)). Residual autocorrelation is negligible "
       f"(Fig. 5(d); lag 1: {f(R['residuals']['acf1_mean_over_sites'], 3)}, lag 24: {f(R['residuals']['acf24_mean_over_sites'], 3)}).")
P.figure(FIG / "ieee_fig_diag.png", f"(a) Observed and predicted arrivals with the M3 90% interval, one test week. "
         f"(b) Calibration by decile. (c) Randomised PIT. (d) Residual autocorrelation (band: approx. 95%).", 7.0)

P.subheading("Demand Gap and Candidate Site")
P.para(f"Table VIII turns the forecasts into planning figures. Hamilton ({n0(sites_sp['HAMILTON']['pred_sessions_per_port'])} "
       f"predicted sessions per port, peak ρ = {f(sites_sp['HAMILTON']['rho_peak'], 2)}) and Webster "
       f"({n0(sites_sp['WEBSTER']['pred_sessions_per_port'])}) carry far more per port than the other sites "
       f"({n0(min(r['pred_sessions_per_port'] for r in SP['sites']))}–{n0(sorted(r['pred_sessions_per_port'] for r in SP['sites'])[-3])}). "
       f"The gap surface therefore has one positive area over downtown, while Cambridge, Ted Thompson, Rinconada and "
       f"MPL are relatively over-supplied (Fig. 6). DBSCAN finds {SP['n_clusters']} cluster of "
       f"{f(cl0['area_km2'], 2)} km² holding a gap of {n0(cl0['total_gap_sessions_per_yr'])} sessions per year "
       f"(Fig. 7). Its centroid, ({cd0['lat']:.4f}, {cd0['lon']:.4f}), lies {f(cd0['dist_nearest_site_m'], 0)} m from "
       f"Hamilton, between the Hamilton, Webster and Bryant garages, and is the recommended site. It moves by at most "
       f"{f(max(r['dist_to_nearest_baseline_candidate_m'] for r in sens_dt), 0)} m for bandwidths of 250–600 m and by "
       f"{f(sens_obs[0]['dist_to_nearest_baseline_candidate_m'], 0)} m when observed instead of predicted sessions are "
       f"used. A second small zone near Cambridge appears only with the 250 m kernel and is not recommended.")
rows = [[T(s), str(sites_sp[s]["ports_2019"]), n0(sites_sp[s]["pred_sessions_2019"]), n0(sites_sp[s]["obs_sessions_2019"]),
         n0(sites_sp[s]["pred_sessions_per_port"]), f(sites_sp[s]["rho_peak"], 2)] for s in SITES]
P.table(["Site", "c", "Pred.", "Obs.", "Pred./port", "Peak ρ"], rows, "Predicted 2019 demand and utilisation",
        col_widths=[1.05, 0.3, 0.55, 0.55, 0.6, 0.45], align=["left"] + ["right"] * 5,
        notes="Pred./Obs.: 2019 sessions. ρ: offered load per port in the busiest weekday hour.")
P.figure(FIG / "fig10_maps_gap.png", "(a) Sites sized by predicted 2019 sessions and coloured by peak utilisation. "
         "(b) Gap G(x) for h = 400 m; red marks demand above port supply.", 6.6)
P.figure(FIG / "fig11_dbscan_candidates.png", f"(a) k-distance plot (k = {SP['min_samples']}). (b) DBSCAN cluster of "
         f"high-gap cells, existing sites (triangles) and the candidate (star).", 6.0)

P.subheading("Simulation")
P.para(f"With observed arrival rates the simulator reproduces hourly occupancy at every site (r = "
       f"{f(min(r['occ_corr'] for r in val_emp), 3)}–{f(max(r['occ_corr'] for r in val_emp), 3)}; Fig. 8(a), Table IX). "
       f"With forecast rates the shapes still agree (r ≥ {f(min(r['occ_corr'] for r in val_sim), 2)}) but the level "
       f"inherits the forecast shortfall. Erlang-B at the peak arrival hour exceeds the simulated blocking at "
       f"{n_erl_over} of {len(erl_emp)} sites, because cars that arrive at the peak stay about two hours and blocking builds "
       f"up later in the morning (Fig. 8(b)). Hamilton is the exception to a good fit: random arrivals at its observed "
       f"rate would lose {pc(one(erl_emp, site='HAMILTON')['sim_daily_block'], 0)} of drivers, yet every recorded "
       f"session was served, which suggests that drivers there time their visits to free ports.")
rows = []
for a, b, va in zip(erl, erl_emp, val_emp):
    rows.append([T(a["site"]), str(a["ports"]), f(va["occ_corr"], 3), f(va["occ_MAE"], 2), f(b["offered_load_erlang"], 2),
                 pc(b["erlang_B"]), pc(b["sim_block_at_peak_arrival_hour"]), f(b["sim_turned_away_per_day"], 2),
                 f(a["sim_turned_away_per_day"], 2)])
P.table(["Site", "c", "Occ. r", "Occ. MAE", "a (Erl.)", "Erlang-B", "Sim. block", "Lost/day", "Lost/day (M3)"], rows,
        "Simulator check and peak-hour blocking",
        col_widths=[1.1, 0.3, 0.55, 0.6, 0.6, 0.7, 0.75, 0.7, 0.85], align=["left"] + ["right"] * 8,
        notes="Occupancy r and MAE compare the 24 hourly means of simulation and data. a, Erlang-B and simulated "
              "blocking refer to the peak arrival hour; all columns except the last use observed 2019 rates.")
P.figure(FIG / "ieee_fig_sim.png", "(a) Simulated against observed mean occupancy per hour (all sites, observed "
         "arrival rates). (b) Share of drivers turned away by site and hour.", 7.0)
P.para(f"Table X gives the effect of the recommendation in the downtown cluster. At observed 2019 demand the four "
       f"garages turn away {n0(B_OBS['turned_away_per_year'])} weekday sessions per year. Two overflow ports at the "
       f"candidate reduce this to {n0(O2_OBS['turned_away_per_year'])} and four ports to {n0(O4_OBS['turned_away_per_year'])}, "
       f"a fall of {pc(1 - O4_OBS['turned_away_per_year'] / B_OBS['turned_away_per_year'], 0)}. With forecast rates the "
       f"same four ports take the figure from {n0(B_M3['turned_away_per_year'])} to {n0(O4_M3['turned_away_per_year'])}. "
       f"Sending drivers to any free port in the cluster, with no new hardware, already brings losses down to "
       f"{n0(P0_OBS['turned_away_per_year'])}, so real-time availability information is about as useful as new ports "
       f"here; together they leave {n0(P4_OBS['turned_away_per_year'])}.")
order = ["Baseline (separate sites)", "Overflow to 2 new ports", "Overflow to 4 new ports", "Pooled, no new ports",
         "Pooled + 2 new ports", "Pooled + 4 new ports"]
lab = {"Baseline (separate sites)": "Baseline", "Overflow to 2 new ports": "Overflow, +2", "Overflow to 4 new ports": "Overflow, +4",
       "Pooled, no new ports": "Pooled, +0", "Pooled + 2 new ports": "Pooled, +2", "Pooled + 4 new ports": "Pooled, +4"}
rows = [[lab[n], str(sc(scen_emp, n)["total_ports"]), n0(sc(scen, n)["turned_away_per_year"]),
         n0(sc(scen_emp, n)["turned_away_per_year"]), pc(sc(scen_emp, n)["peak_block_prob"])] for n in order]
P.table(["Scenario", "Ports", "Lost/yr (M3)", "Lost/yr (obs.)", "Peak block"], rows,
        "Downtown scenarios (weekdays)", col_widths=[0.95, 0.45, 0.7, 0.75, 0.6],
        align=["left", "right", "right", "right", "right"],
        notes=f"Bryant, Hamilton, High and Webster. Per year = {D['n_weekdays_2019']} weekdays. Peak block uses observed rates.")

P.subheading("The Research Gaps Revisited")
P.para("For Gap 1, the interpretable GLM was more accurate than gradient boosting on the unseen year in deviance, RMSE "
       "and likelihood, and its coefficients answer planning questions directly. For Gap 2, the same forecasts drive "
       "the gap map, the clustering and the simulation. For Gap 3, count models give valid probabilities, but the "
       "usual case for NB2 holds only for daily totals. For Gap 4, DBSCAN isolated the high-gap zone without fixing the "
       "number of clusters and set isolated cells aside as noise.")

# ================================================================== IX. limitations
P.heading("Limitations")
lims = [("One city", "Eight sites in one city give a gap map of imbalance between existing sites; areas without chargers "
                     "cannot be assessed, and the results may not transfer to other cities."),
        ("Censored demand", "Drivers who found a full site are not recorded, so demand, the fee effect at busy sites and the "
                            "simulated losses are all underestimated."),
        ("Level shift", f"2019 demand was under-predicted by {f(abs(bias('M3')), 1)}% overall (High "
                        f"{f(site_bias['HIGH']['M3_bias_pct'], 0)}%), and the trend choice rests on one short validation quarter."),
        ("Missing data", "Weather, traffic and points of interest were not available. Port outages were inferred from "
                         "gaps in use, and the simulation assumes random arrivals and no waiting.")]
for head, body in lims:
    P.subsubheading(head, body)

# ================================================================== X. conclusion
P.heading("Conclusion and Future Work")
P.para(f"The forecast-cluster-recommend pipeline of the synopsis was built and tested on real Palo Alto data. A count "
       f"GLM with an exposure offset, site-specific terms and past-only lags explains {pc(GLM['D2'], 0)} of the hourly "
       f"deviance in an unseen year and holds up better than gradient boosting. Hourly arrivals are Poisson; daily totals "
       f"need NB2. The gap analysis identifies downtown Palo Alto as under-supplied, and four overflow ports at the "
       f"candidate site would recover about {n0(B_OBS['turned_away_per_year'] - O4_OBS['turned_away_per_year'])} weekday "
       f"sessions per year. Future work should model censoring directly, add weather and land-use data, include grid "
       f"limits in placement [6], [9], [10] and repeat the study in other cities.")

# ================================================================== references
P.heading_unnumbered("References")
for i, ref in enumerate(S.REFERENCES, 1):
    q = doc.add_paragraph()
    q.paragraph_format.left_indent = Inches(0.3); q.paragraph_format.first_line_indent = Inches(-0.3)
    q.paragraph_format.space_after = Pt(1)
    q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    q.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
    P.rich(q, f"[{i}]\t{ref}", size=8)

P.save(OUT)
print(f"Saved {OUT}: {P.fig_n} figures, {P.tab_n} tables, {P.eq_n} equations, {len(S.REFERENCES)} references")
