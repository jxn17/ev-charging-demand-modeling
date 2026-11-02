"""Summary report: is each model doing the job it was built for? All numbers come from outputs/results.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_style import Report  # noqa: E402

R = json.load(open(HERE.parent / "outputs" / "results.json"))
FIG = HERE.parent / "outputs" / "figures"
OUT = HERE / "Model_Performance_Summary.docx"


def f(x, d=3):
    return "—" if x is None else f"{x:,.{d}f}".replace("-", "−")


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def n0(x):
    return f"{round(x):,}"


def signed(x, d=1):
    return f"{x:+.{d}f}%".replace("-", "−")


def one(lst, **kw):
    return [r for r in lst if all(r.get(k) == v for k, v in kw.items())][0]


T = lambda s: "MPL" if s == "MPL" else s.title()  # noqa: E731
TM = {r["index"]: r for r in R["test_metrics"]}
ANN = R["annual_2019"]
bias = lambda k: 100 * (ANN[k] / ANN["observed"] - 1)  # noqa: E731
GLM, HGB, OLS, NAIVE = TM["M2 Poisson GLM"], TM["M4 HGB-Poisson"], TM["M1 OLS"], TM["M0 Seasonal naive"]
DS, DM, AH = R["dispersion"], R["daily_models"], R["alpha_hourly"]
DP, DN = DM["metrics"]["Daily Poisson"], DM["metrics"]["Daily NB2"]
A = R["audit"]
det = {r["model"]: r for r in A["detection"]}
ps = A["per_site"]
rk = A["rankings"]
SP, SIM, RN = R["spatial"], R["simulation"], R["rinconada"]
sites = {r["site"]: r for r in SP["sites"]}
full = {r["site"]: r for r in RN["observed_full_share"]}
vo = SIM["validation_empirical"]
vm = SIM["validation"]
sens_dt = [r for r in SP["sensitivity"] if r.get("demand_weights") == "M3 predicted 2019" and isinstance(r.get("sites_pooled"), str)]
sens_obs = [r for r in SP["sensitivity"] if r.get("demand_weights") == "observed 2019"][0]
scO = SIM["scenarios_empirical"]
B4 = one(scO, scenario="Baseline (separate sites)")
O4 = one(scO, scenario="Overflow to 4 new ports")
prec = {(r["rates"], r["system"]): r for r in RN["precision"]}
P4x, P5x = prec[("observed", "4 sites pooled")], prec[("observed", "5 sites pooled")]
B5 = one(RN["scenarios"], rates="observed", scenario="Baseline, 5 sites separate")
P50 = one(RN["scenarios"], rates="observed", scenario="Pooled +0, with Rinconada")
share4, share5 = 1 - P4x["lost_per_year"] / B4["turned_away_per_year"], 1 - P5x["lost_per_year"] / B5["lost_per_year"]
cand = SP["candidates"][0]
near_peak = sum(1 for r in ps if r["peak_hour_diff"] <= 1)
missed_peak = [T(r["site"]) for r in ps if r["peak_hour_diff"] > 1]
n_under = sum(1 for r in ps if r["peak_rate_predicted"] < r["peak_rate_observed"])
hm_o = one(vo, site="HAMILTON")
top = R["calibration_top_decile"]
g = det["Poisson GLM"]

D = Report("EV Charging Demand - Model Performance Summary")
D.title("Are the Models Doing Their Jobs?", "Model performance summary · Prathamesh Patil (9) · Srijan Bera (17) · Snehal Tyagi (24)")

D.h1("Bottom Line")
D.p("Each model in this project has one job. This report checks each against evidence it did not train on: the 2019 test year, or the real session log.")
D.items([f"**The forecast works, with one known weakness.** The Poisson GLM explains {pc(GLM['D2'], 0)} of hourly deviance on 2019 and ranks busy hours well (AUC {f(g['auc_busy_hour'], 2)}), but it runs low: {f(abs(bias('M2')), 1)}% under on the year, and more at the peaks.",
         f"**The site-finding step works.** It points to downtown, and the sites it flags as short of ports are also the ones the real log shows filling up most (rank correlation {f(rk['spearman_gap_vs_observed_full_share'], 2)}).",
         f"**The simulator works, except at Hamilton.** It reproduces hourly occupancy at every site (r ≥ {f(min(r['occ_corr'] for r in vo), 3)}), but understates how busy Hamilton is.",
         f"**The Rinconada idea helps less than expected.** Pooling the four downtown garages recovers about {pc(share4, 0)} of simulated losses. Adding Rinconada brings that to about {pc(share5, 0)}."])
D.rule()

D.h1("1. Scorecard")
D.table(["Model", "Its job", "How it was tested", "Result", "Verdict"],
        [["Poisson GLM", "Predict arrivals per site-hour", "Unseen 2019", f"Deviance {f(GLM['mean_poisson_deviance'])}, D² {f(GLM['D2'])}, total {signed(bias('M2'))}", "Works; runs low"],
         ["NB2 GLM", "Capture variance Poisson misses", "Dispersion tests; 2019 coverage", f"Hourly α = 0; daily α = {f(DM['alpha_summary']['alpha'], 4)}", "Needed for daily totals only"],
         ["Gradient boosting", "Show what a black box can add", "Same test as GLM", f"Deviance {f(HGB['mean_poisson_deviance'])}, total {signed(bias('M4'))}", "Does not beat GLM"],
         ["OLS", "Continuous-variable baseline", "Same test", f"Deviance {f(OLS['mean_poisson_deviance'])}; {pc(R['ols_negative_pred_frac'], 0)} negative forecasts", "Ranks well, invalid as counts"],
         ["Seasonal naive", "Floor to beat", "Same test", f"MAE {f(NAIVE['MAE'])}, busy-hour AUC {f(det['Seasonal naive']['auc_busy_hour'], 2)}", "Beaten by all"],
         ["KDE gap + DBSCAN", "Find the under-supplied zone", "Real full-port share; bandwidth changes", f"Rank corr. {f(rk['spearman_gap_vs_observed_full_share'], 2)}; site moves ≤ {f(max(r['dist_to_nearest_baseline_candidate_m'] for r in sens_dt if 'HAMILTON' in r['sites_pooled']), 0)} m", "Works; zone is broad"],
         ["Queueing simulator", "Reproduce occupancy, test fixes", "Real 2019 occupancy", f"r = {f(min(r['occ_corr'] for r in vo), 3)}–{f(max(r['occ_corr'] for r in vo), 3)}", "Works; weak at Hamilton"]],
        [1.1, 1.45, 1.25, 1.65, 1.05], size=9)

D.h1("2. Forecast Models: Predicting Demand")
D.p(f"The job is to predict how many cars arrive at each site in each hour. All models were fitted on 2017–2018 and scored once on the {n0(R['data']['test_rows'])} site-hours of 2019. "
    f"Lower is better for MAE, RMSE and deviance. D² is the share of the variation explained, and deviance is the fairest single score for counts.", keep_next=True)
rows = []
for key, name, code in [("M0 Seasonal naive", "Seasonal naive", "M0"), ("M1 OLS", "OLS", "M1"), ("M2 Poisson GLM", "Poisson GLM", "M2"), ("M3 NB2 GLM", "NB2 GLM", "M3"), ("M4 HGB-Poisson", "Gradient boosting", "M4")]:
    r = TM[key]; d = det[name]
    rows.append([name, f(r["MAE"]), f(r["RMSE"]), f(r["mean_poisson_deviance"]), f(r["D2"]), signed(bias(code)), f(d["auc_busy_hour"]), pc(d["recall_top_decile"], 0)])
D.table(["Model", "MAE", "RMSE", "Deviance", "D²", "2019 total", "Busy-hour AUC", "Busy hours caught"], rows, [1.5, 0.6, 0.6, 0.75, 0.55, 0.75, 0.75, 0.75], bold_rows={2},
        align=["l", "r", "r", "r", "r", "r", "r", "r"],
        note=f"A busy hour has {A['busy_threshold']}+ arrivals ({pc(g['busy_share'])} of site-hours). AUC measures how well a forecast ranks busy above quiet hours (0.5 is chance, 1 is perfect). "
             f"\"Caught\" is the share of busy hours inside the model's top 10% of forecasts.")
D.p("**What this shows:**", keep_next=True)
D.items([f"**The GLM is the best overall.** It has the lowest deviance and RMSE. Gradient boosting has the lowest MAE, partly because MAE rewards predicting zero in quiet hours, and it predicts too many of those ({pc(HGB['pred_zero_frac'])} against {pc(HGB['obs_zero_frac'])}).",
         f"**Telling busy from quiet hours is not what separates them.** OLS, the GLM and boosting all score about {f(g['auc_busy_hour'], 2)}. Their top 10% of forecasts catch about half of busy hours (precision {pc(g['precision_top_decile'], 0)}), so hour-level surges remain hard to call a day ahead.",
         f"**The difference is the level.** Boosting misses the 2019 total by {f(abs(bias('M4')), 1)}%, because trees cannot go beyond levels seen in training and demand rebounded in 2019. That makes it a poor input for sizing capacity.",
         f"**Calibration holds except at the top.** In the busiest tenth of hours the GLM predicts {f(top['M2']['pred'], 2)} arrivals against {f(top['M2']['obs'], 2)} observed."])
D.figure(FIG / "ieee_fig_diag.png", "Fig. 1. (a) One test week at Cambridge. (b) Calibration by decile. (c) Randomised PIT histograms; flat means well calibrated. (d) Residual autocorrelation.")

D.h2("Site by site")
D.p(f"The forecast ranks the sites correctly on volume (rank correlation {f(rk['spearman_pred_vs_obs_sessions'], 2)}) and tracks each site's daily shape (r = {f(min(r['profile_r'] for r in ps), 2)}–{f(max(r['profile_r'] for r in ps), 2)}). "
    f"It under-predicts every site's total, and every site's peak rate.", keep_next=True)
D.table(["Site", "Observed", "Predicted", "Total error", "Daily MAPE", "Shape r", "Peak hour: real / forecast"],
        [[T(r["site"]), n0(r["observed"]), n0(r["predicted"]), signed(r["bias_pct"]), pc(r["daily_mape_pct"] / 100, 0), f(r["profile_r"], 2), f"{r['peak_hour_observed']:02d}:00 / {r['peak_hour_predicted']:02d}:00"] for r in ps],
        [1.3, 0.8, 0.8, 0.8, 0.8, 0.65, 1.3], align=["l", "r", "r", "r", "r", "r", "r"])
D.p(f"**Weak spot.** The forecast gets the busiest weekday hour right to within an hour at {near_peak} of 8 sites. It misses at {', '.join(missed_peak[:-1])} and {missed_peak[-1]}, where morning and evening peaks are close in size and the model under-predicts the larger one. "
    f"High and Rinconada have the largest total errors, at about 18% low each. Treat site-level peak figures as lower bounds.")

D.h1("3. NB2: Capturing Extra Variance")
D.p("The negative binomial exists to handle counts that vary more than a Poisson model allows. The check is whether that extra variance is present, and whether modelling it gives better prediction intervals.", keep_next=True)
D.table(["Check", "Hourly counts", "Daily totals"],
        [["Pearson χ²/df (1 = Poisson)", f(DS["hourly_pearson_dispersion"], 2), f(DM["alpha_summary"]["pearson_dispersion_poisson"], 2)],
         ["Cameron–Trivedi α (t)", f"{f(DS['hourly_ct']['alpha_ct'], 3)} ({f(DS['hourly_ct']['t'], 1)})", f"{f(DM['ct']['alpha_ct'], 3)} ({f(DM['ct']['t'], 1)})"],
         ["NB2 α", "0 (at the boundary)", f"{f(DM['alpha_summary']['alpha'], 4)} ± {f(DM['alpha_summary']['alpha_se'], 4)} SE"],
         ["2019 90% interval coverage, Poisson", pc(GLM["coverage90"]), pc(DP["coverage90"])],
         ["2019 90% interval coverage, NB2", pc(GLM["coverage90"]), pc(DN["coverage90"])]], [2.8, 1.6, 1.6])
D.p(f"**Verdict.** NB2 does its job only for daily totals, where shared shocks such as weather add up across a day's hours. Hourly, it reduces to Poisson, so the two give identical results. Even with NB2, daily coverage ({pc(DN['coverage90'])}) stays below the 90% target because 2019 demand moved beyond the training range. "
    f"Hourly coverage above 90% is partly an effect of whole-number interval limits.")

D.h1("4. KDE Gap + DBSCAN: Finding the Under-Supplied Zone")
D.p(f"The job is to find where demand outruns ports. To test this without using the forecast, we measured from the real 2019 log how often each site had every port occupied. The method flagged the right end of that list.", keep_next=True)
D.table(["Site", "Ports", "Pred. sessions per port", "Gap at site", "Peak ρ", "All ports busy (real log)", "Mean occupancy (real log)"],
        [[T(s), str(sites[s]["ports_2019"]), n0(sites[s]["pred_sessions_per_port"]), n0(sites[s]["gap_at_site_per_km2"]).replace("-", "−"), f(sites[s]["rho_peak"], 2), pc(full[s]["observed_full_share"]), pc(full[s]["observed_mean_occupancy_share"], 0)] for s in rk["sites_by_observed_full_share"]],
        [1.3, 0.5, 0.85, 0.75, 0.6, 1.0, 1.0], align=["l", "r", "r", "r", "r", "r", "r"],
        note="Sorted by the real all-ports-busy share, measured on weekdays 08:00–20:00. Gap: sessions per km² per year; positive means demand above port supply.")
D.figure(FIG / "fig11_dbscan_candidates.png", f"Fig. 2. (a) k-distance plot used to set ε. (b) DBSCAN cluster, existing sites and the candidate at ({f(cand['lat'], 4)}, {f(cand['lon'], 4)}).", 5.6)
D.items([f"**It finds the right area.** The gap surface matches the real all-ports-busy ranking closely (rank correlation {f(rk['spearman_gap_vs_observed_full_share'], 2)}), with Hamilton ({pc(full['HAMILTON']['observed_full_share'], 0)}) and Webster ({pc(full['WEBSTER']['observed_full_share'], 0)}) at the top.",
         f"**The result is stable.** The candidate moves at most {f(max(r['dist_to_nearest_baseline_candidate_m'] for r in sens_dt if 'HAMILTON' in r['sites_pooled']), 0)} m across bandwidths of 250–600 m and {f(sens_obs['dist_to_nearest_baseline_candidate_m'], 0)} m with observed instead of predicted demand.",
         f"**The zone is broader than the problem.** It includes Bryant ({pc(full['BRYANT']['observed_full_share'])}) and High ({pc(full['HIGH']['observed_full_share'])}), which rarely fill. The pressure is really at Hamilton and Webster.",
         f"**Peak ρ alone is weaker** (rank correlation {f(rk['spearman_rho_vs_observed_full_share'], 2)}) because it ignores site size. Rinconada has a low ρ, yet with 3 ports it has the third-highest busy share.",
         "**Its blind spot:** with only 8 demand points, it measures imbalance between existing sites. It cannot see demand in areas with no charger."])

D.h1("5. Queueing Simulator: Reproducing Occupancy")
D.p("The job is to behave like the real system, so that it can be trusted on scenarios that never happened. The test compares simulated occupancy with occupancy rebuilt from real 2019 sessions.", keep_next=True)
D.table(["Site", "Ports", "Hourly r", "MAE (ports)", "Port-hours/day: real", "Simulated", "Sim. with forecast rates"],
        [[T(a["site"]), str(a["ports"]), f(a["occ_corr"], 3), f(a["occ_MAE"], 2), f(a["daily_port_hours_obs"], 1), f(a["daily_port_hours_sim"], 1), f(one(vm, site=a["site"])["daily_port_hours_sim"], 1)] for a in vo],
        [1.3, 0.5, 0.65, 0.75, 1.1, 0.75, 1.0], align=["l", "r", "r", "r", "r", "r", "r"])
D.figure(FIG / "ieee_fig_sim.png", "Fig. 3. (a) Simulated against observed occupancy by hour, using observed arrival rates. (b) Simulated share of drivers turned away, by site and hour.")
D.p(f"**Verdict.** With observed arrival rates the shapes match closely at all 8 sites. Hamilton is the clear exception on level: {f(hm_o['daily_port_hours_sim'], 1)} simulated port-hours a day against {f(hm_o['daily_port_hours_obs'], 1)} real. "
    f"The simulator treats arrivals as random, and it turns away about {pc(one(SIM['erlang_empirical'], site='HAMILTON')['sim_daily_block'], 0)} of Hamilton's drivers even though every logged session was served, which suggests drivers time their visits to free ports. "
    f"This check validates the simulator, not the fix: pooling has not been tried in the field.")

D.h1("6. The Rinconada Pooling Test")
D.p(f"This test asks whether Rinconada's spare capacity can absorb downtown's shortfall before any ports are built. Rinconada is the closest candidate, but it is {f(RN['distances'][0]['dist_to_nearest_downtown_site_m'] / 1000, 2)} km from the nearest downtown garage, while the downtown garages are at most {n0(RN['downtown_max_spread_m'])} m apart.", keep_next=True)
D.table(["System (observed 2019 rates)", "Ports", "Lost sessions/yr", "Recovered"],
        [["Baseline, 4 downtown sites separate", str(B4["total_ports"]), n0(B4["turned_away_per_year"]), "—"],
         ["4 sites with 4 new overflow ports", str(O4["total_ports"]), n0(O4["turned_away_per_year"]), pc(1 - O4["turned_away_per_year"] / B4["turned_away_per_year"], 0)],
         ["4 sites pooled, no new ports", str(B4["total_ports"]), f"{n0(P4x['lost_per_year'])} ± {n0(P4x['lost_ci95'])}", pc(share4, 0)],
         ["Baseline, 5 sites separate", str(B5["ports"]), n0(B5["lost_per_year"]), "—"],
         ["5 sites pooled with Rinconada, no new ports", str(P50["ports"]), f"{n0(P5x['lost_per_year'])} ± {n0(P5x['lost_ci95'])}", pc(share5, 0)]],
        [3.2, 0.6, 1.2, 0.9], bold_rows={4},
        note=f"Pooled rows use the {n0(P5x['n_runs'])}-run check with its 95% interval; other rows use {SIM['n_reps']} runs. Each \"recovered\" figure compares with the baseline for the same set of sites, because five sites have more drivers to lose.")
D.items([f"**Pooling is the strong result; Rinconada is a small add-on.** Most of the recovery comes from sharing ports among the downtown garages.",
         f"**Rinconada is not an idle reservoir.** Its busiest-hour utilization stays near {pc(P50['rinconada_peak_util'], 0)}, so it is not overloaded, but about {n0(P50['downtown_to_rinconada_per_year'])} downtown drivers a year go there and about {n0(P50['rinconada_to_downtown_per_year'])} of its own drivers go downtown. More flows out than in.",
         "**These gains are upper limits on practical benefit.** The simulation assumes every driver is told about a free port and goes there, including the trip across town."])
D.rule()

D.h1("7. Where the Models Fall Short")
D.table(["Weakness", "Effect", "How to treat it"],
        [["2019 demand under-predicted", f"{f(abs(bias('M2')), 1)}% low overall; about 18% low at High and Rinconada", "Read forecast-based figures as conservative"],
         ["Peaks under-predicted", f"Peak rate low at {n_under} of 8 sites; wrong peak hour at {len(missed_peak)}", "Do not size ports from a single peak forecast"],
         ["Demand is censored", "Drivers who left a full site are never logged", "All demand and loss figures are lower bounds"],
         ["Only 8 demand points", "No view of areas without chargers", "Use the map for imbalance, not new-area demand"],
         ["Random-arrival assumption", "Overstates losses at Hamilton", "Treat \"lost sessions\" as estimates, not counts"],
         ["No field test", "Pooling and redirection are simulated only", "Pilot redirection before relying on it"]], [2.0, 2.4, 2.4])
D.save(OUT)
print("saved", OUT)
