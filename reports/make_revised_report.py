"""Revised report: the team's draft with Section 8 implemented and the Section 9 table filled in.
Tables and new figures come from outputs/results.json; retained prose is the draft's, with corrections."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_style import Report  # noqa: E402

R = json.load(open(HERE.parent / "outputs" / "results.json"))
OUT = HERE / "EV_Charging_Demand_Report_Revised.docx"


def f(x, d=3):
    return "—" if x is None else f"{x:,.{d}f}".replace("-", "−")


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def n0(x):
    return f"{round(x):,}"


def small(x):
    return f"{x:.1f}" if x < 10 else n0(x)


def one(lst, **kw):
    return [r for r in lst if all(r.get(k) == v for k, v in kw.items())][0]


TM = {r["index"]: r for r in R["test_metrics"]}
ANN = R["annual_2019"]
bias = lambda k: f"{100 * (ANN[k] / ANN['observed'] - 1):+.1f}%".replace("-", "−")  # noqa: E731
SIM, RN = R["simulation"], R["rinconada"]
sc_m, sc_o = SIM["scenarios"], SIM["scenarios_empirical"]
P_ = RN["scenarios"]
pm = lambda s: one(P_, rates="model", scenario=s)  # noqa: E731
po = lambda s: one(P_, rates="observed", scenario=s)  # noqa: E731
prec = {(r["rates"], r["system"]): r for r in RN["precision"]}
dist = {r["site"]: r for r in RN["distances"]}
full = {r["site"]: r for r in RN["observed_full_share"]}
sites = {r["site"]: r for r in R["spatial"]["sites"]}
IRR = {r["term"]: r for r in R["irr"]}
CON = R["irr_contrasts"]
fee = {r["site"]: r for r in CON if r["kind"] == "fee"}
h9 = one(CON, kind="hour09", site="BRYANT")
sat, sun = [r for r in CON if r["kind"] == "weekend09"]
DS, DM = R["dispersion"], R["daily_models"]
DMP, DMN = DM["metrics"]["Daily Poisson"], DM["metrics"]["Daily NB2"]
B4 = one(sc_o, scenario="Baseline (separate sites)")
B4m = one(sc_m, scenario="Baseline (separate sites)")
P40 = one(sc_o, scenario="Pooled, no new ports")
P40m = one(sc_m, scenario="Pooled, no new ports")
B5, P50 = po("Baseline, 5 sites separate"), po("Pooled +0, with Rinconada")
P4x, P5x = prec[("observed", "4 sites pooled")], prec[("observed", "5 sites pooled")]
R_ALONE_LOST = B5["lost_rinconada_drivers"]
share4 = 1 - P4x["lost_per_year"] / B4["turned_away_per_year"]
share5 = 1 - P5x["lost_per_year"] / B5["lost_per_year"]
n_b_o = sum(1 for a in SIM["erlang_empirical"] if a["erlang_B"] > a["sim_block_at_peak_arrival_hour"])
n_b_m = sum(1 for a in SIM["erlang"] if a["erlang_B"] > a["sim_block_at_peak_arrival_hour"])
vo = SIM["validation_empirical"]

D = Report("EV Charging Demand Report - Revised")
D.title("Predictive Spatiotemporal Demand Modeling for Optimal Electric Vehicle Charging Station Placement",
        "Prathamesh Patil (9) · Srijan Bera (17) · Snehal Tyagi (24)")

D.h1("Abstract (Revised)")
D.p(f"Public EV chargers in Palo Alto were placed where land was available, not where demand is highest. Using "
    f"{n0(R['data']['study_sessions'])} charging sessions (2017–2019) across 8 sites, we forecast hourly charging demand "
    f"with a Poisson regression model, identify one undersupplied zone (downtown) and one lightly used nearby station "
    f"(Rinconada Library), and test two low-cost fixes before recommending new infrastructure: pooling spare capacity, "
    f"and adding overflow ports only for the demand pooling cannot absorb. The Poisson model explains {pc(TM['M3 NB2 GLM']['D2'], 0)} "
    f"of hourly deviance on an unseen year and outperforms gradient boosting on deviance, RMSE and likelihood, while "
    f"remaining interpretable enough for planners to act on directly. In simulation, pooling the four downtown garages "
    f"recovers about {pc(share4, 0)} of lost weekday sessions; adding Rinconada raises this to about {pc(share5, 0)}, which "
    f"leaves little for new ports to do.")
D.rule()

D.h1("1. Problem Statement")
D.p("EV chargers in Palo Alto are overcrowded in some areas and sitting idle in others, because stations were sited by "
    "land availability rather than by where drivers actually need them. This project builds a model that predicts hourly "
    "charging demand at each existing station, uses that forecast to find where demand exceeds supply and where capacity "
    "is going unused, and tests two fixes — rerouting drivers to idle capacity, and adding new ports only where rerouting "
    "isn't enough — before recommending a final solution.")
D.p("Keep this as the one sentence you say out loud if asked \"what problem are you solving\": *stations are mismatched to "
    "demand, and we fix the mismatch before building anything new.*")
D.rule()

D.h1("2. Introduction")
D.p("EV adoption is outpacing planned charging infrastructure. Stations are typically sited by land cost or developer "
    "convenience, not by usage data, so some garages see long waits while others sit mostly empty. This creates two separate "
    "but related problems: unmet demand in hot zones, and wasted capacity in cold zones. Most existing work treats these as "
    "one problem solved by building more stations. We treat them as two problems with two different fixes — redistribute "
    "existing capacity first, build new capacity second — using the City of Palo Alto's own session logs rather than "
    "synthetic or survey-based demand data.")
D.p("**What the project contributes:**", keep_next=True)
D.items(["An interpretable count-regression model of hourly charging demand, validated on a full unseen year.",
         "A demand–supply gap map that identifies the undersupplied zone (downtown), read alongside port occupancy to find lightly used stations (Rinconada Library).",
         "A test of whether pooling idle capacity at Rinconada into the downtown cluster reduces turned-away drivers *before* any new ports are built.",
         "A queueing simulation, checked against real occupancy data, that quantifies how much pooling alone fixes versus how much still needs new infrastructure."], numbered=True)
D.rule()

D.h1("3. Literature Review")
D.p("Most EV charging-demand research falls into three camps: (a) deep-learning forecasters (LSTM, GRU, Transformer, graph "
    "networks) that predict load but don't recommend placement; (b) GIS/multi-criteria siting methods (AHP, TOPSIS, "
    "DBSCAN-based clustering) that rank candidate sites on static criteria rather than live demand forecasts; and (c) "
    "session/energy-level ensemble forecasters that perform well but aren't tied to a siting decision at all. A smaller body "
    "of work studies spatial imbalance directly (e.g., supply–demand mismatch across Chinese megacities, grid-aware siting "
    "frameworks), but typically stops at identifying the mismatch rather than testing a fix.")
D.table(["Study type", "Example", "Forecasts demand?", "Recommends siting?", "Validates against real outcomes?"],
        [["Deep learning (Transformer/GCN)", "[1], [11], [12]", "Yes", "No", "No"],
         ["GIS / multi-criteria siting", "AHP/TOPSIS/DBSCAN studies", "No (static criteria)", "Yes", "No"],
         ["Ensemble session/energy forecasting", "[3]–[5]", "Yes", "No", "No"],
         ["Spatiotemporal imbalance studies", "China megacity study, [10]", "Yes", "Partial", "No"],
         ["This work", "—", "Yes", "Yes", "Partly: simulator checked against observed occupancy; the fix itself is simulated, not field-tested"]],
        [1.7, 1.3, 0.9, 0.9, 1.5], bold_rows={4})
D.p("The gap this table makes visible: almost nobody closes the loop by testing whether the recommended fix actually works "
    "against real operational data, and almost nobody checks whether existing idle capacity could solve part of the problem "
    "before recommending new construction.")
D.rule()

D.h1("4. Research Gaps and Novelty")
D.p("**Primary novelty claim (lead with this):** Unlike prior work, which stops at forecasting demand or at ranking candidate "
    "sites on static criteria, this project closes the loop — it forecasts demand, identifies both oversupplied and "
    "undersupplied stations, tests a zero-construction fix (pooling idle capacity), and evaluates the remaining recommendation "
    "with a queueing simulation that has been checked against real occupancy.")
D.p("**Supporting gaps from the literature:**", keep_next=True)
D.items(["*Interpretability–accuracy tradeoff:* black-box models (CNN/Transformer/GCN) are hard for municipal planners to trust or justify decisions from.",
         "*Forecasting and placement are treated separately:* forecasting papers don't propose placement; siting papers use static criteria instead of live forecasts.",
         "*Session counts are discrete events*, often modeled with continuous regression when count models (Poisson/NB2) are the statistically correct choice.",
         "*Idle capacity is rarely considered as a solution.* Siting literature moves straight from \"demand exceeds supply here\" to \"build here,\" without checking whether underused capacity elsewhere could absorb the shortfall first."], numbered=True)
D.p("Gap 4 is the one this revision adds, and it is the strongest and least-covered of the four — most reviewers will not "
    "have seen a paper that tests reallocation before recommending construction.")
D.rule()

D.h1("5. Dataset and Study Area (condensed)")
D.p(f"City of Palo Alto charging session records, {n0(R['data']['raw_rows'])} raw rows (2011–2020). After cleaning (duplicates, "
    f"excluded stations, bad timestamps, implausible durations, Level-1 sessions, out-of-window rows), "
    f"{n0(R['data']['study_sessions'])} sessions remain across 8 sites for 2017–2019: Bryant, Cambridge, Hamilton, High, MPL, "
    f"Rinconada Library, Ted Thompson, Webster. Active-port count per site-hour is used as model exposure. Training uses "
    f"2017–2018; 2019 is held out for a single final evaluation.")
D.rule()

D.h1("6. Methodology")
D.h2("6.1 Demand Forecasting Model (simplified)")
D.p("Charging sessions are counts, not continuous values, so a **Poisson GLM with an exposure offset** (log of active ports) "
    "is the core model. Features: hour of day, day of week, weekend×hour, month, US federal holiday flag, fee regime, a "
    "linear trend (held at its last training value when forecasting), site effects, site×hour (06:00–23:00) and "
    "site×fee terms, and three lag terms (24h, 168h, 7-day rolling mean) so the 2019 forecast never looks at future data.")
D.equation(r"\log \mu_{st} = \log c_{st} + \mathbf{x}_{st}^\top \boldsymbol{\beta}")
D.p("Here $c_{st}$ is active ports (exposure) and $\\mathbf{x}_{st}$ the feature vector. The model maximizes the Poisson log-likelihood")
D.equation(r"\ell_P(\boldsymbol{\beta}) = \sum_{s,t} \left[ y_{st}\log\mu_{st} - \mu_{st} - \log(y_{st}!) \right]")
D.p(f"Dispersion is checked against the NB2 variance function $\\mathrm{{Var}}(Y)=\\mu+\\alpha\\mu^2$ via the Cameron–Trivedi "
    f"test; hourly α is approximately zero (Poisson suffices), daily α = {f(DM['alpha_summary']['alpha'], 4)} (NB2 needed).")
D.p("Two supporting checks, not two more full models:", keep_next=True)
D.items(["**Dispersion check:** hourly counts are *not* overdispersed (Poisson is sufficient); only *daily totals* need the negative binomial (NB2) version, because weather/events create shared shocks across hours of the same day.",
         f"**Black-box comparison:** gradient boosting is run once as a sanity check that the interpretable GLM isn't sacrificing accuracy. The GLM wins on the unseen year (lower deviance, doesn't undercount total 2019 demand by {bias('M4').lstrip(chr(0x2212))} the way boosting does)."])
D.p(f"Drop OLS and the no-lag ablation from the main narrative — keep them only as one-line footnotes if asked (\"a "
    f"continuous-variable baseline gives negative forecasts in {pc(R['ols_negative_pred_frac'], 0)} of hours, confirming count models are necessary\").")
D.h2("6.2 Spatial Gap Analysis and Candidate Siting")
D.p("From the 2019 forecasts, each site gets predicted demand and peak-hour utilization ρ (offered load per port). Demand and "
    "port-supply are each turned into a smoothed surface (Gaussian kernel density, 400m bandwidth) over a grid around the sites; "
    "subtracting supply from demand gives a gap surface. Cells with the highest gap are clustered with DBSCAN to find the "
    "undersupplied zone (downtown, centered near Hamilton/Webster/Bryant) without having to pre-specify how many zones should exist.")
D.h2("6.3 Underused Station Analysis: Pooling Rinconada Library")
dR, dT, dM = dist["RINCONADA LIB"], dist["TED THOMPSON"], dist["MPL"]
D.p(f"This is the new analysis. From the peak-utilization column (ρ), the two clearly underused stations are MPL "
    f"(ρ={f(sites['MPL']['rho_peak'], 2)}) and Rinconada Library (ρ={f(sites['RINCONADA LIB']['rho_peak'], 2)}), with Ted Thompson "
    f"(ρ={f(sites['TED THOMPSON']['rho_peak'], 2)}) a soft third. **Rinconada is the closest of these to the undersupplied downtown "
    f"cluster:** {f(dR['dist_to_nearest_downtown_site_m'] / 1000, 2)} km in a straight line from the nearest downtown garage, against "
    f"{f(dT['dist_to_nearest_downtown_site_m'] / 1000, 2)} km for Ted Thompson and {f(dM['dist_to_nearest_downtown_site_m'] / 1000, 2)} km for MPL. "
    f"That makes it the natural pooling candidate, but it is a short drive rather than a neighbour: the four downtown garages are at "
    f"most {n0(RN['downtown_max_spread_m'])} m apart. MPL is too far from every other station to pool with, so it is excluded and "
    f"flagged separately (see Limitations).")
D.p(f"Rinconada's low use needs one qualification. Its average daytime occupancy is low ({pc(full['RINCONADA LIB']['observed_mean_occupancy_share'], 0)}, "
    f"second lowest of the 8 sites), but with only {sites['RINCONADA LIB']['ports_2019']} ports all of them are busy for "
    f"{pc(full['RINCONADA LIB']['observed_full_share'])} of weekday daytime in the real 2019 log, the third highest figure of the 8 sites.")
D.p(f"**The test:** does adding Rinconada's {sites['RINCONADA LIB']['ports_2019']} ports into the downtown pool reduce turned-away drivers, before any new ports are built?")
D.h2("6.4 Queueing Simulation")
D.p("Each site/cluster is simulated as a loss system for a typical 2019 weekday: arrivals follow a non-homogeneous Poisson process "
    "at the forecast hourly rate, plug-in durations are resampled from that site's actual 2019 weekday sessions, and a driver who "
    "finds every port busy leaves (lost demand). The simulator is checked against real 2019 occupancy before being trusted for the "
    "new scenario. The original paper already ran this for the downtown cluster (Bryant, Hamilton, High, Webster): baseline, +2 and +4 "
    "overflow ports, and full pooling with 0, 2 or 4 new ports. **This revision adds Rinconada into the pooled scenarios.**")
D.h2("6.5 Model Validation (Original Fit Statistics)")
D.p("For reference, this is how the forecasting model performed against the 2019 held-out year, unchanged by the pooling revision "
    "(the demand model itself is not refit — only the simulation in Sections 8 and 9 uses it differently):", keep_next=True)


def mrow(name, key, code):
    r = TM[key]
    return [name, f(r["MAE"]), f(r["RMSE"]), f(r["mean_poisson_deviance"]), f(r["D2"]), pc(r["coverage90"]), bias(code)]


D.table(["Model", "MAE", "RMSE", "Deviance", "D²", "90% coverage", "2019 bias"],
        [mrow("Seasonal naive", "M0 Seasonal naive", "M0"), mrow("Poisson GLM (main model)", "M2 Poisson GLM", "M2"),
         mrow("Gradient boosting", "M4 HGB-Poisson", "M4")], [2.0, 0.6, 0.6, 0.75, 0.55, 0.9, 0.75])
D.p(f"The GLM is preferred over boosting despite a slightly higher MAE: boosting undercounts the full 2019 total by "
    f"{bias('M4').lstrip(chr(0x2212))} because it cannot forecast beyond demand levels seen in training, while the GLM's deviance and "
    f"calibration are better and its coefficients stay interpretable for planners.")
D.rule()

D.h1("7. Results and Discussion (Original Study Findings)")
D.p("These are the results already established in the intermediate report, carried forward except where noted. They sit here, before "
    "the new Rinconada rerun, because the rerun builds directly on them.")
ctH, ctD = DS["hourly_ct"], DM["ct"]
D.p(f"**Dispersion.** {pc(R['data']['zero_fraction_all'])} of site-hours have zero arrivals. Hourly counts are not overdispersed "
    f"(Pearson χ²/df = {f(DS['hourly_pearson_dispersion'], 2)}, Cameron–Trivedi α = {f(ctH['alpha_ct'], 3)}, t = {f(ctH['t'], 1)}), so "
    f"the NB2 likelihood is maximized at α = 0 and reduces to the plain Poisson model. Daily totals behave differently — weather and "
    f"local events create shared shocks across a day's hours — so the daily NB2 estimate (α = {f(DM['alpha_summary']['alpha'], 4)}, "
    f"likelihood-ratio stat {f(DM['alpha_summary']['LR_stat'], 1)}) is the one case where the heavier model is justified, raising 90% "
    f"interval coverage from {pc(DMP['coverage90'])} to {pc(DMN['coverage90'])}.")
D.p("**Forecast accuracy on the 2019 held-out year:**", keep_next=True)


def mrow2(name, key, code):
    r = TM[key]
    return [name, f(r["MAE"]), f(r["RMSE"]), f(r["mean_poisson_deviance"]), f(r["D2"]), bias(code)]


D.table(["Model", "MAE", "RMSE", "Deviance", "D²", "2019 bias"],
        [mrow2("Seasonal naive", "M0 Seasonal naive", "M0"), mrow2("OLS (continuous baseline)", "M1 OLS", "M1"),
         mrow2("Poisson GLM (chosen model)", "M2 Poisson GLM", "M2"), mrow2("NB2 GLM", "M3 NB2 GLM", "M3"),
         mrow2("Gradient boosting", "M4 HGB-Poisson", "M4")], [2.3, 0.7, 0.7, 0.8, 0.7, 0.8], bold_rows={2})
D.p(f"Gradient boosting wins on raw MAE only because it over-predicts zero hours ({pc(TM['M4 HGB-Poisson']['pred_zero_frac'])} vs. "
    f"{pc(TM['M4 HGB-Poisson']['obs_zero_frac'])} actual) and undercounts the 2019 total by {bias('M4').lstrip(chr(0x2212))} — the GLM is the "
    f"more trustworthy model for planning, not just the more interpretable one.")
D.p(f"**What drives demand (selected rate ratios, 95% CI, cluster-robust by site-day):** a weekday port at Bryant sees about "
    f"{f(h9['IRR'], 0)}× as many arrivals at 09:00–10:00 as at midnight ({f(h9['CI_low'], 0)}–{f(h9['CI_high'], 0)}×). Saturdays run "
    f"{pc(1 - sat['IRR'], 0)} below Monday at the same hour, Sundays {pc(1 - sun['IRR'], 0)} below, and a federal holiday cuts the rate to "
    f"{f(IRR['holiday']['IRR'], 2)}× normal. Recent same-hour demand matters: a doubling of the trailing 7-day average raises the expected rate by "
    f"{f(2 ** IRR['log1p_roll7']['coef'], 2)}×. The fee introduction cut arrivals per port by {pc(1 - fee['TED THOMPSON']['IRR'], 0)} at Ted Thompson "
    f"and {pc(1 - fee['RINCONADA LIB']['IRR'], 0)} at Rinconada Library, but did not measurably change Hamilton (IRR {f(fee['HAMILTON']['IRR'], 2)}, CI "
    f"{f(fee['HAMILTON']['CI_low'], 2)}–{f(fee['HAMILTON']['CI_high'], 2)}) — Hamilton was already running near capacity, so a demand drop there "
    f"doesn't show up in recorded arrivals.")
D.p("**Demand and utilization by site (2019):**", keep_next=True)
D.table(["Site", "Ports", "Predicted sessions", "Pred./port", "Peak ρ"],
        [[("MPL" if s == "MPL" else s.title().replace(" Lib", " Lib")), str(r["ports_2019"]), n0(r["pred_sessions_2019"]), n0(r["pred_sessions_per_port"]), f(r["rho_peak"], 2)]
         for s, r in sites.items()], [1.7, 0.7, 1.5, 1.0, 0.8])
D.p("This table is the basis for both the downtown gap cluster (Hamilton/Webster run hottest) and the underused-station analysis (MPL and Rinconada run coldest).")
D.p(f"**Simulator validation.** Before trusting the queueing simulation for any new scenario, it was checked against real 2019 occupancy: "
    f"simulated hourly occupancy matches observed occupancy closely at every site (r = {f(min(r['occ_corr'] for r in vo), 3)}–{f(max(r['occ_corr'] for r in vo), 3)}). "
    f"Erlang-B theory overstates blocking relative to the simulation at all {n_b_o} sites with observed rates ({n_b_m} of 8 with forecast rates), because cars arriving "
    f"at the peak hour stay roughly two hours and congestion builds through the morning rather than resetting hour to hour — confirming the "
    f"simulator, not the simple formula, is the right tool for the downtown and Rinconada scenarios.")
D.p("**Original downtown pooling/overflow results (pre-Rinconada), weekdays only:**", keep_next=True)
order = [("Baseline (separate sites)", "Baseline"), ("Overflow to 2 new ports", "Overflow, +2"), ("Overflow to 4 new ports", "Overflow, +4"),
         ("Pooled, no new ports", "Pooled, +0"), ("Pooled + 2 new ports", "Pooled, +2"), ("Pooled + 4 new ports", "Pooled, +4")]
D.table(["Scenario", "Ports", "Lost/yr (model)", "Lost/yr (observed)", "Peak block"],
        [[lab, str(one(sc_o, scenario=n)["total_ports"]), n0(one(sc_m, scenario=n)["turned_away_per_year"]), n0(one(sc_o, scenario=n)["turned_away_per_year"]),
          pc(one(sc_o, scenario=n)["peak_block_prob"])] for n, lab in order], [1.6, 0.7, 1.3, 1.5, 1.0])
D.p("This is the table Section 9 below extends: the only change is adding Rinconada's 3 ports into the pool set and rerunning the pooled rows.")
D.rule()

D.h1("8. What Was Rerun")
D.p("**Objective:** determine how much of the downtown shortfall is absorbed by adding Rinconada Library's 3 ports into the \"full pooling\" "
    "simulation, with no new hardware, and compare this with the baseline and the original four-site pool. The code is in `src/pooling.py`.".replace("`", ""))
D.p("**Step 1: adjacency.** Straight-line distances from the sites outside the downtown cluster:", keep_next=True)
D.table(["Site", "Ports", "Peak ρ", "To nearest downtown garage", "To cluster centroid", "To DBSCAN candidate"],
        [[("MPL" if r["site"] == "MPL" else r["site"].title()), str(r["ports"]), f(r["rho_peak"], 2), f"{f(r['dist_to_nearest_downtown_site_m'] / 1000, 2)} km",
          f"{f(r['dist_to_downtown_centroid_m'] / 1000, 2)} km", f"{f(r['dist_to_candidate_m'] / 1000, 2)} km"] for r in RN["distances"]],
        [1.3, 0.6, 0.6, 1.5, 1.2, 1.2])
D.p("**Step 2: pool set.** The pooled scenarios now cover {Bryant, Hamilton, High, Webster, Rinconada}, with Rinconada's own 2019 weekday forecast and "
    "resampled plug-in durations. Each port stays attached to its own site: a driver tries the own site first, then the other sites from nearest to farthest, "
    "and is lost only if every port in the system is busy. Total losses therefore follow the same law as one shared pool, and the simulation can also report "
    "where each session ends up, including Rinconada's own utilization.")
D.p(f"**Step 3: run settings.** Same as the original: two simulated days per run (first discarded), {SIM['n_reps']} runs, seed {R['meta']['seed']}, scaled to "
    f"{SIM['n_weekdays_2019']} weekdays/year. Both forecast (\"model\") and observed 2019 arrival rates are used.")
D.p(f"**Added check.** At {SIM['n_reps']} runs, loss counts this small are noisy, so the key comparison was repeated with {n0(P4x['n_runs'])} runs. A five-site "
    f"baseline row was also added: the original baseline of {n0(B4['turned_away_per_year'])} covers four sites, while the pooled rows include Rinconada's drivers too.")
D.rule()

D.h1("9. Rerun Results")
rows = [["Baseline, 4 sites (original)", "26", n0(B4m["turned_away_per_year"]), n0(B4["turned_away_per_year"]), pc(B4["peak_block_prob"]), "—"],
        ["Baseline, 5 sites separate", "29", n0(pm("Baseline, 5 sites separate")["lost_per_year"]), n0(B5["lost_per_year"]), pc(B5["peak_block"]), pc(B5["rinconada_peak_util"], 0)],
        ["Pooled, +0 (original, no Rinconada)", "26", n0(P40m["turned_away_per_year"]), n0(P40["turned_away_per_year"]), pc(P40["peak_block_prob"]), "—"]]
for k in (0, 2, 4):
    s = f"Pooled +{k}, with Rinconada"
    rows.append([f"Pooled, +{k}, with Rinconada", str(po(s)["ports"]), small(pm(s)["lost_per_year"]),
                 f"{n0(po(s)['lost_per_year'])} ± {n0(po(s)['lost_ci95']) if po(s)['lost_ci95'] >= 1 else f(po(s)['lost_ci95'], 1)}", pc(po(s)["peak_block"]), pc(po(s)["rinconada_peak_util"], 0)])
D.table(["Scenario", "Ports", "Lost/yr (model)", "Lost/yr (observed)", "Peak block %", "Rinconada utilization"], rows, [2.3, 0.5, 0.75, 1.05, 0.7, 0.95],
        bold_rows={3, 4, 5},
        note=f"Weekdays only, {SIM['n_reps']} runs. ± is the 95% Monte Carlo interval. Peak block uses observed rates. Rinconada utilization: occupied share of its "
             f"3 ports in its busiest hour, observed rates; the 5-site baseline shows Rinconada operating alone. With forecast rates the +0, +2 and +4 losses are "
             f"too small to tell apart at this run count.")
D.p(f"**Headline.** Pooling works, but most of the gain comes from sharing ports among the four downtown garages rather than from Rinconada. With {n0(P4x['n_runs'])} "
    f"runs and observed rates, the four-site pool loses {n0(P4x['lost_per_year'])} ± {n0(P4x['lost_ci95'])} sessions a year ({pc(share4, 0)} recovered from the "
    f"{n0(B4['turned_away_per_year'])} baseline). The five-site pool loses {n0(P5x['lost_per_year'])} ± {n0(P5x['lost_ci95'])} ({pc(share5, 0)} recovered from the "
    f"{n0(B5['lost_per_year'])} five-site baseline). Adding Rinconada is a real improvement, worth about {n0(P4x['lost_per_year'] + R_ALONE_LOST - P5x['lost_per_year'])} sessions "
    f"a year against keeping it separate, but the zero-construction result is mostly a downtown result.")
D.table(["System (observed rates)", "Runs", "Lost/yr, all drivers", "Lost/yr, downtown drivers", "Recovered from baseline"],
        [["4 downtown sites pooled", n0(P4x["n_runs"]), f"{n0(P4x['lost_per_year'])} ± {n0(P4x['lost_ci95'])}", f"{n0(P4x['lost_downtown_drivers'])} ± {n0(P4x['lost_downtown_ci95'])}", pc(share4)],
         ["4 sites + Rinconada pooled", n0(P5x["n_runs"]), f"{n0(P5x['lost_per_year'])} ± {n0(P5x['lost_ci95'])}", f"{n0(P5x['lost_downtown_drivers'])} ± {n0(P5x['lost_downtown_ci95'])}", pc(share5)]],
        [2.0, 0.6, 1.2, 1.3, 1.1],
        note=f"This run gives {n0(P4x['lost_per_year'])} for the four-site pool, so the {n0(P40['turned_away_per_year'])} in the original table (and the earlier paper) was a low draw "
             f"from {SIM['n_reps']} runs; both figures are within that run's uncertainty.")
D.p(f"**Is Rinconada overloaded? No.** Its peak-hour utilization moves from {pc(B5['rinconada_peak_util'], 0)} alone to {pc(P50['rinconada_peak_util'], 0)} when pooled, and the share of "
    f"daytime with all 3 ports busy from {pc(B5['rinconada_full_share_8_20'])} to {pc(P50['rinconada_full_share_8_20'])}. The utilization barely moves because the flows are small and run both ways: "
    f"about {n0(P50['downtown_to_rinconada_per_year'])} downtown drivers a year go to Rinconada, while about {n0(P50['rinconada_to_downtown_per_year'])} Rinconada drivers go downtown when its ports are full.")
P52, P54 = po("Pooled +2, with Rinconada"), po("Pooled +4, with Rinconada")
D.p(f"**What is left for new ports?** Little: {n0(P50['lost_per_year'])} lost sessions a year with no new ports, about {n0(P52['lost_per_year'])} with 2 and about {n0(P54['lost_per_year'])} with 4, "
    f"at this run count. If redirection cannot be made to work, overflow ports remain the fallback (4 ports: {n0(one(sc_o, scenario='Overflow to 4 new ports')['turned_away_per_year'])} lost a year).")
D.rule()

D.h1("10. Limitations")
D.items(["**One city.** Results reflect Palo Alto's 8 sites and may not transfer elsewhere.",
         "**Censored demand.** Drivers who found a full site and left are never logged, so true demand, fee sensitivity, and simulated losses are all underestimates.",
         f"**MPL cannot be pooled.** It is {f(dM['dist_to_nearest_downtown_site_m'] / 1000, 1)} km from the nearest downtown garage, so low utilization there needs a different fix (pricing, demand-side promotion, or reassessing whether it should remain a Level-2 site) rather than network pooling — flagged honestly rather than forced into the pooling framework.",
         f"**Level shift.** 2019 demand was under-predicted by {bias('M3').lstrip(chr(0x2212))} overall by the model, so simulated \"model-rate\" figures are conservative relative to observed-rate figures.",
         f"**No real-time routing infrastructure assumed, and full compliance.** The pooling scenario assumes every driver is told about availability at another site and goes there, including the {f(dR['dist_to_nearest_downtown_site_m'] / 1000, 2)} km trip to or from Rinconada. Implementing this (app/signage) is out of scope here but necessary for the result to be realized in practice; partial compliance would reduce the gains.",
         "**The \"observed\" columns are still simulated.** They use observed arrival rates but treat arrivals as random. At Hamilton this overstates losses: the simulator blocks many drivers there although every logged session was served, which suggests drivers time their visits to free ports."], numbered=True)
D.rule()

D.h1("11. Conclusion and Future Work")
D.p(f"The revised pipeline forecasts hourly demand with an interpretable Poisson model, locates the undersupplied downtown zone and the lightly used Rinconada Library station, and tests "
    f"whether pooling reduces lost sessions before recommending new ports. In simulation it does: sharing ports among the four downtown garages recovers about {pc(share4, 0)} of lost weekday "
    f"sessions and adding Rinconada brings this to about {pc(share5, 0)}. This reframes the recommendation from \"build N new ports\" to \"pool existing capacity first, then build only the "
    f"remaining gap\" — a cheaper, more defensible planning conclusion, provided drivers can actually be redirected. Future work: extend pooling analysis to Ted Thompson/Cambridge, model "
    f"censored demand directly, incorporate real-time routing feasibility, and repeat the study in a second city to test transferability.")
D.rule()

D.h1("Appendix: Selected References")
D.items(["Koohfar, Woldemariam & Kumar, \"Prediction of Electric Vehicles Charging Demand: A Transformer-Based Deep Learning Approach,\" Sustainability, 2023.",
         "Marzbani, Osman & Hassan, \"Electric Vehicle Energy Demand Prediction Techniques: An In-Depth and Critical Systematic Review,\" IEEE Access, 2023.",
         "Akshay et al., \"Power Consumption Prediction for Electric Vehicle Charging Stations and Forecasting Income,\" Scientific Reports, 2024.",
         "Akshayaa, Sajidha & Radha, \"Session and Energy Forecasting for Electric Vehicle Charging Station Using Custom Weighted Ensemble Machine Learning,\" Results in Engineering, 2025.",
         "Li et al., \"Data-Driven Planning of Electric Vehicle Charging Infrastructure: A Case Study of Sydney, Australia,\" IEEE Trans. Smart Grid, 2021.",
         "Ester, Kriegel, Sander & Xu, \"A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise\" (DBSCAN), KDD, 1996.",
         "Cameron & Trivedi, \"Regression-Based Tests for Overdispersion in the Poisson Model,\" Journal of Econometrics, 1990.",
         "City of Palo Alto, \"Electric Vehicle Charging Station Usage, July 2011–December 2020,\" City of Palo Alto Open Data.",
         "Gross, Shortle, Thompson & Harris, *Fundamentals of Queueing Theory*, 4th ed., Wiley, 2008."], numbered=True)
D.p("*(Full bibliography carries over unchanged from the intermediate report; abbreviated here for space.)*")
D.save(OUT)
print("saved", OUT)
