"""End-to-end pipeline: data -> EDA -> models -> spatial -> simulation -> figures -> results.json.

Usage:  python run_all.py              (everything)
        python run_all.py --reuse-models   (skip model fitting, reuse outputs/cache/models.pkl)
"""
import argparse
import json
import pickle
import platform
import random
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
import scipy
import sklearn
import statsmodels
from pandas.tseries.holiday import USFederalHolidayCalendar

from src import config as C
from src import data as D
from src import features as F
from src import models as M
from src import spatial as SP
from src import simulate as SIM
from src import figures as FIG
from src import pooling as POOL
from src import audit as AUD

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
random.seed(C.SEED)
np.random.seed(C.SEED)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o is None or not np.isfinite(o)) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return jsonable(o.reset_index().to_dict(orient="records"))
    return o


def main(reuse_models=False):
    R = {"meta": {"seed": C.SEED, "python": platform.python_version(), "pandas": pd.__version__,
                  "numpy": np.__version__, "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
                  "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                  "split": {"train": "2017-01-01 to 2018-12-31", "validation": "2018-10-01 to 2018-12-31 (tuning only)",
                            "test": "2019-01-01 to 2019-12-31"}}}

    # ------------------------------------------------------------------ 1. data
    log("Loading and cleaning data")
    raw = D.load_raw()
    sessions, clean_log = D.clean(raw)
    study = D.study_window(sessions)
    clean_log.loc[len(clean_log)] = ["Outside study window (keep 2017-01-01 to 2019-12-31)",
                                     len(sessions) - len(study), len(study)]
    clean_log.to_csv(C.TAB / "cleaning_log.csv", index=False)
    fee_start = D.fee_start_date(sessions)
    act = D.port_activity(sessions)
    hours = pd.date_range(C.PANEL_START, C.STUDY_END, freq="h")
    ports = D.active_ports_panel(act, hours)
    coords = D.site_coordinates(sessions)
    panel = F.build_panel(sessions, ports, fee_start)
    p = F.model_frame(panel)
    s2019 = study[study["start"].dt.year == 2019]
    hol = USFederalHolidayCalendar().holidays(start="2019-01-01", end="2019-12-31")
    s2019_wk = s2019[(s2019["start"].dt.dayofweek < 5) & (~s2019["start"].dt.normalize().isin(hol))]
    n_weekdays_2019 = int(sum((d.dayofweek < 5) and (d not in hol) for d in pd.date_range("2019-01-01", "2019-12-31")))

    raw_start = pd.to_datetime(raw["Start Date"], format="%m/%d/%Y %H:%M", errors="coerce")
    fee_q = study.assign(q=study["start"].dt.to_period("Q")).groupby("q")["fee"].apply(lambda s: (s > 0).mean())
    R["data"] = {
        "raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1]),
        "raw_first": str(raw_start.min()), "raw_last": str(raw_start.max()),
        "clean_rows_all_years": int(len(sessions)), "study_sessions": int(len(study)),
        "n_stations": int(study["Station Name"].nunique()), "n_l2_ports_2019": int(ports.loc[pd.Timestamp("2019-06-01 12:00")].sum()),
        "n_sites": len(C.SITES), "fee_start": str(fee_start.date()),
        "fee_share_2017Q3": float(fee_q.loc[pd.Period("2017Q3")]), "fee_share_2017Q4": float(fee_q.loc[pd.Period("2017Q4")]),
        "fee_share_2019": float((s2019["fee"] > 0).mean()),
        "median_fee_2019_usd": float(s2019.loc[s2019["fee"] > 0, "fee"].median()),
        "panel_rows": int(len(p)), "train_rows": int((p["split"] == "train").sum()),
        "test_rows": int((p["split"] == "test").sum()), "val_rows": int(p["is_val"].sum()),
        "zero_fraction_all": float((p["y"] == 0).mean()),
        "zero_fraction_train": float((p.loc[p["split"] == "train", "y"] == 0).mean()),
        "mean_hourly_count": float(p["y"].mean()), "var_hourly_count": float(p["y"].var()),
        "max_hourly_count": int(p["y"].max()),
        "mean_duration_h": float(study["duration_h"].mean()), "median_duration_h": float(study["duration_h"].median()),
        "mean_charging_h": float(study["charging_h"].mean()), "mean_energy_kwh": float(study["energy_kwh"].mean()),
        "total_energy_mwh": float(study["energy_kwh"].sum() / 1000), "n_users": int(study["User ID"].nunique()),
        "n_weekdays_2019": n_weekdays_2019,
        "outage_intervals_removed": int(len(act) - act["port_id"].nunique()),
    }
    sher = raw[raw["Station Name"].str.contains("SHERMAN", na=False)]
    sher_t = pd.to_datetime(sher["Start Date"], format="%m/%d/%Y %H:%M")
    R["data"].update({"sherman_n": int(len(sher)), "sherman_first": str(sher_t.min().date()), "sherman_last": str(sher_t.max().date()),
                      "bryant_stray_n": int((raw["Station Name"] == "PALO ALTO CA / BRYANT # 1").sum())})
    yearly = sessions.groupby(sessions["start"].dt.year).size()
    R["data"]["sessions_per_year"] = {int(k): int(v) for k, v in yearly.items()}
    site_sum = (study.groupby("site").agg(sessions=("start", "size"), stations=("Station Name", "nunique"),
                                          mean_duration_h=("duration_h", "mean"), mean_energy_kwh=("energy_kwh", "mean"))
                .join(coords).join(ports.loc[pd.Timestamp("2019-06-01 12:00")].rename("ports_2019")))
    site_sum.to_csv(C.TAB / "site_summary.csv")
    R["site_summary"] = site_sum
    port_tab = act.copy()
    port_tab.to_csv(C.TAB / "port_active_windows.csv", index=False)
    commission = act.groupby("port_id")["active_from"].min()
    R["data"]["ports_commissioned_2017_2018"] = int(((commission >= "2017-01-01") & (commission < "2019-01-01")).sum())

    # ------------------------------------------------------------------ 2. EDA / dispersion
    log("EDA and overdispersion tests")
    tr = (p["split"] == "train")
    X = F.design_matrix(p)
    pois_tr = M.fit_poisson(X[tr], p.loc[tr, "y"].astype(float), p.loc[tr, "offset"])
    ct_hour = M.cameron_trivedi(p.loc[tr, "y"].values, pois_tr.fittedvalues.values)
    # correctness check of the sparse IRLS solver against statsmodels (base design, same data)
    import statsmodels.api as sm
    Xb = F.design_matrix(p, site_x_hour=False, site_x_fee=False)
    sm_fit = sm.GLM(p.loc[tr, "y"].astype(float), Xb[tr], family=sm.families.Poisson(), offset=p.loc[tr, "offset"]).fit()
    fast_fit = M.fit_poisson(Xb[tr], p.loc[tr, "y"], p.loc[tr, "offset"])
    R["solver_check"] = {"max_abs_beta_diff_vs_statsmodels": float(np.abs(fast_fit.beta - sm_fit.params.values).max()),
                         "llf_diff": float(fast_fit.llf - sm_fit.llf)}
    log(f"  solver check: {R['solver_check']}")
    mv = M.mean_variance_cells(p)
    mv.to_csv(C.TAB / "mean_variance_hourly_cells.csv", index=False)
    dly = M.daily_frame(p)
    mv_d = (dly[dly["split"] == "train"].assign(wk=lambda d: (d["dow"] >= 5).astype(int))
            .groupby(["site", "month", "wk"])["y"].agg(["mean", "var", "size"]).reset_index())
    mv_d.to_csv(C.TAB / "mean_variance_daily_cells.csv", index=False)
    mu_bins = pd.qcut(pois_tr.fittedvalues.values, 10, labels=False)
    disp_by_decile = (pd.DataFrame({"y": p.loc[tr, "y"].values, "mu": pois_tr.fittedvalues.values, "b": mu_bins})
                      .groupby("b").apply(lambda g: pd.Series({"mean_mu": g.mu.mean(),
                                                               "resid_var": ((g.y - g.mu) ** 2).mean(),
                                                               "ratio": ((g.y - g.mu) ** 2).mean() / g.mu.mean()}))
                      .reset_index())
    disp_by_decile.to_csv(C.TAB / "hourly_dispersion_by_decile.csv", index=False)
    R["dispersion"] = {
        "hourly_uncond_mean": float(p.loc[tr, "y"].mean()), "hourly_uncond_var": float(p.loc[tr, "y"].var()),
        "hourly_cells_median_var_mean_ratio": float((mv["var"] / mv["mean"]).replace([np.inf], np.nan).median()),
        "hourly_cells_share_ratio_gt1": float(((mv["var"] / mv["mean"]) > 1).mean()),
        "daily_cells_median_var_mean_ratio": float((mv_d["var"] / mv_d["mean"]).median()),
        "daily_cells_share_ratio_gt1": float(((mv_d["var"] / mv_d["mean"]) > 1).mean()),
        "hourly_pearson_dispersion": float(pois_tr.pearson_chi2 / pois_tr.df_resid),
        "hourly_ct": ct_hour, "hourly_by_decile": disp_by_decile,
    }
    eda = {"heat": study[study["start"].dt.year.isin([2017, 2018])].pipe(
               lambda s: s.groupby([s["start"].dt.dayofweek, s["start"].dt.hour]).size().unstack(fill_value=0)),
           "yearly": yearly, "site_sessions": study.groupby(["site", study["start"].dt.year]).size().unstack(),
           "monthly": study.groupby([study["start"].dt.to_period("M")]).agg(n=("start", "size"),
                                                                            fee_share=("fee", lambda s: (s > 0).mean())),
           "monthly_ports": ports.resample("MS").mean().sum(axis=1)}
    R["eda"] = {"busiest_weekday_hour": None}
    heat = eda["heat"] / 104.0   # 104 weeks in 2017-2018 -> mean sessions per hour-slot (all sites)
    i, j = np.unravel_index(np.argmax(heat.values), heat.shape)
    R["eda"] = {"heat_peak_dow": int(heat.index[i]), "heat_peak_hour": int(heat.columns[j]),
                "heat_peak_mean_sessions": float(heat.values[i, j]),
                "weekday_share": float((study["start"].dt.dayofweek < 5).mean()),
                "growth_2017_2019_pct": float(100 * (yearly[2019] / yearly[2017] - 1)),
                "change_2017_2018_pct": float(100 * (yearly[2018] / yearly[2017] - 1)),
                "sessions_2020": int(yearly.get(2020, 0)), "sessions_2019": int(yearly[2019])}

    # ------------------------------------------------------------------ 3-4. models
    if reuse_models:
        log("Reusing cached model outputs")
        with open(C.CACHE / "models.pkl", "rb") as f:
            MOD = pickle.load(f)
    else:
        MOD = {}
        log("HGB tuning on 2018-Q4 validation")
        fit_tr = (p["split"] == "train") & (~p["is_val"])
        Xt = F.tree_matrix(p)
        pp = p["ports"].astype(float)
        best, tune_tab = M.tune_hgb(Xt[fit_tr], p.loc[fit_tr, "y"], pp[fit_tr], Xt[p["is_val"]],
                                    p.loc[p["is_val"], "y"], pp[p["is_val"]])
        tune_tab.to_csv(C.TAB / "hgb_tuning.csv", index=False)
        MOD["hgb_best"] = best
        log(f"  best HGB params {best}")
        log("Validation fit (2017-01..2018-09 -> 2018-Q4)")
        vpred, vfits, vinfo = M.fit_all(p, fit_tr, p["is_val"], hgb_params=best, log=log)
        MOD["val_pred"], MOD["val_info"] = vpred, vinfo
        MOD["val_ols_sigma"] = float(np.sqrt(vfits["M1"].scale))
        log("Final fit (2017-2018 -> 2019)")
        te = p["split"] == "test"
        tpred, tfits, tinfo = M.fit_all(p, p["split"] == "train", te, hgb_params=best, log=log, full=True)
        MOD["test_pred"], MOD["test_info"] = tpred, tinfo
        MOD["ols_sigma"] = float(np.sqrt(tfits["M1"].scale))
        MOD["irr"] = M.irr_table(tfits["M3"], tinfo["M3_model_based_bse"])
        MOD["pois_params"] = tfits["M2"].params
        MOD["n_params"] = int(len(tfits["M3"].params))
        MOD["ols_neg_frac"] = float((tpred["M1"] < 0).mean())
        log("Daily-resolution Poisson vs NB2")
        MOD["daily"] = M.fit_daily(p)
        log("Trend-handling variants (Poisson GLM)")
        MOD["trend_variants"] = M.trend_variants(p)
        log("Specification variants (site interactions)")
        MOD["spec_variants"] = M.spec_variants(p)
        with open(C.CACHE / "models.pkl", "wb") as f:
            pickle.dump(MOD, f)

    R["hgb_best_params"] = MOD["hgb_best"]
    MOD["trend_variants"].to_csv(C.TAB / "trend_variants.csv", index=False)
    R["trend_variants"] = MOD["trend_variants"]
    MOD["spec_variants"].to_csv(C.TAB / "spec_variants.csv", index=False)
    R["spec_variants"] = MOD["spec_variants"]
    names = {"M0": "M0 Seasonal naive", "M1": "M1 OLS", "M2": "M2 Poisson GLM", "M3": "M3 NB2 GLM",
             "M3c": "M3c NB2 (calendar only)", "M4": "M4 HGB-Poisson"}

    def metric_table(pred, y, info, sigma):
        rows = {}
        for m in ["M0", "M1", "M2", "M3", "M3c", "M4"]:
            dist = {"M2": "poisson", "M3": "nb", "M3c": "nb", "M4": "poisson"}.get(m)
            al = info["M3_alpha"] if m == "M3" else info.get("M3c_alpha") if m == "M3c" else None
            if dist == "nb" and not al:   # alpha-hat = 0 (boundary): NB2 is exactly Poisson
                dist, al = "poisson", None
            mu = pred[m].values
            if m == "M1":   # OLS: raw predictions for MAE/RMSE, clipped at 0.01 for deviance
                met = M.hourly_metrics(y, mu, sigma=sigma)
                cl = M.hourly_metrics(y, np.clip(mu, 0.01, None))
                met["mean_poisson_deviance"], met["D2"] = cl["mean_poisson_deviance"], cl["D2"]
            else:
                met = M.hourly_metrics(y, mu, dist, al)
            rows[names[m]] = met
        return pd.DataFrame(rows).T

    yv = p.loc[MOD["val_pred"].index, "y"].values
    val_tab = metric_table(MOD["val_pred"], yv, MOD["val_info"], MOD["val_ols_sigma"])
    val_tab.to_csv(C.TAB / "validation_metrics_2018Q4.csv")
    te = p["split"] == "test"
    pt = p[te].copy()
    for m in MOD["test_pred"].columns:
        pt[m] = MOD["test_pred"][m].values
    yt = pt["y"].values
    test_tab = metric_table(MOD["test_pred"], yt, MOD["test_info"], MOD["ols_sigma"])
    test_tab.to_csv(C.TAB / "test_metrics_2019.csv")
    R["val_metrics"], R["test_metrics"] = val_tab, test_tab
    R["ols_negative_pred_frac"] = MOD["ols_neg_frac"]
    R["n_params_glm"] = MOD["n_params"]
    R["alpha_hourly"] = MOD["test_info"]["M3_alpha_summary"]
    R["alpha_hourly_calendar_only"] = MOD["test_info"]["M3c_alpha"]

    # daily metrics per site for each hourly model (aggregated)
    dm = []
    for m in ["M0", "M1", "M2", "M3", "M4"]:
        t = M.daily_metrics(pt.assign(**{m: pt[m].clip(lower=0) if m == "M1" else pt[m]}), m)
        t.insert(0, "model", names[m])
        dm.append(t)
    dm = pd.concat(dm)
    dm.to_csv(C.TAB / "daily_metrics_per_site_2019.csv", index=False)
    R["daily_metrics"] = dm
    # calibration
    cal = {m: M.calibration_table(yt, pt[m].clip(lower=0).values) for m in ["M1", "M2", "M3", "M4"]}
    for m, t in cal.items():
        t.to_csv(C.TAB / f"calibration_{m}.csv", index=False)
    R["calibration_max_abs_gap"] = {m: float((t["mean_pred"] - t["mean_obs"]).abs().max()) for m, t in cal.items()}
    R["calibration_top_decile"] = {m: {"pred": float(t["mean_pred"].iloc[-1]), "obs": float(t["mean_obs"].iloc[-1])}
                                   for m, t in cal.items()}
    # IRR table
    irr = MOD["irr"]
    irr.to_csv(C.TAB / "nb_irr_full.csv")
    R["irr"] = irr
    if "contrasts" not in MOD:
        log("IRR contrasts (cluster-robust)")
        MOD["contrasts"] = M.irr_contrasts(p, MOD["test_info"]["M3_alpha"])
        with open(C.CACHE / "models.pkl", "wb") as f:
            pickle.dump(MOD, f)
    MOD["contrasts"].to_csv(C.TAB / "irr_contrasts.csv", index=False)
    R["irr_contrasts"] = MOD["contrasts"]
    # residual diagnostics (Pearson residuals on the 2019 test set)
    a = MOD["test_info"]["M3_alpha"]
    pr = (yt - pt["M3"]) / np.sqrt(pt["M3"] + a * pt["M3"] ** 2)
    pt["pearson_resid"] = pr
    lag1 = pt.groupby("site")["pearson_resid"].apply(lambda s: s.autocorr(1)).mean()
    lag24 = pt.groupby("site")["pearson_resid"].apply(lambda s: s.autocorr(24)).mean()
    R["residuals"] = {"test_pearson_dispersion_M3": float(np.mean(pr ** 2)), "mean_pearson_resid": float(pr.mean()),
                      "acf1_mean_over_sites": float(lag1), "acf24_mean_over_sites": float(lag24)}
    pit = {"M2": M.randomized_pit(yt, pt["M2"].values, "poisson"),
           "M3": M.randomized_pit(yt, pt["M3"].values, "nb" if a > 0 else "poisson", a if a > 0 else None)}
    R["pit_hist_M2"] = np.histogram(pit["M2"], bins=10, range=(0, 1))[0] / len(yt)
    # daily models
    DA = MOD["daily"]
    tails = lambda u: float(np.mean((u < 0.05) | (u > 0.95)))   # nominal 0.10 if calibrated  # noqa: E731
    R["pit_tail_share"] = {"hourly_M3": tails(pit["M3"]), "daily_poisson": tails(DA["pit"]["poisson"]),
                           "daily_nb": tails(DA["pit"]["nb"])}
    R["daily_models"] = {"ct": DA["ct"], "alpha_summary": DA["alpha_summary"], "metrics": DA["metrics"],
                         "n_test_days": int((DA["frame"]["split"] == "test").sum())}
    pd.DataFrame(DA["metrics"]).T.to_csv(C.TAB / "daily_model_metrics_2019.csv")
    # 2019 annual totals
    R["annual_2019"] = {"observed": int(yt.sum()), **{m: float(pt[m].clip(lower=0).sum()) for m in ["M0", "M1", "M2", "M3", "M3c", "M4"]}}
    site_bias = pt.groupby("site").agg(obs=("y", "sum"), M3=("M3", "sum"), M4=("M4", "sum"))
    site_bias["M3_bias_pct"] = 100 * (site_bias["M3"] / site_bias["obs"] - 1)
    site_bias["M4_bias_pct"] = 100 * (site_bias["M4"] / site_bias["obs"] - 1)
    site_bias.to_csv(C.TAB / "site_bias_2019.csv")
    R["site_bias_2019"] = site_bias

    # ------------------------------------------------------------------ 5. spatial
    log("Spatial demand-gap analysis")
    sites = SP.site_table(pt, "M3", s2019_wk, coords)
    sites.to_csv(C.TAB / "site_demand_utilisation_2019.csv")
    SPA = SP.run_spatial(sites)
    db = SPA["db"]
    db["clusters"].to_csv(C.TAB / "dbscan_clusters.csv", index=False)
    cand = db["candidates"].copy()
    cand_out = cand.assign(sites_pooled=cand["sites_pooled"].apply(", ".join))
    cand_out.to_csv(C.TAB / "candidate_locations.csv", index=False)
    sens = SP.bandwidth_sensitivity(sites, SPA)
    sens["demand_weights"] = "M3 predicted 2019"
    obs_sites = sites.assign(pred_sessions_2019=sites["obs_sessions_2019"].astype(float))
    SPA_obs = SP.run_spatial(obs_sites)
    bc = SPA["db"]["candidates"]
    for _, cr in SPA_obs["db"]["candidates"].iterrows():
        sens.loc[len(sens)] = {"bandwidth_m": C.BANDWIDTH_M, "eps_m": SPA_obs["db"]["eps"],
                               "n_clusters": len(SPA_obs["db"]["candidates"]), "cluster": int(cr["cluster"]),
                               "lat": cr["lat"], "lon": cr["lon"], "sites_pooled": ", ".join(cr["sites_pooled"]),
                               "dist_to_nearest_baseline_candidate_m": float(np.min(np.hypot(bc["x"] - cr["x"], bc["y"] - cr["y"]))),
                               "demand_weights": "observed 2019"}
    sens.to_csv(C.TAB / "bandwidth_sensitivity.csv", index=False)
    R["spatial"] = {"bandwidth_m": C.BANDWIDTH_M, "grid_res_m": C.GRID_RES_M, "grid_shape": list(SPA["surf"]["G"].shape),
                    "gap_threshold_per_km2": db["tau"], "gap_quantile": C.GAP_QUANTILE, "n_cells_selected": db["n_selected"],
                    "eps_m": db["eps"], "eps_knee_m": db["eps_knee"], "min_samples": C.DBSCAN_MIN_SAMPLES,
                    "n_clusters": int(len(db["clusters"])), "n_noise_cells": int((db["labels"] == -1).sum()),
                    "clusters": db["clusters"], "candidates": cand_out, "sites": SPA["sites"].drop(columns=[]),
                    "sensitivity": sens}

    # ------------------------------------------------------------------ 6. simulation
    log(f"Simulation ({C.N_REPS} replications per system)")
    prof = SIM.weekday_profile(pt, "M3")
    base_res, val_sim, erl = SIM.run_baseline(prof, s2019, s2019_wk, sites, C.N_REPS, C.SEED, hol)
    val_sim.to_csv(C.TAB / "simulator_validation.csv")
    erl.to_csv(C.TAB / "erlang_vs_simulation.csv")
    scen = SIM.run_scenarios(prof, s2019_wk, sites, cand, base_res, C.N_REPS, C.SEED, n_weekdays_2019)
    scen.drop(columns=["hourly_block"], errors="ignore").to_csv(C.TAB / "scenarios.csv", index=False)
    # same simulator driven by the observed 2019 weekday arrival rates: separates simulator error from forecast error
    prof_emp = SIM.weekday_profile(pt, "y")
    base_emp, val_emp, erl_emp = SIM.run_baseline(prof_emp, s2019, s2019_wk, sites, C.N_REPS, C.SEED, hol)
    val_emp.to_csv(C.TAB / "simulator_validation_empirical_lambda.csv")
    erl_emp.to_csv(C.TAB / "erlang_vs_simulation_empirical_lambda.csv")
    scen_emp = SIM.run_scenarios(prof_emp, s2019_wk, sites, cand, base_emp, C.N_REPS, C.SEED, n_weekdays_2019)
    scen_emp.drop(columns=["hourly_block"], errors="ignore").to_csv(C.TAB / "scenarios_empirical_lambda.csv", index=False)
    hourly_block = pd.DataFrame({s: base_res[s]["blocking_prob"] for s in C.SITES})
    hourly_block.index.name = "hour"
    hourly_block.to_csv(C.TAB / "sim_hourly_blocking.csv")
    tot_blk = sum(base_res[s]["blocked"].sum() for s in C.SITES)
    tot_arr = sum(base_res[s]["arrivals"].sum() for s in C.SITES)
    R["simulation"] = {"n_reps": C.N_REPS, "n_weekdays_2019": n_weekdays_2019, "validation": val_sim, "erlang": erl,
                       "scenarios": scen.drop(columns=["hourly_block"], errors="ignore"),
                       "system_turned_away_per_weekday": float(tot_blk),
                       "system_turned_away_per_year": float(tot_blk * n_weekdays_2019),
                       "system_daily_block_prob": float(tot_blk / tot_arr),
                       "system_arrivals_per_weekday": float(tot_arr),
                       "validation_empirical": val_emp, "erlang_empirical": erl_emp,
                       "scenarios_empirical": scen_emp.drop(columns=["hourly_block"], errors="ignore"),
                       "system_turned_away_per_year_empirical": float(sum(base_emp[s]["blocked"].sum() for s in C.SITES) * n_weekdays_2019),
                       "system_daily_block_prob_empirical": float(sum(base_emp[s]["blocked"].sum() for s in C.SITES)
                                                                  / sum(base_emp[s]["arrivals"].sum() for s in C.SITES))}

    # ------------------------------------------------------------------ 6b. Rinconada pooling rerun and model audit
    log(f"Rinconada pooling rerun ({C.N_REPS} replications per scenario)")
    dist_tab, dt_spread = POOL.distance_table(SPA["sites"], cand.iloc[0])
    dist_tab.to_csv(C.TAB / "rinconada_distances.csv", index=False)
    pool_tab, pool_hourly = POOL.run_rinconada(prof, prof_emp, s2019_wk, SPA["sites"], cand.iloc[0], base_res, base_emp,
                                               C.N_REPS, C.SEED, n_weekdays_2019)
    pool_tab.to_csv(C.TAB / "rinconada_pooling.csv", index=False)
    full_tab = pd.DataFrame([POOL.observed_full_share(s2019, s, int(sites.loc[s, "ports_2019"]), hol) for s in C.SITES])
    full_tab.to_csv(C.TAB / "observed_full_share_2019.csv", index=False)
    prec_tab = POOL.precision_check(prof, prof_emp, s2019_wk, SPA["sites"], 10000, C.SEED, n_weekdays_2019)
    prec_tab.to_csv(C.TAB / "rinconada_precision_check.csv", index=False)
    R["rinconada"] = {"distances": dist_tab, "downtown_max_spread_m": dt_spread, "scenarios": pool_tab, "precision": prec_tab,
                      "rinconada_hourly_occupancy": pool_hourly, "observed_full_share": full_tab}
    log("Model audit")
    det_tab = AUD.busy_hour_detection(pt)
    det_tab.to_csv(C.TAB / "audit_busy_hour_detection.csv", index=False)
    site_audit = AUD.per_site(pt, dm)
    site_audit.to_csv(C.TAB / "audit_per_site.csv", index=False)
    R["audit"] = {"busy_threshold": AUD.BUSY_THRESHOLD, "detection": det_tab, "per_site": site_audit,
                  "rankings": AUD.ranking_checks(site_audit, SPA["sites"], full_tab)}

    # ------------------------------------------------------------------ 7. figures
    log("Figures")
    FIG.make_all(p=p, pt=pt, sessions=sessions, study=study, eda=eda, mv=mv, mv_d=mv_d, cal=cal, irr=irr, pit=pit,
                 daily=DA, sites=sites, SPA=SPA, sens=sens, base_res=base_res, scen=scen, prof=prof,
                 base_emp=base_emp, prof_emp=prof_emp, scen_emp=scen_emp,
                 alpha_daily=DA["alpha"], fee_start=fee_start, R=R)

    with open(C.RESULTS_JSON, "w") as f:
        json.dump(jsonable(R), f, indent=1)
    log(f"Done. results.json written ({C.RESULTS_JSON})")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-models", action="store_true")
    main(ap.parse_args().reuse_models)
