"""Checks of whether each model finds what it is meant to find (used by the summary report)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import config as C

BUSY_THRESHOLD = 3     # arrivals in one site-hour


def busy_hour_detection(pt: pd.DataFrame) -> pd.DataFrame:
    """Can each forecast pick out the busy site-hours of 2019? The flag is the top decile of each model's forecast."""
    y = pt["y"].values
    busy = y >= BUSY_THRESHOLD
    rows = []
    for m, name in [("M0", "Seasonal naive"), ("M1", "OLS"), ("M2", "Poisson GLM"), ("M3", "NB2 GLM"), ("M4", "Gradient boosting")]:
        s = pt[m].values
        rank = pd.Series(s).rank(method="first", ascending=False).values
        top = rank <= 0.10 * len(s)
        rows.append({"model": name, "auc_busy_hour": float(roc_auc_score(busy, s)),
                     "precision_top_decile": float(busy[top].mean()), "recall_top_decile": float(busy[top].sum() / busy.sum()),
                     "session_share_top_decile": float(y[top].sum() / y.sum()), "busy_hours": int(busy.sum()),
                     "busy_share": float(busy.mean())})
    return pd.DataFrame(rows)


def per_site(pt: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Per-site check of M3: total, hourly error, daily error and weekday profile (shape and peak hour)."""
    wk = pt[(pt["weekend"] == 0) & (pt["holiday"] == 0)]
    rows = []
    for s in C.SITES:
        t, w = pt[pt["site"] == s], wk[wk["site"] == s]
        po = w.groupby("hour")["y"].mean(); pm = w.groupby("hour")["M3"].mean()
        d = daily[(daily["site"] == s) & (daily["model"] == "M3 NB2 GLM")].iloc[0]
        ho, hm = int(po.idxmax()), int(pm.idxmax())
        rows.append({"site": s, "observed": int(t["y"].sum()), "predicted": float(t["M3"].sum()),
                     "bias_pct": float(100 * (t["M3"].sum() / t["y"].sum() - 1)),
                     "hourly_mae": float(np.mean(np.abs(t["y"] - t["M3"]))), "daily_mae": float(d["daily_MAE"]),
                     "daily_mape_pct": float(d["daily_MAPE_pct"]), "profile_r": float(np.corrcoef(po, pm)[0, 1]),
                     "peak_hour_observed": ho, "peak_hour_predicted": hm, "peak_hour_diff": abs(ho - hm),
                     "peak_rate_observed": float(po.max()), "peak_rate_predicted": float(pm.max())})
    return pd.DataFrame(rows)


def ranking_checks(site_tab: pd.DataFrame, sites: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Do forecast-based rankings of the sites agree with what the 2019 log shows independently?"""
    s = sites.loc[C.SITES]
    f = full.set_index("site").loc[C.SITES]
    st = site_tab.set_index("site").loc[C.SITES]
    sp = lambda a, b: float(stats.spearmanr(a, b).statistic)  # noqa: E731
    obs_per_port = st["observed"].values / s["ports_2019"].values
    hottest = f["observed_full_share"].sort_values(ascending=False)
    return {"spearman_pred_vs_obs_sessions": sp(st["predicted"], st["observed"]),
            "spearman_pred_vs_obs_sessions_per_port": sp(s["pred_sessions_per_port"], obs_per_port),
            "spearman_gap_vs_observed_full_share": sp(s["gap_at_site_per_km2"], f["observed_full_share"]),
            "spearman_rho_vs_observed_full_share": sp(s["rho_peak"], f["observed_full_share"]),
            "sites_by_observed_full_share": [str(x) for x in hottest.index]}
