"""Site x hour panel construction and feature engineering.

All lag / rolling features are built from strictly past values (>= 24 h back),
so the model is a one-day-ahead forecaster and no test-hour information leaks
into its own features.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config as C
from . import data as D


def build_panel(sessions: pd.DataFrame, ports: pd.DataFrame, fee_start: pd.Timestamp) -> pd.DataFrame:
    hours = ports.index
    s = sessions[(sessions["start"] >= hours[0]) & (sessions["start"] <= hours[-1] + pd.Timedelta(minutes=59))]
    counts = (s.assign(hour=s["start"].dt.floor("h"))
                .groupby(["hour", "site"]).size().unstack(fill_value=0)
                .reindex(index=hours, columns=C.SITES, fill_value=0))
    counts.index.name, counts.columns.name = "time", "site"
    pp = ports.copy()
    pp.index.name, pp.columns.name = "time", "site"
    long = pd.concat([counts.stack().rename("y"), pp.stack().rename("ports")], axis=1).reset_index()
    long = long.sort_values(["site", "time"]).reset_index(drop=True)

    g = long.groupby("site")["y"]
    long["lag24"] = g.shift(24)
    long["lag168"] = g.shift(168)
    same_hour = [g.shift(24 * k) for k in range(1, 8)]
    long["roll7"] = pd.concat(same_hour, axis=1).mean(axis=1)   # trailing 7-day same-hour mean

    t = long["time"]
    long["hour"] = t.dt.hour
    long["dow"] = t.dt.dayofweek
    long["weekend"] = (long["dow"] >= 5).astype(int)
    long["month"] = t.dt.month
    hol = USFederalHolidayCalendar().holidays(start="2016-01-01", end="2020-12-31")
    long["holiday"] = t.dt.normalize().isin(hol).astype(int)
    long["fee"] = (t >= fee_start).astype(int)
    long["trend"] = (t - pd.Timestamp(C.STUDY_START)).dt.total_seconds() / (365.25 * 24 * 3600)
    long["date"] = t.dt.normalize()
    long["year"] = t.dt.year
    return long


def model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows inside the study window where the site has >= 1 active port."""
    p = panel[(panel["time"] >= C.STUDY_START) & (panel["time"] <= C.STUDY_END) & (panel["ports"] > 0)].copy()
    assert p[["lag24", "lag168", "roll7"]].notna().all().all()
    p["offset"] = np.log(p["ports"].astype(float))
    p["split"] = np.where(p["time"] >= C.TEST_START, "test", "train")
    p["is_val"] = (p["time"] >= C.VAL_START) & (p["time"] < C.TEST_START)
    return p.reset_index(drop=True)


FEATURE_GROUPS = ["hour", "dow", "weekend_x_hour", "month", "holiday", "fee", "trend", "site", "lags"]


def design_matrix(p: pd.DataFrame, lags: bool = True, site_x_hour: bool = True, site_x_fee: bool = True) -> pd.DataFrame:
    """Dummy-coded design matrix shared by OLS / Poisson / NB.

    Reference levels: hour 00, Monday, January, site BRYANT, no holiday, free charging.
    site_x_hour / site_x_fee add site-specific daily profiles and fee responses
    (selected on the 2018-Q4 validation set, see models.spec_variants).
    """
    hour, dow, wk, month, site = (p["hour"].values, p["dow"].values, p["weekend"].values,
                                  p["month"].values, p["site"].values)
    fee = p["fee"].values
    cols = {"const": np.ones(len(p))}
    for h in range(1, 24):
        cols[f"hour_{h:02d}"] = (hour == h).astype(float)
    for d, name in zip(range(1, 7), ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        cols[f"dow_{name}"] = (dow == d).astype(float)
    for h in range(1, 24):
        cols[f"wkend_x_hour_{h:02d}"] = ((hour == h) & (wk == 1)).astype(float)
    for m in range(2, 13):
        cols[f"month_{m:02d}"] = (month == m).astype(float)
    cols["holiday"] = p["holiday"].values.astype(float)
    cols["fee_regime"] = fee.astype(float)
    cols["trend_years"] = p["trend"].values.astype(float)
    for s in C.SITES[1:]:
        cols[f"site_{s}"] = (site == s).astype(float)
    if site_x_fee:
        for s in C.SITES[1:]:
            cols[f"site_{s}_x_fee"] = ((site == s) & (fee == 1)).astype(float)
    if site_x_hour:
        # site-specific profiles for 06-23 h only: overnight hours are near-empty at every site
        # (some site-hour cells have zero training arrivals, which would make their MLE diverge)
        for s in C.SITES[1:]:
            for h in range(C.SITE_HOUR_FROM, 24):
                cols[f"site_{s}_x_hour_{h:02d}"] = ((site == s) & (hour == h)).astype(float)
    if lags:
        cols["log1p_lag24"] = np.log1p(p["lag24"].values)
        cols["log1p_lag168"] = np.log1p(p["lag168"].values)
        cols["log1p_roll7"] = np.log1p(p["roll7"].values)
    X = pd.DataFrame(cols, index=p.index)
    return X


def tree_matrix(p: pd.DataFrame) -> pd.DataFrame:
    """Same raw information for the gradient-boosting benchmark (trees find interactions)."""
    X = pd.DataFrame({
        "hour": p["hour"], "dow": p["dow"], "weekend": p["weekend"], "month": p["month"],
        "holiday": p["holiday"], "fee": p["fee"], "trend": p["trend"],
        "site": p["site"].map({s: i for i, s in enumerate(C.SITES)}),
        "log1p_lag24": np.log1p(p["lag24"]), "log1p_lag168": np.log1p(p["lag168"]),
        "log1p_roll7": np.log1p(p["roll7"]),
    })
    return X
