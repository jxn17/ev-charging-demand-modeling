"""Count models M0-M4, overdispersion tests and evaluation metrics."""
import itertools
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from statsmodels.discrete.discrete_model import NegativeBinomial

from . import config as C
from . import features as F
from .glm_fast import FastGLM

Z90 = stats.norm.ppf(0.95)


# --------------------------------------------------------------------------- tests
def cameron_trivedi(y, mu):
    """Cameron & Trivedi (1990) auxiliary regression for NB2-type overdispersion.

    ((y - mu)^2 - y) / mu = alpha * mu + e   (no intercept); H0: alpha = 0.
    """
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    aux = ((y - mu) ** 2 - y) / mu
    r = sm.OLS(aux, mu).fit()
    alpha, se, t = float(r.params[0]), float(r.bse[0]), float(r.tvalues[0])
    p_one_sided = float(stats.norm.sf(t))
    return {"alpha_ct": alpha, "se": se, "t": t, "p_one_sided": p_one_sided}


def mean_variance_cells(p: pd.DataFrame):
    """Mean and variance of hourly counts in site x hour-of-day x day-type cells (training data)."""
    tr = p[p["split"] == "train"]
    g = tr.groupby(["site", "hour", "weekend"])["y"].agg(["mean", "var", "size"]).reset_index()
    return g


# --------------------------------------------------------------------------- distributions
def nb_params(mu, alpha):
    n = 1.0 / alpha
    return n, n / (n + mu)


def loglik(y, mu, dist, alpha=None):
    if dist == "poisson":
        return stats.poisson.logpmf(y, mu)
    n, pr = nb_params(mu, alpha)
    return stats.nbinom.logpmf(y, n, pr)


def interval(mu, dist, alpha=None, level=0.90):
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    if dist == "poisson":
        return stats.poisson.ppf(lo_q, mu), stats.poisson.ppf(hi_q, mu)
    n, pr = nb_params(mu, alpha)
    return stats.nbinom.ppf(lo_q, n, pr), stats.nbinom.ppf(hi_q, n, pr)


def p_zero(mu, dist, alpha=None):
    if dist == "poisson":
        return np.exp(-mu)
    return (1 + alpha * mu) ** (-1 / alpha)


def randomized_pit(y, mu, dist, alpha=None, rng=None):
    rng = rng or np.random.default_rng(C.SEED)
    if dist == "poisson":
        F_hi, F_lo = stats.poisson.cdf(y, mu), stats.poisson.cdf(y - 1, mu)
    else:
        n, pr = nb_params(mu, alpha)
        F_hi, F_lo = stats.nbinom.cdf(y, n, pr), stats.nbinom.cdf(y - 1, n, pr)
    return F_lo + rng.uniform(size=len(y)) * (F_hi - F_lo)


# --------------------------------------------------------------------------- metrics
def hourly_metrics(y, mu, dist=None, alpha=None, sigma=None):
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    out = {"MAE": float(np.mean(np.abs(y - mu))), "RMSE": float(np.sqrt(np.mean((y - mu) ** 2)))}
    if np.all(mu > 0):
        dev = mean_poisson_deviance(y, mu)
        dev0 = mean_poisson_deviance(y, np.full_like(y, y.mean()))
        out.update({"mean_poisson_deviance": float(dev), "D2": float(1 - dev / dev0)})
    else:
        out.update({"mean_poisson_deviance": None, "D2": None})
    out["obs_zero_frac"] = float(np.mean(y == 0))
    if dist in ("poisson", "nb"):
        out["loglik_per_obs"] = float(np.mean(loglik(y, mu, dist, alpha)))
        out["pred_zero_frac"] = float(np.mean(p_zero(mu, dist, alpha)))
        lo, hi = interval(mu, dist, alpha)
        out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
        out["mean_PI_width"] = float(np.mean(hi - lo))
    elif sigma is not None:   # Gaussian interval for OLS
        lo, hi = mu - Z90 * sigma, mu + Z90 * sigma
        out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
        out["mean_PI_width"] = float(np.mean(hi - lo))
        out["loglik_per_obs"] = None
        out["pred_zero_frac"] = None
    else:
        out.update({"loglik_per_obs": None, "pred_zero_frac": None, "coverage90": None, "mean_PI_width": None})
    return out


def daily_metrics(frame: pd.DataFrame, col: str):
    d = frame.groupby(["site", "date"]).agg(y=("y", "sum"), mu=(col, "sum")).reset_index()
    rows = []
    for site, g in list(d.groupby("site")) + [("ALL", d)]:
        nz = g[g["y"] > 0]
        rows.append({"site": site, "daily_MAE": float(np.mean(np.abs(g.y - g.mu))),
                     "daily_MAPE_pct": float(100 * np.mean(np.abs(nz.y - nz.mu) / nz.y)),
                     "mean_daily_sessions": float(g.y.mean()), "n_days": int(len(g)),
                     "zero_days_excluded_from_MAPE": int((g["y"] == 0).sum())})
    return pd.DataFrame(rows)


def calibration_table(y, mu, n_bins=10):
    q = pd.qcut(pd.Series(mu).rank(method="first"), n_bins, labels=False)
    t = pd.DataFrame({"y": np.asarray(y), "mu": np.asarray(mu), "bin": q.values})
    return t.groupby("bin").agg(mean_pred=("mu", "mean"), mean_obs=("y", "mean"), n=("y", "size")).reset_index()


# --------------------------------------------------------------------------- fitting
def fit_ols(X, y, ports):
    Xo = X.copy()
    Xo["ports"] = ports.values      # no offset in a linear model -> ports as a regressor
    return sm.OLS(y, Xo).fit(), Xo.columns


def fit_poisson(X, y, offset, beta0=None):
    return FastGLM(X, y, offset, alpha=0.0, beta0=beta0)


def fit_nb2_ml(X, y, offset, start_params):
    """Joint ML of (beta, alpha) for NB2. BFGS works on log(alpha), so alpha stays positive."""
    model = NegativeBinomial(np.asarray(y), np.asarray(X), loglike_method="nb2", offset=np.asarray(offset))
    sp = np.append(start_params, 0.05)
    return model.fit(start_params=sp, method="bfgs", maxiter=5000, gtol=1e-6, disp=False)


def profile_nb2_alpha(X, y, offset, start_params, lo=1e-6, hi=2.0):
    """ML estimate of the NB2 dispersion by maximising the profile log-likelihood
    l_p(alpha) = max_beta l(beta, alpha) over log(alpha) (bounded Brent search).

    The maximiser of the profile likelihood is the joint MLE. If it sits on the lower
    bound the data give no evidence of NB2 overdispersion (alpha-hat = 0, Poisson).
    """
    y_ = np.asarray(y, float)
    state = {"start": np.asarray(start_params, float)}
    memo = {}

    def prof(alpha):
        if alpha in memo:
            return memo[alpha]
        r = FastGLM(X, y, offset, alpha=alpha, beta0=state["start"])
        state["start"] = r.beta
        memo[alpha] = r.llf
        return memo[alpha]

    opt = minimize_scalar(lambda la: -prof(np.exp(la)), bounds=(np.log(lo), np.log(hi)), method="bounded",
                          options={"xatol": 1e-3})
    a = float(np.exp(opt.x))
    boundary = a < 10 * lo
    se = None
    if not boundary:   # observed information of the profile likelihood (central difference)
        h = 0.02 * a
        d2 = (prof(a + h) - 2 * prof(a) + prof(a - h)) / h ** 2
        se = float(1 / np.sqrt(-d2)) if d2 < 0 else None
    return {"alpha": 0.0 if boundary else a, "alpha_search": a, "alpha_se": se, "boundary": bool(boundary),
            "llf_nb": prof(a), "n_evals": len(memo), "prof": prof}


def nb_glm(X, y, offset, alpha, groups=None, beta0=None):
    """NB2 GLM for a fixed alpha (the ML estimate); cluster-robust covariance by site-day if groups given."""
    r = FastGLM(X, y, offset, alpha=max(float(alpha), 0.0), beta0=beta0)
    r.cov = r.cluster_cov(groups) if groups is not None else r.cov_model
    return r


def nb_alpha_summary(prof_res, pois_res, n_params):
    """alpha-hat, its SE, LR test vs Poisson and (at the boundary) a profile-likelihood 95 % upper bound."""
    llf_nb = max(prof_res["llf_nb"], float(pois_res.llf))
    lr = max(0.0, 2 * (llf_nb - pois_res.llf))
    out = {"alpha": prof_res["alpha"], "alpha_se": prof_res["alpha_se"], "boundary": prof_res["boundary"],
           "llf_nb": llf_nb, "llf_poisson": float(pois_res.llf), "LR_stat": lr,
           "LR_p_boundary": float(0.5 * stats.chi2.sf(lr, 1)) if lr > 0 else 0.5,
           "aic_nb": float(-2 * llf_nb + 2 * (n_params + 1)), "aic_poisson": float(pois_res.aic),
           "profile_evals": prof_res["n_evals"]}
    if prof_res["boundary"]:   # one-sided 95 % bound: l_p(alpha_u) = l_max - chi2_{0.90,1}/2
        prof, target = prof_res["prof"], llf_nb - stats.chi2.ppf(0.90, 1) / 2
        lo, hi = 1e-6, 0.2
        for _ in range(12):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if prof(mid) > target else (lo, mid)
        out["alpha_upper95_profile"] = float(hi)
    return out


def hgb_model(params):
    return HistGradientBoostingRegressor(loss="poisson", categorical_features=[7], random_state=C.SEED,
                                         early_stopping=False, **params)


def tune_hgb(Xt_tr, y_tr, ports_tr, Xt_va, y_va, ports_va):
    grid = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
            "max_iter": [200, 500], "min_samples_leaf": [50]}
    rows = []
    for vals in itertools.product(*grid.values()):
        prm = dict(zip(grid.keys(), vals))
        m = hgb_model(prm).fit(Xt_tr, y_tr / ports_tr, sample_weight=ports_tr)
        mu = m.predict(Xt_va) * ports_va
        rows.append({**prm, "val_poisson_deviance": float(mean_poisson_deviance(y_va, mu))})
    tab = pd.DataFrame(rows).sort_values("val_poisson_deviance").reset_index(drop=True)
    best = tab.iloc[0][list(grid.keys())].to_dict()
    best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
    return best, tab


def fit_all(p: pd.DataFrame, train_mask, eval_mask, hgb_params=None, log=print, full=False):
    """Fit M1-M4 on train_mask rows and predict eval_mask rows.

    full=True additionally computes the alpha summary (LR test, profile bound) and
    cluster-robust NB standard errors used for the IRR table.
    """
    X = F.design_matrix(p)
    Xc = F.design_matrix(p, lags=False)
    Xt = F.tree_matrix(p)
    y = p["y"].astype(float)
    off = p["offset"]
    tr, ev = train_mask.values, eval_mask.values
    # frozen trend: out-of-sample rows keep the trend value reached at the end of training
    tmax = p.loc[tr, "trend"].max()
    for M_ in (X, Xc):
        M_.loc[ev, "trend_years"] = np.minimum(p.loc[ev, "trend"], tmax)
    Xt.loc[ev, "trend"] = np.minimum(p.loc[ev, "trend"], tmax)
    preds = pd.DataFrame(index=p.index[ev])
    fits, info = {}, {}
    groups = pd.factorize(p.loc[tr, "site"].astype(str) + p.loc[tr, "date"].astype(str))[0]

    preds["M0"] = p.loc[ev, "lag168"].astype(float)

    t0 = time.time()
    ols, ocols = fit_ols(X[tr], y[tr], p.loc[tr, "ports"])
    Xo_ev = X[ev].copy(); Xo_ev["ports"] = p.loc[ev, "ports"].values
    preds["M1"] = ols.predict(Xo_ev[ocols])
    fits["M1"] = ols
    log(f"  M1 OLS done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pois = fit_poisson(X[tr], y[tr], off[tr])
    preds["M2"] = pois.predict(X[ev], offset=off[ev])
    fits["M2"] = pois
    log(f"  M2 Poisson GLM done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    prof = profile_nb2_alpha(X[tr], y[tr], off[tr], pois.params.values)
    alpha = prof["alpha"]
    nb = nb_glm(X[tr], y[tr], off[tr], alpha, groups=groups if full else None, beta0=pois.beta)
    preds["M3"] = nb.predict(X[ev], offset=off[ev])
    fits["M3"] = nb
    info["M3_alpha"] = alpha          # 0 on the boundary -> NB2 reduces to Poisson
    if full:
        info["M3_alpha_summary"] = nb_alpha_summary(prof, pois, X.shape[1])
        info["M3_model_based_bse"] = pd.Series(np.sqrt(np.diag(nb.cov_model)), index=nb.names)
    log(f"  M3 NB2 done ({time.time()-t0:.1f}s), alpha={alpha:.3g}")

    t0 = time.time()
    pois_c = fit_poisson(Xc[tr], y[tr], off[tr])
    prof_c = profile_nb2_alpha(Xc[tr], y[tr], off[tr], pois_c.params.values)
    nb_c = nb_glm(Xc[tr], y[tr], off[tr], prof_c["alpha"])
    preds["M3c"] = nb_c.predict(Xc[ev], offset=off[ev])
    fits["M3c"] = nb_c
    info["M3c_alpha"] = prof_c["alpha"]
    log(f"  M3c NB2 calendar-only done ({time.time()-t0:.1f}s)")

    if hgb_params is not None:
        t0 = time.time()
        ports = p["ports"].astype(float)
        hgb = hgb_model(hgb_params).fit(Xt[tr], y[tr] / ports[tr], sample_weight=ports[tr])
        preds["M4"] = hgb.predict(Xt[ev]) * ports[ev]
        fits["M4"] = hgb
        log(f"  M4 HGB done ({time.time()-t0:.1f}s)")
    return preds, fits, info


def irr_table(nb_res, model_based_bse=None):
    sf = nb_res.summary_frame(nb_res.cov)
    t = pd.DataFrame({"coef": sf["coef"], "se_cluster": sf["se"], "IRR": np.exp(sf["coef"]),
                      "CI_low": np.exp(sf["CI_low_coef"]), "CI_high": np.exp(sf["CI_high_coef"]),
                      "p_value": sf["p_value"]})
    if model_based_bse is not None:
        t["se_model"] = model_based_bse
    t.index.name = "term"
    return t


def trend_variants(p: pd.DataFrame):
    """Poisson GLM with extrapolated / frozen / no trend, scored on validation and test."""
    X = F.design_matrix(p)
    y, off = p["y"].astype(float), p["offset"]
    tr = (p["split"] == "train").values
    va = p["is_val"].values
    te = (p["split"] == "test").values
    rows = []
    for mode in ["extrapolated", "frozen", "none"]:
        for label, a, b in [("validation 2018-Q4", tr & ~va, va), ("test 2019", tr, te)]:
            Xv = X.drop(columns="trend_years") if mode == "none" else X.copy()
            if mode == "frozen":
                Xv.loc[b, "trend_years"] = np.minimum(p.loc[b, "trend"], p.loc[a, "trend"].max())
            r = fit_poisson(Xv[a], y[a], off[a])
            mu = r.predict(Xv[b], offset=off[b])
            rows.append({"trend": mode, "eval_set": label, "mean_poisson_deviance": float(mean_poisson_deviance(y[b], mu)),
                         "MAE": float(np.mean(np.abs(y[b] - mu))), "total_bias_pct": float(100 * (mu.sum() / y[b].sum() - 1))})
    return pd.DataFrame(rows)


def spec_variants(p: pd.DataFrame):
    """Validation-based choice of site interactions (Poisson GLM, frozen trend)."""
    y, off = p["y"].astype(float), p["offset"]
    tr = (p["split"] == "train").values
    va = p["is_val"].values
    te = (p["split"] == "test").values
    rows = []
    for name, sh, sf in [("base (site fixed effects only)", False, False), ("+ site x fee", False, True),
                         ("+ site x hour", True, False), ("+ site x hour + site x fee", True, True)]:
        X = F.design_matrix(p, site_x_hour=sh, site_x_fee=sf)
        for label, a, b in [("validation 2018-Q4", tr & ~va, va), ("test 2019", tr, te)]:
            Xb = X.copy()
            Xb.loc[b, "trend_years"] = np.minimum(p.loc[b, "trend"], p.loc[a, "trend"].max())
            r = fit_poisson(X[a], y[a], off[a])
            mu = r.predict(Xb[b], offset=off[b])
            sb = pd.DataFrame({"y": y[b], "mu": mu, "s": p.loc[b, "site"]}).groupby("s").sum()
            rows.append({"specification": name, "n_params": X.shape[1], "eval_set": label,
                         "mean_poisson_deviance": float(mean_poisson_deviance(y[b], mu)),
                         "total_bias_pct": float(100 * (mu.sum() / y[b].sum() - 1)),
                         "worst_site_abs_bias_pct": float(100 * (sb.mu / sb.y - 1).abs().max())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- daily resolution
def daily_frame(p: pd.DataFrame) -> pd.DataFrame:
    d = (p.groupby(["site", "date"])
          .agg(y=("y", "sum"), ports=("ports", "mean"), dow=("dow", "first"), month=("month", "first"),
               holiday=("holiday", "first"), fee=("fee", "first"), trend=("trend", "first"),
               split=("split", "first"))
          .reset_index().sort_values(["site", "date"]))
    g = d.groupby("site")["y"]
    d["lag1"], d["lag7"] = g.shift(1), g.shift(7)
    return d.dropna().reset_index(drop=True)


def daily_design(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0}, index=d.index)
    for k, name in zip(range(1, 7), ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        X[f"dow_{name}"] = (d["dow"] == k).astype(float)
    for m in range(2, 13):
        X[f"month_{m:02d}"] = (d["month"] == m).astype(float)
    X["holiday"] = d["holiday"].astype(float)
    X["fee_regime"] = d["fee"].astype(float)
    X["trend_years"] = d["trend"].astype(float)
    for s in C.SITES[1:]:
        X[f"site_{s}"] = (d["site"] == s).astype(float)
    X["log1p_lag1d"] = np.log1p(d["lag1"])
    X["log1p_lag7d"] = np.log1p(d["lag7"])
    return X


def fit_daily(p: pd.DataFrame):
    """Daily site-level Poisson vs NB2 (the planning resolution)."""
    d = daily_frame(p)
    X = daily_design(d)
    off = np.log(d["ports"])
    y = d["y"].astype(float)
    tr, te = (d["split"] == "train").values, (d["split"] == "test").values
    X.loc[te, "trend_years"] = np.minimum(d.loc[te, "trend"], d.loc[tr, "trend"].max())   # frozen trend
    pois = fit_poisson(X[tr], y[tr], off[tr])
    prof = profile_nb2_alpha(X[tr], y[tr], off[tr], pois.params.values)
    a = prof["alpha"]
    nb = nb_glm(X[tr], y[tr], off[tr], a)
    ct = cameron_trivedi(y[tr], pois.fittedvalues)
    summ = nb_alpha_summary(prof, pois, X.shape[1])
    nb_disc = fit_nb2_ml(X[tr], y[tr], off[tr], pois.beta)   # cross-check: joint ML in statsmodels
    summ["alpha_discrete_nb_crosscheck"] = float(nb_disc.params[-1])
    summ["alpha_se_discrete_nb_crosscheck"] = float(nb_disc.bse[-1])
    summ["pearson_dispersion_poisson"] = float(pois.pearson_chi2 / pois.df_resid)
    d.loc[te, "mu_pois"] = pois.predict(X[te], offset=off[te])
    d.loc[te, "mu_nb"] = nb.predict(X[te], offset=off[te])
    yt = y[te].values
    metrics = {"Daily Poisson": hourly_metrics(yt, d.loc[te, "mu_pois"].values, "poisson"),
               "Daily NB2": hourly_metrics(yt, d.loc[te, "mu_nb"].values, "nb", a)}
    pit = {"poisson": randomized_pit(yt, d.loc[te, "mu_pois"].values, "poisson"),
           "nb": randomized_pit(yt, d.loc[te, "mu_nb"].values, "nb", a)}
    return {"frame": d, "ct": ct, "alpha_summary": summ, "metrics": metrics, "pit": pit, "alpha": a}


def irr_contrasts(p: pd.DataFrame, alpha: float):
    """IRRs for linear combinations of M3 coefficients (cluster-robust 95% CI):
    site-specific fee effects and weekday peak-hour effects per site."""
    X = F.design_matrix(p)
    tr = (p["split"] == "train").values
    groups = pd.factorize(p.loc[tr, "site"].astype(str) + p.loc[tr, "date"].astype(str))[0]
    res = nb_glm(X[tr], p.loc[tr, "y"].astype(float), p.loc[tr, "offset"], max(alpha, 1e-8), groups=groups)
    names = list(X.columns)
    V = res.cov
    rows = []

    def add(label, kind, site, terms):
        L = np.zeros(len(names))
        for t in terms:
            L[names.index(t)] = 1.0
        est = float(L @ res.beta)
        se = float(np.sqrt(L @ V @ L))
        rows.append({"contrast": label, "kind": kind, "site": site, "IRR": np.exp(est),
                     "CI_low": np.exp(est - 1.96 * se), "CI_high": np.exp(est + 1.96 * se),
                     "p_value": float(2 * stats.norm.sf(abs(est / se)))})

    for s in C.SITES:
        add(f"Fee regime, {s}", "fee", s, ["fee_regime"] + ([] if s == C.SITES[0] else [f"site_{s}_x_fee"]))
    for s in C.SITES:
        extra = [] if s == C.SITES[0] else [f"site_{s}_x_hour_09"]
        add(f"Weekday 09:00 vs 00:00, {s}", "hour09", s, ["hour_09"] + extra)
    add("Saturday 09:00 vs Monday 09:00 (all sites)", "weekend09", "ALL", ["dow_Sat", "wkend_x_hour_09"])
    add("Sunday 09:00 vs Monday 09:00 (all sites)", "weekend09", "ALL", ["dow_Sun", "wkend_x_hour_09"])
    return pd.DataFrame(rows)
