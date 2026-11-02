"""All paper figures (300 dpi PNG). Every plotted value comes from the pipeline objects."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from . import config as C

# ---- palette (reference categorical order, sequential blue, diverging blue<->red) ----
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SITE_COL = dict(zip(C.SITES, CAT))
MODEL_COL = {"M0": "#8a8984", "M1": "#eb6834", "M2": "#2a78d6", "M3": "#104281", "M4": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ee8a84", "#b8282a"])
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def TT(s):
    """Display name of a site (MPL is an acronym)."""
    return "MPL" if s == "MPL" else s.title()


def save(fig, name):
    fig.savefig(C.FIG / f"{name}.png")
    plt.close(fig)


def panel(ax, letter):
    """Prefix the axis title with the panel letter (avoids collisions with centred titles)."""
    t = ax.get_title()
    ax.set_title(f"({letter}) {t}" if t else f"({letter})")


# ------------------------------------------------------------------ 1 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    boxes = [
        (0.5, "Raw log\n259,415\nsessions"),
        (15, "Cleaning,\nactive-port\nexposure"),
        (29.5, "Site-hour\npanel,\nlag features"),
        (44, "Count\nmodels\nM0-M4"),
        (58.5, "KDE gap\nsurface +\nDBSCAN"),
        (73, "Candidate\nsites"),
        (87.5, "Loss-system\nsimulation"),
    ]
    w, h, y = 12, 19, 16
    for i, (x, t) in enumerate(boxes):
        fc = "#cde2fb" if i in (3, 4, 6) else "#f0efec"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2, color=INK)
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            ax.annotate("", xy=(nx - 0.6, y + h / 2), xytext=(x + w + 0.6, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    notes = [(50, "Poisson / NB2, offset,\nIRRs, test = 2019"), (64.5, "h = 400 m,\neps from k-distance"),
             (93.5, "validated vs. observed\noccupancy, Erlang-B/C")]
    for x, t in notes:
        ax.text(x, 8, t, ha="center", va="center", fontsize=6.4, color=INK2, style="italic")
    ax.text(50, 38.5, "Forecast  ->  cluster  ->  recommend  ->  evaluate", ha="center", fontsize=9, color=INK, fontweight="bold")
    save(fig, "fig01_pipeline")


# ------------------------------------------------------------------ 2 temporal EDA
def fig_temporal(sessions, eda, fee_start):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    yr = eda["yearly"]
    cols = ["#2a78d6" if 2017 <= y <= 2019 else "#b7d3f6" for y in yr.index]
    ax.bar(yr.index, yr.values / 1000, color=cols, width=0.72)
    ax.set_xticks(yr.index); ax.set_xticklabels([str(y)[2:] for y in yr.index])
    ax.set_ylabel("Sessions (thousands)"); ax.set_xlabel("Year (20xx)")
    ax.set_title("Clean sessions per year (dark = study window)")
    ax.annotate("COVID-19", xy=(2020, yr[2020] / 1000), xytext=(2017.6, yr.max() / 1000 * 0.95),
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
    panel(ax, "a")
    ax = axes[1]
    m = eda["monthly"]
    t = m.index.to_timestamp()
    ax.plot(t, m["n"], color="#2a78d6", lw=1.6, label="Sessions per month")
    ax.axvline(fee_start, color="#e34948", lw=1.1, ls="--")
    ax.text(fee_start + pd.Timedelta(days=12), m["n"].max() * 0.98, f"fees start\n{fee_start:%d %b %Y}", fontsize=7, color="#b8282a", va="top")
    for d, lab in [("2017-01-06", "+4 ports\nHigh"), ("2017-07-05", "+6 ports\nBryant"),
                   ("2018-03-21", "+12 ports Cambridge,\nTed Thompson")]:
        ax.axvline(pd.Timestamp(d), color=INK2, lw=0.6, ls=":")
        ax.text(pd.Timestamp(d) + pd.Timedelta(days=8), m["n"].min() * 0.985, lab, fontsize=6.0, color=INK2, va="bottom")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("Sessions per month"); ax.set_title("Monthly sessions 2017-2019 with fee start and new ports")
    panel(ax, "b")
    fig.tight_layout()
    save(fig, "fig02_temporal")


def fig_heat_sites(eda):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    H = eda["heat"] / 104.0
    im = ax.imshow(H.values, aspect="auto", cmap=SEQ, origin="upper")
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW)
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_xlabel("Hour of day (start time)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Mean arrivals / hour (all sites)", fontsize=7.5)
    ax.set_title("Arrival intensity, 2017-2018 (training data)")
    panel(ax, "a")
    ax = axes[1]
    ss = eda["site_sessions"].loc[C.SITES]
    x = np.arange(len(ss)); w = 0.27
    for k, (y, c) in enumerate(zip(ss.columns, ["#9ec5f4", "#3987e5", "#104281"])):
        ax.barh(x + (k - 1) * w, ss[y] / 1000, height=w - 0.03, color=c, label=str(y))
    ax.set_yticks(x); ax.set_yticklabels([TT(s) for s in ss.index], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Sessions (thousands)"); ax.legend(loc="lower right"); ax.set_title("Sessions per site and year")
    panel(ax, "b")
    fig.tight_layout()
    save(fig, "fig03_heatmap_sites")


# ------------------------------------------------------------------ 3 dispersion
def fig_dispersion(mv, mv_d, alpha_daily, R):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    ax = axes[0]
    ax.scatter(mv["mean"], mv["var"], s=9, color="#2a78d6", alpha=0.55, lw=0)
    lim = max(mv["mean"].max(), mv["var"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--", label="Var = mean (Poisson)")
    ax.set_xlabel("Cell mean (arrivals/h)"); ax.set_ylabel("Cell variance")
    ax.set_title("Hourly cells"); ax.legend(loc="upper left"); panel(ax, "a")
    ax = axes[1]
    ax.scatter(mv_d["mean"], mv_d["var"], s=9, color="#eb6834", alpha=0.6, lw=0)
    xx = np.linspace(0, mv_d["mean"].max() * 1.05, 100)
    ax.plot(xx, xx, color=INK2, lw=1, ls="--", label="Var = mean")
    ax.plot(xx, xx + alpha_daily * xx ** 2, color="#104281", lw=1.4, label=fr"NB2, $\hat\alpha$={alpha_daily:.3f}")
    ax.set_xlabel("Cell mean (arrivals/day)"); ax.set_ylabel("Cell variance")
    ax.set_title("Daily cells"); ax.legend(loc="upper left"); panel(ax, "b")
    ax = axes[2]
    dd = pd.DataFrame(R["dispersion"]["hourly_by_decile"])
    ax.plot(dd["mean_mu"], dd["ratio"], marker="o", ms=4, color="#2a78d6")
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xlabel(r"Fitted mean $\hat\mu$ (decile)"); ax.set_ylabel(r"$E[(y-\hat\mu)^2]\,/\,\hat\mu$")
    ax.set_title("Hourly residual dispersion"); panel(ax, "c")
    fig.tight_layout()
    save(fig, "fig04_dispersion")


# ------------------------------------------------------------------ 4 IRRs
def fig_irr(irr):
    keep = ([f"hour_{h:02d}" for h in range(6, 22)] + [f"dow_{d}" for d in ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
            + [f"wkend_x_hour_{h:02d}" for h in (9, 12, 15)] + ["holiday", "fee_regime", "trend_years"]
            + [f"site_{s}_x_fee" for s in C.SITES[1:]] + ["log1p_lag24", "log1p_lag168", "log1p_roll7"])
    t = irr.loc[[k for k in keep if k in irr.index]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    y = np.arange(len(t))
    groups = np.where(t.index.str.startswith("hour_"), "#2a78d6",
              np.where(t.index.str.startswith(("dow_", "wkend")), "#eb6834",
              np.where(t.index.str.contains("_x_fee") | (t.index == "fee_regime"), "#e34948",
              np.where(t.index.str.startswith("log1p"), "#1baf7a", "#4a3aa7"))))
    ax.hlines(y, t["CI_low"], t["CI_high"], color=groups, lw=1.6)
    ax.scatter(t["IRR"], y, color=groups, s=18, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axvline(1, color=INK2, lw=0.9, ls="--")
    ax.set_xscale("log")
    labels = []
    for k in t.index:
        if k.startswith("hour_"):
            labels.append(f"Weekday {k[5:]}:00 vs 00:00 (Bryant)")
        elif k.startswith("dow_"):
            labels.append(f"{k[4:]} vs Mon (00:00)")
        elif k.startswith("wkend_x_hour_"):
            labels.append(f"Weekend x {k[-2:]}:00 (extra)")
        elif k.endswith("_x_fee"):
            labels.append(f"Fee x {TT(k[5:-6])} (vs Bryant)")
        else:
            labels.append({"holiday": "Federal holiday", "fee_regime": "Fee regime (Bryant)", "trend_years": "Trend (per year)",
                           "log1p_lag24": "log(1+y, t-24h)", "log1p_lag168": "log(1+y, t-168h)",
                           "log1p_roll7": "log(1+7-day same-hour mean)"}.get(k, k))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Incidence rate ratio (log scale), 95% CI, site-day cluster-robust SE")
    ax.grid(axis="y", visible=False)
    ax.set_title("M3 incidence rate ratios (selected terms; full table in outputs/tables/nb_irr_full.csv)")
    fig.tight_layout()
    save(fig, "fig05_irr_forest")


# ------------------------------------------------------------------ 5 profiles & diagnostics
def fig_site_profiles(pt):
    wk = pt[(pt["weekend"] == 0) & (pt["holiday"] == 0)]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharex=True)
    for ax, s in zip(axes.ravel(), C.SITES):
        g = wk[wk["site"] == s].groupby("hour")
        ax.plot(g["y"].mean().index, g["y"].mean().values, color=INK2, lw=0, marker="o", ms=2.6, label="Observed")
        ax.plot(g["M3"].mean().index, g["M3"].mean().values, color=MODEL_COL["M3"], lw=1.4, label="M3 NB2")
        ax.plot(g["M4"].mean().index, g["M4"].mean().values, color=MODEL_COL["M4"], lw=1.1, ls="--", label="M4 HGB")
        ax.set_title(TT(s), fontsize=8.5); ax.set_xticks([0, 6, 12, 18, 23])
    for ax in axes[1]:
        ax.set_xlabel("Hour")
    for ax in axes[:, 0]:
        ax.set_ylabel("Arrivals / h")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Mean 2019 weekday arrival profile per site: observed vs. predicted (test set)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig06_site_profiles")


def fig_calibration(cal, pit, daily):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax = axes[0]
    for m, lab in [("M1", "M1 OLS"), ("M3", "M2/M3 GLM"), ("M4", "M4 HGB")]:
        t = cal[m]
        ax.plot(t["mean_pred"], t["mean_obs"], marker="o", ms=3.5, color=MODEL_COL[m], label=lab)
    lim = max(t["mean_obs"].max() for t in cal.values()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.9, ls="--")
    ax.set_xlabel("Mean predicted (decile)"); ax.set_ylabel("Mean observed")
    ax.set_title("Hourly calibration, 2019"); ax.legend(loc="upper left"); panel(ax, "a")
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    ax.hist(pit["M3"], bins=bins, density=True, color="#9ec5f4", edgecolor="white", lw=1)
    ax.axhline(1, color=INK2, ls="--", lw=0.9)
    ax.set_xlabel("Randomised PIT"); ax.set_ylabel("Density"); ax.set_title("Hourly M3 PIT, 2019"); panel(ax, "b")
    ax = axes[2]
    ax.hist(daily["pit"]["poisson"], bins=bins, density=True, histtype="step", color="#2a78d6", lw=1.6, label="Daily Poisson")
    ax.hist(daily["pit"]["nb"], bins=bins, density=True, histtype="step", color="#eb6834", lw=1.6, label="Daily NB2")
    ax.axhline(1, color=INK2, ls="--", lw=0.9)
    ax.set_xlabel("Randomised PIT"); ax.set_title("Daily models PIT, 2019"); ax.legend(loc="upper center"); panel(ax, "c")
    fig.tight_layout()
    save(fig, "fig07_calibration_pit")


def fig_week(pt, R):
    from scipy import stats
    site = pd.Series({s: pt.loc[pt["site"] == s, "y"].sum() for s in C.SITES}).idxmax()
    w = pt[(pt["site"] == site) & (pt["time"] >= "2019-10-07") & (pt["time"] < "2019-10-14")]
    mu = w["M3"].values
    lo, hi = stats.poisson.ppf(0.05, mu), stats.poisson.ppf(0.95, mu)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.fill_between(w["time"], lo, hi, step="mid", color="#cde2fb", label="M3 90% prediction interval")
    ax.step(w["time"], w["y"], where="mid", color=INK, lw=1.0, label="Observed arrivals")
    ax.plot(w["time"], mu, color=MODEL_COL["M3"], lw=1.4, label="M3 mean")
    ax.plot(w["time"], w["M4"], color=MODEL_COL["M4"], lw=1.0, ls="--", label="M4 HGB mean")
    ax.set_ylabel("Arrivals / h"); ax.set_title(f"{site.title()}, Mon 7 - Sun 13 Oct 2019 (test week)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, "fig08_week")
    return site


def fig_residuals(pt):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    b = pd.qcut(pt["M3"].rank(method="first"), 20, labels=False)
    g = pt.groupby(b).agg(mu=("M3", "mean"), r=("pearson_resid", "mean"), sd=("pearson_resid", "std"))
    ax.errorbar(g["mu"], g["r"], yerr=g["sd"], fmt="o", ms=3, color=MODEL_COL["M3"], ecolor="#9ec5f4", lw=1)
    ax.axhline(0, color=INK2, lw=0.9, ls="--")
    ax.set_xlabel(r"Fitted mean $\hat\mu$ (20 bins)"); ax.set_ylabel("Pearson residual (mean +/- sd)")
    ax.set_title("Pearson residuals vs fitted, 2019"); panel(ax, "a")
    ax = axes[1]
    lags = np.arange(1, 169)
    ac = np.mean([[s.autocorr(L) for L in lags] for _, s in pt.groupby("site")["pearson_resid"]], axis=0)
    ax.vlines(lags, 0, ac, color="#2a78d6", lw=0.8)
    ci = 1.96 / np.sqrt(len(pt) / len(C.SITES))
    ax.axhspan(-ci, ci, color="#f0efec")
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("ACF (mean over sites)"); ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
    ax.set_title("Residual autocorrelation, 2019"); panel(ax, "b")
    fig.tight_layout()
    save(fig, "fig09_residuals")


# ------------------------------------------------------------------ 6 maps
def _lonlat_axes(ax, proj, surf=None):
    ax.set_aspect(1 / np.cos(np.radians(proj.lat0)))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False)


def _scalebar(ax, proj, length_m=1000):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dlon = length_m / (6_371_000 * np.cos(np.radians(proj.lat0))) * 180 / np.pi
    xs = x0 + 0.06 * (x1 - x0); ys = y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + dlon], [ys, ys], color=INK, lw=2.2, solid_capstyle="butt")
    ax.text(xs + dlon / 2, ys + 0.02 * (y1 - y0), f"{length_m/1000:g} km", ha="center", fontsize=7)


def _surf_lonlat(SPA):
    proj, surf = SPA["proj"], SPA["surf"]
    lat, lon = proj.inv(surf["GX"], surf["GY"])
    return lon, lat


LABEL_OFFSETS = {"BRYANT": (-12, 22), "WEBSTER": (24, 16), "HAMILTON": (30, -8), "HIGH": (-6, -20),
                 "TED THOMPSON": (34, 12), "CAMBRIDGE": (-36, -12), "RINCONADA LIB": (0, 12), "MPL": (0, 12)}


def _site_labels(ax, sites, dy=None):
    for s, r in sites.iterrows():
        dx_, dy_ = LABEL_OFFSETS.get(s, (0, 12))
        ax.annotate(TT(s), (r["lon"], r["lat"]), xytext=(dx_, dy_), textcoords="offset points", fontsize=6.3,
                    ha="center", va="center", color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=2))


def fig_maps(SPA):
    sites = SPA["sites"]; proj = SPA["proj"]; surf = SPA["surf"]
    lon, lat = _surf_lonlat(SPA)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    ax = axes[0]
    ax.contour(lon, lat, surf["D"], levels=6, colors="#9ec5f4", linewidths=0.6)
    sc = ax.scatter(sites["lon"], sites["lat"], s=sites["pred_sessions_2019"] / 25, c=sites["rho_peak"], cmap=SEQ,
                    vmin=0, vmax=max(0.7, sites["rho_peak"].max()), edgecolor=INK2, linewidth=0.7, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label(r"Peak-hour utilisation $\rho$", fontsize=7.5)
    _site_labels(ax, sites)
    _lonlat_axes(ax, proj); ax.set_xlim(lon.min(), lon.max()); ax.set_ylim(lat.min(), lat.max())
    ax.set_title("Sites: size = predicted 2019 sessions,\ncontours = demand KDE D(x)"); _scalebar(ax, proj); panel(ax, "a")
    ax = axes[1]
    G = surf["G"]
    v = np.abs(G).max()
    im = ax.pcolormesh(lon, lat, G, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Gap G(x), sessions / km$^2$ / yr", fontsize=7.5)
    ax.scatter(sites["lon"], sites["lat"], s=14, color=INK, marker="^", zorder=3)
    _site_labels(ax, sites)
    _lonlat_axes(ax, proj); ax.set_title("Demand-supply gap G(x)=D(x)-S(x)\n(red = demand exceeds port supply)")
    _scalebar(ax, proj); panel(ax, "b"); ax.grid(False)
    fig.tight_layout()
    save(fig, "fig10_maps_gap")


def fig_dbscan(SPA):
    sites = SPA["sites"]; proj = SPA["proj"]; db = SPA["db"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    kd = db["kdist"]
    ax.plot(np.arange(len(kd)), kd, color="#2a78d6", lw=1.4)
    ax.axhline(db["eps"], color="#e34948", lw=1, ls="--", label=f"eps = {db['eps']:.0f} m")
    ax.axhline(db["eps_knee"], color=INK2, lw=0.8, ls=":", label=f"knee = {db['eps_knee']:.0f} m")
    ax.set_xlabel("Selected grid cells (sorted)"); ax.set_ylabel(f"Distance to {C.DBSCAN_MIN_SAMPLES}th neighbour (m)")
    ax.set_title("k-distance plot"); ax.legend(); panel(ax, "a")
    ax = axes[1]
    lon, lat = _surf_lonlat(SPA)
    ax.contour(lon, lat, SPA["surf"]["G"], levels=[0], colors=GRID, linewidths=0.8)
    plat, plon = proj.inv(db["points"][:, 0], db["points"][:, 1])
    lab = db["labels"]
    cl_cols = ["#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
    ax.scatter(plon[lab == -1], plat[lab == -1], s=2, color="#c3c2b7", label="noise")
    for L in sorted(set(lab) - {-1}):
        ax.scatter(plon[lab == L], plat[lab == L], s=2.5, color=cl_cols[L % 4], label=f"cluster {L}")
    ax.scatter(sites["lon"], sites["lat"], s=18, color=INK, marker="^", zorder=3, label="existing site")
    cand = db["candidates"]
    clat, clon = proj.inv(cand["x"].values, cand["y"].values)
    ax.scatter(clon, clat, s=110, marker="*", color="#e34948", edgecolor=INK, linewidth=0.6, zorder=4, label="candidate")
    for i, (a, b) in enumerate(zip(clon, clat)):
        ax.annotate(f"Candidate C{cand['cluster'].iloc[i]}", (a, b), xytext=(48, -30), textcoords="offset points",
                    fontsize=7, color="#b8282a", fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#b8282a", lw=0.6))
    _site_labels(ax, sites, dy=0.0007)
    _lonlat_axes(ax, proj)
    pad = 0.004
    ax.set_xlim(min(plon.min(), sites["lon"].min()) - pad, max(plon.max(), sites["lon"].max()) + pad)
    ax.set_ylim(min(plat.min(), sites["lat"].min()) - pad, max(plat.max(), sites["lat"].max()) + pad)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=4, color="#c3c2b7", label="noise cell")]
    hs += [Line2D([], [], ls="", marker="o", ms=4, color=cl_cols[L % 4], label=f"cluster {L}") for L in sorted(set(lab) - {-1})]
    hs += [Line2D([], [], ls="", marker="^", ms=5, color=INK, label="existing site"),
           Line2D([], [], ls="", marker="*", ms=9, color="#e34948", mec=INK, mew=0.5, label="candidate")]
    ax.legend(handles=hs, loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9)
    ax.set_title(f"DBSCAN on cells with G > q{int(C.GAP_QUANTILE*100)} and candidates")
    _scalebar(ax, proj, 500); panel(ax, "b")
    fig.tight_layout()
    save(fig, "fig11_dbscan_candidates")


def fig_bandwidth(sites_df):
    from . import spatial as SP
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, h, L in zip(axes, C.BANDWIDTH_SENS, "abc"):
        r = SP.run_spatial(sites_df, h)
        lon, lat = _surf_lonlat(r)
        G = r["surf"]["G"]; v = np.abs(G).max()
        ax.pcolormesh(lon, lat, G, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), shading="auto", rasterized=True)
        s = r["sites"]
        ax.scatter(s["lon"], s["lat"], s=8, color=INK, marker="^")
        c = r["db"]["candidates"]
        if len(c):
            ax.scatter(c["lon"], c["lat"], s=60, marker="*", color="#e34948", edgecolor=INK, linewidth=0.5)
        _lonlat_axes(ax, r["proj"]); ax.grid(False)
        ax.set_title(f"h = {h:.0f} m ({len(c)} cluster{'' if len(c) == 1 else 's'})"); ax.tick_params(labelsize=6)
        ax.set_xlabel("Longitude", fontsize=7); ax.set_ylabel("Latitude" if L == "a" else "", fontsize=7)
        panel(ax, L)
    fig.tight_layout()
    save(fig, "fig12_bandwidth_sensitivity")


# ------------------------------------------------------------------ 7 simulation
def fig_sim_validation(base_res, base_emp):
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.5), sharex=True)
    hrs = np.arange(24) + 0.5
    for ax, s in zip(axes.ravel(), C.SITES):
        ax.plot(hrs, base_res[s]["obs_occupancy"], color=INK, lw=0, marker="o", ms=2.4, label="Observed 2019")
        ax.plot(hrs, base_res[s]["occupancy"], color=MODEL_COL["M3"], lw=1.4, label=r"Sim, $\lambda$ from M3")
        ax.plot(hrs, base_emp[s]["occupancy"], color="#eb6834", lw=1.1, ls="--", label=r"Sim, observed $\lambda$")
        ax.set_title(TT(s), fontsize=8.5); ax.set_xticks([0, 6, 12, 18, 24])
    for ax in axes[1]:
        ax.set_xlabel("Hour")
    for ax in axes[:, 0]:
        ax.set_ylabel("Occupied ports")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Simulator validation: mean occupied ports on a 2019 weekday", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig13_sim_validation")


def fig_blocking(base_res, base_emp):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    vmax = max(max(base_res[s]["blocking_prob"].max(), base_emp[s]["blocking_prob"].max()) for s in C.SITES)
    for ax, res, title, L in [(axes[0], base_res, r"$\lambda$ from M3 forecast", "a"),
                              (axes[1], base_emp, r"$\lambda$ = observed 2019 mean", "b")]:
        B = np.array([res[s]["blocking_prob"] for s in C.SITES])
        im = ax.imshow(B * 100, aspect="auto", cmap=SEQ, vmin=0, vmax=vmax * 100)
        ax.set_yticks(range(len(C.SITES))); ax.set_yticklabels([TT(s) for s in C.SITES], fontsize=7.5)
        ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlabel("Hour of day"); ax.grid(False); ax.set_title(f"Blocking probability, {title}")
        panel(ax, L)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("Drivers turned away (%)", fontsize=7.5)
    save(fig, "fig14_blocking_heatmap")


def fig_scenarios(scen, scen_emp):
    order = ["Baseline (separate sites)", "Overflow to 2 new ports", "Overflow to 4 new ports",
             "Pooled, no new ports", "Pooled + 2 new ports", "Pooled + 4 new ports"]
    cols = ["#8a8984", "#86b6ef", "#2a78d6", "#f4b49c", "#eb6834", "#b8282a"]
    clusters = sorted(scen["cluster"].unique())
    fig, axes = plt.subplots(len(clusters), 2, figsize=(7.0, 2.4 * len(clusters)), squeeze=False)
    for i, cl in enumerate(clusters):
        for j, (sc, title) in enumerate([(scen, r"$\lambda$ from M3 forecast"), (scen_emp, r"$\lambda$ = observed 2019 mean")]):
            ax = axes[i, j]
            d = sc[sc["cluster"] == cl].set_index("scenario").loc[order]
            yv = np.arange(len(order))[::-1]
            ax.barh(yv, d["turned_away_per_year"], color=cols, height=0.7)
            for yy, v in zip(yv, d["turned_away_per_year"]):
                ax.text(v + d["turned_away_per_year"].max() * 0.01, yy, f"{v:,.0f}", va="center", fontsize=7, color=INK)
            ax.set_yticks(yv); ax.set_yticklabels(order if j == 0 else [""] * len(order), fontsize=7.5)
            ax.set_xlim(0, d["turned_away_per_year"].max() * 1.18)
            ax.set_xlabel("Turned-away weekday sessions / year"); ax.grid(axis="y", visible=False)
            ax.set_title(f"C{cl}, {title}", fontsize=8.5)
            panel(ax, "abcdefgh"[2 * i + j])
    fig.tight_layout()
    save(fig, "fig15_scenarios")


# ------------------------------------------------------------------ compact figures for the 10-page IEEE paper
def fig_eda_compact(eda):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [0.85, 1.35, 1.0]})
    ax = axes[0]
    yr = eda["yearly"]
    ax.bar(yr.index, yr.values / 1000, color=["#2a78d6" if 2017 <= y <= 2019 else "#b7d3f6" for y in yr.index], width=0.72)
    ax.set_xticks(yr.index[::2]); ax.set_xticklabels([str(y)[2:] for y in yr.index[::2]])
    ax.set_ylabel("Sessions (thousands)"); ax.set_xlabel("Year (20xx)"); ax.set_title("Sessions per year"); panel(ax, "a")
    ax = axes[1]
    H = eda["heat"] / 104.0
    im = ax.imshow(H.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW, fontsize=7)
    ax.set_xticks(range(0, 24, 6)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 6)])
    ax.set_xlabel("Start hour"); ax.grid(False); ax.set_title("Arrivals/h, all sites, 2017-18"); panel(ax, "b")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ax = axes[2]
    ss = eda["site_sessions"].loc[C.SITES]
    x = np.arange(len(ss)); w = 0.27
    for k, (y, c) in enumerate(zip(ss.columns, ["#9ec5f4", "#3987e5", "#104281"])):
        ax.barh(x + (k - 1) * w, ss[y] / 1000, height=w - 0.03, color=c, label=str(y))
    ax.set_yticks(x); ax.set_yticklabels([TT(s) for s in ss.index], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("Sessions (thousands)"); ax.legend(loc="lower right", fontsize=6); ax.set_title("Sessions per site")
    panel(ax, "c")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ieee_fig_eda")


def fig_irr_compact(irr, con):
    rows = []
    for h in (7, 9, 12, 15, 18, 21):
        r = irr.loc[f"hour_{h:02d}"]; rows.append((f"Weekday {h:02d}:00 vs 00:00", r["IRR"], r["CI_low"], r["CI_high"], "#2a78d6"))
    for _, r in con[con["kind"] == "weekend09"].iterrows():
        rows.append((r["contrast"].split(" (")[0].replace(" vs Monday 09:00", " vs Mon"), r["IRR"], r["CI_low"], r["CI_high"], "#eb6834"))
    for term, lab in [("holiday", "Federal holiday"), ("trend_years", "Trend (per year)")]:
        r = irr.loc[term]; rows.append((lab, r["IRR"], r["CI_low"], r["CI_high"], "#4a3aa7"))
    for _, r in con[con["kind"] == "fee"].iterrows():
        rows.append((f"Fee regime, {TT(r['site'])}", r["IRR"], r["CI_low"], r["CI_high"], "#e34948"))
    for term, lab in [("log1p_lag24", "log(1+y), 24 h lag"), ("log1p_lag168", "log(1+y), 168 h lag"),
                      ("log1p_roll7", "log(1+7-day mean)")]:
        r = irr.loc[term]; rows.append((lab, r["IRR"], r["CI_low"], r["CI_high"], "#1baf7a"))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 4.3))
    y = np.arange(len(rows))
    for yy, (lab, v, lo, hi, c) in zip(y, rows):
        ax.hlines(yy, lo, hi, color=c, lw=1.5); ax.plot(v, yy, "o", ms=3.5, color=c)
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.tick_params(axis="x", labelsize=7); ax.grid(axis="y", visible=False)
    ax.set_xlabel("IRR (log scale), 95% cluster-robust CI", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ieee_fig_irr")


def fig_diagnostics(pt, cal, pit, daily):
    from scipy import stats
    fig = plt.figure(figsize=(7.0, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    site = pd.Series({s: pt.loc[pt["site"] == s, "y"].sum() for s in C.SITES}).idxmax()
    w = pt[(pt["site"] == site) & (pt["time"] >= "2019-10-07") & (pt["time"] < "2019-10-14")]
    mu = w["M3"].values
    ax.fill_between(w["time"], stats.poisson.ppf(0.05, mu), stats.poisson.ppf(0.95, mu), step="mid", color="#cde2fb",
                    label="M3 90% PI")
    ax.step(w["time"], w["y"], where="mid", color=INK, lw=0.9, label="Observed")
    ax.plot(w["time"], mu, color=MODEL_COL["M3"], lw=1.3, label="M3 mean")
    ax.plot(w["time"], w["M4"], color=MODEL_COL["M4"], lw=1.0, ls="--", label="M4 HGB")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_ylabel("Arrivals / h"); ax.set_title(f"{TT(site)}, test week 7-13 Oct 2019"); ax.legend(ncol=4, fontsize=6.5, loc="upper left")
    panel(ax, "a")
    ax = fig.add_subplot(gs[1, 0])
    for m, lab in [("M1", "OLS"), ("M3", "GLM"), ("M4", "HGB")]:
        t = cal[m]; ax.plot(t["mean_pred"], t["mean_obs"], marker="o", ms=2.5, lw=1.1, color=MODEL_COL[m], label=lab)
    lim = max(t["mean_obs"].max() for t in cal.values()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Mean predicted"); ax.set_ylabel("Mean observed"); ax.set_title("Calibration (deciles)")
    ax.legend(fontsize=6); panel(ax, "b")
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 11)
    ax.hist(pit["M3"], bins=bins, density=True, histtype="step", color=MODEL_COL["M3"], lw=1.4, label="Hourly GLM")
    ax.hist(daily["pit"]["poisson"], bins=bins, density=True, histtype="step", color="#2a78d6", lw=1.0, ls="--", label="Daily Poisson")
    ax.hist(daily["pit"]["nb"], bins=bins, density=True, histtype="step", color="#eb6834", lw=1.4, label="Daily NB2")
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Randomised PIT"); ax.set_ylabel("Density"); ax.set_title("PIT histograms"); ax.legend(fontsize=5.8)
    panel(ax, "c")
    ax = fig.add_subplot(gs[1, 2])
    lags = np.arange(1, 169)
    ac = np.mean([[s.autocorr(L) for L in lags] for _, s in pt.groupby("site")["pearson_resid"]], axis=0)
    ax.vlines(lags, 0, ac, color="#2a78d6", lw=0.6)
    ci = 1.96 / np.sqrt(len(pt) / len(C.SITES)); ax.axhspan(-ci, ci, color="#f0efec")
    ax.set_xticks([24, 72, 120, 168]); ax.set_xlabel("Lag (h)"); ax.set_ylabel("ACF"); ax.set_title("Residual ACF")
    panel(ax, "d")
    save(fig, "ieee_fig_diag")


def fig_sim_compact(base_res, base_emp):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axes[0]
    for s in C.SITES:
        ax.scatter(base_emp[s]["obs_occupancy"], base_emp[s]["occupancy"], s=7, color=SITE_COL[s], label=TT(s), lw=0)
    lim = max(max(base_emp[s]["obs_occupancy"].max(), base_emp[s]["occupancy"].max()) for s in C.SITES) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Observed occupied ports"); ax.set_ylabel("Simulated occupied ports")
    ax.set_title("Hourly occupancy, 2019 weekday"); ax.legend(fontsize=5.5, ncol=2, loc="upper left", markerscale=1.5)
    panel(ax, "a")
    ax = axes[1]
    B = np.array([base_emp[s]["blocking_prob"] for s in C.SITES]) * 100
    im = ax.imshow(B, aspect="auto", cmap=SEQ, vmin=0)
    ax.set_yticks(range(len(C.SITES))); ax.set_yticklabels([TT(s) for s in C.SITES], fontsize=6.5)
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)], fontsize=7)
    ax.set_xlabel("Hour of day"); ax.grid(False); ax.set_title("Drivers turned away (%), observed rates")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    panel(ax, "b")
    fig.tight_layout()
    save(fig, "ieee_fig_sim")


def make_all(p, pt, sessions, study, eda, mv, mv_d, cal, irr, pit, daily, sites, SPA, sens, base_res, scen, prof,
             base_emp, prof_emp, scen_emp, alpha_daily, fee_start, R):
    fig_eda_compact(eda)
    fig_irr_compact(irr, R["irr_contrasts"])
    fig_diagnostics(pt, cal, pit, daily)
    fig_sim_compact(base_res, base_emp)
    fig_pipeline()
    fig_temporal(sessions, eda, fee_start)
    fig_heat_sites(eda)
    fig_dispersion(mv, mv_d, alpha_daily, R)
    fig_irr(irr)
    fig_site_profiles(pt)
    fig_calibration(cal, pit, daily)
    R["week_plot_site"] = fig_week(pt, R)
    fig_residuals(pt)
    fig_maps(SPA)
    fig_dbscan(SPA)
    fig_bandwidth(sites)
    fig_sim_validation(base_res, base_emp)
    fig_blocking(base_res, base_emp)
    fig_scenarios(scen, scen_emp)
