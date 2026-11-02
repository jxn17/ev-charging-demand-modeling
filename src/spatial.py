"""Spatial demand-gap analysis: KDE demand/supply surfaces, DBSCAN hot-spots, candidates."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from . import config as C

R_EARTH = 6_371_000.0


class Projector:
    """Local equirectangular projection (error < 0.1 % over a few km)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.k = np.cos(np.radians(lat0))

    def fwd(self, lat, lon):
        x = R_EARTH * np.radians(np.asarray(lon) - self.lon0) * self.k
        y = R_EARTH * np.radians(np.asarray(lat) - self.lat0)
        return x, y

    def inv(self, x, y):
        lat = self.lat0 + np.degrees(np.asarray(y) / R_EARTH)
        lon = self.lon0 + np.degrees(np.asarray(x) / (R_EARTH * self.k))
        return lat, lon


def site_table(p: pd.DataFrame, pred_col: str, sessions_2019: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Per-site 2019 demand, capacity and utilisation figures from the model predictions."""
    t = p[p["split"] == "test"]
    wk = t[(t["weekend"] == 0) & (t["holiday"] == 0)]
    prof = wk.groupby(["site", "hour"])[pred_col].mean().unstack()
    rows = []
    for s in C.SITES:
        ts = t[t["site"] == s]
        ss = sessions_2019[sessions_2019["site"] == s]
        lam_peak = float(prof.loc[s].max())
        ES = float(ss["duration_h"].mean())
        c = int(ts["ports"].mode().iloc[0])
        rows.append({"site": s, "lat": coords.loc[s, "lat"], "lon": coords.loc[s, "lon"],
                     "ports_2019": c, "pred_sessions_2019": float(ts[pred_col].sum()),
                     "obs_sessions_2019": int(ts["y"].sum()),
                     "peak_hour": int(prof.loc[s].idxmax()), "peak_lambda_per_h": lam_peak,
                     "mean_duration_h": ES, "median_duration_h": float(ss["duration_h"].median()),
                     "offered_load_peak_erlang": lam_peak * ES,
                     "rho_peak": lam_peak * ES / c,
                     "pred_sessions_per_port": float(ts[pred_col].sum()) / c})
    return pd.DataFrame(rows).set_index("site")


def make_grid(xs, ys, res=C.GRID_RES_M, buffer=C.BUFFER_M):
    gx = np.arange(xs.min() - buffer, xs.max() + buffer + res, res)
    gy = np.arange(ys.min() - buffer, ys.max() + buffer + res, res)
    GX, GY = np.meshgrid(gx, gy)
    return gx, gy, GX, GY


def kde_surface(GX, GY, xs, ys, w, h):
    """sum_i w_i * N2(x; x_i, h^2 I)  -> intensity per km^2."""
    Z = np.zeros_like(GX)
    for xi, yi, wi in zip(xs, ys, w):
        d2 = (GX - xi) ** 2 + (GY - yi) ** 2
        Z += wi * np.exp(-d2 / (2 * h * h)) / (2 * np.pi * h * h)
    return Z * 1e6


def gap_surfaces(sites: pd.DataFrame, proj: Projector, h: float):
    xs, ys = proj.fwd(sites["lat"].values, sites["lon"].values)
    gx, gy, GX, GY = make_grid(xs, ys)
    w_d = sites["pred_sessions_2019"].values
    w_s = sites["ports_2019"].values.astype(float)
    Dz = kde_surface(GX, GY, xs, ys, w_d, h)
    Sz = kde_surface(GX, GY, xs, ys, w_s, h) * (w_d.sum() / w_s.sum())   # same total mass as demand
    return {"gx": gx, "gy": gy, "GX": GX, "GY": GY, "D": Dz, "S": Sz, "G": Dz - Sz, "xs": xs, "ys": ys}


def k_distance(points, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    d, _ = nn.kneighbors(points)
    return np.sort(d[:, k])


def knee(values):
    """Point of maximum distance below the chord of a sorted, increasing curve (Kneedle-style)."""
    v = np.asarray(values, float)
    x = np.linspace(0, 1, len(v))
    yn = (v - v.min()) / (v.max() - v.min() + 1e-12)
    i = int(np.argmax(x - yn))
    return float(v[i]), i


def dbscan_gap(surf, sites: pd.DataFrame, quantile=C.GAP_QUANTILE, min_samples=C.DBSCAN_MIN_SAMPLES, eps=None):
    G = surf["G"]
    tau = float(np.quantile(G, quantile))
    m = G > tau
    pts = np.column_stack([surf["GX"][m], surf["GY"][m]])
    g = G[m]
    kd = k_distance(pts, min_samples)
    eps_knee, _ = knee(kd)
    if eps is None:
        eps = max(eps_knee, np.sqrt(2) * C.GRID_RES_M * 1.01)   # never below the grid diagonal
    weight = g / tau                                            # a cell at the threshold counts as one point
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(pts, sample_weight=weight).labels_
    cell_area_km2 = (C.GRID_RES_M / 1000) ** 2
    rows, cands = [], []
    for L in sorted(set(lab) - {-1}):
        sel = lab == L
        P, w = pts[sel], g[sel]
        cx, cy = np.average(P[:, 0], weights=w), np.average(P[:, 1], weights=w)
        dsite = np.hypot(sites["x"].values[:, None] - P[:, 0][None, :], sites["y"].values[:, None] - P[:, 1][None, :])
        inside = sites.index[(dsite.min(axis=1) <= eps)].tolist()
        # candidate: gap-weighted centroid, moved to the nearest member cell >= MIN_DIST from every site
        dc = np.hypot(sites["x"].values - cx, sites["y"].values - cy).min()
        moved = False
        if dc < C.MIN_DIST_TO_SITE_M:
            ok = dsite.min(axis=0) >= C.MIN_DIST_TO_SITE_M
            if ok.any():
                j = np.argmin(np.hypot(P[ok, 0] - cx, P[ok, 1] - cy))
                cx, cy = P[ok][j]
                moved = True
        rows.append({"cluster": int(L), "n_cells": int(sel.sum()), "area_km2": float(sel.sum() * cell_area_km2),
                     "total_gap_sessions_per_yr": float(w.sum() * cell_area_km2),
                     "peak_gap_per_km2": float(w.max()), "sites_inside": ", ".join(inside)})
        cands.append({"cluster": int(L), "x": float(cx), "y": float(cy), "moved_from_centroid": moved,
                      "dist_nearest_site_m": float(np.hypot(sites["x"].values - cx, sites["y"].values - cy).min()),
                      "nearest_site": sites.index[int(np.argmin(np.hypot(sites["x"].values - cx, sites["y"].values - cy)))],
                      "sites_pooled": inside})
    return {"tau": tau, "eps": float(eps), "eps_knee": eps_knee, "kdist": kd, "labels": lab, "points": pts,
            "gap_vals": g, "n_selected": int(m.sum()), "clusters": pd.DataFrame(rows), "candidates": pd.DataFrame(cands)}


def run_spatial(sites: pd.DataFrame, h=C.BANDWIDTH_M):
    lat0, lon0 = sites["lat"].mean(), sites["lon"].mean()
    proj = Projector(lat0, lon0)
    s = sites.copy()
    s["x"], s["y"] = proj.fwd(s["lat"].values, s["lon"].values)
    surf = gap_surfaces(s, proj, h)
    db = dbscan_gap(surf, s)
    cand = db["candidates"]
    if len(cand):
        cand["lat"], cand["lon"] = proj.inv(cand["x"].values, cand["y"].values)
    # gap at the site locations themselves
    s["gap_at_site_per_km2"] = [float(surf["G"][np.argmin(np.abs(surf["gy"] - yy)), np.argmin(np.abs(surf["gx"] - xx))])
                                for xx, yy in zip(s["x"], s["y"])]
    return {"proj": proj, "sites": s, "surf": surf, "db": db}


def bandwidth_sensitivity(sites: pd.DataFrame, base):
    rows = []
    bc = base["db"]["candidates"]
    for h in C.BANDWIDTH_SENS:
        r = run_spatial(sites, h)
        cand = r["db"]["candidates"]
        for _, cr in cand.iterrows():
            dmin = float(np.min(np.hypot(bc["x"] - cr["x"], bc["y"] - cr["y"]))) if len(bc) else np.nan
            rows.append({"bandwidth_m": h, "eps_m": r["db"]["eps"], "n_clusters": len(cand), "cluster": int(cr["cluster"]),
                         "lat": cr["lat"], "lon": cr["lon"], "sites_pooled": ", ".join(cr["sites_pooled"]),
                         "dist_to_nearest_baseline_candidate_m": dmin})
        if len(cand) == 0:
            rows.append({"bandwidth_m": h, "eps_m": r["db"]["eps"], "n_clusters": 0})
    return pd.DataFrame(rows)
