"""Rinconada Library pooling rerun (Section 8 of the revised draft).

Every port stays attached to its own site. A driver tries the own site first, then the other pools
from nearest to farthest, and is lost only if every port in the system is busy. Total losses therefore
follow the same law as one shared pool (simulate.simulate_system with a single capacity), while this
version also reports where each session ended up, which the single-pool model cannot do.
"""
import heapq

import numpy as np
import pandas as pd

from . import config as C
from .simulate import MIN_ARR_PER_DAY, N_DAYS, duration_pools

DOWNTOWN = ["BRYANT", "HAMILTON", "HIGH", "WEBSTER"]
EXTRA = "RINCONADA LIB"
FULL_WINDOW = (8.0, 20.0)     # hours of day over which "all ports full" is measured


def simulate_routed(lam_by_source, dur_by_source, caps, routes, n_reps, rng, track_pool=None):
    """Loss network with per-pool bookkeeping. Returns per-hour means, per-run losses by source,
    occupancy per pool and hour, the source-to-pool placement matrix (per measured day) and the share
    of FULL_WINDOW during which pool `track_pool` has every port busy."""
    lam_by_source = [np.asarray(l, float) for l in lam_by_source]
    m, n_pools = len(lam_by_source), len(caps)
    lam_tot = np.sum(lam_by_source, axis=0)
    lam_max = lam_tot.max()
    share = np.array(lam_by_source) / np.where(lam_tot > 0, lam_tot, 1)
    T = 24.0 * N_DAYS
    arr = np.zeros(24); blk = np.zeros(24)
    occ = np.zeros((n_pools, 24))
    placed = np.zeros((m, n_pools))
    lost_by_run = np.zeros((n_reps, m))
    full_time = 0.0
    w0, w1 = 24.0 + FULL_WINDOW[0], 24.0 + FULL_WINDOW[1]
    for rep in range(n_reps):
        n = rng.poisson(lam_max * T)
        cand = np.sort(rng.uniform(0, T, n))
        hrs = (cand % 24).astype(int)
        keep = rng.uniform(size=n) < lam_tot[hrs] / lam_max
        times, hrs = cand[keep], hrs[keep]
        if m == 1:
            srcs = np.zeros(len(times), int)
        else:
            u = rng.uniform(size=len(times))
            srcs = np.minimum((u[:, None] > np.cumsum(share[:, hrs].T, axis=1)).sum(axis=1), m - 1)
        u_dur = rng.uniform(size=len(times))
        busy = [[] for _ in caps]
        served = [[] for _ in caps]
        for t, h, src, ud in zip(times, hrs, srcs, u_dur):
            measured = t >= 24.0
            if measured:
                arr[h] += 1
            done = False
            for k in routes[src]:
                hp = busy[k]
                while hp and hp[0] <= t:
                    heapq.heappop(hp)
                if len(hp) < caps[k]:
                    pool = dur_by_source[src][h]
                    sv = pool[int(ud * len(pool))]
                    heapq.heappush(hp, t + sv)
                    served[k].append((t, t + sv))
                    if measured:
                        placed[src, k] += 1
                    done = True
                    break
            if not done and measured:
                blk[h] += 1
                lost_by_run[rep, src] += 1
        for k, s_k in enumerate(served):
            if s_k:
                sv = np.array(s_k)
                for h in range(24):
                    a_, b_ = 24.0 + h, 25.0 + h
                    occ[k, h] += np.clip(np.minimum(sv[:, 1], b_) - np.maximum(sv[:, 0], a_), 0, None).sum()
        if track_pool is not None and served[track_pool]:
            ev = sorted([(a, 1) for a, _ in served[track_pool]] + [(b, -1) for _, b in served[track_pool]])
            n_busy, last = 0, w0
            for tt, d in ev:
                tc = min(max(tt, w0), w1)
                if n_busy >= caps[track_pool]:
                    full_time += tc - last
                last = tc
                n_busy += d
            if n_busy >= caps[track_pool]:
                full_time += w1 - last
    return {"arrivals": arr / n_reps, "blocked": blk / n_reps, "occupancy": occ / n_reps, "placed": placed / n_reps,
            "lost_by_run": lost_by_run,
            "blocking_prob": np.divide(blk, arr, out=np.zeros(24), where=arr / n_reps >= MIN_ARR_PER_DAY),
            "full_share": full_time / (n_reps * (w1 - w0))}


def distance_table(sites: pd.DataFrame, candidate) -> pd.DataFrame:
    """Step 1 of the spec: how far is each non-downtown site from the downtown cluster?"""
    dt = sites.loc[DOWNTOWN]
    cx, cy = dt["x"].mean(), dt["y"].mean()
    rows = []
    for s in [x for x in C.SITES if x not in DOWNTOWN]:
        r = sites.loc[s]
        rows.append({"site": s, "ports": int(r["ports_2019"]), "rho_peak": float(r["rho_peak"]),
                     "dist_to_downtown_centroid_m": float(np.hypot(r["x"] - cx, r["y"] - cy)),
                     "dist_to_nearest_downtown_site_m": float(np.hypot(dt["x"] - r["x"], dt["y"] - r["y"]).min()),
                     "dist_to_candidate_m": float(np.hypot(r["x"] - candidate["x"], r["y"] - candidate["y"]))})
    spread = max(float(np.hypot(dt.loc[a, "x"] - dt.loc[b, "x"], dt.loc[a, "y"] - dt.loc[b, "y"]))
                 for a in DOWNTOWN for b in DOWNTOWN)
    return pd.DataFrame(rows).sort_values("dist_to_downtown_centroid_m"), spread


def observed_full_share(sessions_2019: pd.DataFrame, site: str, c: int, hol_dates) -> dict:
    """From the real 2019 log only: share of weekday 08:00-20:00 with every port occupied, and mean occupancy."""
    s = sessions_2019[sessions_2019["site"] == site]
    t0 = pd.Timestamp("2018-12-31")
    st = ((s["start"] - t0).dt.total_seconds() / 60).round().astype(int).values
    en = st + np.round(s["duration_h"].values * 60).astype(int)
    N = 367 * 24 * 60
    diff = np.zeros(N + 1)
    np.add.at(diff, np.clip(st, 0, N), 1)
    np.add.at(diff, np.clip(en, 0, N), -1)
    ser = pd.Series(np.cumsum(diff)[:N], index=pd.date_range(t0, periods=N, freq="min"))
    ser = ser[(ser.index >= "2019-01-01") & (ser.index < "2020-01-01")]
    idx = ser.index
    keep = (idx.dayofweek < 5) & (~idx.normalize().isin(hol_dates)) & (idx.hour >= FULL_WINDOW[0]) & (idx.hour < FULL_WINDOW[1])
    x = ser[keep]
    return {"site": site, "ports": c, "observed_full_share": float((x >= c).mean()),
            "observed_mean_occupancy_share": float(x.mean() / c)}


def _lost_stats(r, n_weekdays, idx):
    per_run = r["lost_by_run"][:, idx].sum(axis=1)
    mean = float(per_run.mean()) * n_weekdays
    half = 1.96 * float(per_run.std(ddof=1)) / np.sqrt(len(per_run)) * n_weekdays
    return mean, half


def precision_check(prof_model, prof_obs, dur_sessions, sites, n_reps, seed, n_weekdays):
    """High-replication comparison of the two pooled systems, because at 500 runs the small loss counts are noisy."""
    rng = np.random.default_rng(seed + 100)
    rows = []
    for label, prof in [("model", prof_model), ("observed", prof_obs)]:
        for names in (DOWNTOWN, DOWNTOWN + [EXTRA]):
            xy = {s: (sites.loc[s, "x"], sites.loc[s, "y"]) for s in names}
            routes = [[names.index(p) for p in sorted(names, key=lambda p: np.hypot(xy[p][0] - xy[s][0], xy[p][1] - xy[s][1]))]
                      for s in names]
            r = simulate_routed([prof.loc[s].values for s in names], [duration_pools(dur_sessions, s) for s in names],
                                [int(sites.loc[s, "ports_2019"]) for s in names], routes, n_reps, rng)
            lost, ci = _lost_stats(r, n_weekdays, list(range(len(names))))
            lost_dt, ci_dt = _lost_stats(r, n_weekdays, list(range(4)))
            rows.append({"rates": label, "system": f"{len(names)} sites pooled", "n_runs": n_reps, "lost_per_year": lost, "lost_ci95": ci,
                         "lost_downtown_drivers": lost_dt, "lost_downtown_ci95": ci_dt, "peak_block": float(r["blocking_prob"].max())})
    return pd.DataFrame(rows)


def run_rinconada(prof_model, prof_obs, dur_sessions, sites, candidate, base_model, base_obs, n_reps, seed, n_weekdays):
    rng = np.random.default_rng(seed)
    members = DOWNTOWN + [EXTRA]
    i_r = members.index(EXTRA)
    c_r = int(sites.loc[EXTRA, "ports_2019"])
    pools_dur = [duration_pools(dur_sessions, s) for s in members]
    rows, hourly = [], {}

    def routes_for(names, k):
        xy = {s: (sites.loc[s, "x"], sites.loc[s, "y"]) for s in names}
        pool_names = list(names) + (["NEW"] if k else [])
        if k:
            xy["NEW"] = (candidate["x"], candidate["y"])
        out = []
        for s in names:
            others = sorted((p for p in pool_names if p != s),
                            key=lambda p: np.hypot(xy[p][0] - xy[s][0], xy[p][1] - xy[s][1]))
            out.append([pool_names.index(p) for p in [s] + others])
        return out, pool_names

    for label, prof, base in [("model", prof_model, base_model), ("observed", prof_obs, base_obs)]:
        # reference: Rinconada on its own, with occupancy and time-full tracked
        solo = simulate_routed([prof.loc[EXTRA].values], [pools_dur[i_r]], [c_r], [[0]], n_reps, rng, track_pool=0)
        arr5 = np.sum([base[s]["arrivals"] for s in members], axis=0)
        blk5 = np.sum([base[s]["blocked"] for s in members], axis=0)
        prob5 = np.divide(blk5, arr5, out=np.zeros(24), where=arr5 >= MIN_ARR_PER_DAY)
        blk4 = np.sum([base[s]["blocked"] for s in DOWNTOWN], axis=0)
        rows.append({"rates": label, "scenario": "Baseline, 5 sites separate", "ports": int(sum(sites.loc[s, "ports_2019"] for s in members)),
                     "lost_per_year": float(blk5.sum() * n_weekdays), "lost_ci95": None,
                     "lost_downtown_drivers": float(blk4.sum() * n_weekdays),
                     "lost_rinconada_drivers": float(base[EXTRA]["blocked"].sum() * n_weekdays),
                     "peak_block": float(prob5.max()), "daily_block": float(blk5.sum() / arr5.sum()),
                     "rinconada_peak_util": float(solo["occupancy"][0].max() / c_r),
                     "rinconada_mean_util_8_20": float(solo["occupancy"][0][8:20].mean() / c_r),
                     "rinconada_full_share_8_20": solo["full_share"],
                     "downtown_to_rinconada_per_year": 0.0, "rinconada_to_downtown_per_year": 0.0,
                     "served_at_rinconada_per_weekday": float(solo["placed"].sum())})
        hourly[(label, "alone")] = solo["occupancy"][0].tolist()
        # regression check: the routed model on the four original sites should reproduce the single-pool result
        names4 = DOWNTOWN
        r4 = simulate_routed([prof.loc[s].values for s in names4], pools_dur[:4], [int(sites.loc[s, "ports_2019"]) for s in names4],
                             routes_for(names4, 0)[0], n_reps, rng)
        lost4, ci4 = _lost_stats(r4, n_weekdays, list(range(4)))
        rows.append({"rates": label, "scenario": "Pooled +0, 4 sites (routed check)", "ports": int(sum(sites.loc[s, "ports_2019"] for s in names4)),
                     "lost_per_year": lost4, "lost_ci95": ci4, "lost_downtown_drivers": lost4, "lost_rinconada_drivers": None,
                     "peak_block": float(r4["blocking_prob"].max()), "daily_block": float(r4["blocked"].sum() / r4["arrivals"].sum())})
        for k in [0] + list(C.CANDIDATE_PORTS):
            routes, pool_names = routes_for(members, k)
            caps = [int(sites.loc[s, "ports_2019"]) for s in members] + ([k] if k else [])
            r = simulate_routed([prof.loc[s].values for s in members], pools_dur, caps, routes, n_reps, rng, track_pool=i_r)
            lost, ci = _lost_stats(r, n_weekdays, list(range(5)))
            lost_dt, _ = _lost_stats(r, n_weekdays, list(range(4)))
            lost_r, _ = _lost_stats(r, n_weekdays, [i_r])
            d2r = float(r["placed"][:4, i_r].sum()) * n_weekdays
            r2d = float(r["placed"][i_r, [j for j in range(len(caps)) if j != i_r]].sum()) * n_weekdays
            rows.append({"rates": label, "scenario": f"Pooled +{k}, with Rinconada", "ports": sum(caps),
                         "lost_per_year": lost, "lost_ci95": ci, "lost_downtown_drivers": lost_dt, "lost_rinconada_drivers": lost_r,
                         "peak_block": float(r["blocking_prob"].max()), "daily_block": float(r["blocked"].sum() / r["arrivals"].sum()),
                         "rinconada_peak_util": float(r["occupancy"][i_r].max() / c_r),
                         "rinconada_mean_util_8_20": float(r["occupancy"][i_r][8:20].mean() / c_r),
                         "rinconada_full_share_8_20": r["full_share"],
                         "downtown_to_rinconada_per_year": d2r, "rinconada_to_downtown_per_year": r2d,
                         "served_at_rinconada_per_weekday": float(r["placed"][:, i_r].sum())})
            hourly[(label, f"pooled_{k}")] = r["occupancy"][i_r].tolist()
    return pd.DataFrame(rows), {f"{a}|{b}": v for (a, b), v in hourly.items()}
