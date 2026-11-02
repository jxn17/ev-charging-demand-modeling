"""Discrete-event simulation of charging sites as M_t/G/c/c loss systems.

Arrivals: non-homogeneous Poisson process with the model's hourly rate
(Lewis-Shedler thinning). Service: plug-in durations resampled from the site's
empirical 2019 sessions that started in the same hour of day. A driver who
finds all c ports busy is lost (turned away).
"""
import heapq
import math

import numpy as np
import pandas as pd

from . import config as C

MIN_PER_HOUR_SAMPLE = 30
MIN_ARR_PER_DAY = 0.1   # min. expected arrivals in an hour for its blocking probability to be reported
N_DAYS = 2          # day 1 = warm-up (overnight occupancy), day 2 = measured


# --------------------------------------------------------------------------- Erlang formulas
def erlang_b(c: int, a: float) -> float:
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c: int, a: float):
    if a >= c:
        return None
    b = erlang_b(c, a)
    return c * b / (c - a * (1 - b))


# --------------------------------------------------------------------------- inputs
def duration_pools(sessions_2019: pd.DataFrame, site: str):
    s = sessions_2019[sessions_2019["site"] == site]
    pooled = s["duration_h"].values
    by_hour = []
    for h in range(24):
        v = s.loc[s["start"].dt.hour == h, "duration_h"].values
        by_hour.append(v if len(v) >= MIN_PER_HOUR_SAMPLE else pooled)
    return by_hour


def weekday_profile(p: pd.DataFrame, col: str) -> pd.DataFrame:
    t = p[(p["split"] == "test") & (p["weekend"] == 0) & (p["holiday"] == 0)]
    return t.groupby(["site", "hour"])[col].mean().unstack()


# --------------------------------------------------------------------------- core simulator
def simulate_system(lam_by_source, dur_by_source, caps, routes, n_reps, rng):
    """General loss network.

    lam_by_source : list of 24-vectors, arrival rate (per h) of each source site
    dur_by_source : list of by-hour duration pools (hours) of each source site
    caps          : capacity (ports) of each pool
    routes        : for each source, the ordered list of pools a driver tries; if all are full the driver is lost
    Returns per-hour (measured day) mean arrivals, blocked drivers and time-average occupied ports.
    """
    lam_by_source = [np.asarray(l, float) for l in lam_by_source]
    lam_tot = np.sum(lam_by_source, axis=0)
    lam_max = lam_tot.max()
    share = np.array(lam_by_source) / np.where(lam_tot > 0, lam_tot, 1)   # P(source | hour)
    T = 24.0 * N_DAYS
    arr = np.zeros(24); blk = np.zeros(24); occ = np.zeros(24)
    for _ in range(n_reps):
        n = rng.poisson(lam_max * T)                                  # Lewis-Shedler thinning
        cand = np.sort(rng.uniform(0, T, n))
        hrs = (cand % 24).astype(int)
        keep = rng.uniform(size=n) < lam_tot[hrs] / lam_max
        times, hrs = cand[keep], hrs[keep]
        if len(lam_by_source) == 1:
            srcs = np.zeros(len(times), int)
        else:
            u = rng.uniform(size=len(times))
            srcs = np.minimum((u[:, None] > np.cumsum(share[:, hrs].T, axis=1)).sum(axis=1), len(lam_by_source) - 1)
        u_dur = rng.uniform(size=len(times))
        busy = [[] for _ in caps]                                     # heaps of departure times per pool
        served = []
        for t, h, src, ud in zip(times, hrs, srcs, u_dur):
            measured = t >= 24.0
            if measured:
                arr[h] += 1
            placed = False
            for k in routes[src]:
                hp = busy[k]
                while hp and hp[0] <= t:
                    heapq.heappop(hp)
                if len(hp) < caps[k]:
                    pool = dur_by_source[src][h]
                    sv = pool[int(ud * len(pool))]
                    heapq.heappush(hp, t + sv)
                    served.append((t, t + sv))
                    placed = True
                    break
            if not placed and measured:
                blk[h] += 1
        if served:
            sv = np.array(served)
            for h in range(24):
                a_, b_ = 24.0 + h, 25.0 + h
                occ[h] += np.clip(np.minimum(sv[:, 1], b_) - np.maximum(sv[:, 0], a_), 0, None).sum()
    # hourly blocking is reported only for hours with >= MIN_ARR_PER_DAY expected arrivals (ratio of tiny counts is noise)
    return {"arrivals": arr / n_reps, "blocked": blk / n_reps, "occupancy": occ / n_reps,
            "blocking_prob": np.divide(blk, arr, out=np.zeros(24), where=arr / n_reps >= MIN_ARR_PER_DAY)}


# --------------------------------------------------------------------------- validation data
def observed_occupancy(sessions_2019: pd.DataFrame, site: str, hol_dates) -> np.ndarray:
    """Mean number of plugged-in vehicles per hour of a 2019 non-holiday weekday, from start + duration."""
    s = sessions_2019[sessions_2019["site"] == site]
    t0 = pd.Timestamp("2018-12-31")
    st = ((s["start"] - t0).dt.total_seconds() / 60).round().astype(int).values
    en = st + np.round(s["duration_h"].values * 60).astype(int)
    N = 366 * 24 * 60 + 24 * 60
    diff = np.zeros(N + 1)
    np.add.at(diff, np.clip(st, 0, N), 1)
    np.add.at(diff, np.clip(en, 0, N), -1)
    occ_min = np.cumsum(diff)[:N]
    idx = pd.date_range(t0, periods=N, freq="min")
    ser = pd.Series(occ_min, index=idx)
    ser = ser[(ser.index >= "2019-01-01") & (ser.index < "2020-01-01")]
    hourly = ser.resample("h").mean()
    keep = (hourly.index.dayofweek < 5) & (~hourly.index.normalize().isin(hol_dates))
    return hourly[keep].groupby(hourly[keep].index.hour).mean().values, float(ser.max())


# --------------------------------------------------------------------------- experiments
def run_baseline(profile, sessions_2019, dur_sessions, sites, n_reps, seed, hol_dates):
    """dur_sessions: 2019 weekday sessions used for service times; sessions_2019: all 2019 sessions (occupancy)."""
    rng = np.random.default_rng(seed)
    res, val_rows, erl_rows = {}, [], []
    for s in C.SITES:
        c = int(sites.loc[s, "ports_2019"])
        pools = duration_pools(dur_sessions, s)
        r = simulate_system([profile.loc[s].values], [pools], [c], [[0]], n_reps, rng)
        res[s] = r
        obs, obs_max = observed_occupancy(sessions_2019, s, hol_dates)
        res[s]["obs_occupancy"] = obs
        val_rows.append({"site": s, "ports": c, "max_observed_concurrent": obs_max,
                         "occ_MAE": float(np.mean(np.abs(r["occupancy"] - obs))),
                         "occ_corr": float(np.corrcoef(r["occupancy"], obs)[0, 1]),
                         "peak_occ_sim": float(r["occupancy"].max()), "peak_occ_obs": float(obs.max()),
                         "daily_port_hours_sim": float(r["occupancy"].sum()), "daily_port_hours_obs": float(obs.sum())})
        hpk = int(np.argmax(profile.loc[s].values))
        lam = float(profile.loc[s].values[hpk])
        ES = float(dur_sessions.loc[dur_sessions["site"] == s, "duration_h"].mean())
        a = lam * ES
        erl_rows.append({"site": s, "ports": c, "peak_arrival_hour": hpk, "lambda_peak_per_h": lam, "mean_service_h": ES,
                         "offered_load_erlang": a, "erlang_B": erlang_b(c, a), "erlang_C": erlang_c(c, a),
                         "sim_block_at_peak_arrival_hour": float(r["blocking_prob"][hpk]),
                         "sim_max_hourly_block": float(r["blocking_prob"].max()),
                         "sim_max_block_hour": int(np.argmax(r["blocking_prob"])),
                         "sim_daily_block": float(r["blocked"].sum() / r["arrivals"].sum()),
                         "sim_turned_away_per_day": float(r["blocked"].sum())})
    return res, pd.DataFrame(val_rows).set_index("site"), pd.DataFrame(erl_rows).set_index("site")


def run_scenarios(profile, dur_sessions, sites, candidates, base_res, n_reps, seed, n_weekdays):
    """Baseline vs. (i) overflow to k new ports at the candidate and (ii) full pooling of the
    cluster sites with the candidate (drivers redirected to any free port within the cluster)."""
    rng = np.random.default_rng(seed + 1)
    rows = []

    def row(cl, members, name, k, ctot, r_arr, r_blk):
        prob = np.divide(r_blk, r_arr, out=np.zeros(24), where=r_arr >= MIN_ARR_PER_DAY)
        return {"cluster": cl, "sites": ", ".join(members), "scenario": name, "added_ports": k, "total_ports": ctot,
                "peak_block_prob": float(prob.max()), "peak_block_hour": int(np.argmax(prob)),
                "daily_block_prob": float(r_blk.sum() / r_arr.sum()), "turned_away_per_weekday": float(r_blk.sum()),
                "turned_away_per_year": float(r_blk.sum() * n_weekdays), "hourly_block": prob.tolist()}

    for _, cand in candidates.iterrows():
        cl = int(cand["cluster"])
        members = list(cand["sites_pooled"]) or [cand["nearest_site"]]
        m = len(members)
        lam = [profile.loc[s].values for s in members]
        pools = [duration_pools(dur_sessions, s) for s in members]
        cs = [int(sites.loc[s, "ports_2019"]) for s in members]
        b_arr = np.sum([base_res[s]["arrivals"] for s in members], axis=0)
        b_blk = np.sum([base_res[s]["blocked"] for s in members], axis=0)
        rows.append(row(cl, members, "Baseline (separate sites)", 0, sum(cs), b_arr, b_blk))
        for k in C.CANDIDATE_PORTS:   # overflow: try own site first, then the new candidate site
            r = simulate_system(lam, pools, cs + [k], [[i, m] for i in range(m)], n_reps, rng)
            rows.append(row(cl, members, f"Overflow to {k} new ports", k, sum(cs) + k, r["arrivals"], r["blocked"]))
        for k in [0] + list(C.CANDIDATE_PORTS):   # pooling: any free port in the cluster
            r = simulate_system(lam, pools, [sum(cs) + k], [[0]] * m, n_reps, rng)
            name = "Pooled, no new ports" if k == 0 else f"Pooled + {k} new ports"
            rows.append(row(cl, members, name, k, sum(cs) + k, r["arrivals"], r["blocked"]))
    return pd.DataFrame(rows)
