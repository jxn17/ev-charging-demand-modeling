"""Loading and cleaning of the City of Palo Alto EV charging session log.

Every filtering step is logged so that the paper's cleaning table is generated
from the code, not written by hand.
"""
import re

import numpy as np
import pandas as pd

from . import config as C

SITE_RE = re.compile(r"^PALO ALTO CA / (.+?)\s*#?\s*\d+$")


def hms_to_hours(s: pd.Series) -> pd.Series:
    """'h:mm:ss' strings (hours may exceed 24) -> float hours."""
    parts = s.astype(str).str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return h + m / 60.0 + sec / 3600.0


def site_of(station_name: str) -> str:
    m = SITE_RE.match(station_name)
    return m.group(1).strip() if m else station_name


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.DATA_PATH, encoding="utf-8-sig", low_memory=False)


def clean(raw: pd.DataFrame):
    """Return (clean sessions, cleaning log DataFrame)."""
    log = []
    df = raw.copy()
    log.append(("Raw records (Jul 2011 - Dec 2020)", 0, len(df)))

    def step(name, mask_keep):
        nonlocal df
        before = len(df)
        df = df[mask_keep].copy()
        log.append((name, before - len(df), len(df)))

    step("Exact duplicate rows", ~df.duplicated())
    sherman = df["Station Name"].str.contains("SHERMAN", na=False)
    stray = df["Station Name"].isin(C.EXCLUDED_STATIONS)
    step("Excluded stations (SHERMAN, 'BRYANT # 1')", ~(sherman | stray))

    df["start"] = pd.to_datetime(df["Start Date"], format="%m/%d/%Y %H:%M", errors="coerce")
    step("Unparseable start time / UTC-stamped records", df["start"].notna() & (df["Start Time Zone"] != "UTC"))

    df["duration_h"] = hms_to_hours(df["Total Duration (hh:mm:ss)"])
    df["charging_h"] = hms_to_hours(df["Charging Time (hh:mm:ss)"])
    step("Zero / negative plug-in duration", df["duration_h"] > 0)
    step("Plug-in duration > 24 h", df["duration_h"] <= 24)
    step("Energy < 0.1 kWh", df["Energy (kWh)"] >= 0.1)
    step("Level-1 (120 V outlet) sessions", df["Port Type"] == "Level 2")

    df["site"] = df["Station Name"].map(site_of)
    assert set(df["site"].unique()) == set(C.SITES), df["site"].unique()
    df["port_id"] = df["Station Name"] + " | P" + df["Port Number"].astype(str)
    df["energy_kwh"] = df["Energy (kWh)"]
    df["fee"] = df["Fee"]
    df["lat"] = df["Latitude"]
    df["lon"] = df["Longitude"]
    df = df.sort_values("start").reset_index(drop=True)
    log_df = pd.DataFrame(log, columns=["step", "removed", "remaining"])
    return df, log_df


def study_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["start"] >= C.STUDY_START) & (df["start"] <= C.STUDY_END)].copy()


def fee_start_date(df: pd.DataFrame) -> pd.Timestamp:
    """First day from which the 7-day share of paid sessions stays above 50 %."""
    d = df.set_index("start")["fee"].gt(0).resample("D").mean()
    roll = d.rolling(7, min_periods=3).mean()
    after = roll[roll > 0.5]
    first = after.index[0] - pd.Timedelta(days=6)
    # align to the first day that actually has a paid session
    paid_days = d[(d > 0) & (d.index >= first)]
    return paid_days.index[0].normalize()


def port_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Active intervals for every Level-2 port.

    A port is active from the day of its first session to the end of the day of
    its last session (full 2011-2020 history), minus any gap longer than
    OUTAGE_GAP_DAYS between consecutive sessions, which is treated as an outage.
    """
    rows = []
    gap = pd.Timedelta(days=C.OUTAGE_GAP_DAYS)
    for (site, port), g in df.groupby(["site", "port_id"]):
        t = g["start"].sort_values().reset_index(drop=True)
        start = t.iloc[0].normalize()
        d = t.diff()
        for i in np.where(d > gap)[0]:
            end = t.iloc[i - 1].normalize() + pd.Timedelta(days=1)
            rows.append((site, port, start, end, False))
            start = t.iloc[i].normalize()
        rows.append((site, port, start, t.iloc[-1].normalize() + pd.Timedelta(days=1), False))
    act = pd.DataFrame(rows, columns=["site", "port_id", "active_from", "active_to", "_"]).drop(columns="_")
    return act


def active_ports_panel(act: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Number of active Level-2 ports per site per hour (hours x sites)."""
    out = pd.DataFrame(0, index=hours, columns=C.SITES, dtype=np.int16)
    for r in act.itertuples():
        m = (hours >= r.active_from) & (hours < r.active_to)
        out.loc[m, r.site] += 1
    return out


def site_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("site")[["lat", "lon"]].median().loc[C.SITES]
