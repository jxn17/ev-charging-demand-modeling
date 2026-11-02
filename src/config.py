"""Central configuration: paths, study window, seeds and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "raw" / "palo-alto-ev-charging" / "data" / "dataset.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE = OUT / "cache"
RESULTS_JSON = OUT / "results.json"
for _d in (OUT, FIG, TAB, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Study window and temporal split (strict, no shuffling)
PANEL_START = "2016-12-01"      # one extra month so 24 h / 168 h lags exist on 2017-01-01
STUDY_START = "2017-01-01"
STUDY_END = "2019-12-31 23:00"
TRAIN_END = "2018-12-31 23:00"  # train = 2017-2018
VAL_START = "2018-10-01"        # 2018-Q4 held out for tuning only
TEST_START = "2019-01-01"       # test = all of 2019

SITES = ["BRYANT", "CAMBRIDGE", "HAMILTON", "HIGH", "MPL", "RINCONADA LIB", "TED THOMPSON", "WEBSTER"]
EXCLUDED_STATIONS = {"PALO ALTO CA / BRYANT # 1"}   # stray 2020 record; SHERMAN handled by prefix
OUTAGE_GAP_DAYS = 30           # a port with no session for > 30 days is treated as out of service
SITE_HOUR_FROM = 6             # site x hour interactions for hours 06..23 (overnight shares one profile)

# Spatial analysis
GRID_RES_M = 40.0
BUFFER_M = 1000.0
BANDWIDTH_M = 400.0            # ~5-minute walk catchment
BANDWIDTH_SENS = [250.0, 400.0, 600.0]
GAP_QUANTILE = 0.95            # cells above this quantile of G(x) enter DBSCAN
DBSCAN_MIN_SAMPLES = 4         # 2 x dimensionality (Ester et al. 1996 / Sander et al. 1998)
MIN_DIST_TO_SITE_M = 150.0

# Simulation
N_REPS = 500
CANDIDATE_PORTS = [2, 4]
