# Predictive Spatiotemporal Demand Modeling for EV Charging Station Placement

ML lab mini-project (B.Tech CSE AI & ML): Prathamesh Patil, Srijan Bera, Snehal Tyagi.

The pipeline runs **forecast → cluster → recommend → evaluate** on real City of Palo Alto EV
charging sessions. It uses interpretable count models (Poisson / negative binomial with a
log-port exposure offset), a KDE demand–supply gap surface with DBSCAN hotspots, and a
loss-system discrete-event simulation of the recommended capacity.

Every number, table and figure in the paper is produced by the code in this repository.

## Layout

```
src/
  config.py     paths, study window, split dates, seeds, hyper-parameters
  data.py       loading, cleaning (logged), port active windows, exposure
  features.py   site x hour panel, calendar / lag / rolling features, design matrices
  models.py     M0-M4, Cameron-Trivedi test, NB2 profile-likelihood alpha, metrics, daily models
  spatial.py    projection, KDE demand/supply/gap surfaces, k-distance, DBSCAN, candidates
  simulate.py   NHPP-thinning loss-system simulator, Erlang-B/C, scenarios
  pooling.py    Rinconada pooling rerun: per-site routing, utilization, precision check
  audit.py      checks of whether each model finds what it should (busy hours, per-site errors, rankings)
  figures.py    all figures (300 dpi PNG)
run_all.py      runs everything, writes outputs/tables/*.csv, outputs/figures/*.png, outputs/results.json
paper/
  make_paper.py           builds the IEEE paper, paper/EV_Charging_Demand_Paper.docx, from results.json + figures
  docx_utils.py           python-docx helpers (native Word equations via MathML -> OMML)
  synopsis_text.py        retained synopsis sections and the reference list
  docx_to_pdf.ps1         docx -> PDF via Microsoft Word (or use LibreOffice, see below)
reports/
  make_revised_report.py  revised report with the Rinconada rerun filled in
  make_model_summary.py   summary of how well each model does its job
viva_deck/                source of the viva slides
outputs/
  figures/  tables/  results.json  cache/ (intermediate pickles, not committed)
raw/palo-alto-ev-charging/data/dataset.csv   input data (not committed; see step 2)
```

## Reproduce

1. Python 3.12 environment:
   ```bash
   python -m venv .venv
   .venv/Scripts/python -m pip install -r requirements.txt     # Windows (use .venv/bin/python on Linux/macOS)
   ```
2. Data (City of Palo Alto Open Data, mirrored on GitHub/Kaggle):
   ```bash
   git clone --depth 1 https://github.com/andriysavcyn/palo-alto-ev-charging.git raw/palo-alto-ev-charging
   ```
3. Run the full pipeline. It takes about 5 minutes on a laptop:
   ```bash
   .venv/Scripts/python run_all.py
   ```
   Use `--reuse-models` to skip refitting and regenerate only the downstream stages and figures.
4. Build the paper and convert it to PDF:
   ```bash
   .venv/Scripts/python paper/make_paper.py
   powershell -File paper/docx_to_pdf.ps1 -Docx paper/EV_Charging_Demand_Paper.docx
   # or: soffice --headless --convert-to pdf --outdir paper paper/EV_Charging_Demand_Paper.docx
   ```
5. Build the reports:
   ```bash
   .venv/Scripts/python reports/make_revised_report.py
   .venv/Scripts/python reports/make_model_summary.py
   ```

All random seeds are fixed (`SEED = 42` in `src/config.py`); the simulation uses `numpy.random.default_rng(42)`.

## Key design decisions (details in the paper)

- **Station key:** `Station Name`, not `Address 1`, because the addresses are inconsistent. A site is the
  station name with the `PALO ALTO CA / ` prefix and the trailing `#n` removed. SHERMAN and the stray
  `BRYANT # 1` are dropped.
- **Exposure:** active Level-2 ports per site per hour. Each port is active from its first to its last
  session; gaps of more than 30 days count as outages. `log(active ports)` enters the count models as an offset.
- **Split:** train 2017–2018, test 2019, with no shuffling. 2018-Q4 is held out for every tuning or
  specification choice (HGB grid, trend handling, site interactions).
- **No leakage:** every lag and rolling feature looks back at least 24 h. The test-set forecasts are
  one-day-ahead. The trend term is frozen at its end-of-training value outside the training window.
