
# Renewables Forecast Backend (Weather-aware)

Implements 7-day hourly forecasts with Open-Meteo weather, static PVGIS/NSRDB/GWA features, peak-hour extraction, and an animated regional map.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Generate sample data and regions
python scripts/generate_synth.py

# 2) Train (pulls Open-Meteo historical weather for regions in config/regions.json)
python scripts/train.py --data data/synthetic.csv --history_start 2024-01-01 --history_end 2025-10-31

# 3) Forecast artifacts via CLI
python scripts/forecast_cli.py --data data/synthetic.csv --out out/

# 4) Run API
export DATA_PATH=data/synthetic.csv MODEL_DIR=models REGISTRY_PATH=config/regions.json OUT_DIR=out
uvicorn api.main:app --reload --port 8080
```

## Inputs wired
- **Open-Meteo**: `src/external_sources.py` (forecast + historical)
- **PVGIS**: `pvgis_radiation()` enriches region registry with mean GHI
- **NSRDB**: sample helpers to query availability/links (add your API key/email)
- **Global Wind Atlas**: `global_wind_atlas_stub()` placeholder (swap with raster sampling or precomputed CSV)

## Files to customize
- `config/regions.json` — add your regions with `lat/lon` and (optionally) pre-fill static features
- `src/features.py` — add more lags or derived weather features
- `src/models.py` — replace/stack models (Prophet/LightGBM/etc.)

## Endpoints
- `GET /forecast?region=&source=` — 7-day hourly rows with `mw_hat`, `mw_lo`, `mw_hi`
- `GET /peaks?region=&source=` — peak hour per day
- `GET /map` — generates GIF and returns its path
- `GET /map.gif` — serves the latest GIF
```
