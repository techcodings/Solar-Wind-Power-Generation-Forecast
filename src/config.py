
from pathlib import Path

# Horizon / features
FORECAST_HOURS = 24 * 7
SEASONAL_PERIOD = 24
LAGS = [1, 24, 48, 168]
QUANTILES = [0.05, 0.95]

# Data & model paths
DATA_PATH = Path("data/synthetic.csv")  # override via env in api/main.py
MODEL_DIR = Path("models")
OUT_DIR = Path("out")
REGISTRY_PATH = Path("config/regions.json")  # region lat/lon + static features

# External sources
OPEN_METEO_TIMEOUT = 30
PVGIS_TIMEOUT = 30
NREL_TIMEOUT = 30

# If NSRDB is used, set via env or .env for scripts:
# NREL_API_KEY, NREL_EMAIL
