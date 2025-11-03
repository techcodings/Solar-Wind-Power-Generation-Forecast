# scripts/forecast_now.py  (no argparse; robust path handling)
import os, sys, json, pandas as pd
from pathlib import Path

# ---- locate project root (folder that contains 'src' and 'scripts') ----
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- project imports ----
from src.data import load_timeseries
from src.forecast import forecast_per_group
from src.peaks import peak_hours
from src.config import REGISTRY_PATH

# ---- inputs via env vars (with defaults) ----
DATA_PATH   = os.environ.get("DATA_PATH", "data/synthetic.csv")
MODEL_DIR   = os.environ.get("MODEL_DIR", "models")
REGISTRY    = os.environ.get("REGISTRY_PATH", str(REGISTRY_PATH))
OUT_DIR     = os.environ.get("OUT_DIR", "out")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load data & registry ----
df = load_timeseries(DATA_PATH)
with open(REGISTRY, "r", encoding="utf-8") as f:
    registry = json.load(f)
reg_df = pd.DataFrame(registry)

# ---- forecast ----
fc = forecast_per_group(df, reg_df, MODEL_DIR)
if fc.empty:
    raise SystemExit("No forecasts produced; ensure models are trained and registry has regions.")

# ---- outputs ----
fc.to_csv(os.path.join(OUT_DIR, "forecast_7d.csv"), index=False)
peaks = peak_hours(fc)
peaks.to_csv(os.path.join(OUT_DIR, "peak_hours.csv"), index=False)

print("Wrote:", os.path.join(OUT_DIR, "forecast_7d.csv"))
print("Wrote:", os.path.join(OUT_DIR, "peak_hours.csv"))
