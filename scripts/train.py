# scripts/train_now.py  (no argparse import)
import os, sys, json, pandas as pd

# ---- ensure project root on sys.path ----
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- project imports ----
from src.data import load_timeseries
from src.external_sources import openmeteo_history, pvgis_radiation, global_wind_atlas_stub
from src.forecast import train_per_group
from src.config import REGISTRY_PATH

# ---- EDIT THESE IF YOU WANT ----
DATA_PATH = os.environ.get("DATA_PATH", "data/synthetic.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
REGISTRY = os.environ.get("REGISTRY_PATH", str(REGISTRY_PATH))
HIST_START = os.environ.get("HISTORY_START", "2024-01-01")
HIST_END   = os.environ.get("HISTORY_END",   "2025-10-31")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = load_timeseries(DATA_PATH)

# Load region registry
with open(REGISTRY, "r", encoding="utf-8") as f:
    registry = json.load(f)
reg_df = pd.DataFrame(registry)

# Enrich static features (PVGIS & GWA)
if "pvgis_ghi_mean" not in reg_df.columns or reg_df["pvgis_ghi_mean"].isna().any():
    for i, row in reg_df.iterrows():
        try:
            info = pvgis_radiation(row["lat"], row["lon"])
            reg_df.loc[i, "pvgis_ghi_mean"] = info.get("pvgis_ghi_mean")
        except Exception:
            reg_df.loc[i, "pvgis_ghi_mean"] = None

if "gwa_mean_speed_100m" not in reg_df.columns or reg_df["gwa_mean_speed_100m"].isna().any():
    for i, row in reg_df.iterrows():
        reg_df.loc[i, "gwa_mean_speed_100m"] = global_wind_atlas_stub(row["lat"], row["lon"])["gwa_mean_speed_100m"]

# Save enriched registry back
with open(REGISTRY, "w", encoding="utf-8") as f:
    json.dump(reg_df.to_dict(orient="records"), f, indent=2)

# Fetch historical weather per region & merge
hist_weather = []
for _, row in reg_df.iterrows():
    w = openmeteo_history(row["lat"], row["lon"], start_date=HIST_START, end_date=HIST_END)
    w["region"] = row["region"]
    hist_weather.append(w)
import pandas as pd
w_all = pd.concat(hist_weather, ignore_index=True)
df = df.merge(w_all, on=["region","timestamp"], how="left")

# Add site_id
df["site_id"] = df["region"] + "-" + df["source"]

# Train
train_per_group(df, reg_df, MODEL_DIR)
print("Training complete. Models saved to", MODEL_DIR)
