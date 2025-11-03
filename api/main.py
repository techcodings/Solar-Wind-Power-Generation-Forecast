# api/main.py
import os
import json
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.data import load_timeseries
from src.forecast import forecast_per_group
from src.peaks import peak_hours
from src.map_anim import animated_map
from src.config import REGISTRY_PATH

# --------------------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------------------
app = FastAPI(title="Renewables 7-day Forecast API (Weather-aware)")

# Optional CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # replace with your UI origin(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Env/config defaults
DATA_PATH = os.environ.get("DATA_PATH", "data/synthetic.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
REGISTRY = os.environ.get("REGISTRY_PATH", str(REGISTRY_PATH))
OUT_DIR = os.environ.get("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def load_registry_df(path: Optional[str] = None) -> pd.DataFrame:
    """Load regions registry JSON into a DataFrame."""
    reg_path = path or REGISTRY
    with open(reg_path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

# --------------------------------------------------------------------------------------
# Pydantic request models for POST endpoints
# --------------------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    region: Optional[str] = None
    source: Optional[str] = None
    horizon_hours: int = Field(default=168, ge=1, le=168)
    # If you later train more quantiles, you can accept these and route accordingly.
    quantiles: Optional[List[float]] = None  # e.g. [0.05, 0.95]

class PeaksRequest(BaseModel):
    region: Optional[str] = None
    source: Optional[str] = None

class MapRequest(BaseModel):
    regions: Optional[List[str]] = None
    gif_name: Optional[str] = "regional_animation.gif"

class TrainRequest(BaseModel):
    data_path: str = "data/synthetic.csv"
    history_start: str = "2024-01-01"
    history_end: str = "2025-10-31"
    registry_path: str = str(REGISTRY_PATH)
    model_dir: str = "models"

# --------------------------------------------------------------------------------------
# GET endpoints (filter via query params)
# --------------------------------------------------------------------------------------
@app.get("/forecast")
def forecast_get(region: Optional[str] = None, source: Optional[str] = None):
    df = load_timeseries(DATA_PATH)
    reg_df = load_registry_df()
    fc = forecast_per_group(df, reg_df, MODEL_DIR)
    if region:
        fc = fc[fc["region"] == region]
    if source:
        fc = fc[fc["source"] == source]
    return [] if fc.empty else fc.to_dict(orient="records")

@app.get("/peaks")
def peaks_get(region: Optional[str] = None, source: Optional[str] = None):
    df = load_timeseries(DATA_PATH)
    reg_df = load_registry_df()
    fc = forecast_per_group(df, reg_df, MODEL_DIR)
    if fc.empty:
        return []
    pk = peak_hours(fc)
    if region:
        pk = pk[pk["region"] == region]
    if source:
        pk = pk[pk["source"] == source]
    return pk.to_dict(orient="records")

@app.get("/map")
def map_get():
    df = load_timeseries(DATA_PATH)
    reg_df = load_registry_df()
    fc = forecast_per_group(df, reg_df, MODEL_DIR)
    gif_path = os.path.join(OUT_DIR, "regional_animation.gif")
    coords = {r.region: [r.lat, r.lon] for r in reg_df.itertuples()}
    animated_map(fc, coords, gif_path)
    return {"gif_path": gif_path}

@app.get("/map.gif")
def map_gif():
    path = os.path.join(OUT_DIR, "regional_animation.gif")
    return FileResponse(path, media_type="image/gif")

# --------------------------------------------------------------------------------------
# POST endpoints (user-driven payloads)
# --------------------------------------------------------------------------------------
@app.post("/forecast")
def forecast_post(req: ForecastRequest):
    df = load_timeseries(os.environ.get("DATA_PATH", DATA_PATH))
    reg_df = load_registry_df(os.environ.get("REGISTRY_PATH", REGISTRY))
    model_dir = os.environ.get("MODEL_DIR", MODEL_DIR)

    fc = forecast_per_group(df, reg_df, model_dir)
    if fc.empty:
        return []

    # Filter by user inputs
    if req.region:
        fc = fc[fc["region"] == req.region]
    if req.source:
        fc = fc[fc["source"] == req.source]

    # Trim horizon if requested (< 168)
    if req.horizon_hours < 168 and not fc.empty:
        tmax = fc["timestamp"].max()
        tmin = tmax - pd.Timedelta(hours=req.horizon_hours - 1)
        fc = fc[(fc["timestamp"] >= tmin) & (fc["timestamp"] <= tmax)]

    return fc.to_dict(orient="records")

@app.post("/peaks")
def peaks_post(req: PeaksRequest):
    df = load_timeseries(os.environ.get("DATA_PATH", DATA_PATH))
    reg_df = load_registry_df(os.environ.get("REGISTRY_PATH", REGISTRY))
    model_dir = os.environ.get("MODEL_DIR", MODEL_DIR)

    fc = forecast_per_group(df, reg_df, model_dir)
    if fc.empty:
        return []

    pk = peak_hours(fc)
    if req.region:
        pk = pk[pk["region"] == req.region]
    if req.source:
        pk = pk[pk["source"] == req.source]

    return pk.to_dict(orient="records")

@app.post("/map")
def map_post(req: MapRequest):
    df = load_timeseries(os.environ.get("DATA_PATH", DATA_PATH))
    reg_df = load_registry_df(os.environ.get("REGISTRY_PATH", REGISTRY))
    model_dir = os.environ.get("MODEL_DIR", MODEL_DIR)

    fc = forecast_per_group(df, reg_df, model_dir)
    if req.regions:
        fc = fc[fc["region"].isin(req.regions)]

    os.makedirs(OUT_DIR, exist_ok=True)
    gif_path = os.path.join(OUT_DIR, req.gif_name or "regional_animation.gif")
    coords = {r.region: [r.lat, r.lon] for r in reg_df.itertuples()}
    animated_map(fc, coords, gif_path)
    return {"gif_path": gif_path}

@app.post("/train")
def train_post(req: TrainRequest):
    """
    Launch training in-process using the same logic as scripts/train.py.
    Expects files already present on disk (data/registry). For file uploads,
    add a multipart endpoint separately.
    """
    from src.external_sources import openmeteo_history, pvgis_radiation, global_wind_atlas_stub
    from src.forecast import train_per_group

    os.makedirs(req.model_dir, exist_ok=True)

    # Load inputs
    df_hist = load_timeseries(req.data_path)
    with open(req.registry_path, "r", encoding="utf-8") as f:
        reg_df = pd.DataFrame(json.load(f))

    # Enrich static (PVGIS/GWA)
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

    # Historical weather per region
    hist_weather = []
    for _, row in reg_df.iterrows():
        w = openmeteo_history(row["lat"], row["lon"], start_date=req.history_start, end_date=req.history_end)
        w["region"] = row["region"]
        hist_weather.append(w)
    if hist_weather:
        w_all = pd.concat(hist_weather, ignore_index=True)
        df_hist = df_hist.merge(w_all, on=["region", "timestamp"], how="left")

    # Site id for modeling
    df_hist["site_id"] = df_hist["region"] + "-" + df_hist["source"]

    # Train and persist models
    train_per_group(df_hist, reg_df, req.model_dir)

    # Persist enriched registry
    with open(req.registry_path, "w", encoding="utf-8") as f:
        json.dump(reg_df.to_dict(orient="records"), f, indent=2)

    # Report how many models/groups were trained (best-effort)
    groups_csv = os.path.join(req.model_dir, "groups_trained.csv")
    try:
        n = len(pd.read_csv(groups_csv))
    except Exception:
        n = None

    return {"status": "ok", "models_dir": req.model_dir, "groups_trained": n}

# --------------------------------------------------------------------------------------
# Optional: run with `python -m uvicorn api.main:app --reload --port 8080`
# --------------------------------------------------------------------------------------
