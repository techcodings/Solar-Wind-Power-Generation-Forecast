
import pandas as pd
import numpy as np
from src.config import LAGS

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek
    out["dom"] = out["timestamp"].dt.day
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["dow"]>=5).astype(int)
    return out

def add_lags(df: pd.DataFrame, group_cols=("region","source","site_id")) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp")
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(list(group_cols))["mw"].shift(lag)
    return df

def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["region","source","site_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
            out[f"{col}_code"] = out[col].cat.codes
    return out

def merge_weather(power_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in weather_df.columns if c != "timestamp"]
    return power_df.merge(weather_df[["timestamp"] + cols], on="timestamp", how="left")

def attach_static(power_df: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    # registry_df: columns [region, lat, lon, elevation, ... static features ...]
    return power_df.merge(registry_df, on="region", how="left")
