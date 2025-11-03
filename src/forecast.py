# top of src/forecast.py
import os, json, numpy as np, pandas as pd
from joblib import dump, load
from src.models import GBMPointModel, QuantileGBM
from src.config import FORECAST_HOURS, QUANTILES, LAGS
from src.features import add_calendar_features, add_lags, encode_cats, merge_weather, attach_static
from src.external_sources import openmeteo_forecast

def _feat_cols(df_cols):
    base = ["hour","dow","dom","month","is_weekend","region_code","source_code","site_id_code"]
    lags = [f"lag_{L}" for L in LAGS]
    weather = [c for c in df_cols if c not in base+lags and c not in ["timestamp","region","source","site_id","mw"]]
    return base + lags + weather

def train_per_group(df: pd.DataFrame, registry_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = attach_static(df, registry_df)
    df = add_calendar_features(df)
    df = add_lags(df)
    df = encode_cats(df)
    groups = df.groupby(["region","source"], sort=False)

    meta = []
    for (region, source), g in groups:
        if g["mw"].count() < max(LAGS)+24*14:
            continue
        g_feat = g.dropna(subset=[f"lag_{L}" for L in LAGS])
        feats = _feat_cols(g_feat.columns)

        X = g_feat[feats].values
        y = g_feat["mw"].values

        # point model as GBM
        m_point = GBMPointModel()
        m_point.fit(X, y, feats=feats)
        dump(m_point, f"{out_dir}/model_point_{region}_{source}.joblib")

        # quantiles
        for q in QUANTILES:
            m_q = QuantileGBM(q)
            m_q.fit(X, y)
            dump(m_q, f"{out_dir}/model_q{int(q*100)}_{region}_{source}.joblib")

        meta.append({"region": region, "source": source, "features": feats})
    pd.DataFrame(meta).to_csv(f"{out_dir}/groups_trained.csv", index=False)

def forecast_per_group(df_hist: pd.DataFrame, registry_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    rows = []
    groups = df_hist.groupby(["region","source"], sort=False)
    regmap = registry_df.set_index("region").to_dict(orient="index")
    for (region, source), g in groups:
        try:
            m_point = load(f"{model_dir}/model_point_{region}_{source}.joblib")
        except:
            continue

        last_ts = g["timestamp"].max()
        future = pd.DataFrame({"timestamp": pd.date_range(last_ts + pd.Timedelta(hours=1), periods=FORECAST_HOURS, freq="h", tz="UTC")})

        lat, lon = regmap[region]["lat"], regmap[region]["lon"]
        wfc = openmeteo_forecast(lat, lon, days=7)
        wfc = wfc[wfc["timestamp"].isin(future["timestamp"])].reset_index(drop=True)

        fut = future.copy()
        fut["region"] = region
        fut["source"] = source
        fut["site_id"] = f"{region}-{source}"
        fut = attach_static(fut, registry_df)
        fut = merge_weather(fut, wfc)
        fut = add_calendar_features(fut)

        # lags from history + rolling with own mean predictions (simple)
        from collections import deque
        hist = g.sort_values("timestamp")
        lag_buf = deque(hist.set_index("timestamp")["mw"].iloc[-max(LAGS):].values.tolist(), maxlen=max(LAGS))
        for L in LAGS:
            fut[f"lag_{L}"] = np.nan
        # We'll fill lags after we get point predictions iteratively.
        fut = encode_cats(fut)

        feats = m_point.feats or [c for c in fut.columns if c not in ["timestamp","region","source","site_id","mw"]]

        # iterative build: fill lags step by step
        preds = []
        for i in range(len(fut)):
            for L in LAGS:
                fut.loc[i, f"lag_{L}"] = lag_buf[-L] if len(lag_buf) >= L else np.nan
            xi = fut.iloc[[i]][feats].fillna(method="ffill").fillna(method="bfill").to_numpy()
            yhat = float(m_point.forecast(xi))
            preds.append(yhat)
            lag_buf.append(yhat)

        mean = np.array(preds)

        # quantiles
        try:
            qlo = load(f"{model_dir}/model_q{int(QUANTILES[0]*100)}_{region}_{source}.joblib")
            qhi = load(f"{model_dir}/model_q{int(QUANTILES[1]*100)}_{region}_{source}.joblib")
            Xf = fut[feats].fillna(method="ffill").fillna(method="bfill").values
            lo = qlo.predict(Xf); hi = qhi.predict(Xf)
        except:
            lo = mean*0.85; hi = mean*1.15

        rows.append(pd.DataFrame({
            "timestamp": fut["timestamp"],
            "region": region, "source": source,
            "mw_hat": mean, "mw_lo": lo, "mw_hi": hi
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
