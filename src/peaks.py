
import pandas as pd

def peak_hours(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    df["date"] = df["timestamp"].dt.date
    peaks = (
        df.loc[df.groupby(["region","source","date"])["mw_hat"].idxmax()]
          .sort_values(["region","source","date"]).rename(columns={"mw_hat":"peak_mw_forecast"})
          .reset_index(drop=True)
    )
    return peaks
