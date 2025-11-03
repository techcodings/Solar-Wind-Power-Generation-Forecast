
import os, requests, pandas as pd

def openmeteo_forecast(lat: float, lon: float, days: int = 7, timezone: str = "UTC"):
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m","relative_humidity_2m","cloud_cover",
        "wind_speed_10m","wind_speed_100m","wind_speed_120m",
        "wind_gusts_10m","shortwave_radiation","direct_radiation",
        "diffuse_radiation","surface_pressure","precipitation"
    ]
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon, "timezone": timezone,
        "hourly": ",".join(hourly_vars),
        "forecast_days": days
    }, timeout=int(os.getenv("OPEN_METEO_TIMEOUT", 30)))
    r.raise_for_status()
    j = r.json()
    hrs = pd.to_datetime(j["hourly"]["time"], utc=True)
    df = pd.DataFrame({"timestamp": hrs})
    for k, v in j["hourly"].items():
        if k == "time": continue
        df[k] = v
    return df

def openmeteo_history(lat: float, lon: float, start_date: str, end_date: str, timezone: str = "UTC"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = [
        "temperature_2m","relative_humidity_2m","cloud_cover",
        "wind_speed_10m","wind_speed_100m","wind_speed_120m",
        "wind_gusts_10m","shortwave_radiation","direct_radiation",
        "diffuse_radiation","surface_pressure","precipitation"
    ]
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon, "timezone": timezone,
        "hourly": ",".join(hourly_vars),
        "start_date": start_date, "end_date": end_date
    }, timeout=int(os.getenv("OPEN_METEO_TIMEOUT", 30)))
    r.raise_for_status()
    j = r.json()
    hrs = pd.to_datetime(j["hourly"]["time"], utc=True)
    df = pd.DataFrame({"timestamp": hrs})
    for k, v in j["hourly"].items():
        if k == "time": continue
        df[k] = v
    return df

def pvgis_radiation(lat: float, lon: float):
    """Fetch PVGIS radiation summary (annual GHI etc.).
    Returns minimal dict; you can expand fields per your needs.
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    # Using defaults for simplicity; PVGIS supports rich parameters.
    params = {
        "lat": lat, "lon": lon, "raddatabase": "PVGIS-SARAH3",
        "startyear": 2020, "endyear": 2024, "outputformat": "json"
    }
    r = requests.get(url, params=params, timeout=int(os.getenv("PVGIS_TIMEOUT", 30)))
    r.raise_for_status()
    j = r.json()
    # Extract quick aggregates from hourly series if available
    ghi = None
    if "outputs" in j and "hourly" in j["outputs"]:
        df = pd.DataFrame(j["outputs"]["hourly"])
        if "G(i)" in df.columns:
            ghi = float(df["G(i)"].mean())
    return {"pvgis_ghi_mean": ghi}

def nsrdb_data_query(lat: float, lon: float, api_key: str, email: str):
    url = "https://developer.nrel.gov/api/solar/nsrdb_data_query.json"
    r = requests.get(url, params={
        "api_key": api_key, "wkt": f"POINT({lon} {lat})", "email": email
    }, timeout=int(os.getenv("NREL_TIMEOUT", 30)))
    r.raise_for_status()
    return r.json()

def nsrdb_poi_monthly(lat: float, lon: float, api_key: str, email: str, names: str = "psm3-5min", year: int = 2024):
    """Fetch monthly means near a POI from NSRDB PSM3 (if allowed).
    Returns dict with ghi/dni/dhi monthly means when available.
    """
    url = "https://developer.nrel.gov/api/solar/nsrdb_psm3_download.json"
    params = {
        "api_key": api_key, "email": email,
        "wkt": f"POINT({lon} {lat})", "names": names, "interval": 60,
        "attributes": "ghi,dni,dhi", "full_name": "User", "reason": "research",
        "affiliation": "Org", "mailing_list": "false", "year": year
    }
    r = requests.get(url, params=params, timeout=int(os.getenv("NREL_TIMEOUT", 30)))
    r.raise_for_status()
    j = r.json()
    # This endpoint may return links; you might need to follow and parse CSV.
    # Here we expose the JSON for downstream processing.
    return j

def global_wind_atlas_stub(lat: float, lon: float):
    """Placeholder for GWA sampling.
    You can replace this with a raster sampler or precomputed CSV lookup.
    Returns a toy mean speed for demo.
    """
    # Simple heuristic by latitude for demo purposes.
    base = 6.5
    return {"gwa_mean_speed_100m": base + (abs(lat)-30)*0.03}
