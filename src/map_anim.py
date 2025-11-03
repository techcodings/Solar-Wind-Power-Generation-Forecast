
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animated_map(forecast_df: pd.DataFrame, region_coords: dict, out_gif: str):
    daily_region = (
        forecast_df.assign(day=lambda d: d["timestamp"].dt.floor("D"))
                  .groupby(["day","region"], as_index=False)["mw_hat"].sum()
    )
    days = sorted(daily_region["day"].unique())
    coords = pd.DataFrame([{"region": r, "lat": region_coords[r][0], "lon": region_coords[r][1]} for r in region_coords])

    frames = []
    for d in days:
        tmp = daily_region[daily_region["day"]==d].merge(coords, on="region", how="left")
        frames.append(tmp)

    fig, ax = plt.subplots(figsize=(6,4))
    all_lats = [v[0] for v in region_coords.values()]
    all_lons = [v[1] for v in region_coords.values()]
    lat_pad = 5; lon_pad = 5
    ax.set_xlim(min(all_lons)-lon_pad, max(all_lons)+lon_pad)
    ax.set_ylim(min(all_lats)-lat_pad, max(all_lats)+lat_pad)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Daily Regional Generation (Forecast)")

    def update(i):
        ax.clear()
        ax.set_xlim(min(all_lons)-lon_pad, max(all_lons)+lon_pad)
        ax.set_ylim(min(all_lats)-lat_pad, max(all_lats)+lat_pad)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("Daily Regional Generation (Forecast)")
        df = frames[i]
        sizes = 50 + (df["mw_hat"] / max(df["mw_hat"].max(), 1.0)) * 300
        ax.scatter(df["lon"], df["lat"], s=sizes)
        for _, row in df.iterrows():
            ax.text(row["lon"], row["lat"], f"{row['region']}\n{row['mw_hat']:.0f} MW", ha="center", va="bottom")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=800, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=1))
    plt.close(fig)
