
import numpy as np, pandas as pd, os, json

np.random.seed(21)

start = pd.Timestamp("2025-04-01 00:00:00", tz="UTC")
end   = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
idx = pd.date_range(start, end, freq="H")

with open("config/regions.json","r",encoding="utf-8") as f:
    regions = json.load(f)

sources = ["Solar","Wind"]

def gen(ts, region_name, source):
    t = np.arange(len(ts))
    daily = np.sin(2*np.pi*(t % 24)/24)
    weekly = 0.5*np.sin(2*np.pi*(t % (24*7))/(24*7))
    trend = 0.0006*t
    r_scale = 1.0 + 0.1*np.random.rand()
    if source=="Solar":
        base = 14*np.maximum(daily,0.0)**1.4
        noise = np.random.normal(0,1.5,size=len(ts))
        y = r_scale*(base*(1+weekly) + noise + 6 + trend*4)
        y = np.clip(y, 0, None)
    else:
        night = np.cos(2*np.pi*(t % 24)/24)
        base = 11 + 3.2*night + 1.6*weekly
        gust = np.random.normal(0,1.8,size=len(ts))
        y = r_scale*(base + gust + trend*3)
        y = np.clip(y, 0, None)
    return y

recs = []
for reg in regions:
    for s in sources:
        y = gen(idx, reg["region"], s)
        recs.append(pd.DataFrame({"timestamp": idx, "region": reg["region"], "source": s, "mw": y}))

df = pd.concat(recs, ignore_index=True).sort_values("timestamp")
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic.csv", index=False)
print("Wrote data/synthetic.csv with", len(df), "rows")
