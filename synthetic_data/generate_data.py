import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)

BASE_PATH = "data/raw"
os.makedirs(BASE_PATH, exist_ok=True)

NUM_ASSETS = 50
DAYS = 180
START_DATE = datetime(2024, 1, 1)

ASSET_TYPES = ["HVAC", "UPS", "Generator", "ServerRack"]

# ---------------- ASSET METADATA ----------------
assets = []
for asset_id in range(1, NUM_ASSETS + 1):
    assets.append({
        "asset_id": asset_id,
        "asset_type": random.choice(ASSET_TYPES),
        "install_date": START_DATE - timedelta(days=random.randint(365, 3000)),
        "capacity_kw": random.randint(50, 500)
    })

asset_df = pd.DataFrame(assets)

# ---------------- SENSOR TELEMETRY ----------------
sensor_records = []

for _, asset in asset_df.iterrows():
    base_temp = random.uniform(20, 30)
    base_power = random.uniform(40, 70)
    base_vibration = random.uniform(0.2, 0.6)

    for day in range(DAYS):
        timestamp = START_DATE + timedelta(days=day)

        sensor_records.append({
            "asset_id": asset.asset_id,
            "timestamp": timestamp,
            "temperature": round(base_temp + np.random.normal(0, 2), 2),
            "power_load": round(base_power + np.random.normal(0, 5), 2),
            "vibration": round(base_vibration + np.random.normal(0, 0.1), 3),
            "humidity": round(random.uniform(30, 70), 2)
        })

sensor_df = pd.DataFrame(sensor_records)

# ---------------- MAINTENANCE LOGS ----------------
maintenance_logs = []
for asset_id in asset_df.asset_id:
    for _ in range(random.randint(2, 6)):
        maintenance_logs.append({
            "asset_id": asset_id,
            "maintenance_date": START_DATE - timedelta(days=random.randint(1, 180)),
            "maintenance_type": random.choice(["Preventive", "Corrective"]),
            "cost_usd": random.randint(500, 5000)
        })

maintenance_df = pd.DataFrame(maintenance_logs)

# ---------------- FAILURE LABEL ----------------
# ---------- FAILURE LABEL ----------
failure_events = []
avg_metrics = sensor_df.groupby("asset_id").mean(numeric_only=True)

# Compute base risk
risk_df = avg_metrics.copy()
risk_df["risk"] = (
    0.35 * (risk_df["temperature"] / 40) +
    0.35 * (risk_df["vibration"] / 1.0) +
    0.30 * (risk_df["power_load"] / 100)
)

# Rank assets by risk (highest risk first)
risk_df = risk_df.sort_values("risk", ascending=False)

# FORCE top-N risky assets to fail (synthetic control)
MIN_FAILURES = 10
risk_df["failure_within_30_days"] = 0
risk_df.iloc[:MIN_FAILURES, risk_df.columns.get_loc("failure_within_30_days")] = 1

# Build final failure table
failure_df = risk_df[["failure_within_30_days"]].reset_index()


# ---------------- SAVE FILES ----------------
asset_df.to_csv(f"{BASE_PATH}/asset_metadata.csv", index=False)
sensor_df.to_csv(f"{BASE_PATH}/sensor_data.csv", index=False)
maintenance_df.to_csv(f"{BASE_PATH}/maintenance_logs.csv", index=False)
failure_df.to_csv(f"{BASE_PATH}/failure_events.csv", index=False)

print("✅ Synthetic data generated successfully")
