import pandas as pd
import os
from datetime import datetime

DATA_PATH = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join("data", "features")
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data():
    sensor = pd.read_csv(f"{DATA_PATH}/sensor_data.csv", parse_dates=["timestamp"])
    asset = pd.read_csv(f"{DATA_PATH}/asset_metadata.csv", parse_dates=["install_date"])
    maintenance = pd.read_csv(f"{DATA_PATH}/maintenance_logs.csv", parse_dates=["maintenance_date"])
    failure = pd.read_csv(f"{DATA_PATH}/failure_events.csv")
    return sensor, asset, maintenance, failure


def build_sensor_features(sensor_df):
    agg = sensor_df.groupby("asset_id").agg(
        mean_temperature=("temperature", "mean"),
        max_temperature=("temperature", "max"),
        std_temperature=("temperature", "std"),
        mean_power_load=("power_load", "mean"),
        mean_vibration=("vibration", "mean"),
        humidity_mean=("humidity", "mean")
    ).reset_index()
    return agg


def build_maintenance_features(maintenance_df):
    maintenance_df["is_corrective"] = maintenance_df["maintenance_type"].eq("Corrective").astype(int)

    agg = maintenance_df.groupby("asset_id").agg(
        maintenance_count=("maintenance_type", "count"),
        corrective_ratio=("is_corrective", "mean"),
        avg_maintenance_cost=("cost_usd", "mean")
    ).reset_index()
    return agg


def build_asset_features(asset_df):
    today = datetime.today()
    asset_df["asset_age_days"] = (today - asset_df["install_date"]).dt.days
    return asset_df[["asset_id", "asset_type", "capacity_kw", "asset_age_days"]]


def build_feature_table():
    sensor, asset, maintenance, failure = load_data()

    sensor_feat = build_sensor_features(sensor)
    maint_feat = build_maintenance_features(maintenance)
    asset_feat = build_asset_features(asset)

    features = (
        sensor_feat
        .merge(maint_feat, on="asset_id", how="left")
        .merge(asset_feat, on="asset_id", how="left")
        .merge(failure, on="asset_id", how="left")
    )

    features.fillna(0, inplace=True)

    features.to_csv(f"{OUTPUT_PATH}/training_features.csv", index=False)
    print("✅ Feature table created at data/features/training_features.csv")


if __name__ == "__main__":
    build_feature_table()
