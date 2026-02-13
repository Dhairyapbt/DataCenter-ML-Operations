import pandas as pd
import numpy as np
import os

BASELINE_PATH = os.path.join("data", "features", "training_features.csv")
NEW_DATA_PATH = os.path.join("data", "features", "training_features.csv")  # simulate prod
OUTPUT_PATH = os.path.join("monitoring", "drift_report.csv")
os.makedirs("monitoring", exist_ok=True)


def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    psi_value = 0
    for i in range(buckets):
        expected_count = np.mean((expected >= expected_perc[i]) & (expected < expected_perc[i + 1]))
        actual_count = np.mean((actual >= actual_perc[i]) & (actual < actual_perc[i + 1]))

        expected_count = max(expected_count, 1e-6)
        actual_count = max(actual_count, 1e-6)

        psi_value += (actual_count - expected_count) * np.log(actual_count / expected_count)

    return psi_value


def run_data_drift():
    baseline = pd.read_csv(BASELINE_PATH)
    new_data = pd.read_csv(NEW_DATA_PATH)

    numeric_features = baseline.select_dtypes(include=[np.number]).columns
    numeric_features = [c for c in numeric_features if c != "failure_within_30_days"]

    drift_results = []

    for col in numeric_features:
        psi = calculate_psi(baseline[col], new_data[col])
        drift_results.append({
            "feature": col,
            "psi": round(psi, 4),
            "drift_status": (
                "NO_DRIFT" if psi < 0.1 else
                "MODERATE_DRIFT" if psi < 0.25 else
                "SEVERE_DRIFT"
            )
        })

    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv(OUTPUT_PATH, index=False)

    print("📊 Data Drift Report")
    print(drift_df.sort_values("psi", ascending=False))


if __name__ == "__main__":
    run_data_drift()
