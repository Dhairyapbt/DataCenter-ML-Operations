from prefect import flow, task
import subprocess


@task(retries=2, retry_delay_seconds=10)
def generate_data():
    subprocess.run(
        ["python", "synthetic_data/generate_data.py"],
        check=True
    )
    print("✅ Synthetic data generated")


@task(retries=2, retry_delay_seconds=10)
def build_features():
    subprocess.run(
        ["python", "features/feature_engineer.py"],
        check=True
    )
    print("✅ Features created")


@task(retries=2, retry_delay_seconds=10)
def train_model():
    subprocess.run(
        ["python", "training/train_failure_model.py"],
        check=True
    )
    print("✅ Models trained")


@task
def run_drift_check():
    subprocess.run(
        ["python", "monitoring/data_drift.py"],
        check=True
    )
    print("✅ Drift check completed")


@flow(name="DataCenter-ML-Operations-Pipeline")
def datacenter_ml_pipeline():
    generate_data()
    build_features()
    train_model()
    run_drift_check()


if __name__ == "__main__":
    datacenter_ml_pipeline()
