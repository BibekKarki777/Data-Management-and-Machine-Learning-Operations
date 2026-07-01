# ============================================================
# Stage 5 - Model Monitoring
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import sys

import pandas as pd
from sqlalchemy import create_engine

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import ClassificationPreset


# Add project path so we can import config.py
sys.path.append("/home/bibek-karki/mlops")

from pipeline.config import (
    DEFAULT_CONN_STRING,
    OBT_TABLE_NAME,
    FEATURE_COLUMNS,
    REPORT_DIR,
)


# ============================================================
# 2. Global Variables
# ============================================================

PREDICTION_LOG_TABLE = "prediction_log"

MONITORING_REPORT_DIR = REPORT_DIR + "/monitoring"

DRIFT_REPORT_PATH = MONITORING_REPORT_DIR + "/data_drift_report.html"
PERFORMANCE_REPORT_PATH = MONITORING_REPORT_DIR + "/performance_report.html"
DASHBOARD_REPORT_PATH = MONITORING_REPORT_DIR + "/monitoring_dashboard.html"


# ============================================================
# 3. Load Data from MariaDB
# ============================================================

def load_monitoring_data():
    """
    Loads baseline OBT data and prediction_log data from MariaDB.
    """

    os.makedirs(MONITORING_REPORT_DIR, exist_ok=True)

    engine = create_engine(DEFAULT_CONN_STRING)

    # Baseline data from Stage 1 ingestion
    baseline_df = pd.read_sql(
        f"SELECT * FROM {OBT_TABLE_NAME}",
        con=engine,
    )

    # Current prediction data from FastAPI predictions
    prediction_log_df = pd.read_sql(
        f"SELECT * FROM {PREDICTION_LOG_TABLE}",
        con=engine,
    )

    engine.dispose()

    print("Baseline OBT shape:", baseline_df.shape)
    print("Prediction log shape:", prediction_log_df.shape)

    return baseline_df, prediction_log_df


# ============================================================
# 4. Generate Data Drift Report
# ============================================================

def generate_drift_report(baseline_df, prediction_log_df):
    """
    Generates data_drift_report.html using Evidently DataDriftPreset.
    Reference data = baseline OBT table
    Current data = prediction_log input values
    """

    reference = baseline_df[FEATURE_COLUMNS].copy()
    current = prediction_log_df[FEATURE_COLUMNS].copy()

    print("Reference drift data shape:", reference.shape)
    print("Current drift data shape:", current.shape)

    drift_report = Report(metrics=[DataDriftPreset()])

    drift_report.run(
        reference_data=reference,
        current_data=current,
    )

    drift_report.save_html(DRIFT_REPORT_PATH)

    print("Drift report saved to:", DRIFT_REPORT_PATH)


# ============================================================
# 5. Generate Performance Report
# ============================================================

def generate_performance_report(prediction_log_df):
    """
    Generates performance_report.html using Evidently ClassificationPreset.

    This requires actual_class in prediction_log.
    If actual_class is missing or empty, performance report cannot calculate
    real accuracy.
    """

    if "actual_class" not in prediction_log_df.columns:
        print("actual_class column not found. Cannot generate performance report.")
        return

    performance_df = prediction_log_df.dropna(
        subset=["actual_class", "predicted_class"]
    ).copy()

    if performance_df.empty:
        print("No rows with actual_class found. Cannot generate performance report.")
        return

    # Evidently expects target and prediction column names
    performance_df = performance_df.rename(
        columns={
            "actual_class": "target",
            "predicted_class": "prediction",
        }
    )

    # Use only the columns needed for classification performance
    performance_df = performance_df[["target", "prediction"]]

    # Split prediction logs into reference/current parts for performance comparison
    split_index = int(len(performance_df) * 0.5)

    if split_index == 0:
        reference = performance_df.copy()
        current = performance_df.copy()
    else:
        reference = performance_df.iloc[:split_index].copy()
        current = performance_df.iloc[split_index:].copy()

    print("Reference performance data shape:", reference.shape)
    print("Current performance data shape:", current.shape)

    performance_report = Report(metrics=[ClassificationPreset()])

    performance_report.run(
        reference_data=reference,
        current_data=current,
    )

    performance_report.save_html(PERFORMANCE_REPORT_PATH)

    print("Performance report saved to:", PERFORMANCE_REPORT_PATH)


# ============================================================
# 6. Generate Combined Monitoring Dashboard
# ============================================================

def generate_dashboard_report(baseline_df, prediction_log_df):
    """
    Generates combined monitoring_dashboard.html.

    This combines:
    - DataDriftPreset
    - ClassificationPreset if actual_class is available
    """

    reference_drift = baseline_df[FEATURE_COLUMNS].copy()
    current_drift = prediction_log_df[FEATURE_COLUMNS].copy()

    if (
        "actual_class" in prediction_log_df.columns
        and "predicted_class" in prediction_log_df.columns
        and prediction_log_df["actual_class"].notna().sum() > 0
    ):
        performance_df = prediction_log_df.dropna(
            subset=["actual_class", "predicted_class"]
        ).copy()

        performance_df = performance_df.rename(
            columns={
                "actual_class": "target",
                "predicted_class": "prediction",
            }
        )

        performance_df = performance_df[["target", "prediction"]]

        split_index = int(len(performance_df) * 0.5)

        if split_index == 0:
            reference_perf = performance_df.copy()
            current_perf = performance_df.copy()
        else:
            reference_perf = performance_df.iloc[:split_index].copy()
            current_perf = performance_df.iloc[split_index:].copy()

        dashboard_report = Report(
            metrics=[
                DataDriftPreset(),
                ClassificationPreset(),
            ]
        )

        # For combined report, use performance data because ClassificationPreset
        # needs target and prediction columns.
        dashboard_report.run(
            reference_data=reference_perf,
            current_data=current_perf,
        )

    else:
        dashboard_report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        dashboard_report.run(
            reference_data=reference_drift,
            current_data=current_drift,
        )

    dashboard_report.save_html(DASHBOARD_REPORT_PATH)

    print("Monitoring dashboard saved to:", DASHBOARD_REPORT_PATH)


# ============================================================
# 7. Main Function
# ============================================================

def run_monitoring():
    """
    Runs the full monitoring process.
    """

    baseline_df, prediction_log_df = load_monitoring_data()

    generate_drift_report(
        baseline_df=baseline_df,
        prediction_log_df=prediction_log_df,
    )

    generate_performance_report(
        prediction_log_df=prediction_log_df,
    )

    generate_dashboard_report(
        baseline_df=baseline_df,
        prediction_log_df=prediction_log_df,
    )

    print("Monitoring completed successfully.")


# ============================================================
# 8. Run File Directly
# ============================================================

if __name__ == "__main__":
    run_monitoring()