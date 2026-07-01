from datetime import datetime, timedelta

import sys

# Add mlops project folder so Airflow can import pipeline modules
sys.path.append("/home/bibek-karki/mlops")

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.ingestion import (
    extract_data,
    validate_data,
    load_to_mariadb,
)

from pipeline.preprocessing import preprocess_data
from pipeline.training import train_model
from pipeline.evaluation import evaluate_model


# ============================================================
# Airflow DAG Definition
# ============================================================

default_args = {
    "owner": "bibek",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="body_performance_ml_pipeline",
    default_args=default_args,
    description="Body Performance ML pipeline using separate pipeline files",
    schedule_interval=None,
    catchup=False,
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    load_task = PythonOperator(
        task_id="load_to_mariadb",
        python_callable=load_to_mariadb,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    extract_task >> validate_task >> load_task >> preprocess_task >> train_task >> evaluate_task