import os
import json

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pipeline.config import (
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    X_TRAIN_KEY,
    Y_TRAIN_KEY,
    MODEL_METADATA_KEY,
)

from pipeline.utils import (
    load_df_from_redis,
    get_redis_client,
)


# ============================================================
# Stage 3 - Model Training
# ============================================================

def train_model():
    """
    Redis cached training datasets
    -> PyArrow deserialization
    -> Train RandomForestClassifier and LogisticRegression
    -> MLflow experiment tracking
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train = load_df_from_redis(X_TRAIN_KEY)
    y_train = load_df_from_redis(Y_TRAIN_KEY)

    y_train = y_train["target"].values.ravel()

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }

    model_metadata = []

    for model_name, model in models.items():

        with mlflow.start_run(run_name=f"{model_name}_Training") as run:

            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)

            candidate_model_path = MODEL_DIR + f"/{model_name}_model.pkl"
            joblib.dump(model, candidate_model_path)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)

            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", 100)

            if model_name == "LogisticRegression":
                mlflow.log_param("max_iter", 1000)

            mlflow.log_metric("train_accuracy", train_accuracy)

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(candidate_model_path)

            model_metadata.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "model_path": candidate_model_path,
                    "mlflow_model_uri": f"runs:/{run.info.run_id}/model",
                    "train_accuracy": train_accuracy,
                }
            )

            print(f"{model_name} training completed.")
            print("Training accuracy:", train_accuracy)
            print("MLflow run id:", run.info.run_id)

    redis_client = get_redis_client()
    redis_client.set(
        MODEL_METADATA_KEY,
        json.dumps(model_metadata).encode("utf-8"),
    )

    print("All models trained successfully.")
    print("Model metadata saved in Redis.")