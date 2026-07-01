import json
import shutil

import joblib
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from pipeline.config import (
    MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    BEST_MODEL_INFO_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    ACCURACY_THRESHOLD,
    CLASS_NAMES,
    X_TEST_KEY,
    Y_TEST_KEY,
    MODEL_METADATA_KEY,
    BEST_MODEL_INFO_KEY,
)

from pipeline.utils import (
    load_df_from_redis,
    get_redis_client,
)

# ============================================================
# Helper - Save Confusion Matrix as PNG
# ============================================================

def save_confusion_matrix_png(cm, class_names, output_path, title="Confusion Matrix"):
    """
    Save confusion matrix as a PNG heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "black"

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Stage 3 - Model Evaluation + MLflow Registry
# ============================================================

def evaluate_model():
    """
    Model evaluation
    -> MLflow experiment tracking
    -> Register best model in MLflow Model Registry
    -> Save best model info for FastAPI deployment
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_test = load_df_from_redis(X_TEST_KEY)
    y_test = load_df_from_redis(Y_TEST_KEY)

    y_test = y_test["target"].values.ravel()

    redis_client = get_redis_client()
    metadata_bytes = redis_client.get(MODEL_METADATA_KEY)

    if metadata_bytes is None:
        raise ValueError(
            "No model metadata found in Redis. "
            "The train_model task may not have completed successfully."
        )

    model_metadata = json.loads(metadata_bytes.decode("utf-8"))

    evaluation_results = []

    for model_info in model_metadata:

        model_name = model_info["model_name"]
        run_id = model_info["run_id"]
        model_path = model_info["model_path"]
        mlflow_model_uri = model_info["mlflow_model_uri"]

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, predictions)

        model_report_path = CLASSIFICATION_REPORT_PATH.replace(
            "classification_report.txt",
            f"{model_name}_classification_report.txt",
        )

        model_confusion_matrix_path = CONFUSION_MATRIX_PATH.replace(
            "confusion_matrix.txt",
            f"{model_name}_confusion_matrix.png",
        )

        with open(model_report_path, "w") as file:
            file.write(f"{model_name} Classification Report\n")
            file.write("====================================\n\n")
            file.write(report)

        save_confusion_matrix_png(
            cm=cm,
            class_names=CLASS_NAMES,
            output_path=model_confusion_matrix_path,
            title=f"{model_name} Confusion Matrix",
        )

        with mlflow.start_run(run_id=run_id):

            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision_weighted", precision)
            mlflow.log_metric("test_recall_weighted", recall)
            mlflow.log_metric("test_f1_weighted", f1)

            mlflow.log_artifact(model_report_path)
            mlflow.log_artifact(model_confusion_matrix_path)

        evaluation_results.append(
            {
                "model_name": model_name,
                "run_id": run_id,
                "model_path": model_path,
                "mlflow_model_uri": mlflow_model_uri,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "classification_report_path": model_report_path,
                "confusion_matrix_path": model_confusion_matrix_path,
            }
        )

        print(f"{model_name} evaluation completed.")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    best_model = max(evaluation_results, key=lambda x: x["f1"])

    print("Best model selected:")
    print("Model:", best_model["model_name"])
    print("Accuracy:", best_model["accuracy"])
    print("F1 Score:", best_model["f1"])

    best_model_object = joblib.load(best_model["model_path"])
    joblib.dump(best_model_object, MODEL_PATH)

    with open(best_model["classification_report_path"], "r") as src:
        best_report = src.read()

    with open(CLASSIFICATION_REPORT_PATH, "w") as dst:
        dst.write(best_report)

    final_confusion_matrix_path = CONFUSION_MATRIX_PATH.replace(
        "confusion_matrix.txt",
        "confusion_matrix.png",
    )

    shutil.copyfile(
        best_model["confusion_matrix_path"],
        final_confusion_matrix_path,
    )

    registered_model = mlflow.register_model(
        model_uri=best_model["mlflow_model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    client = mlflow.tracking.MlflowClient()

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="best_model_name",
        value=best_model["model_name"],
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="test_accuracy",
        value=str(best_model["accuracy"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="test_f1_weighted",
        value=str(best_model["f1"]),
    )

    try:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="production",
            version=registered_model.version,
        )

        production_model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"

    except Exception:
        production_model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}/{registered_model.version}"
        )

    best_model_info = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": registered_model.version,
        "production_model_uri": production_model_uri,
        "best_model_name": best_model["model_name"],
        "best_run_id": best_model["run_id"],
        "accuracy": best_model["accuracy"],
        "precision": best_model["precision"],
        "recall": best_model["recall"],
        "f1": best_model["f1"],
        "local_model_path": MODEL_PATH,
        "classification_report_path": CLASSIFICATION_REPORT_PATH,
        "confusion_matrix_path": final_confusion_matrix_path,
    }

    with open(BEST_MODEL_INFO_PATH, "w") as file:
        json.dump(best_model_info, file, indent=4)

    redis_client.set(
        BEST_MODEL_INFO_KEY,
        json.dumps(best_model_info).encode("utf-8"),
    )

    print("Best model registered in MLflow Model Registry.")
    print("Production model URI:", production_model_uri)
    print("Best model info saved at:", BEST_MODEL_INFO_PATH)

    if best_model["accuracy"] < ACCURACY_THRESHOLD:
        raise ValueError(
            f"Best model accuracy {best_model['accuracy']} "
            f"is below threshold {ACCURACY_THRESHOLD}"
        )

    print("Evaluate task completed successfully.")