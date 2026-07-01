# ============================================================
# Body Performance Prediction API
# Stage 4 - Model Deployment
#
# Flow:
# User JSON Request
# -> FastAPI POST /predict
# -> Input Validation
# -> Load Registered Production Model from MLflow
# -> Predict Class
# -> Store Prediction Log in MariaDB
# -> Return Prediction Response
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import sys
import json
import uuid
from datetime import datetime
from typing import Literal, Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text


# Add mlops project folder so API can import pipeline config
sys.path.append("/home/bibek-karki/mlops")

from pipeline.config import (
    DEFAULT_CONN_STRING,
    MODEL_PATH,
    SCALER_PATH,
    BEST_MODEL_INFO_PATH,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    CLASS_NAMES,
)


# ============================================================
# 2. Global Variables
# ============================================================

PREDICTION_LOG_TABLE = "prediction_log"

MODEL = None
SCALER = None

MODEL_INFO = {
    "model_source": None,
    "model_name": None,
    "model_version": None,
    "production_model_uri": None,
}


# ============================================================
# 3. FastAPI App
# Similar to installation guide style
# ============================================================

app = FastAPI(title="Body Performance Prediction API")


# ============================================================
# 4. Request and Response Schema
# ============================================================

class BodyPerformanceInput(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["M", "F"]

    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=20, le=200)
    body_fat_percent: float = Field(..., ge=1, le=60)

    diastolic: int = Field(..., ge=40, le=130)
    systolic: int = Field(..., ge=60, le=220)

    grip_force: float = Field(..., ge=0, le=100)
    sit_and_bend_forward_cm: float = Field(..., ge=-50, le=250)

    sit_ups_counts: int = Field(..., ge=0, le=100)
    broad_jump_cm: int = Field(..., ge=0, le=400)

    # Optional field for monitoring performance later.
    # If actual_class is provided, monitoring can calculate accuracy.
    # If not provided, it is stored as NULL.
    actual_class: Optional[Literal["A", "B", "C", "D"]] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_class: str
    actual_class: Optional[str]
    confidence_score: Optional[float]
    model_source: str
    model_version: Optional[str]
    prediction_timestamp: str


# ============================================================
# 5. Helper Functions
# ============================================================

def get_input_dict(data: BodyPerformanceInput):
    """
    Supports both Pydantic v1 and Pydantic v2.
    """

    if hasattr(data, "model_dump"):
        return data.model_dump()

    return data.dict()


def create_db_engine():
    """
    Creates SQLAlchemy engine for MariaDB.
    """

    return create_engine(DEFAULT_CONN_STRING)


def create_prediction_log_table():
    """
    Creates prediction_log table if it does not already exist.

    Also tries to add actual_class column if the table already exists
    from an older version of the API.
    """

    engine = create_db_engine()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {PREDICTION_LOG_TABLE} (
        prediction_id VARCHAR(64),
        age INT,
        gender VARCHAR(5),
        height_cm DOUBLE,
        weight_kg DOUBLE,
        body_fat_percent DOUBLE,
        diastolic INT,
        systolic INT,
        grip_force DOUBLE,
        sit_and_bend_forward_cm DOUBLE,
        sit_ups_counts INT,
        broad_jump_cm INT,
        predicted_class VARCHAR(5),
        actual_class VARCHAR(5),
        confidence_score DOUBLE,
        model_source VARCHAR(100),
        model_name VARCHAR(255),
        model_version VARCHAR(100),
        prediction_timestamp DATETIME
    )
    """

    with engine.begin() as conn:
        conn.execute(text(create_table_query))

        # If prediction_log table already existed before actual_class was added,
        # this safely tries to add the column.
        try:
            conn.execute(
                text(
                    f"ALTER TABLE {PREDICTION_LOG_TABLE} "
                    f"ADD COLUMN actual_class VARCHAR(5)"
                )
            )
            print("actual_class column added to prediction_log table.")

        except Exception:
            # Column probably already exists.
            print("actual_class column already exists or could not be added.")

    engine.dispose()


def load_best_model_info():
    """
    Reads best_model_info.json created by the evaluation stage.
    """

    try:
        with open(BEST_MODEL_INFO_PATH, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        return None


def load_production_model():
    """
    Loads registered production model from MLflow Model Registry.

    If MLflow registered model loading fails, it falls back to the
    local best model saved by the evaluation stage.
    """

    global MODEL, SCALER, MODEL_INFO

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_model_info = load_best_model_info()

    production_model_uri = None
    model_version = None
    best_model_name = None

    if best_model_info is not None:
        production_model_uri = best_model_info.get("production_model_uri")
        model_version = str(best_model_info.get("registered_model_version"))
        best_model_name = best_model_info.get("best_model_name")

    if production_model_uri is None:
        production_model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"

    try:
        MODEL = mlflow.sklearn.load_model(production_model_uri)

        MODEL_INFO = {
            "model_source": "mlflow_registry",
            "model_name": best_model_name or REGISTERED_MODEL_NAME,
            "model_version": model_version,
            "production_model_uri": production_model_uri,
        }

        print("Production model loaded from MLflow:")
        print(production_model_uri)

    except Exception as error:
        print("MLflow model loading failed. Falling back to local model.")
        print("Reason:", error)

        MODEL = joblib.load(MODEL_PATH)

        MODEL_INFO = {
            "model_source": "local_joblib_fallback",
            "model_name": best_model_name or "local_best_model",
            "model_version": model_version,
            "production_model_uri": MODEL_PATH,
        }

        print("Local model loaded from:", MODEL_PATH)

    SCALER = joblib.load(SCALER_PATH)

    print("Scaler loaded from:", SCALER_PATH)


def validate_logical_rules(input_data):
    """
    Custom validation rule:
    systolic must be greater than diastolic.
    """

    if input_data["systolic"] <= input_data["diastolic"]:
        raise HTTPException(
            status_code=422,
            detail="Validation error: systolic must be greater than diastolic.",
        )


def preprocess_input(input_data):
    """
    Applies the same preprocessing used during training:
    - remove actual_class because it is not an input feature
    - encode gender
    - arrange columns in training order
    - scale numerical features using saved scaler
    """

    processed_data = input_data.copy()

    # actual_class is only for monitoring, not prediction input
    processed_data.pop("actual_class", None)

    processed_data["gender"] = 1 if processed_data["gender"] == "M" else 0

    input_df = pd.DataFrame([processed_data])

    input_df = input_df[FEATURE_COLUMNS]

    input_df[NUMERIC_FEATURE_COLUMNS] = SCALER.transform(
        input_df[NUMERIC_FEATURE_COLUMNS]
    )

    return input_df


def predict_class(input_df):
    """
    Predicts class and confidence score.
    """

    prediction_numeric = MODEL.predict(input_df)[0]

    try:
        prediction_numeric = int(prediction_numeric)
        predicted_class = CLASS_NAMES[prediction_numeric]

    except Exception:
        predicted_class = str(prediction_numeric)

    confidence_score = None

    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(input_df)[0]

        try:
            class_list = list(MODEL.classes_)
            prediction_index = class_list.index(prediction_numeric)
            confidence_score = float(probabilities[prediction_index])

        except Exception:
            confidence_score = float(max(probabilities))

    return predicted_class, confidence_score


def store_prediction_log(
    prediction_id,
    input_data,
    predicted_class,
    confidence_score,
    prediction_timestamp,
):
    """
    Stores prediction log in MariaDB prediction_log table.
    Includes actual_class if provided.
    """

    engine = create_db_engine()

    insert_query = f"""
    INSERT INTO {PREDICTION_LOG_TABLE} (
        prediction_id,
        age,
        gender,
        height_cm,
        weight_kg,
        body_fat_percent,
        diastolic,
        systolic,
        grip_force,
        sit_and_bend_forward_cm,
        sit_ups_counts,
        broad_jump_cm,
        predicted_class,
        actual_class,
        confidence_score,
        model_source,
        model_name,
        model_version,
        prediction_timestamp
    )
    VALUES (
        :prediction_id,
        :age,
        :gender,
        :height_cm,
        :weight_kg,
        :body_fat_percent,
        :diastolic,
        :systolic,
        :grip_force,
        :sit_and_bend_forward_cm,
        :sit_ups_counts,
        :broad_jump_cm,
        :predicted_class,
        :actual_class,
        :confidence_score,
        :model_source,
        :model_name,
        :model_version,
        :prediction_timestamp
    )
    """

    log_data = {
        "prediction_id": prediction_id,
        "age": input_data["age"],
        "gender": input_data["gender"],
        "height_cm": input_data["height_cm"],
        "weight_kg": input_data["weight_kg"],
        "body_fat_percent": input_data["body_fat_percent"],
        "diastolic": input_data["diastolic"],
        "systolic": input_data["systolic"],
        "grip_force": input_data["grip_force"],
        "sit_and_bend_forward_cm": input_data["sit_and_bend_forward_cm"],
        "sit_ups_counts": input_data["sit_ups_counts"],
        "broad_jump_cm": input_data["broad_jump_cm"],
        "predicted_class": predicted_class,
        "actual_class": input_data.get("actual_class"),
        "confidence_score": confidence_score,
        "model_source": MODEL_INFO["model_source"],
        "model_name": MODEL_INFO["model_name"],
        "model_version": MODEL_INFO["model_version"],
        "prediction_timestamp": prediction_timestamp,
    }

    with engine.begin() as conn:
        conn.execute(text(insert_query), log_data)

    engine.dispose()


# ============================================================
# 6. Startup Event
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Runs when FastAPI starts:
    - creates prediction_log table
    - adds actual_class column if needed
    - loads production model
    - loads scaler
    """

    create_prediction_log_table()
    load_production_model()


# ============================================================
# 7. API Endpoints
# ============================================================

@app.get("/")
def read_root():
    return {
        "message": "Body Performance Prediction API is running",
        "main_prediction_endpoint": "POST /predict",
        "model_source": MODEL_INFO["model_source"],
        "model_version": MODEL_INFO["model_version"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "prediction_log_table": PREDICTION_LOG_TABLE,
    }


@app.get("/model/info")
def model_info():
    return MODEL_INFO


@app.post("/predict", response_model=PredictionResponse)
def predict(data: BodyPerformanceInput):
    """
    Main prediction endpoint.

    If actual_class is provided, it is stored for monitoring accuracy.
    If actual_class is not provided, prediction still works normally.
    """

    input_data = get_input_dict(data)

    validate_logical_rules(input_data)

    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=500,
            detail="Model or scaler is not loaded.",
        )

    prediction_id = str(uuid.uuid4())
    prediction_timestamp = datetime.now()

    try:
        input_df = preprocess_input(input_data)

        predicted_class, confidence_score = predict_class(input_df)

        store_prediction_log(
            prediction_id=prediction_id,
            input_data=input_data,
            predicted_class=predicted_class,
            confidence_score=confidence_score,
            prediction_timestamp=prediction_timestamp,
        )

        return {
            "prediction_id": prediction_id,
            "predicted_class": predicted_class,
            "actual_class": input_data.get("actual_class"),
            "confidence_score": confidence_score,
            "model_source": MODEL_INFO["model_source"],
            "model_version": MODEL_INFO["model_version"],
            "prediction_timestamp": prediction_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )