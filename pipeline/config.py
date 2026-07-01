import os
from urllib.parse import quote_plus

# ============================================================
# Global Variables
# Lab-style configuration
# ============================================================

RANDOM_STATE = 42

# ============================================================
# Project folder configuration
# ============================================================

PROJECT_HOME = "/home/bibek-karki/mlops"

AIRFLOW_HOME = PROJECT_HOME + "/airflow"
DAGS_FOLDER = AIRFLOW_HOME + "/dags"

DATA_DIR = PROJECT_HOME + "/data"
DATA_PATH = DATA_DIR + "/body_performance.csv"

MODEL_DIR = PROJECT_HOME + "/mlflow/model"
REPORT_DIR = PROJECT_HOME + "/reports"
VALIDATION_REPORT_DIR = REPORT_DIR + "/validation"
EVALUATION_REPORT_DIR = REPORT_DIR + "/evaluation"


MODEL_PATH = MODEL_DIR + "/body_performance_model.pkl"
SCALER_PATH = MODEL_DIR + "/scaler.pkl"

VALIDATION_REPORT_PATH = VALIDATION_REPORT_DIR + "/validation_report.txt"
PREPROCESSING_VALIDATION_REPORT_PATH = VALIDATION_REPORT_DIR + "/preprocessing_validation_report.txt"

CLASSIFICATION_REPORT_PATH = EVALUATION_REPORT_DIR + "/classification_report.txt"
CONFUSION_MATRIX_PATH = EVALUATION_REPORT_DIR + "/confusion_matrix.txt"
BEST_MODEL_INFO_PATH = EVALUATION_REPORT_DIR + "/best_model_info.json"

# ============================================================
# Database configuration
# Based on your installation guide
# docker run -d -p 3307:3306 --name mymcs mariadb/columnstore
# ============================================================

DB_USER = "mariadbuser"
DB_PASSWORD = quote_plus("Sunway@123")
DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_NAME = "body_performance_db"

OBT_TABLE_NAME = "body_performance_obt"

DEFAULT_CONN_STRING = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

DEFAULT_SERVER_CONN_STRING = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"
)

# ============================================================
# Redis configuration
# Based on your installation guide
# docker run -d -p 9000:6379 --name myredis redis
# ============================================================

REDIS_CONN_INFO = {
    "host": "localhost",
    "port": 9000,
    "db": 0,
}

# ============================================================
# Redis keys
# ============================================================

RAW_DATA_KEY = "body_raw"
TRANSFORMED_DATA_KEY = "body_transformed"
VALIDATED_DATA_KEY = "body_validated"
RAW_OBT_REDIS_KEY = "body_raw_obt"

CLEAN_DATA_KEY = "body_clean_dataset"

X_TRAIN_KEY = "body_X_train"
X_TEST_KEY = "body_X_test"
Y_TRAIN_KEY = "body_y_train"
Y_TEST_KEY = "body_y_test"

MODEL_METADATA_KEY = "body_model_metadata"
BEST_MODEL_INFO_KEY = "body_best_model_info"

# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Body_Performance_ML_Pipeline"
REGISTERED_MODEL_NAME = "BodyPerformanceClassifier"

# ============================================================
# Model configuration
# ============================================================

TEST_SIZE = 0.2
ACCURACY_THRESHOLD = 0.70

# ============================================================
# Dataset columns
# ============================================================

DEFAULT_COLUMNS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body fat_%",
    "diastolic",
    "systolic",
    "gripForce",
    "sit and bend forward_cm",
    "sit-ups counts",
    "broad jump_cm",
    "class",
]

RAW_COLUMNS = DEFAULT_COLUMNS

RENAME_MAP = {
    "body fat_%": "body_fat_percent",
    "gripForce": "grip_force",
    "sit and bend forward_cm": "sit_and_bend_forward_cm",
    "sit-ups counts": "sit_ups_counts",
    "broad jump_cm": "broad_jump_cm",
    "class": "fitness_class",
}

FEATURE_COLUMNS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat_percent",
    "diastolic",
    "systolic",
    "grip_force",
    "sit_and_bend_forward_cm",
    "sit_ups_counts",
    "broad_jump_cm",
]

NUMERIC_FEATURE_COLUMNS = [
    "age",
    "height_cm",
    "weight_kg",
    "body_fat_percent",
    "diastolic",
    "systolic",
    "grip_force",
    "sit_and_bend_forward_cm",
    "sit_ups_counts",
    "broad_jump_cm",
]

TARGET_COLUMN = "fitness_class"

VALID_RANGES = {
    "age": (10, 100),
    "height_cm": (100, 250),
    "weight_kg": (20, 200),
    "body_fat_percent": (1, 60),
    "diastolic": (40, 130),
    "systolic": (60, 220),
    "grip_force": (0, 100),
    "sit_and_bend_forward_cm": (-50, 250),
    "sit_ups_counts": (0, 100),
    "broad_jump_cm": (0, 400),
}

INTEGER_COLUMNS = [
    "age",
    "systolic",
    "diastolic",
    "sit_ups_counts",
    "broad_jump_cm",
]

TARGET_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

CLASS_NAMES = ["A", "B", "C", "D"]