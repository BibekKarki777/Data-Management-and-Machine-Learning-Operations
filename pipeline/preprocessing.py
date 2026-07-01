import joblib
import pandas as pd
import great_expectations as gx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAP,
    TEST_SIZE,
    RANDOM_STATE,
    RAW_OBT_REDIS_KEY,
    CLEAN_DATA_KEY,
    X_TRAIN_KEY,
    X_TEST_KEY,
    Y_TRAIN_KEY,
    Y_TEST_KEY,
    SCALER_PATH,
    PREPROCESSING_VALIDATION_REPORT_PATH,
)

from pipeline.utils import (
    load_df_from_redis,
    save_df_to_redis,
    create_required_folders,
)


# ============================================================
# Stage 2 - Data Preprocessing
# ============================================================

def validate_preprocessed_train_test_data(X_train, X_test, y_train, y_test):
    """
    Validates train-test datasets after preprocessing.
    """

    validation_success = True
    validation_messages = []

    if len(X_train) != len(y_train):
        validation_success = False
        validation_messages.append("X_train and y_train row counts do not match.")

    if len(X_test) != len(y_test):
        validation_success = False
        validation_messages.append("X_test and y_test row counts do not match.")

    if len(X_train) == 0:
        validation_success = False
        validation_messages.append("X_train is empty.")

    if len(X_test) == 0:
        validation_success = False
        validation_messages.append("X_test is empty.")

    train_df = X_train.copy()
    train_df["target"] = y_train["target"].values

    test_df = X_test.copy()
    test_df["target"] = y_test["target"].values

    train_validator = gx.from_pandas(train_df)

    for column in FEATURE_COLUMNS + ["target"]:
        train_validator.expect_column_to_exist(column)
        train_validator.expect_column_values_to_not_be_null(column)

    train_validator.expect_column_values_to_be_between(
        "target",
        min_value=0,
        max_value=3,
    )

    train_result = train_validator.validate()

    test_validator = gx.from_pandas(test_df)

    for column in FEATURE_COLUMNS + ["target"]:
        test_validator.expect_column_to_exist(column)
        test_validator.expect_column_values_to_not_be_null(column)

    test_validator.expect_column_values_to_be_between(
        "target",
        min_value=0,
        max_value=3,
    )

    test_result = test_validator.validate()

    if not train_result.success:
        validation_success = False
        validation_messages.append("Great Expectations validation failed for training data.")

    if not test_result.success:
        validation_success = False
        validation_messages.append("Great Expectations validation failed for testing data.")

    with open(PREPROCESSING_VALIDATION_REPORT_PATH, "w") as file:
        file.write("Body Performance Preprocessing Validation Report\n")
        file.write("================================================\n\n")
        file.write(f"Overall Validation Success: {validation_success}\n\n")

        file.write("Dataset Shapes:\n")
        file.write(f"X_train shape: {X_train.shape}\n")
        file.write(f"X_test shape: {X_test.shape}\n")
        file.write(f"y_train shape: {y_train.shape}\n")
        file.write(f"y_test shape: {y_test.shape}\n\n")

        file.write("Manual Validation Messages:\n")

        if validation_messages:
            for message in validation_messages:
                file.write(f"- {message}\n")
        else:
            file.write("- No manual validation errors found.\n")

        file.write("\nGreat Expectations Results:\n")
        file.write(f"Train validation success: {train_result.success}\n")
        file.write(f"Test validation success: {test_result.success}\n")

    if not validation_success:
        raise ValueError(
            f"Preprocessing validation failed. "
            f"Report generated at: {PREPROCESSING_VALIDATION_REPORT_PATH}"
        )

    print("Preprocessing validation passed.")


def preprocess_data():
    """
    Redis cached OBT
    -> PyArrow deserialization
    -> encode gender
    -> separate X/y
    -> split train-test
    -> fit scaler only on X_train
    -> transform X_train and X_test
    -> encode target class
    -> validate train-test data
    -> PyArrow serialize train-test data
    -> Redis cache
    """

    create_required_folders()

    df = load_df_from_redis(RAW_OBT_REDIS_KEY)

    print("Raw OBT loaded from Redis for preprocessing:", df.shape)

    df["gender"] = df["gender"].map({"F": 0, "M": 1})

    if df["gender"].isnull().any():
        raise ValueError("Gender encoding failed. Check gender values.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    X_train[NUMERIC_FEATURE_COLUMNS] = scaler.fit_transform(
        X_train_raw[NUMERIC_FEATURE_COLUMNS]
    )

    X_test[NUMERIC_FEATURE_COLUMNS] = scaler.transform(
        X_test_raw[NUMERIC_FEATURE_COLUMNS]
    )

    y_train_mapped = y_train_raw.map(TARGET_MAP)
    y_test_mapped = y_test_raw.map(TARGET_MAP)

    if y_train_mapped.isnull().any() or y_test_mapped.isnull().any():
        raise ValueError("Target encoding failed. Check class values.")

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = pd.DataFrame({"target": y_train_mapped.values})
    y_test = pd.DataFrame({"target": y_test_mapped.values})

    clean_train_dataset = X_train.copy()
    clean_train_dataset["target"] = y_train["target"]

    clean_test_dataset = X_test.copy()
    clean_test_dataset["target"] = y_test["target"]

    clean_dataset = pd.concat(
        [clean_train_dataset, clean_test_dataset],
        axis=0,
        ignore_index=True,
    )

    save_df_to_redis(clean_dataset, CLEAN_DATA_KEY)

    validate_preprocessed_train_test_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    joblib.dump(scaler, SCALER_PATH)

    save_df_to_redis(X_train, X_TRAIN_KEY)
    save_df_to_redis(X_test, X_TEST_KEY)
    save_df_to_redis(y_train, Y_TRAIN_KEY)
    save_df_to_redis(y_test, Y_TEST_KEY)

    print("Train-test datasets serialized with PyArrow and cached in Redis.")
    print("Preprocess task completed successfully.")