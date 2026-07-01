import os

import pandas as pd
import great_expectations as gx
from sqlalchemy import text

from pipeline.config import (
    DATA_PATH,
    RAW_COLUMNS,
    RENAME_MAP,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_RANGES,
    INTEGER_COLUMNS,
    RAW_DATA_KEY,
    TRANSFORMED_DATA_KEY,
    VALIDATED_DATA_KEY,
    RAW_OBT_REDIS_KEY,
    VALIDATION_REPORT_PATH,
    DB_NAME,
    OBT_TABLE_NAME,
)

from pipeline.utils import (
    save_df_to_redis,
    load_df_from_redis,
    create_database_if_not_exists,
    create_db_engine,
    create_required_folders,
)


# ============================================================
# Stage 1 - Data Ingestion
# ============================================================

def basic_cleaning_transform(df):
    """
    Transform / Basic Cleaning:
    - Standardizes column names
    - Converts required numeric columns
    - Accepts only whole number sit_ups_counts
    - Removes impossible values
    - Applies systolic > diastolic rule
    - Converts selected columns to integer
    - Removes duplicate rows
    """

    df = df.rename(columns=RENAME_MAP)

    print("Columns after standardizing:")
    print(list(df.columns))

    for column in VALID_RANGES.keys():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_invalid_numeric = len(df)
    df = df.dropna(subset=list(VALID_RANGES.keys()) + ["gender", TARGET_COLUMN])
    after_invalid_numeric = len(df)

    print(
        "Rows removed because of invalid or missing numeric values:",
        before_invalid_numeric - after_invalid_numeric,
    )

    before_situps = len(df)
    df = df[df["sit_ups_counts"] % 1 == 0]
    after_situps = len(df)

    print(
        "Rows removed because sit_ups_counts was not whole number:",
        before_situps - after_situps,
    )

    before_ranges = len(df)

    for column, (minimum, maximum) in VALID_RANGES.items():
        df = df[
            (df[column] >= minimum) &
            (df[column] <= maximum)
        ]

    after_ranges = len(df)

    print(
        "Rows removed because of impossible numeric ranges:",
        before_ranges - after_ranges,
    )

    before_bp = len(df)
    df = df[df["systolic"] > df["diastolic"]]
    after_bp = len(df)

    print(
        "Rows removed because systolic was not greater than diastolic:",
        before_bp - after_bp,
    )

    for column in INTEGER_COLUMNS:
        df[column] = df[column].astype(int)

    before_duplicates = len(df)
    df = df.drop_duplicates()
    after_duplicates = len(df)

    print("Duplicate rows removed:", before_duplicates - after_duplicates)

    return df


def extract_data():
    """
    CSV -> pandas -> raw Redis copy -> transform/basic cleaning -> Redis
    """

    create_required_folders()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Put body_performance.csv inside /home/bibek-karki/mlops/data/"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)
    print("Original columns:", list(df.columns))

    save_df_to_redis(df, RAW_DATA_KEY)

    missing_columns = [column for column in RAW_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    df = basic_cleaning_transform(df)

    print("Transformed dataset shape:", df.shape)

    save_df_to_redis(df, TRANSFORMED_DATA_KEY)

    print("Extract and transform task completed successfully.")


def validate_data():
    """
    Great Expectations validation.
    If validation fails, DAG stops and a readable validation report is generated.
    """

    create_required_folders()

    df = load_df_from_redis(TRANSFORMED_DATA_KEY)

    validator = gx.from_pandas(df)

    # ------------------------------------------------------------
    # Required column and non-null checks
    # ------------------------------------------------------------
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        validator.expect_column_to_exist(column)
        validator.expect_column_values_to_not_be_null(column)

    # ------------------------------------------------------------
    # Categorical value checks
    # ------------------------------------------------------------
    validator.expect_column_values_to_be_in_set("gender", ["M", "F"])
    validator.expect_column_values_to_be_in_set(TARGET_COLUMN, ["A", "B", "C", "D"])

    # ------------------------------------------------------------
    # Numeric range checks
    # ------------------------------------------------------------
    for column, (minimum, maximum) in VALID_RANGES.items():
        validator.expect_column_values_to_be_between(
            column,
            min_value=minimum,
            max_value=maximum,
        )

    # ------------------------------------------------------------
    # Blood pressure logic check
    # systolic must be greater than diastolic
    # ------------------------------------------------------------
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="systolic",
        column_B="diastolic",
        or_equal=False,
    )

    # ------------------------------------------------------------
    # Integer type checks
    # ------------------------------------------------------------
    for column in INTEGER_COLUMNS:
        validator.expect_column_values_to_be_in_type_list(
            column,
            ["int64", "int32", "int"],
        )

    result = validator.validate()

    # ------------------------------------------------------------
    # Readable validation report
    # ------------------------------------------------------------
    with open(VALIDATION_REPORT_PATH, "w") as file:
        file.write("Body Performance Raw Data Validation Report\n")
        file.write("===========================================\n\n")

        file.write(f"Overall Validation Success: {result.success}\n")
        file.write(f"Total Expectations Checked: {len(result.results)}\n\n")

        file.write("-" * 80 + "\n")
        file.write("Detailed Expectation Results\n")
        file.write("-" * 80 + "\n\n")

        for index, item in enumerate(result.results, start=1):

            try:
                expectation_type = item.expectation_config.expectation_type
                kwargs = item.expectation_config.kwargs
            except Exception:
                expectation_type = "expectation_check"
                kwargs = {}

            success = item.success

            try:
                result_details = item.result
            except Exception:
                result_details = {}

            column = kwargs.get("column", None)
            column_A = kwargs.get("column_A", None)
            column_B = kwargs.get("column_B", None)

            min_value = kwargs.get("min_value", None)
            max_value = kwargs.get("max_value", None)
            value_set = kwargs.get("value_set", None)
            type_list = kwargs.get("type_list", None)

            unexpected_count = result_details.get("unexpected_count", "N/A")
            unexpected_percent = result_details.get("unexpected_percent", "N/A")

            # Column information
            if column is not None:
                column_info = column
            elif column_A is not None and column_B is not None:
                column_info = f"{column_A}, {column_B}"
            else:
                column_info = "Table-level check"

            # Human-readable rule
            if expectation_type == "expect_column_to_exist":
                rule = f"Column '{column}' must exist."

            elif expectation_type == "expect_column_values_to_not_be_null":
                rule = f"Column '{column}' must not contain null values."

            elif expectation_type == "expect_column_values_to_be_in_set":
                rule = f"Column '{column}' values must be in {value_set}."

            elif expectation_type == "expect_column_values_to_be_between":
                rule = (
                    f"Column '{column}' values must be between "
                    f"{min_value} and {max_value}."
                )

            elif expectation_type == "expect_column_values_to_be_in_type_list":
                rule = (
                    f"Column '{column}' values must have one of these types: "
                    f"{type_list}."
                )

            elif expectation_type == "expect_column_pair_values_A_to_be_greater_than_B":
                rule = (
                    f"Column '{column_A}' must be greater than "
                    f"column '{column_B}'."
                )

            else:
                rule = f"Expectation rule: {expectation_type}"

            file.write(f"Check {index}\n")
            file.write(f"Column(s): {column_info}\n")
            file.write(f"Expectation: {expectation_type}\n")
            file.write(f"Rule: {rule}\n")
            file.write(f"Success: {success}\n")
            file.write(f"Unexpected Count: {unexpected_count}\n")
            file.write(f"Unexpected Percentage: {unexpected_percent}\n")
            file.write("-" * 80 + "\n\n")

    if not result.success:
        raise ValueError(
            f"Validation failed. Report generated at: {VALIDATION_REPORT_PATH}"
        )

    save_df_to_redis(df, VALIDATED_DATA_KEY)

    print("Great Expectations validation passed.")
    print(f"Readable validation report generated at: {VALIDATION_REPORT_PATH}")

def create_columnstore_obt_table(engine):
    """
    Creates the raw OBT table using MariaDB ColumnStore engine.
    This matches the pipeline design where validated raw OBT data
    is stored in a ColumnStore table.
    """

    drop_query = f"DROP TABLE IF EXISTS {OBT_TABLE_NAME}"

    create_query = f"""
    CREATE TABLE {OBT_TABLE_NAME} (
        age INT,
        gender VARCHAR(10),
        height_cm DOUBLE,
        weight_kg DOUBLE,
        body_fat_percent DOUBLE,
        diastolic INT,
        systolic INT,
        grip_force DOUBLE,
        sit_and_bend_forward_cm DOUBLE,
        sit_ups_counts INT,
        broad_jump_cm INT,
        fitness_class VARCHAR(5)
    ) ENGINE=Columnstore
    """

    with engine.begin() as conn:
        conn.execute(text(drop_query))
        conn.execute(text(create_query))

    print(f"ColumnStore OBT table created: {OBT_TABLE_NAME}")


def load_to_mariadb():
    """
    Validated data -> SQLAlchemy -> MariaDB ColumnStore OBT table
    MariaDB OBT -> PyArrow serialization -> Redis cache
    """

    create_database_if_not_exists()

    df = load_df_from_redis(VALIDATED_DATA_KEY)

    engine = create_db_engine(use_database=True)

    # Create the OBT table manually using ColumnStore engine
    create_columnstore_obt_table(engine)

    # Append data into the already-created ColumnStore table
    df.to_sql(
        name=OBT_TABLE_NAME,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi",
    )

    print("Validated raw OBT data stored in MariaDB ColumnStore.")
    print(f"Table created/replaced: {DB_NAME}.{OBT_TABLE_NAME}")
    print("Rows loaded:", len(df))

    df_from_db = pd.read_sql(
        f"SELECT * FROM {OBT_TABLE_NAME}",
        con=engine,
    )

    engine.dispose()

    save_df_to_redis(df_from_db, RAW_OBT_REDIS_KEY)

    print("OBT read from MariaDB, serialized with PyArrow, and cached in Redis.")