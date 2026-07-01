import os
from io import BytesIO

import redis
import pyarrow as pa
import pyarrow.parquet as pq

from sqlalchemy import create_engine, text

from pipeline.config import (
    DB_NAME,
    DEFAULT_CONN_STRING,
    DEFAULT_SERVER_CONN_STRING,
    REDIS_CONN_INFO,
    MODEL_DIR,
    REPORT_DIR,
)


# ============================================================
# Helper Functions
# ============================================================

def create_required_folders():
    """
    Creates folders required for models and reports.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VALIDATION_REPORT_DIR, exist_ok=True)


def create_db_engine(use_database=True):
    """
    Creates SQLAlchemy engine for MariaDB ColumnStore.
    """

    if use_database:
        return create_engine(DEFAULT_CONN_STRING)

    return create_engine(DEFAULT_SERVER_CONN_STRING)


def create_database_if_not_exists():
    """
    Creates database if it does not exist.
    """

    engine = create_db_engine(use_database=False)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))

    engine.dispose()

    print(f"Database is ready: {DB_NAME}")


def get_redis_client():
    """
    Creates Redis client.
    """

    return redis.Redis(
        host=REDIS_CONN_INFO["host"],
        port=REDIS_CONN_INFO["port"],
        db=REDIS_CONN_INFO["db"],
        decode_responses=False,
    )


def save_df_to_redis(df, key):
    """
    Serializes pandas DataFrame using PyArrow Parquet
    and saves it into Redis.
    """

    redis_client = get_redis_client()

    buffer = BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buffer)

    redis_client.set(key, buffer.getvalue())

    print(f"Saved DataFrame to Redis key: {key}")


def load_df_from_redis(key):
    """
    Loads PyArrow serialized DataFrame from Redis.
    """

    redis_client = get_redis_client()
    data = redis_client.get(key)

    if data is None:
        raise ValueError(f"No data found in Redis for key: {key}")

    buffer = BytesIO(data)
    table = pq.read_table(buffer)
    df = table.to_pandas()

    print(f"Loaded DataFrame from Redis key: {key}")

    return df
    



      
