"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from pathlib import Path

import pandas as pd

from pipeline.db import copy_labeled_frame, get_conn

logger = logging.getLogger(__name__)


def append_labeled_batch(batch_info: dict) -> dict:
    """Дописывает parquet батча в конец labeled_transactions.

    Читает путь из batch_info. Записывает данные и отметку месяца одной транзакцией:
    повтор после commit не вставляет строки второй раз.
    Возвращает тот же словарь с числом записанных строк, чтобы следующая
    задача знала, какой месяц только что появился в витрине.
    """
    batch_path = Path(batch_info["batch_path"])
    if not batch_path.exists():
        raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")

    frame = pd.read_parquet(batch_path)
    months = pd.to_datetime(frame["transaction_time"]).dt.to_period("M")
    if frame.empty or not (months == pd.Period(batch_info["batch_month"], freq="M")).all():
        raise ValueError("Батч должен содержать строки только заданного месяца.")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS supplied_batches (
                    batch_month DATE PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    row_count BIGINT NOT NULL
                )"""
            )
            cur.execute(
                """INSERT INTO supplied_batches (batch_month, run_id, row_count)
                   VALUES (%s, %s, %s) ON CONFLICT (batch_month) DO NOTHING
                   RETURNING batch_month""",
                (batch_info["batch_month"], batch_info["run_id"], len(frame)),
            )
            inserted = cur.fetchone() is not None
        if inserted:
            copy_labeled_frame(conn, frame, commit=False)
        conn.commit()
    finally:
        conn.close()
    logger.info(
        "Месяц %s: %s, строк: %s",
        batch_info["batch_month"],
        "добавлен в витрину" if inserted else "уже загружен, повтор пропущен",
        len(frame),
    )
    return {
        "batch_path": batch_info["batch_path"],
        "batch_month": batch_info["batch_month"],
        "rows": int(len(frame)),
        "inserted": inserted,
    }
