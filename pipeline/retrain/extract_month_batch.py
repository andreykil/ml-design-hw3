"""Извлечение очередного месяца из размеченной витрины."""

import logging
import json
from datetime import date

import pandas as pd
from airflow.exceptions import AirflowSkipException

from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive, read_training_state
from pipeline.artifacts import run_directory, write_frame, write_json

logger = logging.getLogger(__name__)


def extract_month_batch(run_id: str) -> dict:
    """Забирает из labeled_transactions месяц, указанный в training_state.

    Читает next_batch_month, выбирает наблюдения этого календарного месяца
    и сохраняет их в parquet. Указатель месяца не меняет.
    Возвращает путь к файлу батча и месяц в формате ISO.
    Если строк нет, помечает задачу как skipped: новый месяц ещё не поставлен.
    Месяц сохраняется в каталоге запуска: повтор не читает новый указатель.
    """
    directory = run_directory("retrain", run_id)
    manifest = directory / "batch.json"
    conn = get_conn()
    try:
        state = read_training_state(conn)
        if manifest.exists():
            batch_month = date.fromisoformat(json.loads(manifest.read_text())["batch_month"])
        else:
            batch_month = state["next_batch_month"]
            write_json(manifest, {"batch_month": batch_month.isoformat()})
        batch_end = month_end_exclusive(batch_month)
        batch_path = directory / "batch.parquet"
        if batch_path.exists():
            return {"batch_path": str(batch_path), "batch_month": batch_month.isoformat()}
        logger.info(
            "Текущее окно %s — %s. Извлекаем месяц %s.",
            state["window_start"],
            state["window_end"],
            batch_month,
        )
        columns = ", ".join(LABELED_COLUMNS)
        frame = pd.read_sql(
            f"""
            SELECT {columns}
            FROM labeled_transactions
            WHERE transaction_time >= %s AND transaction_time < %s
            ORDER BY transaction_time
            """,
            conn,
            params=(batch_month, batch_end),
        )
    finally:
        conn.close()

    if frame.empty:
        raise AirflowSkipException(
            f"В витрине нет наблюдений за {batch_month}. Новый месяц ещё не поставлен."
        )

    write_frame(batch_path, frame)
    logger.info(
        "Батч %s: %s строк, %s — %s",
        batch_month,
        len(frame),
        frame["transaction_time"].min(),
        frame["transaction_time"].max(),
    )
    return {
        "batch_path": str(batch_path),
        "batch_month": batch_month.isoformat(),
    }
