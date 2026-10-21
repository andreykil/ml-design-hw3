"""Сдвиг трёхмесячного окна обучения на один месяц вперёд."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from dateutil.relativedelta import relativedelta

from pipeline.artifacts import write_frame
from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive
from pipeline.settings import WINDOW_MONTHS

logger = logging.getLogger(__name__)


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к текущему окну и выбрасывает самый старый.

    Читает фиксированные месяцы витрины и пишет окно в каталог запуска.
    Снимок сохраняется для повторов; current.parquet и указатель не меняет.
    """
    batch_month = date.fromisoformat(batch_info["batch_month"])
    pending_window = Path(batch_info["batch_path"]).parent / "pending_window.parquet"
    window_start = batch_month - relativedelta(months=WINDOW_MONTHS - 1)
    if not pending_window.exists():
        conn = get_conn()
        try:
            updated = pd.read_sql(
                f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
                "WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time",
                conn,
                params=(window_start, month_end_exclusive(batch_month)),
            )
        finally:
            conn.close()
        months = pd.to_datetime(updated["transaction_time"]).dt.to_period("M")
        expected = set(pd.period_range(window_start, batch_month, freq="M"))
        if set(months.unique()) != expected:
            raise RuntimeError(f"В витрине нет полного окна {window_start} — {batch_month}")
        write_frame(pending_window, updated)
    else:
        updated = pd.read_parquet(pending_window)

    logger.info(
        "Окно после сдвига: %s строк, %s — %s. Начальный месяц %s.",
        len(updated),
        updated["transaction_time"].min(),
        updated["transaction_time"].max(),
        window_start,
    )
    return {
        "dataset_path": str(pending_window),
        "batch_month": batch_month.isoformat(),
    }
