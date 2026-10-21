"""Временной train/test split трёхмесячного окна."""

import logging
from pathlib import Path

import pandas as pd

from pipeline.artifacts import write_frame

logger = logging.getLogger(__name__)


def time_train_test_split(window_info: dict) -> dict:
    """Делит окно по времени: два более ранних месяца в train, свежий месяц в test.

    Читает parquet окна из window_info, сортирует по transaction_time и
    сохраняет train_raw.parquet и test_raw.parquet в каталог запуска.
    Колонка времени остаётся, потому что препроцессинг сам строит календарные признаки.
    Возвращает пути к частям, месяц батча и подписи месяцев train и test.
    Повтор перезаписывает только файлы своего запуска.
    """
    splits_dir = Path(window_info["dataset_path"]).parent
    frame = pd.read_parquet(window_info["dataset_path"])
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    frame = frame.sort_values("transaction_time")
    periods = frame["transaction_time"].dt.to_period("M")
    latest = periods.max()
    train = frame.loc[periods < latest]
    test = frame.loc[periods == latest]
    if train.empty or test.empty:
        raise RuntimeError(
            f"Не удалось разрезать окно по времени: train={len(train)}, test={len(test)}"
        )

    train_path = splits_dir / "train_raw.parquet"
    test_path = splits_dir / "test_raw.parquet"
    write_frame(train_path, train)
    write_frame(test_path, test)
    train_months = sorted(periods.loc[periods < latest].unique().astype(str))
    logger.info(
        "Train месяцы %s (%s строк), test %s (%s строк).",
        train_months,
        len(train),
        latest,
        len(test),
    )
    return {
        "train_raw_path": str(train_path),
        "test_raw_path": str(test_path),
        "dataset_path": window_info["dataset_path"],
        "batch_month": window_info["batch_month"],
        "train_months": ",".join(train_months),
        "test_month": str(latest),
    }
