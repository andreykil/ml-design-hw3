"""Публикация бинарника модели и фиксация нового окна обучения."""

import json
import logging
import os
import shutil
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.artifacts import write_json
from pipeline.settings import CURRENT_DATASET, MODELS_DIR

logger = logging.getLogger(__name__)


def publish_model(train_info: dict) -> str:
    """Читает решение и кандидата; публикует разрешённую модель и фиксирует месяц один раз."""
    version = train_info["version"]
    batch_month = date.fromisoformat(train_info["batch_month"])
    decision = json.loads(Path(train_info["evaluation_path"]).read_text())
    staging_model = Path(train_info["staging_model"])
    pending_window = Path(train_info["dataset_path"])

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT next_batch_month FROM training_state WHERE id = 1 FOR UPDATE")
            next_month = cur.fetchone()[0]
        if next_month > batch_month:
            logger.info("Месяц %s уже зафиксирован, повтор публикации пропущен.", batch_month)
            return version
        if next_month != batch_month:
            raise RuntimeError("Публикация пропустила ожидаемый месяц.")
        if not staging_model.exists() or not pending_window.exists():
            raise FileNotFoundError("Нет сохранённого кандидата или снимка окна.")

        if decision["accepted"]:
            final_model = MODELS_DIR / f"my_catboost_{version}.cbm"
            temporary = final_model.with_suffix(".tmp.cbm")
            shutil.copyfile(staging_model, temporary)
            os.replace(temporary, final_model)
            write_json(final_model.with_suffix(".metrics.json"), decision)

            latest_tmp = MODELS_DIR / f".latest.{version}.txt"
            latest_tmp.write_text(version + "\n", encoding="utf-8")
            os.replace(latest_tmp, MODELS_DIR / "latest.txt")
            logger.info("Опубликована модель %s (%s)", version, final_model)
        else:
            logger.info("Модель %s отклонена: %s", version, decision["reason"])

        # Снимок остаётся для восстановления после сбоя между файлом и commit БД.
        current_tmp = CURRENT_DATASET.with_suffix(".tmp.parquet")
        shutil.copyfile(pending_window, current_tmp)
        os.replace(current_tmp, CURRENT_DATASET)
        advance_training_state(conn, batch_month)
    finally:
        conn.close()
    return version
