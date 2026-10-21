"""Изолированные файлы запусков и атомарная запись артефактов."""

import hashlib
import json
import os
from pathlib import Path

from pipeline.settings import TRAINING_DIR


def run_directory(pipeline: str, run_id: str) -> Path:
    """Возвращает постоянный каталог запуск. Повторы используют те же файлы."""
    key = hashlib.sha256(run_id.encode()).hexdigest()[:24]
    directory = TRAINING_DIR / "runs" / pipeline / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: Path, value: dict) -> None:
    """Атомарно заменяет JSON, не оставляя частичный файл после сбоя."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_frame(path: Path, frame) -> None:
    """Атомарно заменяет parquet. Повтор перезаписывает только свой артефакт."""
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
