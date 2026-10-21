"""Обучение CatBoost на новом трёхмесячном окне."""

import logging
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.settings import CAT_FEATURES, LABEL, MODELS_DIR
from pipeline.artifacts import write_json

logger = logging.getLogger(__name__)


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет target от признаков и приводит категориальные колонки к строкам.

    Категории совпадают с теми, что ожидает онлайн-скоринг после run_preproc.
    Возвращает матрицу признаков и вектор целевой переменной.
    """
    if LABEL not in frame.columns:
        raise ValueError(f"В датасете нет колонки {LABEL}")
    features = frame.drop(columns=[LABEL]).copy()
    for column in CAT_FEATURES:
        if column not in features.columns:
            raise ValueError(f"После препроцессинга нет категориального признака {column}")
        features[column] = features[column].astype(str)
    return features, frame[LABEL]


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает CatBoost на train и считает ROC-AUC и F1 на свежем месяце.

    Читает обработанные parquet из preproc_info. Учится 100 итераций без
    подбора по test. Сохраняет кандидата; повтор возвращает сохранённый результат.
    Возвращает версию, путь к черновику, метрики и путь к новому окну.
    """
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    result_path = Path(preproc_info["dataset_path"]).parent / "training.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    train = pd.read_parquet(preproc_info["train_proc_path"])
    test = pd.read_parquet(preproc_info["test_proc_path"])
    x_train, y_train = prepare_features(train)
    x_test, y_test = prepare_features(test)

    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=20,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=CAT_FEATURES,
    )

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba > 0.5).astype(int)
    f1 = float(f1_score(y_test, pred, zero_division=0))
    if y_test.nunique() < 2:
        roc_auc = None
        logger.warning("В test одна метка, ROC-AUC не определён. F1=%.4f", f1)
    else:
        roc_auc = float(roc_auc_score(y_test, proba))
        logger.info(
            "Качество на test %s: ROC-AUC=%.4f, F1=%.4f",
            preproc_info["test_month"],
            roc_auc,
            f1,
        )

    run_key = hashlib.sha256(str(result_path).encode()).hexdigest()[:12]
    version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{run_key}"
    candidates_dir = MODELS_DIR / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    staging_model = candidates_dir / f"my_catboost_{version}.cbm"
    temporary = staging_model.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    os.replace(temporary, staging_model)
    logger.info("Черновик модели сохранён: %s", staging_model)
    result = {
        "version": version,
        "staging_model": str(staging_model),
        "dataset_path": preproc_info["dataset_path"],
        "batch_month": preproc_info["batch_month"],
        "train_months": preproc_info["train_months"],
        "test_month": preproc_info["test_month"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "roc_auc": roc_auc,
        "f1": f1,
        "test_proc_path": preproc_info["test_proc_path"],
    }
    write_json(result_path, result)
    return result
