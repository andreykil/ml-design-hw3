"""Сравнение кандидата и действующей модели на одном test-месяце."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.artifacts import write_json
from pipeline.settings import LABEL, MODELS_DIR

logger = logging.getLogger(__name__)


def model_metrics(model, test: pd.DataFrame) -> dict:
    """Считает ROC-AUC и F1 при пороге 0.5 по признакам конкретной модели."""
    from sklearn.metrics import f1_score, roc_auc_score

    frame = test[list(model.feature_names_)].copy()
    for index in model.get_cat_feature_indices():
        column = model.feature_names_[index]
        frame[column] = frame[column].astype(str)
    proba = model.predict_proba(frame)[:, 1]
    if not np.isfinite(proba).all():
        raise ValueError("Модель вернула неконечные вероятности.")
    return {
        "roc_auc": float(roc_auc_score(test[LABEL], proba)),
        "f1": float(f1_score(test[LABEL], proba > 0.5, zero_division=0)),
    }


def evaluate_model(train_info: dict) -> dict:
    """Читает кандидата, production-модель и test; сохраняет неизменное решение для повторов."""
    from catboost import CatBoostClassifier

    evaluation_path = Path(train_info["staging_model"]).with_suffix(".metrics.json")
    if evaluation_path.exists():
        return {**train_info, **json.loads(evaluation_path.read_text()), "evaluation_path": str(evaluation_path)}

    test = pd.read_parquet(train_info["test_proc_path"])
    latest_path = MODELS_DIR / "latest.txt"
    baseline_version = latest_path.read_text().strip() if latest_path.exists() else None
    candidate_metrics = baseline_metrics = None
    accepted = False
    if set(test[LABEL].dropna().unique()) != {0, 1} or test[LABEL].isna().any():
        reason = "Для сравнения на test нужны оба класса 0 и 1 без пропусков."
    else:
        candidate = CatBoostClassifier()
        candidate.load_model(train_info["staging_model"])
        candidate_metrics = model_metrics(candidate, test)
        if baseline_version is None:
            accepted = True
            reason = "Первая модель: production-версия отсутствует."
        else:
            baseline = CatBoostClassifier()
            baseline.load_model(str(MODELS_DIR / f"my_catboost_{baseline_version}.cbm"))
            baseline_metrics = model_metrics(baseline, test)
            accepted = all(candidate_metrics[key] >= baseline_metrics[key] for key in ("roc_auc", "f1"))
            reason = "Обе метрики не ухудшились." if accepted else "ROC-AUC или F1 ухудшилась."

    decision = {
        **{key: train_info[key] for key in ("version", "batch_month", "train_months", "test_month", "train_rows", "test_rows")},
        "baseline_version": baseline_version,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "accepted": accepted,
        "reason": reason,
    }
    write_json(evaluation_path, decision)
    logger.info("Кандидат %s: accepted=%s, %s", train_info["version"], accepted, reason)
    return {**train_info, **decision, "evaluation_path": str(evaluation_path)}
