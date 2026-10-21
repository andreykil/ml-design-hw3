"""Проверки обучающего окна и HTML-отчет"""

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.artifacts import run_directory, write_frame, write_json
from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive, read_training_state
from pipeline.settings import CAT_FEATURES, REPORTS_DIR, WINDOW_MONTHS


def read_dataset(run_id: str) -> dict:
    """Читает согласованное окно БД и сохраняет снимок; повтор использует тот же parquet."""
    directory = run_directory("dq", run_id)
    manifest = directory / "snapshot.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    conn = get_conn()
    try:
        conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
        state = read_training_state(conn)
        frame = pd.read_sql(
            f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
            "WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time",
            conn,
            params=(state["window_start"], month_end_exclusive(state["window_end"])),
        )
    finally:
        conn.close()
    dataset_path = directory / "dataset.parquet"
    write_frame(dataset_path, frame)
    info = {
        "dataset_path": str(dataset_path),
        "expected_months": list(pd.period_range(state["window_start"], state["window_end"], freq="M").astype(str)),
        "run_id": run_id,
    }
    write_json(manifest, info)
    return info


def dataset_rules(frame: pd.DataFrame, expected_months: list[str]) -> list[dict]:
    """Проверяет схему, значения и временное окно, не изменяя данные."""
    checks = []

    def add(rule, violations):
        checks.append({"rule": rule, "passed": int(violations) == 0, "violations": int(violations)})

    add("Выборка не пустая", int(frame.empty))
    missing = set(LABELED_COLUMNS) - set(frame.columns)
    add("Все обязательные столбцы присутствуют", len(missing))
    if missing:
        return checks

    times = pd.to_datetime(frame["transaction_time"], errors="coerce")
    add("Время заполнено и корректно", times.isna().sum())
    add("Target заполнен и принимает только 0 или 1", (~frame["target"].isin([0, 1])).sum())
    for column, title in [("amount", "Сумма"), ("population_city", "Население")]:
        values = pd.to_numeric(frame[column], errors="coerce")
        add(f"{title}: конечное неотрицательное число", (~np.isfinite(values) | (values < 0)).sum())

    invalid_coordinates = pd.Series(False, index=frame.index)
    for column, bound in [("lat", 90), ("lon", 180), ("merchant_lat", 90), ("merchant_lon", 180)]:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_coordinates |= ~np.isfinite(values) | (values.abs() > bound)
    add("Координаты заполнены, широта ±90, долгота ±180", invalid_coordinates.sum())

    categories = frame[CAT_FEATURES].astype("string")
    invalid_categories = categories.isna() | categories.apply(lambda column: column.str.strip().eq(""))
    add("Категориальные признаки заполнены", invalid_categories.any(axis=1).sum())

    months = times.dt.to_period("M").astype(str)
    expected = pd.PeriodIndex(expected_months, freq="M")
    consecutive = len(expected) == WINDOW_MONTHS and list(expected) == list(pd.period_range(expected.min(), periods=WINDOW_MONTHS, freq="M"))
    add("Окно содержит ровно три ожидаемых последовательных месяца", len(set(months) ^ set(expected_months)) + int(not consecutive))
    latest = max(expected_months) if expected_months else ""
    invalid_classes = sum(
        set(frame.loc[mask, "target"].unique()) != {0, 1}
        for mask in (months < latest, months == latest)
    )
    add("В train и test присутствуют оба класса", invalid_classes)
    return checks


def check_dataset(snapshot_info: dict) -> dict:
    """Читает снимок, сохраняет JSON правил; повтор не меняет витрину."""
    frame = pd.read_parquet(snapshot_info["dataset_path"])
    result = {
        **snapshot_info,
        "rows": len(frame),
        "checks": dataset_rules(frame, snapshot_info["expected_months"]),
    }
    results_path = Path(snapshot_info["dataset_path"]).parent / "checks.json"
    write_json(results_path, result)
    return {"results_path": str(results_path)}


def write_report(check_info: dict) -> str:
    """Пишет HTML до ошибки DQ; повтор атомарно заменяет отчёт своего запуска."""
    results_path = Path(check_info["results_path"])
    result = json.loads(results_path.read_text())
    passed = all(check["passed"] for check in result["checks"])
    rows = "".join(
        f"<tr><td>{escape(check['rule'])}</td><td>{'PASS' if check['passed'] else 'FAIL'}</td>"
        f"<td>{check['violations']}</td></tr>"
        for check in result["checks"]
    )
    document = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Data Quality</title>'
        '<style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse}'
        'td,th{border:1px solid #ccc;padding:8px;text-align:left}</style>'
        f"<h1>Data Quality: {'PASS' if passed else 'FAIL'}</h1>"
        f"<p>Запуск: {escape(result['run_id'])}</p>"
        f"<p>Окно: {escape(', '.join(result['expected_months']))}; строк: {result['rows']}</p>"
        f"<p>Создан: {datetime.now(timezone.utc).isoformat()}</p>"
        f"<table><tr><th>Правило</th><th>Статус</th><th>Нарушения</th></tr>{rows}</table></html>"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"dq_{results_path.parent.name}.html"
    for path in (report_path, REPORTS_DIR / "latest.html"):
        temporary = path.with_suffix(".tmp.html")
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, path)
    if not passed:
        raise ValueError(f"Data Quality FAIL; отчёт сохранён: {report_path}")
    return str(report_path)
