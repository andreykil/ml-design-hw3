"""DAG проверки обучающего окна; реализация задач вынесена в pipeline."""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.data_quality import check_dataset, read_dataset, write_report


@dag(
    dag_id="data_quality",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    tags=["mlops", "dq"],
)
def data_quality():
    """Связывает снимок витрины, проверки и сохранение HTML-отчёта."""
    snapshot = task(read_dataset, task_id="read_dataset")(run_id="{{ run_id }}")
    checked = task(check_dataset, task_id="check_dataset")(snapshot)
    task(write_report, task_id="write_report")(checked)


data_quality()
