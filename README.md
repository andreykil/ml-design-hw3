# Запуск и проверка проекта

Нужны работающий Docker Desktop и Docker Compose v2 или новее. Порты 2181, 5432, 8080, 8081 и 9095 должны быть свободны.

Команды выполняются из корня репозитория — папки с `docker-compose.yaml`.

## Запуск

1. Положите размеченный `train.csv` в папку `data/` в корне репозитория.
2. Создайте `.env`. В шаблоне уже указан `TRAIN_DATA_CSV=./data/train.csv`; при другом расположении файла измените этот путь.
3. Запустите:

   ```bash
   docker compose up --build -d
   docker compose ps -a
   ```

Первый запуск скачивает образы и зависимости, затем загружает CSV. `data_loader` и `kafka-setup` должны завершиться с кодом `0`; остальные сервисы продолжают работать.

Доступны:

- Airflow: http://localhost:8081 — `admin` / `admin`.
- Kafka UI: http://localhost:8080 — топики `transactions` и `scoring`.
- PostgreSQL: `localhost:5432`, БД `fraud_db`, пользователь `fraud_user`, пароль из `.env`.
- Kafka с хоста: `localhost:9095`.

## Проверка скоринга

```bash
docker compose logs --tail=30 data_loader fraud_detector scoring_writer
docker compose exec postgres psql -U fraud_user -d fraud_db -c \
  "SELECT transaction_id, score, fraud_flag, model_version, created_at FROM scores ORDER BY id DESC LIMIT 10;"
```

Повторите SQL через несколько секунд: должны появиться новые транзакции. В Kafka UI в сообщениях `scoring` есть `score`, `fraud_flag`, `model_version`.

## Проверка DAG и обновления модели

Собственный препроцессинг в `fraud_detector/src/preprocessing.py` дополняет семинарские признаки: `amount_log = log1p(amount)`, `time_sin` и `time_cos` для циклического времени суток с учетом минут, `is_night` для операций с 00:00 до 05:59. Преобразования одинаково применяются в DAG и онлайн-скоринге.

Все три DAG выполняются каждые 5 минут, `catchup=False`, одновременно выполняется не более одного запуска каждого DAG.

1. `supply_labeled_batch`: выборка до 100 000 строк → перенос в следующий месяц → запись → удаление истории старше 3 лет.
2. `retrain_catboost`: извлечение месяца → сборка окна → временной split → preprocessing → обучение → сравнение моделей → условная публикация.
3. `data_quality`: снимок текущего окна → проверки → HTML-отчет. Успешные проверки дают зеленый DAG; нарушения — красную задачу `write_report`, при этом отчет уже сохранен.

Окно обучения — 3 месяца: два ранних месяца идут в train, последний в test. У DAG поставки и переобучения разные указатели: переобучение сначала проходит исторические месяцы CSV.

Каждый запуск сохраняет свой месяц и снимок данных, а указатель обновляется только после сохранения кандидата, решения о публикации и окна. Повтор не сдвигает окно второй раз и не возвращает старую production-версию.

Проверить файлы модели и текущую версию:

```bash
docker compose exec airflow sh -c 'cat /opt/airflow/models/latest.txt; ls -lh /opt/airflow/models'
docker compose exec postgres psql -U fraud_user -d fraud_db -c \
  "SELECT model_version, COUNT(*) FROM scores GROUP BY model_version ORDER BY model_version;"
docker compose exec postgres psql -U fraud_user -d fraud_db -c \
  "SELECT window_start, window_end, next_batch_month FROM training_state;"
```

При принятии кандидата появляются `my_catboost_<версия>.cbm` и файл метрик, меняется `latest.txt`. Через несколько секунд новые строки `scores` должны содержать эту версию. При отклонении production-версия сохраняется. До первой публикации версия — `initial`.

## Проверка качества модели

CatBoost обучается на CPU, фиксированные 100 итераций. Test-месяц не используется для подбора числа итераций. Задача `evaluate_model` заново оценивает кандидата и действующую модель на одном и том же свежем test-месяце, выбирая признаки каждой модели по `feature_names_`.

Публикация разрешена, если одновременно `ROC-AUC кандидата >= ROC-AUC production` и `F1 кандидата >= F1 production`. Для F1 используется тот же порог `score > 0.5`, что и в скоринге.

После сохранения и проверки кандидата окно сдвигается на месяц даже при отклонении: следующий запуск обрабатывает новые данные, а скоринг продолжает использовать прежнюю модель.

Метрики обеих моделей и решение о публикации находятся в `/opt/airflow/models/candidates/*.metrics.json`.

## Data Quality и HTML-отчёты

Проверяется все текущее зафиксированное трехмесячное окно из PostgreSQL. Это отдельный диагностический DAG, он не блокирует переобучение. Правила требуют нулевого числа нарушений:

1. Выборка не пустая.
2. Присутствуют все столбцы `LABELED_COLUMNS`.
3. `transaction_time` заполнен и корректен.
4. `target` заполнен и принимает только `0` или `1`.
5. `amount` — конечное неотрицательное число.
6. `population_city` — конечное неотрицательное число.
7. Все координаты заполнены и конечны; широта в `[-90, 90]`, долгота в `[-180, 180]`.
8. Шесть категориальных признаков `CAT_FEATURES` заполнены и не содержат пустых строк.
9. В окне ровно три ожидаемых последовательных календарных месяца.
10. И в train, и в test присутствуют оба класса.

В HTML для каждого правила показаны PASS/FAIL и число нарушений.

Каталог отчетов `/opt/airflow/reports` расположен рядом с `/opt/airflow/models` и смонтирован в локальную папку `reports/`. История хранится в `dq_<id>.html`, последний отчет — `latest.html`.
