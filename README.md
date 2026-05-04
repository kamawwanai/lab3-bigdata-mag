# Лабораторная 3: Lakehouse на Polars и Delta Lake

Проект строит небольшой lakehouse для датасета US Flight Delays: сырые данные попадают в Bronze, очищаются и нормализуются в Silver, затем из них собираются Gold-витрины для аналитики и машинного обучения. ML-эксперименты логируются в MLflow вместе с версиями Delta-таблиц.

Источник данных: [Flight Delay на Kaggle](https://www.kaggle.com/datasets/arvindnagaonkar/flight-delay).

## Что внутри

- `src/bronze/load_bronze.py` читает исходный файл и пишет Bronze Delta-таблицу батчами в `append`.
- `src/silver/transform_silver.py` чистит данные, добавляет признаки и обновляет Silver через Delta `MERGE`.
- `src/gold/build_analytics.py` собирает аналитические агрегаты.
- `src/gold/build_features.py` собирает feature table для ML.
- `src/ml/` обучает регрессию и классификацию и логирует результаты в MLflow.
- `src/delta_ops/manage_delta.py` демонстрирует дополнительные возможности Delta Lake.

В задании указан CSV, но в проекте поддержаны и CSV, и Parquet. Для текущего набора данных используется `data/Flight_Delay.parquet`; это не меняет сути Bronze-слоя, потому что данные всё равно загружаются в Delta инкрементальными батчами.

## Быстрый запуск

Локально:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.pipeline.run_all --mlflow-uri http://127.0.0.1:5000
```

Через Docker:

```powershell
docker-compose up --build
```

`docker-compose` поднимает два сервиса: `mlflow` на `http://localhost:5000` и `app`, который запускает весь пайплайн `bronze -> silver -> gold -> ml`.

## Данные и таблицы

Ожидаемый исходный файл:

```text
data/Flight_Delay.parquet
```

Основные Delta-таблицы:

- Bronze: `lakehouse/bronze/flights`
- Silver: `lakehouse/silver/flights`
- Gold analytics: `lakehouse/gold/analytics`
- Gold features: `lakehouse/gold/features`

Минимально нужны колонки `year`, `month` и `arr_delay`. Если в датасете есть `dep_delay`, `cancelled`, `carrier`, `origin`, `dest`, `flight_num`, `dep_time` и день месяца, они используются для очистки, признаков и стабильного ключа рейса.

## Bronze

Bronze-слой читает `csv` или `parquet` через Polars lazy API и пишет данные в Delta по временным батчам. В текущей реализации батч делится по году и месяцу: так получается несколько Delta-версий, и загрузка похожа на реальный инкрементальный приход данных.

```powershell
.\.venv\Scripts\python -m src.bronze.load_bronze
```

## Silver

Silver-слой делает основную подготовку данных:

- убирает отменённые рейсы и строки без ключевых полей;
- отсекает очевидные выбросы по `arr_delay`;
- нормализует категориальные поля в верхний регистр;
- добавляет `hour`, `day_of_week`, `season`, `route` и `is_delayed_15`;
- пишет таблицу с `partition_by=["year", "month"]`;
- при повторном запуске делает Delta `MERGE`, а не дублирует строки.

```powershell
.\.venv\Scripts\python -m src.silver.transform_silver
```

`flight_key` строится из бизнес-полей рейса, если они есть в источнике. Если таких полей недостаточно, используется стабильный hash строки, чтобы повторный запуск всё равно был идемпотентным.

## Gold

Аналитическая витрина считает средние задержки и долю задержанных рейсов по аэропорту, авиакомпании, часу и сезону:

```powershell
.\.venv\Scripts\python -m src.gold.build_analytics
```

Feature table содержит признаки для обучения и два таргета:

- `arr_delay` для регрессии;
- `is_delayed_15` для классификации задержки больше 15 минут.

```powershell
.\.venv\Scripts\python -m src.gold.build_features
```

## MLflow и модели

Для регрессии сравниваются `LinearRegression` и `HistGradientBoostingRegressor`. Для классификации сравниваются `LogisticRegression` и `GradientBoostingClassifier`. В MLflow логируются параметры, метрики, модели, версия Gold feature table и feature importance для обеих задач.

```powershell
.\.venv\Scripts\python -m src.ml.train_regression --tracking-uri http://127.0.0.1:5000
.\.venv\Scripts\python -m src.ml.train_classification --tracking-uri http://127.0.0.1:5000
```

## Delta Lake extras

Помимо обязательного `MERGE`, в проекте есть отдельный скрипт для Delta-возможностей:

```powershell
.\.venv\Scripts\python -m src.delta_ops.manage_delta --table lakehouse\gold\features --time-travel-version 0
```

Он демонстрирует:

- time travel, то есть чтение предыдущей версии таблицы;
- compaction через `OPTIMIZE`, если метод доступен в установленной версии `delta-rs`;
- Z-ORDER или fallback-кластеризацию через сортировку и overwrite;
- `VACUUM` в dry-run режиме;
- schema evolution через добавление служебной колонки.

## Почему партиции `year`, `month`

Для рейсов время является естественной осью данных: инкрементальные загрузки приходят по периодам, аналитика часто ограничивается годом или месяцем, а ML-обучение обычно делится по времени. Партиционирование `year/month` помогает Delta и Polars читать меньше файлов при фильтрах по периоду, но не создаёт слишком мелкие партиции на уровне дня.

## Пример Polars `.explain()`

Запрос:

```python
import polars as pl

lf = (
    pl.scan_delta("lakehouse/silver/flights")
    .filter((pl.col("year") == 2019) & (pl.col("arr_delay") > 15))
    .select(["carrier", "arr_delay"])
    .group_by("carrier")
    .agg(pl.mean("arr_delay").alias("avg_arr_delay"))
)

print(lf.explain())
```

Вывод:

```text
AGGREGATE
  GROUP BY: [col("carrier")]
  AGGREGATE: [col("arr_delay").mean().alias("avg_arr_delay")]
  FROM
    SELECT [col("carrier"), col("arr_delay")]
      FILTER [([(col("year")) == (2019)]) & ([(col("arr_delay")) > (15)])]
      FROM
        Parquet SCAN [...]
        PROJECT 3/16 COLUMNS
        SELECTION: [([(col("year")) == (2019)]) & ([(col("arr_delay")) > (15)])]
```

