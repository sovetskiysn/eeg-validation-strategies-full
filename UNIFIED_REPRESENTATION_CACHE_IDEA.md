# Unified representation cache — идея для обсуждения

## Цель

Убрать из `preparation` ответственность за маршрутизацию и проверку кэша.
Experiment runner должен получать не «каталог датасета», а готовое
представление данных для конкретного семейства декодеров.

Вместо `get_dataset_dir(dataset, preparation)` предполагается публичная точка:

```python
get_representation_dir(
    dataset_config,
    preparation_config,
    features_config,
) -> Path
```

## Предлагаемая раскладка

```text
python_project/datasets/experiment_cache/
└── <representation-hash>/
    ├── config.yaml
    ├── epochs-epo.fif
    └── features.npy             # только для classical_ml
```

Deep representation содержит готовые очищенные EEG windows (`epochs-epo.fif`).
Classical representation содержит те же windows и вычисленную матрицу
handcrafted features. Поэтому Logistic Regression и XGBoost обращаются к
одному и тому же classical artifact.

## Identity

Адрес representation artifact строится из resolved-конфигов:

```python
{
    "dataset": dataset_config,
    "preparation": preparation_config,
    "features": features_config,
}
```

Ресурсные ручки (`feature_n_jobs`, внутреннее число потоков) в identity не
входят: они меняют скорость, но не содержимое features. Для изменения логики
feature extraction нужен явный version bump в feature recipe.

Каталог публикуется атомарно через staging directory и никогда не
перезаписывается. При cache hit сохранённый `config.yaml` обязан точно
совпадать с текущей identity; иначе runner падает громко.

## Разделение ответственности

* `preparation.py` — чистая научная стадия: BIDS + preparation recipe → Epochs;
  не адресует и не публикует cache entry.
* `validation.py` — получает уже готовые Epochs/representation directories и
  строит train/test splits; не знает о preparation и файловом кэше.
* `scripts/run_experiment.py` — единственное место, которое выбирает cache hit
  или miss и собирает source/target representations.

Для transfer runner получает две representation directories, читает их Epochs
и строит тот же объединённый input и CV, что и сейчас.

## Осознанный trade-off

Один и тот же prepared EEG будет храниться отдельно для `deep_learning` и
`classical_ml`. В текущем дизайне это максимум два representation families;
дублирование диска приемлемо ради одной прямой модели кэша. Оно устраняет
скрытый in-memory state и не требует отдельного Stage-2 cache namespace.

## Ожидаемая выгода

На test sweep handcrafted features занимали около 81% wall time, поскольку
Logistic Regression и XGBoost пересчитывали одинаковую матрицу. Unified
representation cache позволяет вычислить её один раз на состав данных и
переиспользовать между этими decoder recipes, а также между отдельными sweep.

## Не решает

Эта идея не устраняет повторное обучение одинаковой source-модели для разных
target scenarios. По завершённому sweep таких повторных fit около 34.8%; это
отдельная задача реорганизации source→many-target experiments и не должна
смешиваться с implementation representation cache.
