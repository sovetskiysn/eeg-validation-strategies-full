# Задание: анализ низких within-dataset метрик

Проанализируй причины сравнительно низких метрик **только в within-dataset
сценариях** текущего run-а
`python_project/results/scenario_decoder (2026-08-30 % 15-48-47)_copy`:

- `baseline`;
- `cross_subject`;
- `cross_session`.

Не рассматривай `cross_task` и `cross_dataset` как проблему: низкий transfer
score в этих протоколах ожидаем и относится к отдельной гипотезе статьи.

Цель — понять, почему даже в пределах одного датасета и одной постановки
balanced accuracy остаётся умеренной, и разделить следующие причины:

1. недостатки или компромиссы preparation/representation — например, число и
   выбор каналов, длина и overlap окон, фильтрация, ICA, нормализация, качество
   recording units;
2. недообучение, переобучение или неоптимальные гиперпараметры конкретных
   моделей;
3. ограниченная различимость EEG-сигнала для protocol-defined contrast при
   текущем окне и протоколе;
4. нестабильность результатов между субъектами и сессиями.

Используй все сохранённые артефакты run-а: `windows.parquet`,
`folds.parquet`, `importances.parquet`, Hydra-конфиги, preparation logs и HTML
reports.

Для каждого декодера — Logistic Regression, XGBoost, EEGNet, ShallowNet и
EEGConformer — оцени:

- baseline и held-out метрики;
- разрыв baseline → cross-subject / cross-session;
- разброс score по subject и session/fold;
- признаки вероятного underfit или overfit;
- ограничения входного representation и training recipe.

Дай выводы с уровнем уверенности и расположи причины по приоритету. Для каждой
причины предложи конкретную проверку или абляцию, не меняющую labels и не
использующую target data.

Если сохранённых данных недостаточно, точно укажи, каких артефактов не хватает
для вывода — например, train score по fold, learning curves, loss/accuracy по
эпохам, QC ICA по recording unit — и предложи минимальное изменение runner-а,
которое их сохранит.
