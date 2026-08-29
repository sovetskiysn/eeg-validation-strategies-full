# Что из диагностики закрыто, а что нет

Сверка четырёх диагностических отчётов (`WITHIN_DATASET_METRICS_DIAGNOSIS.md`,
`WITHIN_DATASET_METRICS_DIAGNOSIS_CODEX.md`, `SAM40_ICA_EOG_PROXY_DIAGNOSIS.md`,
`CHANNEL_QUALITY_DIAGNOSIS.md`) с рабочим деревом поверх `0cabba5`.

## Закрыто

| Пункт | Где | Статус |
| --- | --- | --- |
| ICA выбрасывала 55–76% дисперсии SAM-40 по EOG-прокси F3/F4/F7/F8 | `src/preparation.py`, `configs/preparation/5_ica_cleaning/ica.yaml`, `pyproject.toml` (`mne-icalabel`) | заменено на ICLabel, версия `v11 -> v12`, конфиг валидируется в Step 1. Осталось: асимметрия датасетов 11.7% vs 0.6% и работа ICLabel вне полосы 1–100 Hz — no_ica-абляция обязательна |
| Дубликаты oversampling по обе стороны inner `ValidSplit` у нейронок | `src/pipeline.py::BalancedClassWeight`, три DL-конфига | шаг `train_resampling` удалён, вес читается с фолда |
| `RandomOverSampler` у XGBoost как асимметрия | `src/pipeline.py::BalancedXGBClassifier` | все пять декодеров ребалансируются одной формулой `balanced` |
| Плохие каналы / QC-порог | — | закрыт диагностикой: ≤1 канал на запись везде, вмешательства не требует |

## Не закрыто — по убыванию цены

1. **Нет верхней ступени `within_recording`.** `build_baseline`
   (`src/validation.py:28`) группирует по `recording_unit`, то есть baseline
   уже leave-recording-out. Лестница начинается с 0.62 и имеет размах 10
   пунктов. Time-blocked within-recording даёт 0.73 / 0.58 без leakage.
   Каталог `configs/validation_strategy/` содержит пять протоколов.
2. **`overlap: 0.5` — это 0.5 секунды, а не 50%.**
   `configs/preparation/.../epoching` не тронут; `make_fixed_length_events`
   получает секунды (`src/preparation.py:545`). Шаг 4.5 с → 5 окон на
   25-секундную запись SAM-40. Сплиты recording-disjoint, поэтому более
   плотные окна не создадут leakage, но инвалидируют кэш Stage 1.
3. **Inner split нейронок всё ещё не group-aware.** Дубликаты убраны, но
   `ValidSplit(0.2, stratified=True)` по-прежнему режет окна случайно, и окна
   одной записи попадают по обе стороны. Early stopping выбирает эпоху по
   оптимистичному лоссу. Закрыта половина проблемы из §5 CODEX-отчёта.
4. **Решение по нормализации внутри записи не принято.** Единственная
   нормализация — `StandardScaler` по train-фолду
   (`configs/pipeline/1_input_scaling/`). Z-скор внутри записи даёт +0.056 /
   +0.042, но это transductive unsupervised adaptation и прямо конфликтует с
   формулировкой «zero-shot, без адаптации на target». Нужно либо объявить
   частью декодера во всех пяти протоколах, либо отказаться явно.
5. **Метрики главной таблицы считаются пулом по всем фолдам, включая ROC-AUC**
   (`src/analysis.py:165`). Для `cross_subject_sam40` это пул из 40 по-разному
   откалиброванных `predict_proba`. На нынешних числах совпадает с per-fold
   средним до третьего знака, но для LOSO корректнее среднее по фолдам с
   разбросом.
7. **ICA фитится на полной записи, включая будущий test-unit.** Меток не
   использует, но при строгом zero-shot это recording-level transductive
   preprocessing. ICLabel сменил правило исключения, а не место фита. Нужно
   назвать в Methods либо оценивать только по train-части.
8. **Единица анализа SAM-40 — окно, а не запись.** Агрегация вероятностей до
   записи даёт 0.572 → 0.572 (ошибки скоррелированы), потолок на 5-секундном
   окне ≈0.62 при любом сплите. Постановка не менялась.
9. **Переименование сценариев** (`baseline` → within-dataset cross-recording и
   т.д.) и вывод рядом с метрикой числа train windows / units / subjects — не
   сделано.

## Мелочи, оставшиеся на месте

- Комментарий в `epoching` всё ещё объясняет `reject_peak_to_peak_uv` через
  «blinks via the EOG-proxy channels» — правило ICA уже другое.
- `build_baseline` (`src/validation.py:24`) вычисляет `m["subject_id"]`, не
  использует его и мутирует `epochs.metadata` на месте.
- XGBoost с полностью дефолтными гиперпараметрами без early stopping.
