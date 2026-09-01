# `scenario_maybe_new` — подсчёт job'ов и направлений

Состояние на 2026-09-01, после удаления `cross_session_distinguishing.yaml` из
`configs/experiments/scenario_maybe_new/`.

19 scenario-файлов × 5 decoder-опций (`eegconformer`, `eegnet`,
`logistic_regression`, `shallownet`, `xgboost`) — как задаёт
`configs/experiments/scenario_decoder_maybe_new.yaml`
(`+experiments/scenario_maybe_new: glob(*)` × `+experiments/decoder: glob(*)`).

**Hydra jobs = 19 × 5 = 95.**

Это число execution job'ов, а не число научных направлений — per
`.claude/rules/python_project/results_layout.md`, transfer-сценарий
(cross-task, cross-dataset) обучается один раз на source и тестируется на
**списке** targets внутри одного job'а. Поэтому job'ов меньше, чем строк
итоговой таблицы.

## Таблица направлений (аналог Table 3 рукописи)

| Scenario | Validation method | Кол-во направлений |
|---|---|---|
| **Baseline (within)** | Stratified K-fold(Distinguishing) | 1 |
| | Stratified K-fold(SAM-40 Full) | 1 |
| | Stratified K-fold(SAM-40: Arithmetic) | 1 |
| | Stratified K-fold(SAM-40: Mirror) | 1 |
| | Stratified K-fold(SAM-40: Stroop) | 1 |
| **Cross-subject** | Leave-one-subject-out(Distinguishing) | 1 |
| | Leave-one-subject-out(SAM-40 Full) | 1 |
| | Leave-one-subject-out(SAM-40: Arithmetic) | 1 |
| | Leave-one-subject-out(SAM-40: Mirror) | 1 |
| | Leave-one-subject-out(SAM-40: Stroop) | 1 |
| **Cross-task (SAM-40)** | Full → Stroop, Full → Arithmetic, Full → Mirror | 3 |
| | Arithmetic → Mirror, Arithmetic → Stroop, Arithmetic → Full | 3 |
| | Mirror → Arithmetic, Mirror → Stroop, Mirror → Full | 3 |
| | Stroop → Arithmetic, Stroop → Mirror, Stroop → Full | 3 |
| **Cross-dataset** | Distinguishing → Full, Distinguishing → Stroop, Distinguishing → Arithmetic, Distinguishing → Mirror | 4 |
| | Full → Distinguishing | 1 |
| | Arithmetic → Distinguishing | 1 |
| | Mirror → Distinguishing | 1 |
| | Stroop → Distinguishing | 1 |

**Итого направлений: 30** (5 baseline + 5 cross-subject + 12 cross-task + 8 cross-dataset).

## Как считаются оба числа

- **Job'ы (95)** = число scenario-файлов (19) × число decoder-файлов (5).
  Hydra `glob(*)` в `scenario_decoder_maybe_new.yaml` перемножает все файлы в
  `configs/experiments/scenario_maybe_new/` на все файлы в
  `configs/experiments/decoder/`; каждая пара — отдельный execution job со
  своим `results/.../<decoder>/<scenario>/`.
- **Направления (30)** = для non-transfer scenario (`baseline_*`,
  `cross_subject_*`) — 1 направление на файл, потому что `preparation` там одна
  сторона. Для transfer scenario (`cross_task_from_*`, `cross_dataset_from_*`)
  — по числу ключей в `preparation.targets` этого файла (список
  `/preparation@preparation.targets.<name>` в `defaults`), потому что один
  source-fold внутри job'а тестируется сразу на всех targets.
- Итоговых строк в per-decoder таблице статьи будет 30 × 5 decoder-таблиц =
  150 метрик, но реальных Hydra-прогонов (95) меньше, потому что
  transfer-source обучается один раз на весь список targets, а не по разу на
  каждое направление.
