# Перевод experiment-конфигов на Hydra config groups

## Summary

Перенести внешний `experiments/` в единое дерево `configs/experiment/`. В нём `scenario` будет задавать данные и preparation, `decoder` — согласованную пару features+pipeline, а `validation_strategy` останется независимой осью запуска.

## Key changes

- Заменить запуск через `--config-dir experiments --config-name ...` на обычную композицию Hydra:
  ```bash
  uv run python scripts/run_experiment.py -m \
    +experiment=scenario_decoder
  ```
  В единственной цели `make experiment` оставить эту команду, чтобы вы меняли выбранный `+experiment=...` прямо в Makefile.

- Перенести пять decoder-рецептов в `configs/experiment/decoder/`; каждый продолжит менять только `features` и `pipeline`.

- Перенести scenario-рецепты в `configs/experiment/scenario/`, переименовав их по составу данных и направлению, без `baseline`/`cross_*` в имени. Каждый scenario явно выбирает dataset и preparation, но не validation.

- Для transfer-сценариев перенести выбор `dataset: transfer`, source и targets из `validation_strategy` в scenario. Удалить из `cross_dataset.yaml` и `cross_task.yaml` скрытую замену dataset: validation-конфиг будет описывать только validation protocol.

- Перенести `scenario_decoder.yaml`, `baseline_logistic_regression.yaml` и `ica_ablation.yaml` в `configs/experiment/` как редактируемые sweep templates. Для ICA template расширить sweep до полного Cartesian product выбранных scenario и validation, как зафиксировано.

- Добавить в base config поле имени запуска; выбранный root experiment preset устанавливает его своим именем. Базовая Hydra-раскладка будет писать, например, `results/scenario_decoder (timestamp)`, независимо от того, какие nested scenario/decoder options развернулись в jobs.

- Удалить внешний `experiments/` после переноса и обновить `.claude/rules/python_project/{overview,scripts,results_layout}.md`, чтобы они описывали новую структуру, команды и владение result directory.

## Test plan

- Проверить `make -n experiment`.
- Проверить Hydra help и композицию representative presets через `--cfg job`, включая обычный scenario, transfer scenario и каждый template.
- Запустить только лёгкие import/compile smoke checks затронутых Python entrypoints.

Не запускать подготовку данных, multirun с вычислениями, обучение моделей или любой GPU/RAM-intensive код.

## Assumptions

- Физические вложенные папки внутри `configs/experiment/` используются для удобной группировки; выбор nested option остаётся Hydra-композицией.
- `scenario` фиксирует dataset composition и preparation; `decoder` фиксирует features+pipeline; `validation_strategy` выбирается отдельно.
- Старые имена scenario заменяются именами, отражающими данные и source→target направление.
