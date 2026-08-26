---
paths:
  - "python_project/**"
---

# Результаты experiment sweeps

Раскладкой `results/` владеет базовый Hydra-конфиг, не Makefile и не
Hydra-рецепты в `experiments/`.

- Базовый Hydra-конфиг пишет в
  `results/${hydra.job.config_name} (${now:%Y-%m-%d | %H-%M-%S})`. Поэтому
  имя каталога совпадает с выбранным experiment YAML, а timestamp изолирует
  повторные прогоны.
- Makefile только выбирает YAML и запускает runner; прямой вызов runner'а и
  вызов через Makefile создают результаты по одному правилу.
- `experiments/` описывает только состав запусков (`scenario × decoder`). В
  этих YAML не должно быть `hydra.sweep.dir`, `hydra.sweep.subdir` или полей,
  влияющих на путь результата.

## Execution job и logical scenario result — не одно и то же

Transfer-scenario держит один source и **список** targets: decoder обучается
один раз на каждом source-fold и проверяется на всех targets этого job. Число
job'ов от этого меньше числа научных направлений, и разложение обратно —
ответственность раннера, а не анализа.

- Non-transfer job пишет три parquet прямо в свой каталог, как и раньше.
- Transfer job пишет по одному самодостаточному результату на направление в
  `targets/<target-recipe-name>/`: те же три parquet плюс `scenario.yaml`.
- Имя листовой папки выведено из resolved target recipe (`name` плюс
  отсортированные `exclude_conditions`), например `sam40__ex-stroop-mirror`.
  Это выведенное читаемое имя, а не рукописная метка и **не ключ кэша**:
  content-hash есть только у артефактов Stage 1 (`datasets.md`).
- `scenario.yaml` — не Hydra config, а проекция одного направления под теми же
  путями полей. Анализ читает его вместо `.hydra/config.yaml` и не узнаёт,
  сколько направлений разделили один job.
- `target_index` живёт только внутри прогона (validation → runner) и в parquet
  не попадает.
