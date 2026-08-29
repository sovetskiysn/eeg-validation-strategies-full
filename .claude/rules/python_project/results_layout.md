---
paths:
  - "python_project/**"
---

# Результаты experiment sweeps

Раскладкой `results/` владеет базовый Hydra-конфиг, не Makefile и не
experiment-рецепты в `configs/experiments/`.

- Базовый Hydra-конфиг (`configs/config.yaml`) пишет в
  `results/<имя пресета> (${now:%Y-%m-%d %% %H-%M-%S})` — одним и тем же
  шаблоном для `hydra.run.dir` (одиночный прогон) и `hydra.sweep.dir`
  (пресет-sweep). Имя берётся из самого выбора группы,
  `${oc.select:hydra.runtime.choices.experiments,adhoc}`: Hydra записывает
  выбранную option в `runtime.choices`, поэтому отдельного поля-метки в конфиге
  нет и рассинхронизировать имя файла с именем каталога невозможно. Запуск без
  корневого пресета попадает в `results/adhoc (...)`, timestamp изолирует
  повторные прогоны.
- Makefile только выбирает `+experiments=<preset>` и запускает runner; прямой
  вызов runner'а и вызов через Makefile создают результаты по одному правилу.
- `configs/experiments/` описывает только состав запусков (`scenario × decoder`,
  плюс независимо выбираемый `validation_strategy`). В этих YAML не должно
  быть `hydra.sweep.dir`, `hydra.sweep.subdir` или других полей, влияющих на
  путь результата: имя каталога выводится из имени файла пресета, а не
  назначается им. `hydra.mode` пресету принадлежит — он задаёт режим, а не
  раскладку.

## Execution job и logical scenario result — не одно и то же

Transfer-scenario держит один source и **список** targets: decoder обучается
один раз на каждом source-fold и проверяется на всех targets этого job. Число
job'ов от этого меньше числа научных направлений, и разложение обратно —
ответственность раннера, а не анализа.

- Non-transfer job пишет три parquet прямо в свой каталог, как и раньше.
- После Stage 1 каждый execution job сохраняет автономные HTML-отчёты
  MNE-BIDS-Pipeline в `preparation_reports/<dataset>/`, сохраняя относительные
  subject/session-пути из derivatives. Transfer-targets используют один общий
  snapshot job, а не дублируют его в каждой листовой папке.
- Transfer job пишет по одному самодостаточному результату на направление в
  `targets/<target-composition>/`: те же три parquet плюс `scenario.yaml`.
- Имя листовой папки выведено из скомпонованной target-стороны: dataset и
  выбранные `task`, например `sam40__task-relax-stroop`. Это выведенное
  читаемое имя, а не рукописная метка и **не ключ кэша**.
- `scenario.yaml` — не Hydra config, а компактная проекция одного направления
  с `preparation.source` и `preparation.target`. Анализ читает её вместо
  `.hydra/config.yaml` и не узнаёт, сколько направлений разделили один job.
- `target_index` живёт только внутри прогона (validation → runner) и в parquet
  не попадает.
- Hydra создаёт `.hydra/config.yaml` до входа в runner. Конфигурация уже
  окончательная: preparation options скомпозированы, а source и targets
  не меняются в Python. После успешного fit runner повторно сохраняет config,
  чтобы зафиксировать также входную частоту, которую узнал feature extractor.
