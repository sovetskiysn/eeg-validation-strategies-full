# План миграции подготовки EEG на MNE-BIDS-Pipeline

Дата проверки: 2026-08-28. Целевая версия для первого PoC:
`mne-bids-pipeline==1.10.1`, `mne-bids==0.19.x`, Python 3.11.

## Статус проверки

План сверён с официальной документацией и исходным кодом тега
`mne-bids-pipeline==1.10.1` (релиз 2026-04-20). Без запуска EEG-данных или
нейросетей прошли:

- `compileall` для `src/` и `scripts/`;
- Hydra-композиция трёх корневых experiment-рецептов;
- Hydra-композиция representative `scenario × decoder`.

В окружении уже установлены `mne==1.12.1`, `mne-bids==0.19.0` и
`mne-icalabel==0.9.0`; `autoreject` и `mne-bids-pipeline` пока намеренно не
добавлялись. Поэтому это статическая проверка контракта и не замена малого PoC
на реальных записях из Фазы 1.

### Найденный контрактный разрыв

`configs/preparation/2_recording_selection/selection.yaml` объявляет
`exclude_run_values`, но текущая `src/preparation.py` читает и применяет только
`exclude_session_values`. Сейчас все run-level списки пусты, поэтому это не
меняет ни одного результата. До реализации миграции нужно принять отдельное
решение: либо удалить неиспользуемый ключ (предпочтительно, пока в `scans.tsv`
нет обобщаемого критерия), либо явно добавить его в selector с теми же
fail-fast проверками. Молча переносить в новую ветку ключ, который старая
ветка игнорировала, нельзя: это изменило бы выборку под видом рефакторинга.

## Решение в одном абзаце

MNE-BIDS-Pipeline стоит попробовать и, при успешном сравнительном прогоне,
отдать ему фильтрацию, resampling, reference, встроенные ICA и ICLabel,
применение ICA к continuous Raw, отчёты и тяжёлый cache. Это действительно позволит отказаться
от `datasets/prepared_cache` как от собственного монолитного cache формата.
Однако Pipeline не является полной заменой нынешнего Stage 1: он не выражает
отбор сессий по произвольным колонкам `sessions.tsv`, не делает overlapping
окна отдельно внутри каждого condition block. EEG-QC при этом настраивается
штатно через связку local AutoReject до и после ICA, поэтому собственные
`annotate_amplitude()` и LOF можно удалить. Его стандартная единица ICA всё же
отличается от SAM40 в текущем коде. Поэтому целевая схема —
MNE-BIDS-Pipeline до очищенного Raw, затем короткий проектный шаг
condition-aware windowing с той же AutoReject policy, который пишет Epochs в
BIDS derivatives; validation только находит эти файлы через
`mne_bids.find_matching_paths()`, читает `mne.read_epochs()` и объединяет
`mne.concatenate_epochs()`.

Это миграция с проверочными воротами, а не big-bang. Удалять нынешнюю
реализацию и старый cache можно только после численного сравнения обеих веток.

## Что подтвердилось по документации и исходникам

### Отбор субъектов, сессий и runs

- `exclude_subjects` есть.
- `sessions` есть как include-list или `'all'`.
- `exclude_runs` есть, но это словарь исключений по участнику; `runs` также
  может задаваться отдельно по task.
- `exclude_sessions` нет.
- Значит, правило вида

  ```yaml
  exclude_session_values:
    session_role: [habituation]
    session_quality: [incomplete]
  ```

  остаётся нашим декларативным правилом. Небольшой selector читает
  `sessions.tsv` и компилирует его в конкретные `subjects`/`sessions` для
  Pipeline. Для нынешнего Distinguishing результат, вероятно, будет
  `sessions: ["03", "04", "05", "06", "07"]`.

Есть важная граница: `sessions` у Pipeline глобален для выбранных subjects. Если
в будущем `session_quality=incomplete` будет различаться между участниками, один
глобальный список уже не выразит точную матрицу subject × session. Selector
обязан это обнаружить и либо сгруппировать участников с одинаковыми наборами
сессий в отдельные вызовы, либо упасть. Молча расширять выборку нельзя.

Официальное описание всех этих параметров находится в
[General settings MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/stable/settings/general.html).

### Conditions и metadata

Здесь нужно разделять три разных операции.

| Операция | Кто её делает после миграции | Почему |
| --- | --- | --- |
| Отбор SAM40 conditions | Selector преобразует conditions в `task=[...]` | В SAM40 условие представлено BIDS task; исключённый task не должен входить и в ICA fit |
| Отбор condition blocks внутри Distinguishing Raw | Проектный windowing | Несколько conditions живут внутри одной continuous записи; весь Raw всё равно проходит preprocessing |
| Отбор сессий по metadata | Selector читает `sessions.tsv`, затем задаёт `sessions` | Pipeline не знает семантики `session_role` и `session_quality` |

`conditions` в самом MNE-BIDS-Pipeline относится к event-locked epoching. Оно
не является универсальным фильтром continuous Raw по длительным блокам. Поэтому
ключевое для проекта `exclude_conditions` остаётся, но превращается либо в
список task до запуска Pipeline (SAM40), либо в фильтр annotations при нарезке
окон (Distinguishing).

### Overlapping fixed-length окна внутри condition block

Pipeline умеет `rest_epochs_duration` и `rest_epochs_overlap`, когда task
обрабатывается как rest. В task-ветке он создаёт epochs относительно events.
Встроенной операции «для каждой annotation с condition C перезапустить сетку
fixed-length окон, не выходить за границу блока и задать overlap» нет. Это
видно и в [настройках epoching](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/epochs.html),
и непосредственно в `make_epochs()` версии 1.10.1.

Следовательно, нынешний маленький алгоритм окон нужен и после миграции:

1. найти annotations выбранных conditions;
2. для каждого блока отдельно вызвать `mne.make_fixed_length_events()` с
   `start`, `stop`, `duration` и `overlap`;
3. собрать `mne.Epochs` с нынешними metadata;
4. применить rejection и сохранить один derivative Epochs файл на исходную
   recording;
5. не позволять окну пересечь condition boundary.

Это существенно меньше нынешнего Stage 1, потому что он получает уже
отфильтрованный и ICA-clean continuous Raw.

### ICA на нескольких task-записях

ICA и MNE-ICALabel уже штатно сочетаются в одной конфигурации Pipeline:

```python
spatial_filter = "ica"
ica_use_icalabel = True
```

Следовательно, в новой проектной ветке не должно быть отдельного вызова
`mne_icalabel.label_components()`: Pipeline сам fit'ит ICA, классифицирует
компоненты, записывает их в `*_proc-ica_components.tsv` и применяет решение к
Raw. Чтобы сохранить текущую политику (исключать только `eye blink` и
`muscle artifact` при вероятности ≥ 0.8), bridge задаёт её полностью, а не
полагается на pipeline defaults:

```python
ica_use_eog_detection = False
ica_use_ecg_detection = False
ica_icalabel_include = (
    "brain", "other", "heart beat", "line noise", "channel noise",
)
ica_exclusion_thresholds = {"eye blink": 0.8, "muscle artifact": 0.8}
```

`ica_icalabel_include` здесь перечисляет сохраняемые классы. Без явного
расширения его default (`brain`, `other`) изменил бы научную политику и начал
бы исключать также channel/line/heart components. Затем каждое решение
проверяется в Pipeline report и, при необходимости, в этом TSV.

Pipeline это умеет. Для каждого subject/session он:

1. собирает выбранные `(run, task)`;
2. готовит временные epochs каждого Raw;
3. объединяет epochs;
4. обучает одну ICA;
5. применяет её обратно к каждому Raw при `process_raw_clean=True`.

Последний флаг создаёт `*_proc-clean_raw.fif`; это официальный контракт
[process_raw_clean](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/ssp_ica.html#process_raw_clean).

Но единица объединения не совпадает с текущим проектом во всех случаях:

- Distinguishing сейчас использует `(subject, session)` — совпадает с Pipeline;
- SAM40 сейчас использует `(subject, run)` и объединяет выбранные tasks только
  внутри одного run;
- Pipeline при отсутствии session по умолчанию объединит все выбранные runs
  данного subject в одну ICA.

Объединять все SAM40 runs в одну ICA нельзя считать безобидной оптимизацией:
общий transform увидит данные из всех run одного subject. SAM40 следует
запускать отдельным Pipeline derivative root для каждого run (`run-01`,
`run-02`, `run-03`), каждый раз с одним run и всеми выбранными tasks. Внутри
такого вызова Pipeline fit остаётся один на `subject × run`.

Объединять все SAM40 runs в одну ICA можно рассматривать только как отдельное
научное изменение и отдельную абляцию, не как техническую миграцию.

### Достаточный материал для ICA

Стандартная task-ветка Pipeline обучает ICA на event-locked epochs. Для
Distinguishing короткий `epochs_tmax` использовал бы лишь начало каждого
600-секундного блока, а не весь continuous материал; это хуже нынешней
семантики. Самый простой PoC — использовать:

```python
task_is_rest = True
epochs_tmin = 0.0
rest_epochs_duration = 5.0
rest_epochs_overlap = 0.0
```

Термин `task_is_rest` здесь означает только «сделать равномерные технические
epochs для ICA по всему Raw». Исходные BIDS task entities сохраняются, а
condition labels позже снова берутся из annotations проектным windowing. Эти
временные Pipeline epochs не подаются декодерам.

Это решение нужно подтвердить PoC, потому что rest-режим также отключает часть
event-specific логики Pipeline. Для нынешних данных она, вероятно, не нужна,
но это должно быть проверено по сохранности annotations и составу clean Raw.

### Cache и derivatives

Pipeline по умолчанию использует `joblib.Memory` и кладёт cache в `_cache`
внутри `deriv_root`. Изменение параметров конфигурации инвалидирует затронутые
шаги; входные и выходные файлы контролируются по `mtime` либо по hash. Это
описано в [официальном разделе Caching](https://mne.tools/mne-bids-pipeline/stable/settings/caching.html).

Это позволяет удалить собственные:

- SHA256-адреса `prepared_cache/<hash>`;
- staging монолитного `epochs-epo.fif`;
- ручное решение, надо ли заново выполнять фильтрацию/ICA;
- собственный общий HTML report подготовки.

При этом cache Pipeline отвечает только за его собственные steps. Наш
последующий condition-aware windowing не становится автоматически частью
этого cache. Для него нужен очень маленький и явный контракт: рядом с готовыми
Epochs хранить resolved window config и fingerprints clean Raw; совпало —
использовать, не совпало — атомарно пересобрать только окна. Это не второй
pipeline и не content-addressed cache; это проверка актуальности собственного
derivative, который Pipeline создать не умеет.

### Hydra и Python config Pipeline

Hydra действительно умеет Python structured configs и обычные dict через
ConfigStore; это подтверждает [Hydra Config Store API](https://hydra.cc/docs/1.3/tutorials/structured_config/config_store/).
Но MNE-BIDS-Pipeline ожидает путь к исполняемому `config.py`, который он сам
импортирует. Hydra `DictConfig` и такой Python module — разные конфигурационные
контракты. Регистрация MNE-настроек в ConfigStore сама по себе их не соединяет.

Рекомендуемый мост без второго `@hydra.main`:

1. единственный `scripts/run_experiment.py` как сейчас композит YAML;
2. `preparation.py` преобразует только узел `cfg.preparation.pipeline` в
   resolved JSON внутри derivative root;
3. один checked-in `scripts/mne_bids_pipeline_config.py` читает путь к этому
   JSON из project-specific environment variable и выставляет его ключи как
   module globals;
4. код запускает public CLI `mne_bids_pipeline --config=... --steps=preprocessing`
   через `subprocess.run(..., check=True)`;
5. Pipeline валидирует итоговые Python settings и включает фактические значения
   в cache keys.

Второй Hydra entrypoint не нужен. Private API MNE-BIDS-Pipeline импортировать
не следует: CLI — публичная и документированная граница
[Basic usage](https://mne.tools/mne-bids-pipeline/stable/getting_started/basic_usage.html).

## Целевая архитектура

```text
Hydra YAML experiment
        |
        v
resolved dataset + preparation
        |
        +--> selector: BIDS paths, sessions.tsv, exclude_conditions
        |          |
        |          v
        |    exact subjects/sessions/tasks/runs + ICA units
        |
        +--> JSON config bridge --> MNE-BIDS-Pipeline CLI
                                   |
                                   +--> filtering/resampling/reference
                                   +--> local AutoReject on ICA epochs
                                   +--> ICA + ICLabel + reports
                                   +--> local AutoReject on Pipeline epochs
                                   +--> Pipeline _cache
                                   +--> proc-clean_raw.fif
                                                   |
                                                   v
                         project condition-aware windowing + AutoReject
                                                   |
                                                   v
                                  desc-windows_proc-<recipe>_epo.fif
                                                   |
                                                   v
                         find_matching_paths -> read_epochs -> concatenate_epochs
                                                   |
                                                   v
                                      existing validation protocols
```

Граница ответственности:

| Остаётся у проекта | Переходит MNE-BIDS-Pipeline |
| --- | --- |
| `DATASET_MAPPING` и бинарная семантика labels | BIDS Raw import |
| `exclude_session_values` → конкретная выборка | frequency/notch filtering |
| `exclude_conditions` → tasks/blocks | resampling |
| сохранение ICA unit SAM40 `(subject, run)` | EEG reference |
| применение той же AutoReject policy к нестандартным condition windows | ICA fit/apply, ICLabel и AutoReject на Pipeline epochs |
| overlapping окна внутри condition blocks | Pipeline reports |
| window metadata и готовые Epochs derivatives | invalidation/cache тяжёлых steps |
| validation split semantics | стандартные intermediate derivatives |

## Предлагаемая раскладка derivatives

Собственный `prepared_cache` больше не нужен. Нужны читаемые namespaces,
потому что разные научные recipes не должны перезаписывать друг друга. Это не
hash-адресация.

Пример для SAM40 default, где исключён `mirror`:

```text
datasets/bids/sam40/
└── derivatives/
    ├── mne-bids-pipeline/
    │   └── default/
    │       └── sam40__ex-mirror/
    │           ├── run-01/
    │           │   ├── dataset_description.json
    │           │   ├── _cache/
    │           │   └── sub-*/eeg/*_proc-clean_raw.fif
    │           ├── run-02/
    │           └── run-03/
    └── eeg-validation-strategies/
        └── default/
            └── sam40__ex-mirror/
                ├── dataset_description.json
                ├── resolved-preparation.json
                └── sub-*/eeg/*_desc-windows_proc-default_epo.fif
```

Для Distinguishing отдельные `run-*` roots не нужны: Pipeline сам делает ICA
по subject/session. `preparation` recipe должен получить короткое стабильное
имя (`default`, `noica`), а dataset composition продолжает иметь нынешнее
читаемое имя `sam40__full`, `sam40__ex-mirror`,
`distinguishing__ex-drowsy` и т. п.

Два derivative producers разделены намеренно:

- `mne-bids-pipeline/` честно принадлежит внешней Pipeline;
- `eeg-validation-strategies/` честно описывает наш нестандартный windowing и
  не приписывает его внешней библиотеке.

`find_matching_paths()` официально поддерживает фильтры `subjects`,
`sessions`, `tasks`, `runs`, `processings`, `descriptions`, `suffixes`,
`extensions` и `datatypes`; см.
[MNE-BIDS API](https://mne.tools/mne-bids/stable/generated/mne_bids.find_matching_paths.html).
Для финальных файлов поиск будет примерно таким:

```python
paths = mne_bids.find_matching_paths(
    root=window_deriv_root,
    datatypes="eeg",
    descriptions="windows",
    processings=preparation_name,
    suffixes="epo",
    extensions=".fif",
    check=False,
)
epochs = mne.concatenate_epochs(
    [mne.read_epochs(path, preload=True, verbose=False) for path in paths],
    on_mismatch="raise",
    verbose=False,
)
```

Перед concatenate loader должен сортировать пути по BIDS entities, исключать
повторное чтение split-parts и проверять одинаковые channel order, event IDs и
схему metadata.

## Маппинг нынешнего preparation config

### Перенести в `preparation.pipeline`

| Нынешний параметр | Pipeline setting / действие |
| --- | --- |
| `filtering.l_freq` | `l_freq` |
| `filtering.h_freq` | `h_freq` |
| `filtering.notch_frequencies` | `notch_freq` |
| `filtering.resample_sfreq` | `raw_resample_sfreq` |
| `filtering.reference` | `eeg_reference` |
| 12 transfer channels | вычислить точный `drop_channels` для каждого dataset config |
| `ica.ica_enabled` | `spatial_filter="ica"` либо `None` |
| ICLabel activation | `ica_use_icalabel=True` — без отдельного `label_components()` |
| retired EOG proxy | `ica_use_eog_detection=False`, `ica_use_ecg_detection=False` |
| `ica_n_components` | `ica_n_components` |
| random state | `random_state` |
| extended Infomax | `ica_algorithm="extended_infomax"` или `"picard-extended_infomax"` |
| исключать только eye blink / muscle artifact | явный `ica_icalabel_include` со всеми прочими 5 классами |
| вероятность 0.8 | `ica_exclusion_thresholds={"eye blink": 0.8, "muscle artifact": 0.8}` |
| QC epochs до ICA | `ica_reject="autoreject_local"` |
| QC epochs после ICA | `reject="autoreject_local"` |
| максимум один ремонтируемый канал | `autoreject_n_interpolate=[1]` |
| clean continuous output | `process_raw_clean=True` |
| Pipeline cache | `memory_location=True`, сначала `memory_file_method="mtime"` |

Для ICLabel Pipeline 1.10.1 требует average reference, extended Infomax,
`ica_l_freq=1.0` и `ica_h_freq=100.0`. При 128 Hz Nyquist равен 64 Hz; исходник
Pipeline специально превращает 100 Hz low-pass в `None` для совместимости с
ICLabel. Это соответствует нынешней отдельной wide-band ICA branch, но должно
быть проверено численно.

`analyze_channels` здесь не подходит: документация прямо говорит, что он не
ограничивает preprocessing. Чтобы сохранить нынешний average reference и ICA
именно в общем 12-канальном пространстве, лишние EEG channels нужно убрать до
preprocessing через `drop_channels`.

### Оставить отдельными узлами

```yaml
preparation:
  name: default
  selection:
    exclude_session_values: ...
  pipeline:
    # только официальные settings MNE-BIDS-Pipeline
    ica_reject: autoreject_local
    reject: autoreject_local
    autoreject_n_interpolate: [1]
  windowing:
    window_size_seconds: 5.0
    overlap_seconds: 0.5
    channels: [...12 names...]
```

`selection` и `windowing` не надо маскировать под параметры Pipeline: это
проектная семантика. Такая граница делает очевидным, какие изменения
инвалидирует внешняя Pipeline, а какие — только финальный derivative Epochs.

## EEG quality control: штатная конфигурация AutoReject

Нынешний код до average reference и ICA:

- создаёт `BAD_peak`/`BAD_flat` через `mne.preprocessing.annotate_amplitude()`;
- ищет bad EEG channels через `find_bad_channels_lof()`;
- ограничивает число auto-bads;
- интерполирует их до reference и ICA;
- не даёт BAD spans попасть в ICA fit и Epochs.

Настраиваемая Pipeline-native замена находится не в MEG-разделе
`find_flat_channels_meg`/`find_noisy_channels_meg`, а в двух местах
artifact-rejection config:

```python
ica_reject = "autoreject_local"
reject = "autoreject_local"
autoreject_n_interpolate = [1]
```

Поведение шагов различается намеренно:

1. `ica_reject="autoreject_local"` обучает локальные пороги на технических
   epochs до ICA, отмечает epochs с чрезмерным числом плохих каналов и не
   допускает их в ICA fit. Интерполяции перед ICA нет, чтобы искусственно
   восстановленный сигнал не формировал decomposition.
2. `reject="autoreject_local"` после ICA обнаруживает плохие EEG channels
   отдельно в каждом epoch, интерполирует допустимое число и удаляет epoch,
   если повреждение шире выбранного лимита.
3. `autoreject_n_interpolate=[1]` соответствует консервативному смыслу
   нынешнего `max_auto_bad_channels: 1`: разрешить ремонт одного канала, а не
   постепенно синтезировать значительную часть 12-канального пространства.
   Пороги амплитуды при этом обучаются на данных по каналам, а не задаются одним
   глобальным `150 µV`.

Это рекомендуемый самим Pipeline workflow: local AutoReject до и после ICA.
Он документирован в разделах
[ICA rejection](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/ssp_ica.html#ica_reject)
и
[amplitude-based artifact rejection](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/artifacts.html).

Глобально известные плохие каналы остаются фактом BIDS и маркируются
`status=bad` в `*_channels.tsv`; Pipeline исключает их из AutoReject и
последующего анализа. Local AutoReject отвечает за transient/channel-specific
повреждения отдельных epochs. Поэтому ручной LOF, `BAD_peak`/`BAD_flat`,
интерполяция Raw и `reject_peak_to_peak_uv` больше не являются целевым кодом.
Это осознанная замена QC-метода на штатную adaptive policy, а не попытка
параметрически воспроизвести LOF. Постоянно или глобально плоский электрод
должен быть отражён в `channels.tsv`; transient-качество финальных окон
контролирует AutoReject.

Остаётся одна техническая граница, но не нерешённый вопрос QC: штатный
`reject` очищает epochs, созданные самой Pipeline, и не изменяет
`proc-clean_raw`. Наши condition-aware overlapping windows создаются позже,
поэтому оконный слой должен вызвать `autoreject.AutoReject` с тем же
`autoreject_n_interpolate=[1]`. Это несколько прямых строк вокруг готовой
библиотеки, а не собственная эвристика качества. Настройки берутся из того же
узла `preparation.pipeline`, чтобы Pipeline epochs и проектные windows не могли
получить разные policies.

PoC теперь нужен не для выбора механизма, а для калибровки и подтверждения
новой policy: проверить reject log, долю интерполяций/удалений по conditions и
убедиться, что AutoReject получает валидные sensor locations из BIDS montage.

## Изменения по файлам после успешного PoC

| Файл/область | Изменение | Эффект |
| --- | --- | --- |
| `python_project/pyproject.toml` | добавить pin `mne-bids-pipeline==1.10.1` и прямую зависимость `autoreject`; согласовать диапазоны MNE/MNE-BIDS | воспроизводимая внешняя реализация preprocessing и явная зависимость оконного слоя |
| `python_project/configs/preparation/**` | разделить `selection`, `pipeline`, `windowing`; добавить стабильное `name` | YAML остаётся Hydra-рецептом, но официальные Pipeline keys видны напрямую |
| `python_project/scripts/mne_bids_pipeline_config.py` | один тонкий JSON→module adapter без Hydra | публичный config contract внешней CLI |
| `python_project/src/preparation.py` | удалить ручные filter/reference/ICA/report/cache; оставить selector, запуск CLI, windowing и loader | основное сокращение и устранение самого хрупкого кода |
| `python_project/src/validation.py` | заменить повторяющиеся `get_dataset_dir()+read_epochs` единым loader | чтение нескольких BIDS derivative Epochs и concatenate |
| `python_project/src/utils.py` | удалить `PREPARED_CACHE_ROOT`; при необходимости оставить только корни raw BIDS | прекращение custom cache addressing |
| `python_project/Makefile` | удалить очистку `datasets/prepared_cache`; добавить явную очистку только `_cache` Pipeline либо не удалять derivatives вообще | нельзя случайно смешать старый и новый cache lifecycle |
| `python_project/datasets/prepared_cache/.gitkeep` | удалить после миграции | каталог больше не часть архитектуры |
| `.gitignore` | игнорировать dataset-local `derivatives/` и временные window files | производные данные не попадают в git |
| `.claude/rules/python_project/overview.md` | Stage 1 описать как Pipeline + project windowing | постоянный контекст совпадает с кодом |
| `.claude/rules/python_project/datasets.md` | заменить `prepared_cache` на два BIDS derivative producers и их provenance | правила данных перестают требовать старый hash artifact |
| `.claude/rules/python_project/scripts.md` | описать единственный Hydra entrypoint и subprocess Pipeline | не появляется второй Hydra runner |
| `notebooks/check_preparation.ipynb` | позже переключить diagnostic paths на derivatives | ручная проверка смотрит реальный новый артефакт |

Не нужно заводить пакет адаптеров, classes, registry или второй config
framework. Вся проектная часть может остаться одной процедурной
`preparation.py` и одним десятком официальных Pipeline settings в YAML.

## Пошаговый план реализации

### Фаза 0. Зафиксировать эталон

1. Не менять нынешнюю подготовку.
2. Выбрать маленький, но диагностичный срез:
   - Distinguishing: 2 subjects × sessions 03 и 04, все три blocks;
   - SAM40: 2 subjects × runs 01 и 02 × все четыре tasks;
   - обе recipes `default` и `no_ica`.
3. Сохранить из текущего Stage 1:
   - число windows по dataset/subject/session/run/task/condition;
   - channel names/order и sfreq;
   - rejected windows;
   - LOF/amplitude bads;
   - ICA labels/probabilities/excluded components;
   - PSD до/после;
   - короткий baseline decoder result с фиксированным seed.

Это oracle сравнения, а не новый постоянный тестовый framework.

### Фаза 1. Минимальный Pipeline PoC без изменения validation

1. Добавить dependency pin и lock-файл через `uv sync`.
2. Создать один ручной config.py для каждого PoC dataset, без Hydra bridge.
3. Выставить только preprocessing settings:
   - `ch_types=["eeg"]`, `data_type="eeg"`;
   - exact subjects/sessions/tasks/runs;
   - `drop_channels` до общих 12;
   - filter/notch/resample/reference;
   - ICA/ICLabel;
   - `process_raw_clean=True`;
   - `task_is_rest=True` и технические 5-s non-overlapping ICA epochs;
   - sensor/source analysis не запускать.
4. Для SAM40 выполнить отдельный deriv root на run 01 и run 02.
5. Проверить, что:
   - annotations сохранились в `proc-clean_raw`;
   - одна ICA создана на нужную unit;
   - выбранные tasks и sessions точны;
   - excluded tasks не участвовали в SAM40 ICA;
   - cache даёт no-op на повторном вызове и корректно инвалидируется при
     изменении filter/ICA settings.

Ворота: если exact selection или ICA-unit нельзя доказать по outputs/report,
миграцию остановить до исправления конфигурации.

### Фаза 2. Проверить настроенную AutoReject policy

1. Выставить `ica_reject="autoreject_local"`,
   `reject="autoreject_local"`, `autoreject_n_interpolate=[1]`.
2. Прогнать оба Pipeline AutoReject steps на эталонном срезе.
3. Сопоставить reject logs с текущими `BAD_peak`, `BAD_flat`, LOF и
   interpolation.
4. Проверить не только число epochs, но и распределение потерь по conditions:
   rejection не должен систематически менять классовый баланс.
5. Просмотреть ICA и AutoReject reports вручную — сама документация Pipeline требует
   инспекции найденных components.
6. Записать AutoReject policy в preparation config и постоянные
   `.claude` rules.

Ворота: на 12-канальном montage AutoReject должен успешно интерполировать один
канал, удалять более широко повреждённые epochs и не создавать систематической
разницы rejection между classes.

### Фаза 3. Реализовать condition-aware window derivative

1. Найти `proc-clean_raw` (`proc-filt_raw` для `no_ica`) через
   `find_matching_paths()`.
2. Проверить BIDS entities против selector output: ни одного лишнего или
   пропущенного файла.
3. Нарезать окна по annotations, перезапуская сетку внутри каждого block.
4. Применить `autoreject.AutoReject` с тем же
   `autoreject_n_interpolate=[1]`, который записан для Pipeline; сохранить
   reject log/summary по recording и condition.
5. Сохранить прежние metadata:
   `dataset`, `subject`, `session`, `task`, `run`, `recording_unit`, `label`,
   `condition`, `window_start_s`, `window_stop_s`.
6. Писать один Epochs derivative на recording с `desc-windows` и
   `proc-<preparation.name>`.
7. Публиковать файл атомарно и сохранять resolved selection/window config плюс
   source fingerprints.
8. Повторный вызов не должен трогать совпадающий derivative; изменение clean
   Raw или window config пересобирает только соответствующие окна.

Ворота: counts, onsets и labels должны совпасть с oracle там, где QC не изменил
состав.

### Фаза 4. Подключить Hydra bridge и validation

1. Добавить стабильное `preparation.name`.
2. Реализовать selector для conditions и `exclude_session_values`.
3. Сериализовать только официальные Pipeline settings в resolved JSON.
4. Подключить статический `mne_bids_pipeline_config.py` и public CLI.
5. Обернуть один derivative namespace файловым lock, чтобы параллельные Hydra
   jobs не писали один output одновременно.
6. Сделать `load_prepared_epochs(dataset, preparation)`:
   - обеспечить актуальные Pipeline/window derivatives;
   - найти все matching Epochs;
   - прочитать и объединить;
   - вернуть тот же `mne.Epochs`, который ожидает validation.
7. Заменить все прямые обращения к `epochs-epo.fif` в `validation.py` этим
   loader. Сами protocol splits не менять.

Ворота: все шесть validation strategies должны построить те же fold units и
не увидеть ни одной лишней recording.

### Фаза 5. Численное сравнение и переключение

1. На эталонном срезе сравнить:
   - shape, sfreq, channels, metadata schema;
   - window counts/onsets;
   - rejection per condition;
   - PSD/bandpower summaries;
   - ICA fit scope и ICLabel decisions;
   - baseline decoder metrics/folds.
2. Каждое отличие классифицировать как:
   - ожидаемое улучшение внешней реализации;
   - принятое изменение QC policy;
   - ошибка миграции.
3. Не требовать побитового равенства FIR/ICA, но требовать равенства научной
   выборки, unit boundaries и absence of leakage.
4. Прогнать полный `scenario_decoder` и `ica_ablation` только после малой
   проверки.

### Фаза 6. Удалить старую реализацию

1. Удалить custom hash/cache/staging/report и ручной filter/ICA код.
2. Удалить `prepared_cache` path, Make target и `.gitkeep`.
3. Не удалять существующие пользовательские cache artifacts автоматически;
   сначала сообщить их объём и путь, затем удалить только по явному решению.
4. Обновить `.claude` rules и diagnostic notebook.
5. Ещё раз выполнить один cached и один invalidated run.

## Acceptance criteria

Миграция считается успешной, если одновременно выполнено следующее:

- metadata-based session selection даёт точный список BIDS recordings;
- excluded SAM40 conditions не входят в Pipeline task list и ICA fit;
- Distinguishing windows берутся только из выбранных condition blocks;
- ни одно окно не пересекает block boundary;
- SAM40 ICA по-прежнему fit отдельно на каждый `(subject, run)`;
- Distinguishing ICA fit отдельно на каждый `(subject, session)`;
- clean Raw сохраняет annotations и имеет ожидаемые 12 channels, reference,
  sfreq и filter band;
- ICLabel policy воспроизводима и доступна для ручной проверки в report;
- до ICA применяется `ica_reject="autoreject_local"`, а к финальным
  condition windows — та же local AutoReject policy;
- `autoreject_n_interpolate=[1]`, sensor locations валидны, reject log сохранён;
- изменение Pipeline setting инвалидирует нужный внешний step;
- изменение только windowing не перезапускает filter/ICA;
- validation находит Epochs по BIDS entities, а не по одному магическому пути;
- cross-session/cross-task folds сохранили прежние grouping units;
- новый `preparation.py` существенно короче и больше не реализует
  filter/reference/ICA/cache/report самостоятельно.

## Риски и решения

| Риск | Решение |
| --- | --- |
| Pipeline объединит все SAM40 runs в ICA | отдельный deriv root и вызов на каждый run |
| Metadata sessions не образуют общий список | группировать subjects по одинаковой selection matrix или fail-fast |
| `task_is_rest` изменит event handling | использовать только для технических ICA epochs; проверить сохранность annotations в PoC |
| `no_ica` не создаёт `proc-clean_raw` | читать `proc-filt_raw`, а финальные Epochs маркировать собственным recipe processing |
| Local AutoReject требует валидных EEG sensor locations | montage уже фиксируется в raw BIDS; PoC падает, если positions отсутствуют или несовместимы |
| Pipeline `reject` не очищает custom windows в `proc-clean_raw` | оконный слой вызывает тот же `autoreject.AutoReject` с тем же config, без собственной QC-эвристики |
| Два Hydra jobs пишут один derivative | lock на readable namespace и атомарная публикация окон |
| Старые Epochs останутся после изменения selection | exact manifest + проверка множества ожидаемых BIDS entities; удалять только файлы этого namespace |
| ICLabel повторный fit изменит component indices | хранить reports/components TSV и повторно инспектировать после upstream change |
| Обновление Pipeline незаметно меняет результаты | exact pin 1.10.1; обновление только отдельным проверенным change |
| `find_matching_paths` захватит Pipeline technical epochs | искать одновременно `desc=windows`, project derivative root и exact processing |

## Итоговая оценка

Миграция реалистична и архитектурно выгодна. Она удаляет наиболее сложную
часть собственного кода — heavy continuous preprocessing, ICA application,
reports и cache invalidation. Conditions и session metadata не мешают
интеграции: они остаются небольшим selector слоем. Hydra также не мешает:
нужен тонкий config bridge, а не второй framework.

Overlapping condition-block windows остаются проектным шагом. EEG-QC при этом
не требует прежних LOF/amplitude эвристик: Pipeline настраивает local
AutoReject до и после ICA, а короткий оконный слой применяет ту же policy к
нестандартным финальным windows.

Главное оставшееся техническое уточнение — сохранить SAM40 ICA unit по run.
Если это сделать, MNE-BIDS-Pipeline сможет стать основой подготовки без
изменения семантики validation protocols.
