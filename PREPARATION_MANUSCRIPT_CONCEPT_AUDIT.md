# Соответствие текущего preparation цели clean manuscript

Дата проверки: 2026-08-30.

## Краткий вывод

**Да, с текущей общей идеей preparation исследование осуществимо.** Оно будет
проверять именно то, что теперь корректно заявлено в статье: zero-shot
переносимость *protocol-defined* контраста «high task demand vs low task
demand» между субъектами, задачами и двумя разными датасетами. Единый вход
действительно обеспечен: 12 одинаково названных каналов, 128 Hz, average
reference, один диапазон фильтрации и один ICA-рецепт для обеих сторон.

Это не превращает два протокола в измерения одного и того же психологического
состояния — и clean manuscript этого больше не утверждает. Для benchmark'а
предадаптационного переноса это не изъян, а именно измеряемый source--target
shift.

Но **в текущем коде есть один критический концептуальный разрыв с рукописью**:
Dataset A включает `drowsy` в класс `low_attention`. Пока он не устранён,
нельзя говорить, что получающиеся результаты отвечают исследовательскому
вопросу clean manuscript.

## Что согласовано и достаточно для вопроса статьи

| Требование исследования | Текущее состояние | Оценка |
| --- | --- | --- |
| Общая геометрия входа в cross-dataset decoder | `analyze_channels` в обоих recipes: F3, F4, F7, F8, FC5, FC6, O1, O2, P7, P8, T7, T8 | Соответствует. Эти 12 каналов действительно присутствуют в обоих BIDS-наборах. |
| Sampling rate | Оба исходника 128 Hz; `raw_resample_sfreq: 128.0` | Соответствует. |
| Одинаковая техническая чистка | 50-Hz notch, 1--40 Hz band-pass, average reference, extended Infomax ICA + ICLabel в обоих recipes | Концептуально соответствует: нет dataset-specific ручной подгонки. |
| Окна и leakage barrier | 5 s, overlap 0.5 s; метаданные `recording_unit`; окна нарезаются после cleaning | Соответствует общему дизайну, если validation действительно группирует по этому идентификатору. |
| Исключение habituation Dataset A | `sessions: ['03', '04', '05', '06', '07']` | Соответствует описанию статьи. |
| Заявленный предмет вывода | Кодовые имена классов и общая рамка: protocol-defined high/low contrast | Соответствует современному, осторожному claim статьи; не требует, чтобы задачи измеряли один латентный «уровень внимания». |

## Критический blocker: `drowsy` реально возвращён в анализ

Статья однозначно исключает `Drowsy`: он отличается от `Focused`/`Unfocused`
не только demand, но и закрыванием глаз, arousal и ocular activity. Это
правильное ограничение, особенно для cross-dataset переноса.

Однако [`preparation.py`](python_project/src/preparation.py) задаёт:

```python
"distinguishing": {
    "low_attention": ["unfocused", "drowsy"],
    "high_attention": ["focused"],
}
```

Далее `prepare_epochs()` берёт все annotations, чьё описание есть в этой
mapping. В BIDS Dataset A каждая запись действительно содержит трёх 600-s
блока: `focused`, `unfocused`, `drowsy`. Поэтому `drowsy` не остаётся только
для ICA: из него создаются labelled windows с меткой low.

Последствие не косметическое: Dataset A меняется с контраста
`Focused vs Unfocused` на `Focused vs (Unfocused + Drowsy)`. Это добавляет в
low class сильный arousal/eyes-closed contrast, прямо исключённый в статье.
Тогда cross-dataset loss или apparent discrimination невозможно чисто читать
как переносимость заявленного high-demand/low-demand контраста.

**Минимальное условие продолжения:** исключить `drowsy` из `DATASET_MAPPING`
или независимо фильтровать его до epoching, затем заново подготовить Dataset A
и пересчитать все связанные baseline/cross-subject/cross-session/cross-dataset
сценарии. Не следует лишь переписать рукопись так, будто drowsy является
обычной low-demand меткой: это существенно меняет construct и делает его ещё
менее сопоставимым с SAM-40 relax.

## Несовпадения описания с текущим implementation: не blocker идеи, но blocker воспроизводимости текста

| В clean manuscript | Текущий implementation | Что делать |
| --- | --- | --- |
| 0.5--45 Hz | 1--40 Hz (`l_freq: 1.0`, `h_freq: 40.0`) | Обновить Methods/S6 либо намеренно вернуть прежний рецепт. |
| FastICA; EOG-proxy F3/F4/F7/F8; z=2.4; fixed random state | `extended_infomax`, `ica_use_icalabel: true`, threshold 0.8 для ICA-label категорий; явного random state нет | Описать фактическое автоматизированное ICA-cleaning, а не старую процедуру. |
| ICA fit на конкатенированных записях одного recording unit | В текущих YAML нет настройки, выражающей такую конкатенацию; runner отдаёт MNE-BIDS-Pipeline набор BIDS raw-записей | Проверить в отчёте нового прогона фактическую единицу ICA fit и описать только её. |
| 5 bands до 45 Hz + 4 entropy + 5 статистик = 168 features | `classical_ml.yaml`: 4 power bands 1--40 Hz плюс sample entropy и 4 статистики = 9/channel, т.е. 108 features | Обновить Models/S6. Это особенно важно, потому что иначе заявляется несуществующий gamma 40--45 диапазон. |
| Dataset A: 3,059 / 3,059 prepared windows | С нынешним `drowsy` mapping ожидается существенно больше low окон | После исправления mapping заново получить и записать реальные counts. |

## Статус имеющихся производных

Текущие recipes пишут в новые scenario-specific roots:

- Dataset A: `.../derivatives/mne-bids-pipeline/attention`;
- SAM-40: `.../relax-stroop[-arithmetic][-mirror]`.

В рабочем дереве чистые FIF найдены только в прежних общих roots
`.../derivatives/mne-bids-pipeline/`; по актуальным путям число clean FIF равно
нулю. Поэтому прежние таблицы/figures нельзя приписать текущим YAML без
отдельной сверки их resolved configuration.

## Практическое решение

1. Сначала восстановить целевой label mapping: Dataset A = `focused` против
   `unfocused`, SAM-40 = `relax` против выбранных high-demand tasks.
2. Выполнить один preparation-прогон по новым roots и проверить состав окон,
   channels, class counts и ICA reports.
3. После этого исследование можно проводить: его валидный вывод будет о
   переносе protocol-defined contrast при compound shift, а не об
   «универсальном EEG-маркере внимания».
4. Только затем синхронизировать Data preparation, feature paragraph, S6 и
   таблицу compositions с resolved configs и новыми артефактами.

