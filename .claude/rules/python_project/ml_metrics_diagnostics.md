---
paths:
  - "python_project/**"
---

# Как находить причины низких метрик в ML-пайплайне

## Главная мысль

Низкий score редко исправляется одной «более сильной» моделью. Обычно наибольший прирост дают в таком порядке:

1. проверка данных, таргета и схемы разбиения;
2. устранение ошибок preprocessing и evaluation;
3. анализ ошибок модели;
4. улучшение признаков и данных;
5. настройка модели, гиперпараметров и decision threshold.

Важно различать две проблемы:

- **реально низкое качество** — модель не извлекает полезный сигнал;
- **ошибочная оценка качества** — выбран неправильный split/metric либо возник leakage. Leakage обычно завышает offline-score, а не уменьшает его, но затем вызывает сильное падение на новых данных.

## Ошибки, которые часто реально уменьшают score

### 1. Ошибки в данных и таргете

- неверные или шумные labels;
- перепутанное соответствие между `X` и `y` после merge/shuffle;
- дубликаты и конфликтующие метки;
- неправильные единицы измерения, типы, категории или временные зоны;
- пропуски, выбросы и sentinel-значения (`-999`, пустая строка), обработанные как обычные данные;
- признаки, доступные в train, но отсутствующие или иначе вычисляемые в inference;
- слишком слабый сигнал в исходных признаках.

**Проверка:** вручную просмотреть случайную выборку объектов, сложные ошибки и примеры каждого класса; сравнить статистики train/validation; проверить label distribution и долю ошибок разметки.

### 2. Неподходящее разбиение данных

- случайный split для временных данных;
- один субъект, пациент, клиент или документ попадает в разные folds;
- train и validation относятся к разным доменам, но этот shift не анализируется;
- слишком маленький validation set создаёт нестабильную оценку;
- редкие классы почти отсутствуют в отдельных folds.

Split должен моделировать реальное применение. Для повторных измерений нужен group-aware split (`GroupKFold`, `StratifiedGroupKFold`), для временного прогноза — временной split, для переноса между датасетами — отдельная cross-dataset evaluation.

### 3. Неконсистентный preprocessing

- scaler/encoder обучен, но не применён к validation/test;
- разный порядок колонок;
- отдельно написанный preprocessing для training и inference;
- категориальные значения, неизвестные encoder;
- трансформации ухудшают сигнал: агрессивная фильтрация, ресемплинг, обрезка или некорректная нормализация.

Все обучаемые преобразования лучше объединять с моделью в единый `Pipeline`. Это обеспечивает одинаковую обработку и снижает риск leakage. См. [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

### 4. Неверная метрика

- `accuracy` при сильном class imbalance;
- F1 без уточнения `binary`, `macro`, `micro` или `weighted`;
- ROC-AUC при задаче, где важнее precision среди небольшого числа top predictions;
- усреднение по всем samples, когда научный вывод делается по subjects/datasets;
- оптимизация одной метрики и сравнение моделей по другой.

Метрика должна соответствовать исследовательскому или бизнес-вопросу. При дисбалансе обычно полезно одновременно смотреть `balanced accuracy`, macro-F1, per-class recall, confusion matrix и PR-AUC. Не следует выбирать метрику только потому, что на ней число выше.

### 5. Underfitting и плохие гиперпараметры

Признаки underfitting: низкие и train-, и validation-метрики. Причины:

- слишком сильная регуляризация;
- слишком простая модель;
- недостаточное число эпох/итераций;
- неудачный learning rate;
- чрезмерный dropout или augmentation;
- неподходящий масштаб признаков;
- недостаточно информативные признаки.

Нужно строить learning curves и сравнивать train/validation score. Если train-score тоже низкий, увеличение данных само по себе часто не решает проблему — сначала требуется повысить способность модели извлекать сигнал.

### 6. Overfitting

Признак: высокий train-score и заметно более низкий validation-score.

Возможные меры: больше независимых данных, регуляризация, упрощение модели, early stopping, корректная augmentation, сокращение шумных признаков. Но сначала необходимо исключить distribution shift и ошибочный split — они могут выглядеть как overfitting.

### 7. Дисбаланс классов

Типичные ошибки:

- оценивать только accuracy;
- выполнять oversampling до разбиения данных;
- автоматически применять SMOTE, не проверив природу признаков;
- использовать class weights и одновременно oversampling без отдельной проверки эффекта.

Resampling должен происходить **только внутри training fold**. Сначала стоит сравнить простой baseline, `class_weight`, threshold tuning и лишь затем методы resampling.

### 8. Порог `0.5` используется без проверки

У бинарного классификатора хороший ranking может сочетаться с плохим F1/recall из-за неподходящего decision threshold. Порог следует выбирать на validation data или во внутреннем CV, а test set оставить нетронутым. В scikit-learn для этого предусмотрен [`TunedThresholdClassifierCV`](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Ошибки, которые делают score недостоверно высоким

- scaling, imputation, feature selection, PCA или oversampling до cross-validation;
- подбор гиперпараметров и отчёт результата на тех же folds без nested CV;
- многократный просмотр test-score и ручная подгонка решений;
- признаки, прямо или косвенно вычисленные из таргета;
- одинаковые субъекты, группы или почти идентичные записи по обе стороны split;
- использование будущей информации в time-series задаче.

Все операции, которые что-либо **обучают по данным**, должны обучаться только на training fold. Исследования показывают, что leakage через feature selection и повторяющихся субъектов способен существенно завышать оценку качества ([Rosenblatt et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901797/)). При настройке гиперпараметров nested CV даёт более честную оценку всей процедуры model selection ([scikit-learn example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).

## Практический порядок улучшения метрик

### Шаг 1. Зафиксировать evaluation protocol

- определить unit of generalization: sample, subject, session, organization, dataset или future period;
- выбрать primary metric и 2–4 diagnostic metrics;
- отделить финальный test set;
- зафиксировать seeds и сохранять индексы splits.

### Шаг 2. Построить простой baseline

- dummy predictor;
- линейная/логистическая модель;
- простое дерево или boosting baseline;
- один прозрачный preprocessing pipeline.

Если сложная модель не превосходит baseline, сначала искать дефект данных, признаков или обучения, а не расширять hyperparameter search.

### Шаг 3. Выполнить data audit

- shape, schema, missingness, duplicates;
- class/group distribution по каждому fold;
- диапазоны и распределения признаков;
- корректность labels и merge keys;
- признаки train–serving skew;
- отдельно проверить наиболее уверенные неправильные predictions.

### Шаг 4. Провести error analysis

Сгруппировать ошибки по классу, субъекту, источнику данных, времени, качеству входа и другим domain slices. Искать не просто «плохие объекты», а повторяющийся failure mode. Именно он подсказывает, нужны ли новые данные, другой preprocessing, признаки или loss.

### Шаг 5. Делать контролируемые эксперименты

Менять один существенный фактор за раз и сохранять:

- конфигурацию;
- commit и версию данных;
- split indices;
- метрики по folds и группам;
- mean, standard deviation и confidence interval;
- predictions для последующего анализа.

### Шаг 6. Улучшать в порядке ожидаемой отдачи

1. исправить labels и pipeline bugs;
2. добавить качественные данные для слабых slices;
3. улучшить domain-specific features/preprocessing;
4. подобрать loss, class weights и threshold;
5. выполнить разумный randomized/Bayesian hyperparameter search;
6. только затем пробовать более сложную архитектуру или ensemble.

Google также рекомендует начинать с простой модели, сначала выстроить метрики и проверять инфраструктуру независимо от модели: [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).

## Минимальный checklist перед доверием к результату

- [ ] Split соответствует реальному сценарию обобщения.
- [ ] Один group/subject не пересекает train и validation.
- [ ] Test set не использовался для принятия решений.
- [ ] Preprocessing и feature selection обучаются внутри каждого fold.
- [ ] Resampling выполняется только внутри training fold.
- [ ] Метрика соответствует задаче и дисбалансу.
- [ ] Есть dummy и простой baseline.
- [ ] Проверены train и validation learning curves.
- [ ] Посчитана вариативность по folds/seeds/groups, а не только одно среднее.
- [ ] Выполнен slice-based error analysis.
- [ ] Улучшение воспроизводится на неизменённом protocol.
- [ ] Финальный test запускается один раз после фиксации решения.

## Короткий вывод

Наиболее надёжный способ повысить метрики — не «перебирать модели», а локализовать bottleneck. Если низок train-score — проблема чаще в сигнале, признаках, оптимизации или capacity. Если train высокий, validation низкий — проверять overfitting, shift и split. Если CV высокий, а внешний test низкий — в первую очередь искать leakage, group overlap и mismatch между train и deployment. Любое улучшение считается реальным только тогда, когда оно воспроизводится на корректном независимом evaluation protocol.

## Основные источники

- [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn — Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [scikit-learn — Decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Rosenblatt et al. (2024) — Data leakage inflates prediction performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901797/)

---

# Особенности EEG

## Почему EEG требует отдельного evaluation protocol

EEG — иерархические, нестационарные временные данные:

```text
dataset → subject → session → run/trial → epoch/window → time samples
```

Окна одного субъекта или одной записи не являются независимыми наблюдениями. Они разделяют индивидуальную нейрофизиологию, расположение электродов, импеданс, оборудование, условия записи и часто одни и те же артефакты. Поэтому большое число эпох не компенсирует малое число субъектов: 10 000 окон от 20 человек по-прежнему представляют только 20 независимых субъектов для cross-subject вывода.

## Наиболее частые ошибки при работе с EEG

### 1. Случайное разбиение эпох вместо разбиения субъектов

Самая опасная ошибка — сначала нарезать EEG на epochs/windows, а затем случайно распределить их между train и test. Модель может узнавать subject-specific или recording-specific паттерны и показывать высокий score без способности работать на новом человеке.

Выбор split зависит от утверждения исследования:

| Цель | Корректная проверка |
| --- | --- |
| Новые окна того же субъекта и той же сессии | разбиение по trials/непересекающимся временным блокам |
| Новая сессия известных субъектов | holdout целых sessions внутри subject |
| Новый субъект | `GroupKFold`/LOSO по `subject_id` |
| Новый датасет или парадигма | train на source dataset, test на полностью отдельном target dataset |

Работа Brookshire et al. показывает, что segment-based split способен существенно переоценивать качество EEG-моделей на новых субъектах: [Data leakage in deep learning studies of translational EEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/).

### 2. Пересечение перекрывающихся окон

При sliding windows с overlap соседние epochs содержат часть одних и тех же samples. Если такие окна оказываются в разных folds, возникает почти прямое дублирование сигнала. Даже без overlap близкие по времени окна сильно автокоррелированы.

Правильно сначала разделить subjects/sessions/trials или непрерывные временные блоки, а уже затем нарезать каждый split на окна. Нельзя генерировать все окна и потом применять обычный `train_test_split`.

### 3. Leakage через preprocessing

К leakage могут привести операции, параметры которых оценены на всех данных до CV:

- scaling и нормализация;
- PCA/CSP и supervised spatial filters;
- feature selection;
- подбор частотных диапазонов и каналов;
- covariance estimation и alignment;
- oversampling;
- подбор параметров фильтрации по test-score.

Такие шаги должны обучаться внутри training fold. Для CSP, PCA, feature selection и alignment это обязательно. Фиксированный заранее band-pass сам по себе не обучается, но выбор его границ после просмотра test results уже является подгонкой.

### 4. Неоднозначность ICA и artifact removal

ICA — data-dependent преобразование. Его нельзя бездумно обучать на объединённых данных разных субъектов или на train+test как на едином массиве. ICA обычно оценивают отдельно для каждого субъекта/сессии на достаточном объёме непрерывного EEG, а компоненты удаляют по воспроизводимому правилу.

Но здесь важна постановка задачи:

- если production-сценарий допускает calibration recording нового субъекта, ICA/normalization могут оцениваться на этой calibration-части;
- если модель должна предсказывать сразу по короткому unseen epoch, использование всей test recording для адаптации является transductive preprocessing и должно быть явно описано;
- если ICA-компоненты выбирались вручную с учётом labels или результата классификации, возникает leakage.

Не следует считать, что более сильная очистка автоматически улучшает научную валидность или score. Артефакт может коррелировать с условием, а aggressive cleaning способен удалить и полезную нейронную активность. Исследование Kessler et al. показало значительное влияние reference, filtering, baseline correction и artifact correction на decoding performance: [How EEG preprocessing shapes decoding performance](https://www.nature.com/articles/s42003-025-08464-3). Поэтому preprocessing выбирают по физиологическому обоснованию и фиксируют до финального test, а не оптимизируют только по максимальной accuracy.

### 5. Фильтрация создаёт искажения

Типичные проблемы:

- чрезмерно высокий high-pass удаляет медленные компоненты;
- слишком низкий low-pass сохраняет мышечные и высокочастотные помехи;
- notch без необходимости искажает соседние частоты;
- фильтр с неподходящей длиной создаёт ringing и edge artifacts;
- фильтрация коротких epochs вместо непрерывной записи усиливает краевые эффекты;
- различный line frequency (`50/60 Hz`) между датасетами не учитывается;
- фильтр использует будущие samples в online-задаче.

Zero-phase filtering подходит для offline-анализа, но не имитирует causal online inference. Для online claims preprocessing должен быть causal либо ограничение нужно явно указать.

### 6. Неправильная reference и монтаж

EEG измеряет разность потенциалов, поэтому reference меняет все каналы. Ошибки:

- разные references в train и test;
- average reference при недостаточном или неодинаковом покрытии головы;
- потеря reference channel без документирования;
- смешивание каналов с одинаковыми названиями, но разным физическим положением;
- неправильное сопоставление названий электродов и координат.

Перед объединением датасетов нужно привести montage, channel names, units и reference к согласованному представлению либо использовать модель, явно устойчивую к различным channel layouts.

### 7. Неверные единицы, sampling rate и временная ось

EEG-файлы могут хранить амплитуду в `V`, `mV` или `µV`. Ошибка масштаба на `10^3–10^6` иногда не вызывает исключения, но разрушает признаки и обучение. Также часто встречаются:

- неверный `sfreq` в metadata;
- off-by-one при переводе event samples во время;
- потеря/сдвиг triggers;
- resampling EEG без согласованного resampling событий;
- aliasing при downsampling без low-pass;
- неодинаковая длина эпох после округления.

Нужно проверять амплитуды, длительность записи, число samples, event timing и визуально сопоставлять события с сигналом.

### 8. Плохие каналы, пропуски и интерполяция

Плохой канал может доминировать в variance, спектральных признаках или CNN. Но интерполяция всех данных до split также способна использовать статистику тестовой записи. Политику bad-channel detection, rejection и interpolation необходимо зафиксировать и применять одинаково.

Для cross-dataset анализа особенно опасно «лечить» отсутствующие каналы нулями без mask: модель может определять датасет по шаблону отсутствующих каналов.

### 9. Нормализация удаляет либо добавляет нужный сигнал

Варианты normalization отвечают на разные вопросы:

- global normalization может быть смещена крупными субъектами/сессиями;
- per-subject normalization использует статистику нового субъекта и предполагает доступ к calibration data;
- per-epoch normalization может удалить абсолютную мощность — потенциально важный EEG-маркер;
- normalization по всей записи может использовать будущие данные в online-сценарии.

Нельзя выбирать способ нормализации только по score. Нужно определить, какая информация должна сохраняться и доступна ли статистика нового subject в реальном применении.

### 10. Псевдорепликация при расчёте метрик

Если считать каждую epoch независимой и затем строить confidence interval по тысячам epochs, неопределённость будет искусственно мала. Для cross-subject исследования статистической единицей обычно является subject, а не window.

Следует сохранять predictions с `subject_id`, считать метрики по каждому субъекту и агрегировать их с mean/median, dispersion и confidence interval. Полезно дополнительно показывать pooled confusion matrix, но она не должна заменять subject-level variability.

## Особенности attention EEG datasets

### 1. «Attention» не является единым таргетом

Под одним словом могут скрываться разные конструкты:

- sustained attention/vigilance;
- selective attention;
- auditory или visual attention;
- concentration;
- engagement;
- mental workload;
- fatigue/drowsiness;
- mind wandering.

Нельзя объединять их как взаимозаменяемые классы без операционального определения. Модель, обученная различать `relaxing` и `concentrating`, не обязательно измеряет sustained attention, а workload не равен attention.

### 2. Labels часто косвенные и шумные

Attention label может происходить из:

- экспериментального condition;
- performance/response time;
- self-report;
- поведения глаз;
- экспертной аннотации;
- временного положения внутри задания.

Condition label означает, что участнику **предписали** состояние, а не что он фактически находился в нём весь блок. Self-report субъективен и имеет низкое временное разрешение; behavioral label может отражать моторную скорость или сложность задания. Нужно хранить происхождение label, оценивать его надёжность и не превращать длинный блок в тысячи якобы точно размеченных epochs.

Например, MEMA содержит состояния `neutral`, `relaxing` и `concentrating`, а также эмоциональные и персональные labels; это multi-label контекст, в котором психологические состояния могут коррелировать друг с другом: [MEMA dataset paper](https://arxiv.org/html/2411.09879v2), [официальный репозиторий](https://github.com/XJTU-EEG/MEMA).

### 3. Confounds вместо внимания

Модель может классифицировать не attention, а:

- движения глаз и моргание;
- мышечное напряжение;
- тип стимула или задачи;
- звук/визуальный контент;
- порядок блоков;
- fatigue и time-on-task;
- motor response;
- session или dataset identity.

Если все attentive trials используют один stimulus/task, а relaxed trials — другой, отделить neural attention marker от task identity невозможно только моделью. Нужны counterbalancing, одинаковые сенсорные условия, контроль порядка и оценка confound-only baselines.

### 4. Временной drift и порядок блоков

Во многих экспериментах внимание снижается, а усталость растёт со временем. Если классы записаны последовательными блоками, label коррелирует с временем. Random window split затем позволяет модели использовать drift.

Нужно рандомизировать/counterbalance порядок при сборе данных или использовать блоки и время как группы/ковариаты. Полезная проверка — попытаться предсказать label только по `subject_id`, `session`, `block index`, времени и metadata без EEG.

### 5. Неверное объединение датасетов

Cross-dataset transfer осложняют различия в:

- определении attention и labels;
- экспериментальной парадигме;
- числе и расположении каналов;
- reference, sampling rate и hardware;
- preprocessing;
- длине окон и class balance;
- популяции участников.

Простое объединение и случайный CV отвечает на вопрос о mixed-domain classification, а не о переносе. Для честного cross-dataset эксперимента target dataset полностью исключается из model selection; если используются target labels для настройки, это supervised domain adaptation и должно называться именно так.

### 6. Сопоставление labels между датасетами

Нельзя автоматически отождествлять `focused`, `concentrating`, `alert`, `high workload` и `attentive`. Перед harmonization нужна таблица:

```text
исходный label → экспериментальная инструкция → способ разметки → временное разрешение → итоговый общий класс
```

Если mapping требует сильных предположений, лучше оставить dataset-specific задачи и сравнить transfer, чем создать искусственно единый таргет.

## Как корректно улучшать метрики EEG-модели

### Приоритет 1. Исправить evaluation

- определить `within-subject`, `cross-session`, `cross-subject` или `cross-dataset` цель;
- группировать split на самом высоком требуемом уровне;
- разделять данные до windowing;
- выполнять data-dependent preprocessing внутри fold;
- использовать nested subject-wise CV для hyperparameter selection, когда нет отдельного validation cohort.

Subject-based и nested validation дают более надёжную оценку cross-subject моделей, чем segment-based и non-nested схемы: [Del Pup et al., 2025](https://www.sciencedirect.com/science/article/pii/S001048252500959X).

### Приоритет 2. Провести EEG data audit

- визуально просмотреть raw и preprocessed записи каждого subject/session;
- проверить PSD, line noise, flat/noisy channels и амплитудные диапазоны;
- проверить montage, reference, units, `sfreq` и triggers;
- сравнить число trials/epochs и rejection rate между классами;
- убедиться, что preprocessing не удаляет один класс чаще другого;
- проверить, может ли metadata-only baseline предсказывать label.

### Приоритет 3. Сравнить обоснованные preprocessing ablations

Вместо поиска одной «идеальной» цепочки сравнить небольшой заранее определённый набор:

- reference;
- разумные band-pass варианты;
- с/без baseline correction;
- политика bad channels/epochs;
- с/без artifact correction;
- несколько физиологически мотивированных window lengths.

Выбор следует делать во внутреннем CV и оценивать не только score, но также стабильность между subjects и физиологическую правдоподобность.

### Приоритет 4. Использовать сильные, но простые baselines

- majority/dummy classifier;
- metadata-only baseline;
- band-power features + Logistic Regression/LDA;
- covariance/Riemannian baseline;
- компактная EEG-архитектура, например EEGNet;
- одинаковый evaluation protocol для всех моделей.

Сложная нейросеть не является улучшением, если она выигрывает только при segment-level split или на части субъектов.

### Приоритет 5. Анализировать перенос, а не только среднее

Нужно показывать:

- метрики каждого subject/session/dataset;
- mean/median и dispersion;
- worst-subject или нижний квартиль;
- confusion matrix и per-class recall;
- разницу между within-subject и cross-subject;
- результаты на полностью внешнем dataset, если заявляется transfer.

## EEG checklist

- [ ] Ясно определено, что именно означает attention label.
- [ ] Subjects/sessions/trials не пересекаются между folds согласно цели исследования.
- [ ] Разбиение выполнено до создания overlapping windows.
- [ ] Preprocessing не обучается на test data.
- [ ] ICA/CSP/PCA/alignment соответствуют реальному inference-сценарию.
- [ ] Проверены units, `sfreq`, montage, reference и event timing.
- [ ] Фильтрация не создаёт edge artifacts и соответствует offline/online claim.
- [ ] Bad channels и rejected epochs анализируются отдельно по классам.
- [ ] Метрики агрегируются на уровне subject, а не только epoch.
- [ ] Проверены metadata-only и artifact/confound baselines.
- [ ] Attention не подменён workload, fatigue, task identity или block order.
- [ ] Cross-dataset label mapping описан и обоснован.
- [ ] Все модели сравниваются на одних и тех же splits.
- [ ] Финальный test dataset не использовался для preprocessing/model selection.

## Вывод по EEG

Для EEG высокий score особенно легко получить по неверной причине. Главная проверка — не «насколько хорошо классифицируются окна», а **какая независимая единица действительно обобщается**: новый trial, session, subject, paradigm или dataset. Для attention-задач дополнительно необходимо доказать, что модель извлекает сигнал внимания, а не артефакты, task identity, fatigue или порядок блоков. Только после фиксации этой логики имеет смысл улучшать preprocessing, features и архитектуру.

## Основные источники по EEG

- [Brookshire et al. (2024) — Data leakage in deep learning studies of translational EEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/)
- [Kessler et al. (2025) — How EEG preprocessing shapes decoding performance](https://www.nature.com/articles/s42003-025-08464-3)
- [Del Pup et al. (2025) — The role of data partitioning on EEG model performance](https://www.sciencedirect.com/science/article/pii/S001048252500959X)
- [Kinahan et al. (2024) — Achieving Reproducibility in EEG-Based Machine Learning](https://dl.acm.org/doi/fullHtml/10.1145/3630106.3658983)
- [Carlson et al. (2025) — NERVE-ML reproducibility checklist](https://pmc.ncbi.nlm.nih.gov/articles/PMC11948487/)
- [Liu et al. (2025) — Multi-label EEG Dataset for Mental Attention State Classification](https://arxiv.org/html/2411.09879v2)
