# Диагностика `scenario_decoder` — run 2026-08-30

## Итог

Главная причина слабых метрик — не одна неудачная настройка модели. Это
сочетание (1) protocol-defined меток, жёстко связанных с временем/типом
задачи, и (2) очень сильного междатасетного domain shift. Текущий preparation
может добавлять вариативность, но по сохранённому run-у нет свидетельства, что
именно он — первичный блокер. Полный hyperparameter sweep сейчас с высокой
вероятностью улучшит только baseline, но не решит cross-dataset transfer.

## Что подтверждают результаты

Метрики ниже пересчитаны из всех `windows.parquet` и `folds.parquet` run-а;
это balanced accuracy / ROC-AUC на всех test windows.

| Протокол | Средняя BA по 5 моделям | Диапазон BA моделей | Средняя AUC |
| --- | ---: | ---: | ---: |
| Baseline | 0.689 | 0.641–0.782 | 0.755 |
| Cross-subject | 0.662 | 0.633–0.707 | 0.721 |
| Cross-session | 0.647 | 0.629–0.661 | 0.691 |
| Cross-task | 0.624 | 0.552–0.691 | 0.668 |
| Cross-dataset | 0.516 | 0.422–0.597 | 0.529 |

- Картина повторяется у логистической регрессии, XGBoost, EEGNet,
  ShallowFBCSPNet и EEGConformer. ShallowNet чаще лучший, но его преимущество
  не меняет порядок протоколов: это против версии о поломке конкретной
  архитектуры или её оптимизатора.
- Внутри SAM-40 лучший baseline — ShallowNet на Stroop, BA=0.782 и AUC=0.857;
  на полном SAM-40 — BA=0.720/AUC=0.813. Значит, сигнал и текущий конвейер
  способны извлечь различимость внутри одного домена.
- Наиболее сильный cross-task результат также не случайный: ShallowNet,
  source `stroop+arithmetic` → `mirror`, BA=0.666; но transfer
  `stroop` → `arithmetic` уже BA=0.599. В общих для задач компонентах есть
  информация, однако она значительно слабее task-specific информации.
- Междатасетный перенос почти исчезает даже по AUC, а не только при пороге
  0.5. Например, `distinguishing` → полный SAM-40 у ShallowNet:
  BA=0.488/AUC=0.477; у логистической регрессии BA=0.521/AUC=0.536.
  Следовательно, это не устраняется простым подбором порога или
  `class_weight`.
- Вариабельность по субъектам велика. Для ShallowNet на SAM-40 baseline
  subject-wise BA лежит в 0.489–0.911, при cross-subject — 0.500–0.933;
  для `distinguishing` всего 5 субъектов (cross-subject: 0.589–0.720).
  Оконные метрики нельзя читать как тысячи независимых наблюдений.

## Наиболее вероятные причины

### 1. Метка не отделима от порядка блоков и task identity — высокая уверенность

В `distinguishing` исходный `events.tsv` задаёт всегда одну и ту же
последовательность: `focused` 0–600 s, `unfocused` 600–1200 s, затем `drowsy`.
Текущий recipe обрезает запись до 1200 s, и потому конечный набор ровно
сбалансирован: 3,192 окна `focused` (старт 0–594 s) и 3,192 `unfocused`
(600–1194 s), по 24 recording units. Время начала окна само по себе
предсказывает метку с полной точностью.

В SAM-40 метка равна task-level condition целого 25-секундного run-а:
`relax` — low, `stroop`/`arithmetic`/`mirror` — high. Поэтому baseline может
учить не общий физиологический demand, а различия сенсорики, инструкции,
движений, структуры стимула или времени задачи. Это не ошибка кода; это
ограничение доступного дизайна и именно объясняет, почему cross-task и
cross-dataset хуже baseline.

### 2. Несопоставимость двух датасетов — высокая уверенность

Даже после harmonization до одинаковых 12 каналов, 1–45 Hz, 128 Hz, average
reference и ICA, классы формируются разными манипуляциями: непрерывный
последовательный focused/unfocused блок в Dataset A против rest и трёх разных
когнитивных задач в Dataset B. Почти-chance AUC в обоих направлениях означает,
что единого ранжирующего сигнала в текущем representation не обнаружено.
Это ожидаемо для cross-dataset zero-shot и не является доказательством, что
«нейронного маркера нет».

### 3. Эффективный размер выборки намного меньше числа окон — высокая уверенность

Окна имеют длину 5 s и overlap 0.5 s (90%). В полном SAM-40 это 2,400 окон,
но лишь 480 task-runs: по 120 runs на `relax`, `stroop`, `arithmetic`,
`mirror`; пять сильно перекрывающихся окон приходятся на один run. Group-aware
разбиение по recording unit не допускает прямого leakage между train/test —
это корректно, — но не превращает окна в независимые образцы. Особенно это
ограничивает сети с 170k+ параметрами и делает оконные доверительные интервалы
нереалистично узкими.

### 4. Hyperparameters сетей — вторичный, но ещё непроверенный фактор — средняя уверенность

Все сети обучались фиксированно 100 эпох без внутреннего validation split,
early stopping или сохранённой epoch history. В артефактах нет train scores:
в `folds.parquet` есть только membership training windows, без их предсказаний;
сами fitted estimators и skorch history не сохранены. Поэтому нельзя строго
исключить недо-/переобучение отдельных DL рецептов. Но высокий ShallowNet
baseline и та же деградация у классических моделей показывают, что это не
главное объяснение плохого cross-dataset score.

## Оценка конкретных моделей

Это диагностика по held-out результатам и архитектурной ёмкости, а не
окончательный verdict об overfit: train metrics и кривые не были сохранены.

| Модель | Наблюдение | Наиболее вероятный статус | Практическое действие |
| --- | --- | --- | --- |
| Logistic regression | BA в среднем 0.589; не проваливается относительно сложных моделей и в части направлений cross-dataset — лучший classical контроль | Не явный underfit/overfit; скорее потолок текущих handcrafted features | Оставить контрольной. Затем искать `C` и L1/L2 только train-only nested CV. |
| XGBoost | BA 0.594; не даёт стабильного выигрыша над linear model, включая cross-dataset | Не доказан overfit; trees не нашли важной нелинейности в нынешних features | Не расширять сразу число деревьев/глубину. Проверить малый grid regularisation после QC representation. |
| EEGNet | Всего 1,938 параметров; средняя BA 0.571 и baseline/cross-task ниже ShallowNet | Кандидат на **недообучение либо неподходящий inductive bias**, но не доказано | Сначала сохранить learning curve; затем проверить LR, temporal kernel/F1/D и early stopping. |
| ShallowFBCSPNet | 23,282 параметра; лучшая средняя BA 0.610, лучший SAM-40 baseline 0.782; сохраняет полезный cross-task signal | Наиболее здоровый текущий decoder; заметен domain/subject shift, но нет признака катастрофического overfit | Использовать как основной diagnostic model и аккуратно тюнить после абляций. |
| EEGConformer | 170,466 параметров; baseline не превосходит ShallowNet, хотя на SAM-40 cross-subject достигает 0.707 | Главный кандидат на **избыточную ёмкость/неоптимальное обучение** при малом числе независимых runs; доказательств пока нет | Сохранить train curves; сравнить меньший вариант, LR/weight decay и early stopping на source-only validation. |

Важно: почти-chance cross-dataset AUC у **всех** пяти семейств исключает
сценарий, в котором исправление одного DL-рецепта даст большой переносимый
score. Даже если EEGNet недообучен или Conformer переобучен, это может
повысить внутри-доменный score и несколько cross-task направлений, но не
объясняет общий провал linear и tree моделей в cross-dataset.

## Отдельно: почему baseline / cross-subject / cross-session не выше

Эта часть не использует cross-task или cross-dataset как объяснение.

### Generalization не является главным источником падения

| Датасет / протокол | BA пяти моделей | Вывод |
| --- | --- | --- |
| `distinguishing` baseline | 0.641–0.670 | Внутри датасета различимость умеренная у всех моделей. |
| `distinguishing` cross-subject | 0.633–0.668 | Практически нет отдельного штрафа за нового субъекта. |
| `distinguishing` cross-session | 0.629–0.661 | Пуленная метрика близка к baseline, но отдельные session-folds очень нестабильны. |
| SAM-40 all baseline | 0.671–0.720 | Декодируемый within-dataset signal есть, максимум у ShallowNet. |
| SAM-40 all cross-subject | 0.652–0.707 | Нет систематического коллапса относительно baseline. |

То есть типичный паттерн сильного overfit — «очень высокий baseline, затем
резкий collapse на новом subject/session» — здесь отсутствует. Основная
проблема уже находится в baseline: текущие 5-секундные EEG representations
содержат лишь умеренно различимый сигнал. Cross-session особенно шумный,
поскольку один test fold — одна короткая session одного субъекта; у ShallowNet
fold BA лежит в 0.346–0.883, хотя pooled BA=0.661.

### Самый вероятный технический потолок для SAM-40: искусственно суженная montage

Все протоколы используют один common set из 12 каналов:
`F3, F4, F7, F8, FC5, FC6, O1, O2, P7, P8, T7, T8`. В SAM-40 из нативной
montage специально исключены ещё 20 каналов, включая `Cz`, `Fz`, `C3/C4`,
`P3/P4`, `Pz` и `Oz`. Это корректная цена за единое representation с
`distinguishing` для cross-dataset benchmark, но она может существенно
ограничивать baseline/cross-subject SAM-40: пространственный паттерн task
demand часто включает именно центрально-париетальные каналы.

Это не повод молча заменить headline benchmark: тогда модели в разных
протоколах получают разные входы. Но это приоритетная **native-montage
диагностическая абляция**: повторить SAM-40 baseline/cross-subject на всех
валидных EEG-каналах, сохранив тот же preprocessing, split и модель. Если
прирост будет устойчивым на subject-wise BA, текущий потолок вызван главным
образом выбором общего представления, а не плохим обучением.

### Что в обучении действительно требует проверки

- У трёх сетей фиксированы 100 эпох, нет inner validation/early stopping, а
  кривые не сохранены. Это делает EEGConformer (170,466 параметров) риском
  overfit, а EEGNet (1,938 параметров) — риском underfit, но не позволяет
  доказать ни одно утверждение по текущему run-у.
- ShallowNet (23,282 параметра) стабильно лучше остальных на SAM-40; значит,
  raw signal не «необучаем», а его spectral-variance inductive bias лучше
  соответствует данным. Он является правильной отправной точкой для
  hyperparameter diagnosis.
- Пять 5-секундных окон из одного 25-секундного SAM-40 run сильно
  перекрываются. Для обучения сети это даёт мало новой независимой информации;
  для frequency representation 10–12.5-секундные неперекрывающиеся окна могут
  оказаться информативнее, даже если их меньше.

### Следующая проверка — в таком порядке

1. На SAM-40 повторить `baseline_sam40_all` и `cross_subject_sam40_all` для
   ShallowNet и logistic regression с полной native montage; это отвечает на
   самый сильный технический кандидат без изменения labels.
2. На исходной 12-channel montage сравнить 5 s / 10 s / 12.5 s windows без
   overlap, агрегируя BA по subject, а не по окнам.
3. Сохранить train BA/loss и validation curves для ShallowNet, EEGNet и
   Conformer. Решение: высокий train при низком held-out — уменьшать
   capacity/epochs или добавлять early stopping; низкие оба — увеличивать
   capacity/epochs или менять representation.
4. Лишь после этих трёх проверок проводить компактный train-only search по
   learning rate, weight decay, epoch budget и ширине сети.

### 5. Quality-control ICA/preparation недостаточно наблюдаем — средняя уверенность

Конфиги согласованы между датасетами, Pipeline логи не содержат явного
падения preparation, а HTML snapshots сохранены. Но run не экспортирует
таблицу по recording unit с bad channels, числом ICA components, исключёнными
IC, residual line noise и долей отброшенных данных. Поэтому качество очистки
можно проверить по отчётам, но нельзя количественно связать с плохими
субъектами/направлениями transfer.

## Что делать в приоритетном порядке

1. **Не запускать сразу широкий hyperparameter sweep.** Зафиксировать текущий
   run как zero-shot benchmark и интерпретировать его как перенос
   protocol-defined contrast, а не универсального уровня внимания.
2. **Добавить диагностические артефакты в runner.** На каждый fold сохранять
   `fold_metrics.parquet` (train/test BA, AUC, class counts, число
   recording units и субъектов); для DL — per-epoch loss/accuracy/LR и seed.
   Это минимально ответит на вопрос underfit/overfit при следующем запуске.
3. **Сделать короткий QC/representation audit до смены моделей.** Для каждого
   recording unit экспортировать ICA/bad-channel/rejection summary, PSD и
   channel-wise robust scale после очистки; сопоставить их с subject-wise BA.
   Проверить одинаковость единиц, montage/channel order и residual 50-Hz noise
   между датасетами.
4. **Сделать две дешёвые абляции окон.** Повторить representative baseline,
   лучший cross-task и оба направления cross-dataset с (a) неперекрывающимися
   5-s окнами и (b) удалённым onset-переходом каждого блока/run. Сравнивать
   subject/run-level BA, не только pooled windows. Если baseline резко меняется,
   нынешний score существеннее связан с переходом/автокорреляцией, чем с
   устойчивым состоянием.
5. **Проверить допустимую фиксированную нормализацию сигнала.** До обучения
   сравнить raw current representation с per-window/channel RMS или robust
   normalization, одинаково заданной для source и target и не использующей
   target labels. Это тестирует амплитудно-reference shift без target-adaptive
   подбора. Per-dataset fit на target cohort отдельно считать adaptation и не
   смешивать с zero-shot результатом.
6. **Только после абляций настроить модели.** Для каждого семейства провести
   небольшой train-only nested search на source: LR (`C`, L1/L2), XGBoost
   (depth/regularisation), ShallowNet/EEGNet (LR, weight decay, capacity,
   early stopping). Выбирать рецепт на source validation/cross-task только;
   cross-dataset target не использовать для выбора.
7. **Если цель — повысить именно cross-dataset score, отделить новую задачу.**
   Понадобится явно заявленный harmonization/domain-adaptation эксперимент
   (например, unlabeled target normalization/alignment), отдельный от
   нынешнего zero-shot benchmark. Без нового контрбалансированного протокола
   он не устранит фундаментальную неоднозначность labels.

## Практическое решение

Для ближайшей итерации рекомендую не менять архитектуры, а сначала выполнить
пункты 2–5 для ShallowNet (лучший общий индикатор), logistic regression
(интерпретируемый контроль) и одного направления каждого protocol. Критерий
решения: если AUC cross-dataset остаётся около 0.5 после QC, non-overlap,
onset exclusion и фиксированной нормализации, причиной считать
непереносимость целевого контраста/доменов, а не hyperparameters. Если только
одна preparation-абляция одновременно поднимет AUC у обоих направлений,
переносить её в полный sweep и лишь затем тюнить модели.
