# XGBoost: ребалансировка внутри эстиматора, а не шагом пайплайна

Разбор по запросу «сделать так, чтобы cross-validation работал с XGBoost и
ребалансировка классов жила внутри самого XGBoost». Всё проверено на
установленных версиях: `xgboost 3.2.0`, `scikit-learn 1.9.0`,
`imbalanced-learn 0.14.2`.

Это ревизия вывода из `TRAIN_RESAMPLING_TO_CLASS_WEIGHTS.md`, где XGBoost
единственный из пяти декодеров остался с `RandomOverSampler`. Тот вывод
опирался на перечисление трёх вариантов; **четвёртый вариант существует, и он
работает** — тонкий подкласс `XGBClassifier`.

---

## Что XGBoost на самом деле предоставляет (проверено, не по памяти)

`XGBClassifier().get_params()` → 40 ключей, `class_weight` среди них **нет**.
По дисбалансу есть ровно два механизма:

| механизм | где | ограничение |
| --- | --- | --- |
| `scale_pos_weight` | параметр конструктора | один скаляр, **только бинарная** задача; не может следовать за составом 1:1 → 2:1 → 3:1 |
| `sample_weight` | аргумент `fit`, per-sample | `imblearn.Pipeline` не роутит fit-параметры через `fit_resample` |

Официальный tutorial по тюнингу подтверждает первый как штатный:
«you can balance positive and negative weights using `scale_pos_weight`» — и
там же оговорка, что при потребности в калиброванных вероятностях
ребалансировать не следует вовсе, а вместо этого ставить конечный
`max_delta_step`.

## Ключевой факт, который снимает тупик

`sample_weight` — не «другой» механизм по сравнению с `class_weight`. В
sklearn это буквально одна формула:

```
compute_sample_weight("balanced", y)  ==  compute_class_weight("balanced", ...)[y]
```

Проверено: при y = 30×class0 / 10×class1 веса равны `0.667` и `2.0`, суммы по
классам совпадают (20.0 против 20.0). То есть `class_weight="balanced"` у
логрега — это и есть per-sample вектор, просто развёрнутый sklearn'ом внутри.

Структурная проблема была не в самой формуле, а в том, что вектор нужно было
**провезти снаружи** через `imblearn.Pipeline`. Но его не нужно везти снаружи:
он целиком выводится из `y`, который эстиматор и так получает в `fit`.

## Решение: подкласс на 15 строк

```python
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that reads `class_weight` off the fold it is fitted on."""

    def __init__(self, class_weight=None, **kwargs):
        self.class_weight = class_weight
        super().__init__(**kwargs)

    def get_xgb_params(self):
        # get_params() feeds the booster, which warns about keys it does not
        # know. class_weight is consumed here in Python, so drop it.
        params = super().get_xgb_params()
        params.pop("class_weight", None)
        return params

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        if self.class_weight is not None and sample_weight is None:
            sample_weight = compute_sample_weight(self.class_weight, y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)
```

Три детали, каждая нужна:

1. **`get_xgb_params`.** Без него первая же версия ловила
   `UserWarning: Parameters { "class_weight" } are not used` — XGBoost отдаёт
   весь `get_params()` в бустер, и незнакомый ключ уезжает в C++. Это был
   реальный дефект, а не косметика.
2. **`sample_weight is None`.** Явно переданный вес пользователя имеет
   приоритет; автоматика не затирает ручной ввод.
3. **Атрибут `class_weight` под тем же именем, что параметр `__init__`** —
   контракт sklearn для `get_params`/`clone`.

Это ровно тот же приём, что уже принят в проекте для нейронок:
`pipeline.BalancedClassWeight` читает `y` фолда в `on_train_begin` и ставит
`criterion_.weight`. Одно правило на все шесть декодеров, а не три разных.

## Что проверено прогоном

| проверка | результат |
| --- | --- |
| `get_params` содержит `class_weight`, всего 41 ключ | ✔ |
| `clone()` и `set_params()` переживают round-trip | ✔ |
| предсказания **идентичны** явному `fit(sample_weight=...)` | ✔ `allclose` |
| предсказания **отличаются** от невзвешенного фита | ✔ (вес не молчит) |
| ни одного warning от бустера (`simplefilter("error")`) | ✔ |
| работает внутри `imblearn.Pipeline` + `cross_validate` + `GroupKFold` | ✔ |
| **вес следует за фолдом** | ✔ см. ниже |
| мультикласс, где `scale_pos_weight` не работает в принципе | ✔ |

Про-фолдовые веса на синтетике с группами 1:3 / 1:1 / 3:1:

```
[{0: 1.333, 1: 0.8}, {0: 1.0, 1: 1.0}, {0: 0.8, 1: 1.333}]
```

Это те же самые числа, что зафиксированы для `BalancedClassWeight` у трёх
нейронок в `TRAIN_RESAMPLING_TO_CLASS_WEIGHTS.md` — то есть XGBoost получает
буквально ту же ребалансировку, что и остальные декодеры.

Сравнение на 600 окнах при дисбалансе 3:1, `StratifiedKFold(5)`:

```
RandomOverSampler       balanced_acc = 0.7427
class_weight balanced   balanced_acc = 0.7443
no rebalancing          balanced_acc = 0.7213
```

Разница между первыми двумя — шум, и так и должно быть: для бустинга
дублирование строки и вес 2 неразличимы, гистограмма считает веса как
счётчики. Ценность не в метрике.

## Почему тогда менять, если метрика та же

Три причины, все про корректность и однородность, а не про accuracy:

1. **Шаг `2_train_resampling` исчезает целиком.** Сейчас `classical_oversampling.yaml`
   существует ради одного декодера из пяти; после изменения обе ветки —
   `none.yaml` и XGBoost — идут через `passthrough`, и ось конфига можно
   удалить.
2. **Мультикласс перестаёт быть тупиком.** `scale_pos_weight` бинарный;
   если состав когда-нибудь станет трёхклассовым, `compute_sample_weight`
   уже работает (проверено), а `scale_pos_weight` пришлось бы выбрасывать.
3. **Одно правило на шесть декодеров.** Сейчас в Methods приходится писать
   «четыре взвешены, один пересэмплен, но это эквивалентно». После — одна
   фраза про `balanced` на всех.

Против: подкласс — это своя абстракция в `src/`, а правило проекта требует
принимать каждую отдельным решением. Здесь критерий выполняется тем же
аргументом, что и для уже принятого `BalancedClassWeight`: библиотека не даёт
точки расширения, а число нельзя записать в рецепт, потому что дисбаланс —
свойство состава (1.02:1 на `distinguishing`, до 3.02:1 на полном SAM-40).

## Чего это изменение НЕ чинит

Дубликаты у XGBoost и так были безвредны: внутреннего валидационного сплита,
через который они могли бы протечь, у него нет — в отличие от нейронок с
`ValidSplit`. Так что leakage здесь не было и чинить нечего. Мотив —
однородность и снятая ось конфига.

Отдельно остаётся не закрытым то, что и раньше: `early_stopping_rounds`/
`eval_set` у XGBoost так и не подключены (см. комментарий в
`configs/pipeline/3_estimator/xgboost.yaml`), бустинг всегда идёт полные 100
раундов без проверки сходимости. Взвешивание этого не касается.

## Что сделано

Изменение применено.

1. `python_project/src/pipeline.py` — `BalancedXGBClassifier` рядом с
   `BalancedClassWeight`.
2. `configs/pipeline/3_estimator/xgboost.yaml` — `_target_:
   pipeline.BalancedXGBClassifier` + `class_weight: balanced`.
3. `configs/pipeline/2_train_resampling/classical_oversampling.yaml` — удалён.
4. `experiments/decoder/xgboost.yaml` — переведён на `none`.

Проверки (реальных обучений не запускалось, GPU нет; нейронки не трогались):

- **Композиция** всех пяти декодеров через
  `run_experiment.py --config-dir experiments --config-name scenario_decoder
  +decoder=<...> +scenario=baseline_sam40 -c job` — собирается, у всех пяти
  `train_resampling: passthrough`, у xgboost `_target_:
  pipeline.BalancedXGBClassifier` и `class_weight: balanced`.
- **Инстанцирование из настоящего рецепта** `configs/pipeline/3_estimator/xgboost.yaml`
  через `instantiate` → `BalancedXGBClassifier`, `device: cpu` на месте.
- **`clone()` внутри `cross_validate`** сохраняет `class_weight`.
- **Про-фолдовые веса** на синтетике (360×10 float, CPU, группы 1:3 / 1:1 / 3:1):
  `[{0: 1.333, 1: 0.8}, {0: 1.0, 1: 1.0}, {0: 0.8, 1: 1.333}]` — вес следует за
  фолдом, а не записан в рецепт. Те же числа, что у `BalancedClassWeight`.
- **Ни одного warning от бустера** при `warnings.simplefilter("error")` —
  `get_xgb_params` действительно снимает `class_weight` с пути в C++.
- `grep` по `configs/`, `src/`, `experiments/`, `scripts/`, `Makefile` на
  `classical_oversampling` — пусто. В `results/` старые снапшоты конфигов
  остаются: это история прогонов.

Кэш Stage 1 не инвалидируется: граница хеша проходит по `preparation`, модель
в адрес артефакта не входит.

## Ось `2_train_resampling` удалена целиком

После перевода XGBoost на веса в оси остался единственный вариант
`none.yaml` (`step: passthrough`), на который ссылались все пять декодеров, —
то есть ось перестала что-либо выбирать. Удалено:

- `configs/pipeline/2_train_resampling/` — папка целиком (обе оставшиеся
  ссылки на неё были на `none`);
- строка группы и `${pipeline_components.train_resampling.step}` из
  `configs/pipeline/default.yaml` — в `make_pipeline` теперь два позиционных
  шага вместо трёх;
- строка `override /pipeline/2_train_resampling@...` из всех пяти
  `experiments/decoder/*.yaml`.

Проверено, что ключ никем не читался: в `src/` и `scripts/` из
`pipeline_components` используется только `model.name`
(`analysis.py`, `run_experiment.py`). Композиция всех пяти декодеров после
удаления проходит, `train_resampling` в собранном конфиге больше не
встречается.

Нумерация групп осталась `1_input_scaling` / `3_estimator` с пропуском.
Переименовывать `3_` в `2_` я не стал: имена групп попадают в
`hydra.job.override_dirname`, то есть в имена папок под `results/`, и
переименование разошлось бы с уже сохранённой историей прогонов.

### Побочно: устаревший комментарий в рецепте логрега

`configs/pipeline/3_estimator/logistic_regression.yaml` объяснял отказ от
`LogisticRegressionCV` тем, что её внутреннему сплиту нужен `groups`, который
не проехать мимо `RandomOverSampler`. **Этот блокер исчез** — ресэмплящих
шагов в пайплайне не осталось вовсе, и роутинг `groups` теперь упирался бы
только в `enable_metadata_routing`. Комментарий переписан: `C: 1.0` — это
осознанно не-тюненный параметр, а не вынужденный. Если тюнинг C когда-нибудь
понадобится, дорога открыта, но inner split должен быть group-aware.

## Sanity-прогон до полного sweep'а

xgboost и логрег на `baseline_sam40` (дисбаланс 3:1) — это как раз то, что
запускается без GPU. Ожидание стоит держать трезвым: на синтетике при 3:1
расхождение с `RandomOverSampler` было 0.7427 против 0.7443, то есть в
пределах шума.
