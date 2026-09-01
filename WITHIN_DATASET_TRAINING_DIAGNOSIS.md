# Почему низки baseline, cross-subject и cross-session

## Короткий вывод

В текущем run-е нет признака, что слабые within-dataset метрики вызваны
коллапсом generalization или одной неисправной моделью. Низкая различимость
уже присутствует в baseline, а held-out subject/session почти не ухудшает её.
Наиболее вероятный технический потолок для SAM-40 — использование только
12-канальной common montage вместо нативной. Для сетей остаётся реальная, но
пока недоказанная, проблема training recipe: 100 фиксированных эпох без
validation/early stopping и без сохранённых learning curves.

## Доказательства из этого run-а

| Сценарий | Logistic | XGBoost | EEGNet | ShallowNet | Conformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distinguishing baseline | .641 | .670 | .663 | .660 | .647 |
| Distinguishing cross-subject | .633 | .658 | .655 | .668 | .648 |
| Distinguishing cross-session | .629 | .652 | .635 | .661 | .659 |
| SAM-40 all baseline | .683 | .680 | .671 | .720 | .699 |
| SAM-40 all cross-subject | .662 | .652 | .667 | .665 | .707 |

Это pooled balanced accuracy, пересчитанная из сохранённых test predictions.

- Для `distinguishing` разница baseline → cross-subject/session составляет
  считанные сотые. Значит, модели не выучили только subject/session identity;
  исходная EEG-различимость этого контраста сама по себе умеренна.
- Для SAM-40 all baseline → cross-subject также нет общего резкого падения.
  Следовательно, новая персона не является главным объяснением BA около .7.
- ShallowNet стабильно сильнее остальных на SAM-40 (до .782 на Stroop), то есть
  в prepared data есть обучаемый сигнал. Он лучше соответствует spectral
  variance / band-power структуре EEG, чем остальные fixed recipes.
- Отдельные session-folds `distinguishing` очень нестабильны: ShallowNet
  0.346–0.883 при pooled BA .661. Один fold — короткая одна session одного
  человека, поэтому pooled metric не означает одинаковое качество на каждой
  сессии.

## Что, вероятнее всего, ограничивает score

1. **Общий 12-channel вход для SAM-40.** Для честного cross-dataset
   representation оставлены `F3/F4/F7/F8/FC5/FC6/O1/O2/P7/P8/T7/T8`, но
   отброшены 20 нативных SAM-40 каналов, включая `Cz`, `Fz`, `C3/C4`,
   `P3/P4`, `Pz`, `Oz`. Это не ошибка preparation, а потенциально дорогой
   компромисс: within-dataset decoder теряет центрально-париетальную
   пространственную информацию.
2. **Мало независимых тренировочных единиц.** У SAM-40 пять 5-s окон
   с шагом 4.5 s получены из одного 25-s run-а. Окна полезны для optimisation,
   но почти не добавляют независимой EEG-информации. У `distinguishing` всего
   24 recording units и 5 subjects.
3. **5-s representation может быть слишком коротким.** Для band-power и
   особенно для устойчивого состояния 10–12.5-s windows могут дать более
   стабильный spectral estimate; это нужно проверять, а не предполагать.
4. **Training recipe сетей не диагностируем.** Все DL модели получают ровно
   100 эпох, `train_split: null`, без early stopping. В результатах нет ни
   train BA, ни loss/history, поэтому exact underfit/overfit не восстановить.

Корректность within-dataset labels не отменяет эти факторы: валидная
protocol-defined метка может иметь только умеренный instantaneous EEG effect
в пятиминутном/пятисекундном окне, а не быть легко separable на уровне .9 BA.

## Вердикт по моделям

| Модель | Статус по имеющимся данным |
| --- | --- |
| Logistic regression | Хороший контроль; нет признака, что именно ей не хватает capacity. Текущий handcrafted representation имеет ограниченный потолок. |
| XGBoost | Нет устойчивого выигрыша над linear model, поэтому нелинейность feature-space не является главным отсутствующим элементом. |
| EEGNet, 1,938 parameters | Наиболее вероятный кандидат на underfit или mismatch architecture/data; это гипотеза до появления train curves. |
| ShallowNet, 23,282 parameters | Лучший текущий decoder; не показывает симптома тяжёлого overfit, потому что удерживает результаты на held-out subjects. |
| EEGConformer, 170,466 parameters | Риск overfit или неудачной optimisation высок из-за capacity и отсутствия early stopping, но cross-subject .707 на SAM-40 не позволяет объявить его переобученным без train history. |

## Минимальный причинный эксперимент

Выполнять по порядку, не меняя labels и не смешивая результат с cross-dataset
benchmark.

1. Повторить SAM-40 all baseline и cross-subject с полной native montage для
   ShallowNet и logistic regression. Рост subject-wise BA у обеих моделей
   подтвердит, что common-channel выбор был главным потолком.
2. На исходных 12 каналах проверить 5 s, 10 s и 12.5 s неперекрывающиеся
   окна. Сравнить subject-wise BA/AUC; это определит, ограничивает ли короткое
   окно representation.
3. Сохранить на каждом fold train BA/AUC, test BA/AUC, число subjects/recording
   units и, для сетей, loss/accuracy/LR по эпохам.
4. Только затем менять learning rate, weight decay, число эпох, EEGNet
   capacity и Conformer capacity. Решение однозначно: большой train–test gap
   означает overfit; низкие оба score — underfit или слабое input
   representation.
