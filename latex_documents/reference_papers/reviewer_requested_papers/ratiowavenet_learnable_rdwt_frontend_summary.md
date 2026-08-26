# Siino et al. (2025): RatioWaveNet для устойчивого MI-EEG

## Идентификация

- **Полное название прочитанной версии:** *RatioWaveNet: A Learnable RDWT
  Front-End for Robust and Interpretable EEG Motor-Imagery Classification*.
- **Авторы:** Marco Siino, Giuseppe Bonomo, Rosario Sorbello, Ilenia
  Tinnirello.
- **Версия, прочитанная здесь:** локальный
  `ratiowavenet_learnable_rdwt_frontend.pdf`, препринт
  [arXiv:2510.21841](https://arxiv.org/abs/2510.21841), v1 от 22 октября 2025.
  Это именно работа, названная Reviewer #3. Не смешивать её с более поздней
  журнальной статьёй авторов с похожим, но другим названием (см. `SOURCE.md`).
- **Предмет:** классификация motor imagery, не внимание и не cross-task
  generalization между когнитивными парадигмами.

## Коротко

RatioWaveNet — гибридная end-to-end архитектура: обучаемый RDWT front-end перед
TCFormer-подобной цепочкой «multi-kernel CNN → grouped-query Transformer →
causal TCN». Авторы специально измеряют performance у худшего субъекта в каждом
seed, а не только среднюю accuracy. Перед TCFormer улучшения над худшим
субъектом есть, но на BCI-IV-2a они очень малы (+0.17 и +0.42 п.п.); более
заметны на BCI-IV-2b (+1.07 и +2.54 п.п.).

## Knowledge gap и цель

**Пробел, сформулированный авторами.** Даже сильные CNN/Transformer MI-decoder'ы
уязвимы к низкому SNR, нестационарности и межсубъектной вариативности. Обычная
фиксированная preprocessing-схема не даёт адаптивной, сдвиго-инвариантной
многомасштабной декомпозиции, согласованной с downstream model. Кроме того,
средняя метрика может скрывать неудачу именно на участниках, для которых BCI
наиболее ненадёжен.

**Цель и вопрос.** Проверить, улучшает ли обучаемый RDWT front-end robustness
сильного Transformer backbone'а у самого сложного субъекта и сохраняется ли
выигрыш между случайными инициализациями в intra-subject и LOSO режимах.

## Данные и экспериментальный дизайн

- BCI Competition IV-2a: 9 здоровых участников, по две сессии в разные дни,
  4 класса (левая рука, правая рука, обе стопы, язык), 288 trial на сессию.
- BCI Competition IV-2b: 9 участников, 5 сессий, бинарный left/right hand MI,
  160 trial на сессию; в части сессий есть online feedback.
- Сравниваются RatioWaveNet, TCFormer, EEGNet и ShallowConvNet. Baseline-модели,
  по словам авторов, используют исходные hyperparameters; pipeline заявлен
  одинаковым для сравниваемых моделей.
- Два протокола: **Sub-Dependent** и leave-one-subject-out (**LOSO**). Пять
  random seed. Основной robustness endpoint: в каждом seed и протоколе выбирают
  субъекта с наименьшей тестовой accuracy, затем усредняют эти худшие значения
  по пяти seed. Дополнительно заявлены average-case results, но центральные
  таблицы посвящены worst-subject accuracy.
- Training: PyTorch/CUDA, Adam (lr 0.001), batch size 64, до 1000 эпох с early
  stopping после 100 эпох без улучшения. Метрики — micro-accuracy и Cohen's
  kappa.

## Архитектура и методология

1. **Обучаемый четырёхуровневый RDWT.** Недециимируемая декомпозиция сохраняет
   длину временной оси и заявленную shift-invariance. Рациональные масштабы
   инициализированы около 1.5, 5/3, 7/4 и 9/5, затем обучаются; фильтры
   Daubechies-4 слегка адаптируются. В detail bands есть обучаемый soft
   thresholding, gain и level dropout.
2. **Гибридный вход.** Модель может через обучаемые веса смешивать raw stream и
   восстановленный RDWT stream, поэтому выгода не обязана происходить от одного
   лишь удаления шума.
3. **Feature extractor.** Multi-kernel CNN извлекает локальные
   temporal--spatial признаки; далее идут Transformer с grouped-query attention
   и RoPE для long-range context, а также TCN с causal dilated convolutions для
   временной интеграции.
4. **Интерпретируемость в заявлении авторов.** Она относится прежде всего к
   сохранению scale identity, волновым subband/scalogram representations и
   обучаемым параметрам front-end. В статье нет отдельной количественной
   валидации того, что эти представления являются нейрофизиологически верными
   объяснениями решения классификатора.

## Основные результаты

Worst-subject accuracy RatioWaveNet против TCFormer (среднее пяти seed):

| Dataset | Sub-Dependent | LOSO |
| --- | ---: | ---: |
| BCI-IV-2a | 73.43% vs 73.26% (**+0.17 п.п.**) | 40.35% vs 39.93% (**+0.42 п.п.**) |
| BCI-IV-2b | 70.57% vs 69.50% (**+1.07 п.п.**) | 65.98% vs 63.44% (**+2.54 п.п.**) |

Авторы также сообщают среднюю latency forward pass около 4.53 ms/trial на RTX
6000 Ada и трактуют overhead как умеренный. Значение «worst subject» может
относиться к разным участникам в разных seed: это summary неблагоприятного
случая, а не подтверждённая оценка одного фиксированного участника.

## Что из статьи следует — и чего нет

**Поддержано данными:** на двух MI benchmark-ах в описанном эксперименте
RDWT-enhanced hybrid model превзошла TCFormer по выбранной worst-subject
accuracy; эффект существенно больше на BCI-IV-2b, чем на 2a. Работа показывает
пример deep architecture и LOSO-оценки, адресующий межсубъектную вариативность.

**Не поддержано напрямую:** что она найдёт task-invariant маркер, что внимание
можно декодировать так же, что RDWT полезен на нашем 12-channel cross-dataset
протоколе, или что результаты представляют general state of the art для всех
EEG задач. Она не тестирует attention labels, sleep, cross-task transfer между
разными cognitive paradigms, cross-dataset transfer либо сравнение с нашим
набором шести декодеров.

## Критическая оценка и ограничения

- Это препринт в версии, которую потребовал Reviewer #3; к нему следует
  относиться как к непрошедшему peer review источнику, пока не будет выбран
  окончательный журнальный вариант для bibliography.
- Всего два небольших MI набора (по 9 субъектов). Это не тест обобщения на
  другие когнитивные конструкты или recording setups.
- Прирост над сильным TCFormer на BCI-IV-2a очень мал. В тексте нет
  доверительных интервалов, p-values или заранее заданного статистического
  теста именно для этих model differences, поэтому слово «reliable» сильнее,
  чем предоставленная статистическая опора.
- Worst-subject выбирается заново для каждого seed. Такой adaptive selection
  полезен как stress-test, но он не заменяет distribution subject-wise scores,
  фиксированного hold-out subject или paired statistical analysis.
- Для воспроизводимости указаны optimizer, batch size, early stopping и
  hardware, но одной этой информации недостаточно, чтобы независимо
  воспроизвести все preprocessing и split decisions без кода/полной
  конфигурации.

## Как корректно использовать в нашей рукописи

Безопасно цитировать как свежий **MI-EEG** пример hybrid CNN--Transformer--TCN
архитектуры с адаптивным wavelet front-end и LOSO stress-test. Корректное
ограничение рядом с цитатой: «Работа проверяла MI benchmarks; перенос её
архитектуры и её выигрыша на protocol-defined contrasts нашего исследования
нельзя предполагать без прямого эксперимента». Так ссылка расширяет методический
фон, но не подменяет evidence по attention decoding или по нашим validation
scenarios.
