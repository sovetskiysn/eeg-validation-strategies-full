---
paths:
  - "python_project/**"
---

# Ловушки specific to attention-EEG transfer

Group-aware CV, recording unit, границы стадий и leakage через preprocessing —
уже описаны точно для этого репозитория в `overview.md` и `datasets.md`.
Здесь — то, что не покрыто там: ловушки, специфичные для того, что метка
называется «attention», а перенос идёт между субъектами/сессиями/задачами/
датасетами.

## «Attention» — не единый конструкт

Sustained attention, selective attention, workload, engagement, fatigue,
mind wandering — разные вещи. Нельзя обсуждать их как взаимозаменяемые классы.
В этом проекте класс — контраст «высокий протокольный task demand» против
«отдых/отключение от задачи» (см. «Рамка исследования» в `CLAUDE.md`), а не
уровень внимания; формулировки в анализе и тексте должны это отражать.

## Confounds вместо внимания

Модель может учиться на движениях глаз/моргании, типе стимула, звуке/визуальном
контенте, порядке блоков, fatigue/time-on-task, motor response, session или
dataset identity — а не на нейронном коррелятe demand. Если все high-demand
trials используют один stimulus/task, а low-demand — другой, отделить эффект
от task identity одной моделью нельзя. Полезная проверка: попробовать
предсказать label только по `subject_id`/`session`/block index/времени без
EEG — если получается, drift или порядок блоков коррелируют с меткой.

## Cross-dataset merging

Dataset A и Dataset B расходятся в определении condition, парадигме, каналах,
reference, sampling rate, preprocessing и class balance. Простое объединение с
случайным CV отвечает на вопрос о mixed-domain classification, а не о
переносе — в этом проекте перенос всегда zero-shot: target полностью исключён
из подбора модели и порога (см. «Рамка исследования»).

## Сопоставление меток между датасетами

Нельзя автоматически отождествлять условия по созвучным названиям. Если
harmonization двух releases требует сильных предположений — лучше оставить
dataset-specific классы и явно сравнить transfer, чем придумать общий таргет,
который на самом деле у датасетов разный.
