# Источники PDF в этой папке

Статьи, описывающие два датасета исследования, которые конвертируются в
`python_project/datasets/bids/`: **Dataset A** = `distinguishing` (mental
attention state, симулятор поезда) и **Dataset B** = `sam40` (SAM-40,
нейрокогнитивные тесты). Здесь только происхождение файлов — что из них
следует для конверсии, записано в README самих BIDS-датасетов.

Состав папки замкнут составом исследования: датасетов ровно два, поэтому и
строк здесь ровно две. Статья про датасет, который в состав не входит, в эту
папку не кладётся — если она нужна как background для Introduction/Discussion,
её место в `generalization_papers/`, если как источник конкретного утверждения
— в `citation_papers/`.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`. Оба датасета в
`manuscript.tex` процитированы, поэтому прочерков в этой папке нет.

| файл | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- |
| `attention_states_passive_bci.pdf` | Acı Ç.İ., Kaya M., Mishchenko Y. Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods. *Expert Systems with Applications* 134:153–166 (2019). doi:10.1016/j.eswa.2019.05.057 | опубликованная, журнал платный | `aci2019attention` |
| `sam40_stress_dataset.pdf` | Ghosh P. и др. SAM 40: Dataset of 40 subject EEG recordings to monitor the induced-stress while performing Stroop color-word test, arithmetic task, and mirror image recognition task. *Data in Brief* 40:107772 (2022). doi:10.1016/j.dib.2021.107772 | опубликованная, open access (Data in Brief) | `ghosh2022sam40` |
