# Источники PDF в этой папке

Статьи, описывающие датасеты, которые конвертируются в `python_project/datasets/bids/`.
Здесь только происхождение файлов — что из них следует для конверсии, записано в
README самих BIDS-датасетов.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`, если статья
процитирована в `manuscript.tex`. Прочерк — статья нужна только коду
(`python_project/`), в тексте статьи не цитируется и записи в `refs.bib` нет.

| файл | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- |
| `gradcpt_eeg_fmri_dwi_dataset.pdf` | Cha Y. и др. Sustained attention task (gradCPT) dataset using simultaneous EEG-fMRI and DTI. *Scientific Data* 13:573 (2026). doi:10.1038/s41597-026-06616-6 | опубликованная, open access (nature.com) | — |
| `proactive_selective_attention_competition.pdf` | Aguado-López B. и др. Proactive selective attention across competition contexts. *Cortex* 176:113–128 (2024). doi:10.1016/j.cortex.2024.04.009 | принятая рукопись (postprint) из репозитория DIGIBUG Университета Гранады, CC-BY-NC-ND; сам журнал платный | — |
| `eeg_nirs_cognitive_tasks_dataset.pdf` | Shin J. и др. Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset. *Scientific Data* 5:180003 (2018). doi:10.1038/sdata.2018.3 | опубликованная, open access (nature.com) | — |
| `attention_states_passive_bci.pdf` | Acı Ç.İ., Kaya M., Mishchenko Y. Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods. *Expert Systems with Applications* 134:153–166 (2019). doi:10.1016/j.eswa.2019.05.057 | опубликованная, журнал платный | `aci2019attention` |
| `sam40_stress_dataset.pdf` | Ghosh P. и др. SAM 40: Dataset of 40 subject EEG recordings to monitor the induced-stress while performing Stroop color-word test, arithmetic task, and mirror image recognition task. *Data in Brief* 40:107772 (2022). doi:10.1016/j.dib.2021.107772 | опубликованная, open access (Data in Brief) | `ghosh2022sam40` |
| `AttentioNet Monitoring Student Attention Type in.pdf` | AttentioNet: Monitoring Student Attention Type in Learning with EEG-Based Measurement System. arXiv:2311.02924 (2023) | препринт, open access (arxiv.org) | `verma2023attentionet` |
| `sadt_cao2019_scidata.pdf` | Cao Z., Chuang C.-H., King J.-K., Lin C.-T. Multi-channel EEG recordings during a sustained-attention driving task. *Scientific Data* 6:19 (2019). doi:10.1038/s41597-019-0027-4 | опубликованная, open access (nature.com) | — |
| `mema_multilabel_attention.pdf` | A Multi-Label EEG Dataset for Mental Attention State Classification in Online Learning. arXiv:2411.09879 (2024); также *ICASSP 2025*, doi:10.1109/ICASSP49660.2025.10889126 | препринт, open access (arxiv.org) | — |
| `transformer_attentive_states.pdf` | Decoding Human Attentive States from Spatial-temporal EEG Patches Using Transformers. arXiv:2502.03736 (2025) | препринт, open access (arxiv.org) | — |

Первые три файла (`gradcpt_...`, `proactive_...`, `eeg_nirs_cognitive_...`) —
исходная тройка статей про BIDS-датасеты. Остальные семь перенесены сюда из
`latex_documents/reference_papers/` (одиночные PDF в корне) и из `for_future/papers/`
при реструктуризации `reference_papers/` в корень репозитория — это не все
статьи про сами датасеты BIDS, часть просто про смежные датасеты внимания
(SAM-40, MEMA, sustained-attention driving), релевантные для Introduction/Discussion.

Соответствие датасетам: gradCPT — OpenNeuro `ds006040`; selective attention —
OpenNeuro `ds005089`; cognitive tasks — релиз EEG-NIRS TU Berlin.

Словарь маркеров релиза TU Berlin сюда не копировался: он лежит в самом архиве
(`python_project/datasets/archive/cognitive_tasks_eeg/Dataset description_BrainVision and NIRx.pdf`),
и это документ релиза, а не публикация.
