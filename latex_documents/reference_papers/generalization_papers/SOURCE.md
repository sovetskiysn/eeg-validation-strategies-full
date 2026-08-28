# Источники PDF в этой папке

Статьи про cross-task / cross-subject / cross-session / cross-dataset
generalization в EEG-декодировании — обоснование новизны и материал для
Introduction/Discussion статьи (transferability нейромаркеров между задачами
и датасетами). Список и ссылки собраны пользователем в Notion, экспорт
которого лежал в корне репозитория (`e6de926c-...-ExportBlock-....zip`); эта
таблица переносит его в постоянную форму — сам zip после переноса можно
удалить.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`, если статья
процитирована в `manuscript.tex`. Прочерк — статья собрана как обоснование
новизны/background, но в текст пока не процитирована и записи в `refs.bib`
нет.

| файл | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- |
| `domain_adaptation_eeg_emotion_deap_seed.pdf` | Domain Adaptation Techniques for EEG-based Emotion Recognition: A Comparative Study on Two Public Datasets (DEAP, SEED) | опубликованная копия из репозитория NTU (dr.ntu.edu.sg), open access | `lan2019domain` |
| `hybrid_feature_learning_cross_session_attention.pdf` | Hybrid EEG Feature Learning Method for Cross-Session Human Mental Attention State Classification. *Brain Sciences* 15(8):805 (2025). doi:10.3390/brainsci15080805 | опубликованная, CC-BY, open access; скачана пользователем вручную из браузера (mdpi.com блокирует автоматические запросы с этого сервера) | `chen2025hybrid` |
| `cross_task_consistency_psd_vs_erp_workload.pdf` | Cross-Task Consistency of Electroencephalography-Based Mental Workload Indicators: Comparisons Between Power Spectral Density and Task-Irrelevant Auditory Event-Related Potentials. *Frontiers in Neuroscience* (2021). doi:10.3389/fnins.2021.703139 | опубликованная, open access (frontiersin.org) | `ke2021crosstask` |
| `evaluating_cognitive_tasks_transfer_learning.pdf` | Evaluating the Structure of Cognitive Tasks with Transfer Learning. arXiv:2308.02408 (2023). doi:10.48550/arXiv.2308.02408 | авторский препринт, open access; скачан с arxiv.org | `aristimunha2023tasks` |
| `eeg_foundation_challenge_cross_task_subject.pdf` | EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding. arXiv:2506.19141 | препринт, open access (arxiv.org) | `aristimunha2025foundation` |
| `single_channel_attention_kalman_elm.pdf` | Single-channel attention classification algorithm based on robust Kalman filtering and norm-constrained ELM. *Frontiers in Human Neuroscience* (2024). doi:10.3389/fnhum.2024.1481493 | опубликованная, open access (frontiersin.org) | `he2025kalman` |
| `deep_metric_learning_adversarial_training.pdf` | Cross-Subject EEG-Based Emotion Recognition Using Deep Metric Learning and Adversarial Training. *IEEE Access* (2024). doi:10.1109/ACCESS.2024.3458833 | опубликованная, CC-BY, open access (IEEE Access — журнал полностью OA); скачана пользователем вручную из браузера (ieeexplore.ieee.org блокирует автоматические запросы с этого сервера) | — |
| `id3rsnet_cross_subject_drowsiness.pdf` | ID3RSNet: cross-subject driver drowsiness detection from raw single-channel EEG with an interpretable residual shrinkage network. *Frontiers in Neuroscience* (2024). doi:10.3389/fnins.2024.1508747 | опубликованная, open access (frontiersin.org) | — |
| `workload_estimation_across_affective_contexts.pdf` | EEG-based workload estimation across affective contexts. *Frontiers in Neuroscience* (2014). doi:10.3389/fnins.2014.00114 | опубликованная, open access (frontiersin.org) | `muhl2014workload` |
| `cross_dataset_variability_deep_learning.pdf` | Cross-Dataset Variability Problem in EEG Decoding With Deep Learning. *Frontiers in Human Neuroscience* (2020). doi:10.3389/fnhum.2020.00103 | опубликованная, open access (frontiersin.org) | — |
| `signal_alignment_cross_datasets_p300.pdf` | Signal alignment for cross-datasets in P300 brain--computer interfaces. *Journal of Neural Engineering* 21(3):036007 (2024). doi:10.1088/1741-2552/ad430d | опубликованная версия PDF с iopscience.iop.org; доступна через страницу издателя | — |
| `toward_cross_subject_cross_session_generalization.pdf` | Toward cross-subject and cross-session generalization in EEG-based emotion recognition: Systematic review, taxonomy, and methods. *Neurocomputing* 610:128354 (2024). doi:10.1016/j.neucom.2024.128354 | опубликованная, hybrid open access (автор оплатил OA); скачана пользователем вручную из браузера (sciencedirect.com отдаёт статью только через JS-редирект с сессионными куками, curl не проходит) | — |
| `generalized_dl_vigilance_decrement_cross_task.pdf` | Generalized Deep Learning EEG Models for Cross-Participant and Cross-Task Detection of the Vigilance Decrement in Sustained Attention Tasks. *Sensors* 21(16):5617 (2021). doi:10.3390/s21165617 | опубликованная, CC-BY, open access; скачана пользователем вручную из браузера (mdpi.com блокирует автоматические запросы с этого сервера) | `kamrud2021vigilance` |
| `cross_task_workload_domain_adaptation.pdf` | Cross-Task Cognitive Workload Recognition Based on EEG and Domain Adaptation. *IEEE Transactions on Neural Systems and Rehabilitation Engineering* (2022). doi:10.1109/TNSRE.2022.3140456 | CC-BY, open access, скачано напрямую с ieeexplore.ieee.org (ielx7-путь не за WAF-челленджем) | `zhou2022crosstask` |
| `foundation_models_cross_domain_survey.pdf` | Li H. и др. Foundation models for cross-domain EEG analysis application: A survey. arXiv:2508.15716 (2025). doi:10.48550/arXiv.2508.15716 | препринт, open access (arxiv.org) | `li2025foundation` |
| `attentionet_student_attention_monitoring.pdf` | Verma S. и др. AttentioNet: Monitoring Student Attention Type in Learning with EEG-Based Measurement System. arXiv:2311.02924 (2023). doi:10.48550/arXiv.2311.02924 | препринт, open access (arxiv.org); лежал в `datasets_papers/` под именем `AttentioNet Monitoring Student Attention Type in.pdf`, перенесён сюда вместе с переименованием в слаг — датасет этой работы в состав исследования не входит, а в тексте она стоит в Introduction рядом с `li2025foundation` | `verma2023attentionet` |
| `transfer_learning_eeg_bci_review_wu.pdf` | Wu D., Xu Y., Lu B.-L. Transfer learning for EEG-based brain–computer interfaces: A review of progress made since 2016. *IEEE Transactions on Cognitive and Developmental Systems* (2022). doi:10.1109/TCDS.2020.3007453 | препринт с arXiv (2004.06286), open access | `wu2022transfer` |
| `transfer_learning_eeg_bci_review_zhang.pdf` | Zhang K. и др. Application of transfer learning in EEG decoding based on brain–computer interfaces: A review. *Sensors* 20(21):6321 (2020). doi:10.3390/s20216321 | опубликованная, CC-BY, open access; скачана пользователем вручную из браузера (mdpi.com блокирует автоматические запросы с этого сервера). Название почти совпадает с `transfer_learning_eeg_bci_review_wu.pdf` (другая статья, другие авторы) — суффикс `_wu`/`_zhang` разводит их | `zhang2020transfer` |
| `negative_transfer_survey.pdf` | A Survey on Negative Transfer. *IEEE/CAA Journal of Automatica Sinica* 10(2):305--329 (2023). doi:10.1109/JAS.2022.106004 | авторский препринт arXiv:2009.00909, open access; скачан с arxiv.org | `zhang2023negative` |

## Не найдено в исходном списке пользователя, но упомянуто в описаниях как related work

Заметки пользователя к статьям про cross-dataset variability и cross-task
consistency (`cross_dataset_variability_deep_learning.pdf`,
`cross_task_consistency_psd_vs_erp_workload.pdf`) ссылаются на дополнительные
методы/статьи (Riemannian Procrustes Analysis, Jayaram & Barachant 2018 и
т.д.) — эти работы не входили в список на скачивание, при необходимости
искать отдельно по цитатам внутри самих PDF.
