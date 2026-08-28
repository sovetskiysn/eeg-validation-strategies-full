# Источники PDF в этой папке

**Предназначение папки.** Здесь лежат чужие работы, которые считали **на тех же
датасетах, что и мы**. Нужны они не как background для текста, а как рабочий
ориентир при построении своего pipeline: какой препроцессинг на этих записях
себя оправдал, какие каналы и полосы выбрасывают, какие модели уже пробовали и
с каким результатом. Часть решений можно взять готовыми со ссылкой, вместо того
чтобы заново перебирать варианты; заодно это те числа, против которых читаются
наши.

Состав определяется составом датасетов исследования: **Dataset A** =
`distinguishing` (EEG data for Mental Attention State Detection, симулятор
поезда) и **Dataset B** = `sam40` (SAM-40). Колонка `датасет` в таблице говорит,
на чём считала работа; A и B перемешаны в одной таблице намеренно — папка
собирается по назначению, а не по датасету.

Отсюда граница папки. Статья, **вводящая** датасет, лежит в `datasets_papers/`
(`attention_states_passive_bci.pdf`, `sam40_stress_dataset.pdf`) и сюда не
дублируется. Работа на другом датасете, взятая как материал для
Introduction/Discussion, — в `generalization_papers/`. Сюда попадает только то,
что запускается на нашем же датасете.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`. Прочерк значит, что
статья собрана как методический ориентир и в рукописи пока не процитирована.

| файл | датасет | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- | --- |
| `novel_ml_brain_attention_detection.pdf` | A | Wang J., Kim S.-K. Novel Machine Learning-Based Brain Attention Detection Systems. *Information* 16(1):25 (2025). doi:10.3390/info16010025 | опубликованная, open access (CC BY) | — |
| `situational_awareness_snn_eeg.pdf` | A | Hadad Y., Bensimon M., Ben-Shimol Y., Greenberg S. Situational Awareness Classification Based on EEG Signals and Spiking Neural Network. *Applied Sciences* 14(19):8911 (2024). doi:10.3390/app14198911 | опубликованная, open access (CC BY) | — |
| `students_attention_online_learning.pdf` | A | Al-Nafjan A., Aldayel M. Predict Students' Attention in Online Learning Using EEG Data. *Sustainability* 14(11):6553 (2022). doi:10.3390/su14116553 | опубликованная, open access (CC BY) | — |
| `time_frequency_analysis_snn.pdf` | A | Bensimon M., Hadad Y., Ben-Shimol Y., Greenberg S. Time–frequency analysis using spiking neural network. *Neuromorphic Computing and Engineering* 4(4):044001 (2024). doi:10.1088/2634-4386/ad80bc | опубликованная, open access (CC BY) | — |
| `stress_effect_personal_identification.pdf` | B | Abdel-Ghaffar E.A., Salama M. The Effect of Stress on a Personal Identification System Based on Electroencephalographic Signals. *Sensors* 24(13):4167 (2024). doi:10.3390/s24134167 | опубликованная, open access (CC BY) | — |
| `vggish_cnn_stress_cognitive_tasks.pdf` | B | Afify H.M., Mohammed K.K., Hassanien A.E. Stress detection based EEG under varying cognitive tasks using convolution neural network. *Neural Computing and Applications* 37(7):5381–5395 (2025). doi:10.1007/s00521-024-10737-7 | опубликованная, open access (CC BY) | — |
| `attention_multifeature_fusion_stress.pdf` | B | Ejaz S. и др. Attention-based multi-feature fusion neuromarker for EEG-driven stress classification in learners. *International Journal of Clinical and Health Psychology* 26(1):100678 (2026). doi:10.1016/j.ijchp.2026.100678 | опубликованная, open access (журнал полностью OA) | — |
| `dynamic_connectivity_tvdtf_stress.pdf` | B | Acharya S. и др. Rewiring Human Brain Networks via Lightweight Dynamic Connectivity Framework: An EEG-Based Stress Validation. arXiv:2511.05505 (2025) | **препринт**; опубликованная версия вышла под другим названием — Reconfiguring brain networks via lightweight dynamic connectivity framework. *Computers in Biology and Medicine* 213:111801 (2026), doi:10.1016/j.compbiomed.2026.111801 | — |
| `geometric_ml_stgcn_stress.pdf` | B | Koszut S., Nallaperuma-Herzberg S., Liò P. Decoding the Stressed Brain with Geometric Machine Learning. arXiv:2506.00587 (2025). doi:10.48550/arXiv.2506.00587 | **препринт** | — |
| `identity_trap_eeg_foundation_models.pdf` | B | Lin J.-Y., Wu Y.C., Jung T.-P. The Identity Trap in EEG Foundation Models: A Diagnostic Audit. arXiv:2606.06647 (2026). doi:10.48550/arXiv.2606.06647 | **препринт**; SAM-40 использован как внешняя когорта, не основной датасет работы | — |
| `stress_gpt_neurogpt_finetune.pdf` | B | Lloyd C. и др. Stress-GPT: Stress detection with an EEG-based foundation model. *Proc. 30th Annual Int. Conf. on Mobile Computing and Networking (MobiCom '24)*, 2341–2346 (2024). doi:10.1145/3636534.3698121 | опубликованная, ACM (не OA) | — |
| `eegnet_realtime_stress_thesis.pdf` | B | Tiraboschi G. EEGNet for Real-time EEG-Based Stress Analysis. Магистерская диссертация, Politecnico di Torino (2025). <https://webthesis.biblio.polito.it/36232/> | **не рецензированная публикация**, а выпускная работа — брать как источник методических деталей, не как ссылку в тексте | — |

Три статьи MDPI скачаны с CDN издателя (`mdpi-res.com`): прямые ссылки
`www.mdpi.com/.../pdf` закрыты бот-защитой Akamai и отдают «Access Denied».
`time_frequency_analysis_snn.pdf` сохранён пользователем вручную из браузера —
IOPscience целиком закрыт бот-защитой Radware (и landing page, и `/pdf`),
репозиторной копии нет ни в OpenAlex, ни в Unpaywall, ни в Wayback. Восемь
работ по SAM-40 собраны пользователем и получены готовым архивом; исходные
имена файлов в архиве были вида `Автор+год`, здесь они переименованы в
описательные слаги по конвенции `reference_papers/`.
