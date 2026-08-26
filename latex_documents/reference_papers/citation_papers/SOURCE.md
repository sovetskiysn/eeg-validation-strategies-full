# Источники PDF в этой папке

Статьи, которые не про датасеты (`datasets_papers/`) и не про обоснование
новизны через cross-task/subject/session/dataset generalization
(`generalization_papers/`), а просто нужны как источник конкретного
утверждения в тексте манускрипта (`latex_documents/revision/manuscript_clean/manuscript.tex`) —
психологическая/нейрофизиологическая база (Stroop, mental rotation,
sustained attention и т.п.), обзоры по теме внимания. Список собран сверкой
`refs.bib` против `\cite` в manuscript.tex.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`. В этой папке у всех
файлов ключ есть (иначе файл был бы не нужен здесь по определению папки).

| файл | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- |
| `stroop_color_word_test_review.pdf` | Scarpina F., Tagini S. The Stroop color and word test. *Frontiers in Psychology* 8:557 (2017). doi:10.3389/fpsyg.2017.00557 | опубликованная, open access (frontiersin.org) | `scarpina2017stroop` |
| `mental_rotation_network_meta_analysis.pdf` | Tomasino B., Gremese M. Effects of stimulus type and strategy on mental rotation network: An activation likelihood estimation meta-analysis. *Frontiers in Human Neuroscience* 9:693 (2016). doi:10.3389/fnhum.2015.00693 | опубликованная, open access (frontiersin.org) | `tomasino2016mental` |
| `stroop_effect_half_century_review.pdf` | MacLeod C.M. Half a century of research on the Stroop effect: An integrative review. *Psychological Bulletin* 109(2):163–203 (1991). doi:10.1037/0033-2909.109.2.163 | опубликованная копия, скачана пользователем вручную из браузера (APA PsycNet, платный доступ) | `macleod1991stroop` |
| `mental_rotation_three_dimensional_objects.pdf` | Shepard R.N., Metzler J. Mental rotation of three-dimensional objects. *Science* 171(3972):701–703 (1971). doi:10.1126/science.171.3972.701 | опубликованная копия, скачана пользователем вручную из браузера (science.org, платный доступ). Скан старой Science-страницы, текст-экстракция первой страницы PDF цепляет хвост соседней заметки того же разворота (двухколоночная вёрстка) — сам PDF корректный, проверено по содержанию эксперимента (1600 пар perspective line drawings, congruent/mirror, рычаг) | `shepard1971mental` |
| `no_one_knows_what_attention_is.pdf` | Hommel B. и др. No one knows what attention is. *Attention, Perception, & Psychophysics* 81(7):2288–2303 (2019). doi:10.3758/s13414-019-01846-w | опубликованная копия, скачана пользователем вручную из браузера (Springer, платный доступ) | `hommel2019noone` |
| `visual_working_memory_mental_rotation_substrate.pdf` | Hyun J.-S., Luck S.J. Visual working memory as the substrate for mental rotation. *Psychonomic Bulletin & Review* 14(1):154–158 (2007). doi:10.3758/BF03194043 | опубликованная копия, скачана пользователем вручную из браузера (Springer, платный доступ) | `hyun2007visual` |
| `mathematical_competence_parietal_activation.pdf` | Grabner R.H. и др. Individual differences in mathematical competence predict parietal brain activation during mental calculation. *NeuroImage* 38(2):346–356 (2007). doi:10.1016/j.neuroimage.2007.07.041 | опубликованная копия, скачана пользователем вручную из браузера (sciencedirect.com, платный доступ + Cloudflare-челлендж на автоматические запросы) | `grabner2007math` |
| `in_the_zone_or_zoning_out_sustained_attention.pdf` | Esterman M. и др. In the zone or zoning out? Tracking behavioral and neural fluctuations during sustained attention. *Cerebral Cortex* 23(11):2712–2723 (2013). doi:10.1093/cercor/bhs261 | опубликованная копия, скачана пользователем вручную из браузера (academic.oup.com, платный доступ; не в PMC/Europe PMC) | `esterman2013zone` |
| `cortical_theta_alpha_oscillatory_attention_task.pdf` | Kitaura Y. и др. Functional localization and effective connectivity of cortical theta and alpha oscillatory activity during an attention task. *Clinical Neurophysiology Practice* 2:193–200 (2017). doi:10.1016/j.cnp.2017.09.002 | опубликованная, open access (Elsevier OA-журнал), скачана пользователем вручную из браузера (sciencedirect.com отдаёт Cloudflare-челлендж на автоматические запросы) | `kitaura2017functional` |
| `attention_detection_eeg_ml_review.pdf` | Sun Q. и др. Attention detection using EEG signals and machine learning: A review. *Machine Intelligence Research* 22(2):219–238 (2025). doi:10.1007/s11633-024-1492-6 | опубликованная копия, скачана пользователем вручную из браузера (link.springer.com, бот-защита на прямой PDF-ссылке) | `sun2025attention` |
| `eeg_bci_status_challenges_review.pdf` | Rashid M. и др. Current status, challenges, and possible solutions of EEG-based brain-computer interface: A comprehensive review. *Frontiers in Neurorobotics* 14:25 (2020). doi:10.3389/fnbot.2020.00025 | опубликованная, CC-BY, open access (frontiersin.org), скачана автоматически | `rashid2020bci` |
| `knudsen2007_fundamental_components_attention.pdf` | Knudsen E.I. Fundamental components of attention. *Annual Review of Neuroscience* 30:57–78 (2007). doi:10.1146/annurev.neuro.30.051606.094256 | опубликованная копия из публичного университетского course archive; издательская версия платная | `knudsen2007components` |
| `petersen_posner2012_attention_system.pdf` | Petersen S.E., Posner M.I. The attention system of the human brain: 20 years after. *Annual Review of Neuroscience* 35:73–89 (2012). doi:10.1146/annurev-neuro-062111-150525 | опубликованная копия из публичного университетского course archive; авторская рукопись также доступна в PMC (PMCID: PMC3413263) | `petersen2012attention` |
| `lawhern2018_eegnet.pdf` | Lawhern V.J. и др. EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering* 15(5):056013 (2018). doi:10.1088/1741-2552/aace8c | препринт arXiv:1611.08024, open access | `lawhern2018eegnet` |
| `schirrmeister2017_deep_convnet_eeg.pdf` | Schirrmeister R.T. и др. Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping* 38(11):5391–5420 (2017). doi:10.1002/hbm.23730 | препринт arXiv:1703.05051, open access | `schirrmeister2017deep` |
| `song2023_eeg_conformer.pdf` | Song Y. и др. EEG Conformer: Convolutional transformer for EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 31:710–719 (2023). doi:10.1109/TNSRE.2022.3230250 | авторская версия с сайта автора; статья опубликована open access под CC-BY 4.0 | `song2023conformer` |
| `wang2024_eegpt.pdf` | Wang G. и др. EEGPT: Pretrained transformer for universal and reliable representation of EEG signals. *Advances in Neural Information Processing Systems* 37 (2024). doi:10.52202/079017-1239 | опубликованная conference-версия, open access (NeurIPS Proceedings) | `wang2024eegpt` |
| `chen_guestrin2016_xgboost.pdf` | Chen T., Guestrin C. XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794 (2016). doi:10.1145/2939672.2939785 | препринт arXiv:1603.02754, open access | `chen2016xgboost` |

`zhang2020transfer` (transfer learning review, Sensors) тематически ближе к
`generalization_papers/` — файл и запись о нём там же
(`transfer_learning_eeg_bci_review_zhang.pdf`), не здесь.
