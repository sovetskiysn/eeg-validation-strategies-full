# Источники PDF в этой папке

Статьи, которые **рецензент прямо потребовал процитировать** (Reviewer 4,
comment 4.4 в `latex_documents/revision/journal_feedback.md`). Назначение у
папки отдельное: это не источник цитаты по существу текста
(`citation_papers/`), не обоснование работы с датасетами (`datasets_papers/`)
и не материал по generalization-новизне (`generalization_papers/`) — это
внешнее требование, на которое нужно дать ответ в
`response_to_reviewers/main.tex`, независимо от того, войдут ли эти работы в
итоговый текст манускрипта.

Обе работы — про motor imagery BCI, то есть вне paradigm scope нашей статьи;
хранятся здесь, чтобы ответ рецензенту опирался на прочитанный текст, а не на
аннотацию.

Колонка `bib-ключ` — соответствующая запись в
`latex_documents/revision/manuscript_clean/refs.bib`, если статья
процитирована. Прочерк — статья скачана как материал для ответа рецензенту и в
`refs.bib` пока не заведена.

| файл | публикация | что за версия | bib-ключ |
| --- | --- | --- | --- |
| `rdwt_motor_imagery_deep_learning_impact.pdf` | Siino M., Bonomo G., Sorbello R., Tinnirello I. Investigating the Impact of Rational Dilated Wavelet Transform on Motor Imagery EEG Decoding With Deep Learning Models. *IEEE Access* 13:214223–214235 (2025). doi:10.1109/ACCESS.2025.3645762 | препринт с arXiv (2510.09242, CC-BY 4.0), open access. Опубликованная версия — IEEE Access (журнал полностью OA), но прямая ссылка ielx-путём отдаёт чужой PDF, а копии на ieeexplore.ieee.org и iris.unipa.it за Cloudflare-челленджем; итоговую вёрстку нужно взять руками из браузера, если она понадобится | — |
| `ratiowavenet_learnable_rdwt_frontend.pdf` | Siino M., Bonomo G., Sorbello R., Tinnirello I. RatioWaveNet: A Learnable RDWT Front-End for Robust and Interpretable EEG Motor-Imagery Classification. arXiv:2510.21841 (2025) | препринт, CC-BY 4.0, open access (arxiv.org). Журнальной версии на момент скачивания не заявлено; код авторов — github.com/Bonomo31/RatioWaveNet | — |
