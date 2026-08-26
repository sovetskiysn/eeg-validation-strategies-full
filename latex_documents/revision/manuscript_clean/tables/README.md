# Table source files

The completed analysis was pulled from `/home/datalake/models_test/analysis` with
`make pull-artifacts`. Its five decoder-specific scenario tables are preserved as
supporting artifacts in `supplementary/` and are cited as S1--S5 Tables in
`manuscript.tex`. The analysis preview, including its local LaTeX build files, is kept
separately in `analysis_preview/` and is not part of the manuscript source.

| Supporting label | File |
| --- | --- |
| S1 Table (logistic regression) | `supplementary/S1_table_logistic_regression.tex` |
| S2 Table (XGBoost) | `supplementary/S2_table_xgboost.tex` |
| S3 Table (EEGNet) | `supplementary/S3_table_eegnet.tex` |
| S4 Table (ShallowFBCSPNet) | `supplementary/S4_table_shallownet.tex` |
| S5 Table (EEGConformer) | `supplementary/S5_table_eegconformer.tex` |

`table2_current_sweep.tex` is the logistic-regression table embedded in Results.
It is the same completed-analysis artifact as S1 Table, retained as a main-text copy
so readers can inspect the values underlying Fig. 1 without opening supplementary files.

The small label-mapping table (`tab:label_mapping`) stays inline in the Methodology.

## Historical placeholder

`results_scenarios.tex` is retained as a **placeholder containing old-pipeline
numbers**. It is not cited or included in the manuscript.

`results_scenarios.tex` is the Frontiers Table 2 (from
`latex_documents/superseded_frontiers_article/frontiers.tex`) re-fitted to the PLOS
single-column layout: `adjustwidth` instead of `table*`, plain `tabular` instead of
`tabularx`, `multirow` instead of `makecell`. It carries the numbers of the **old**
pipeline. The Results section deliberately removed those values; the file is here as the
structural placeholder for the table the section must contain, and its rows must not be
cited in the prose until the runs are redone.

The table needs `\usepackage{multirow}` and the `L{<width>}` column type; both are
declared in the preamble of `manuscript.tex`.

## Naming, and what `make pull-artifacts` overwrites

`python_project/scripts/run_analysis.py` writes `tables/results.tex` and
`tables/paired_tests.tex` into an analysis directory, and

```
make pull-artifacts ANALYSIS_DIR=/abs/path/to/analysis
```

copies that directory's contents over this one. Those two names are therefore reserved
for generated output — hand-written files must use other names, which is why this one is
`results_scenarios.tex`. Generated files are bare `tabular` bodies from
`DataFrame.to_latex`; wrapping one in a float with a caption and a label stays a manual
step, as does deciding which of them the manuscript `\input`s.

## `\input` is expanded on export

`make prepare-plos-submission` inlines every `\input{...}` that stands alone on its line
into the exported manuscript, so the PLOS submission is a single `.tex`. Keep each
`\input{tables/...}` on its own line and unindented-safe (leading whitespace is fine, a
trailing comment is fine, anything else on the line is not).
