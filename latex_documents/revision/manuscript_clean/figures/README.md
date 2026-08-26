# Figures for the clean manuscript

The completed analysis was pulled from `/home/datalake/models_test/analysis` with
`make pull-artifacts`, then arranged by editorial role. `manuscript.tex` embeds the three
files in `main/` as complete figure--caption blocks in Results. The neural-network
per-decoder figures are supporting files, cited in the Results text and captioned after
the bibliography.

| Role | Manuscript label | Raster source |
| --- | --- | --- |
| Main | Fig 1 | `main/fig1_logistic_regression_all_scenarios.png` |
| Main | Fig 2 | `main/xgboost_all_scenarios.png` |
| Main | Fig 3 (cross-subject model comparison) | `main/cross_subject_model_comparison.png` |
| Supplementary | S1 Fig. (EEGNet) | `supplementary/S3_fig_eegnet_all_scenarios.png` |
| Supplementary | S2 Fig. (ShallowFBCSPNet) | `supplementary/S4_fig_shallownet_all_scenarios.png` |
| Supplementary | S3 Fig. (EEGConformer) | `supplementary/S5_fig_eegconformer_all_scenarios.png` |

`source_svg/` retains the vector originals from the analysis, including the
vector version of the main figure. The historical `fig*.jpg` files remain
unchanged for provenance only and are not cited by the clean manuscript.

## Do not strip the graphics by hand

The PLOS-shaped project — manuscript without `\includegraphics` plus `figures/Fig<N>.tif` —
is generated:

```
make prepare-plos-submission SOURCE=revision/manuscript_clean OUTPUT=revision/manuscript_clean_plos
```

The target converts every embedded graphic to TIFF (LZW, 300 dpi), numbers the files in
citation order, and warns when a figure falls outside the PLOS pixel limits. Upload each
`Fig<N>.tif` as an individual file in Editorial Manager.

## Known deviation

`fig2.jpg` is 673 px wide, below the PLOS minimum of 789 px; `fig1.jpg` (846x547) is inside
the limits. The regenerated figures must be produced at print size, not upscaled.

## PLOS ONE specifications

Source: <https://journals.plos.org/plosone/s/figures> (verified 2026-08-08).

- **Format:** TIFF or EPS only. TIFF preferred (EPS frequently has font problems).
- **Naming:** `Fig1.tif`, `Fig2.tif` — numeric order, matching the citation and caption labels.
- **Width:** min 789 px (2.63 in / 6.68 cm) at 300 dpi; max 2250 px (7.5 in / 19.05 cm).
  Keep to <= 5.2 in (13.2 cm) wide for text-column alignment.
- **Height:** max 2625 px (8.75 in / 22.23 cm) at 300 dpi.
- **Resolution:** 300-600 dpi at the intended print size. Below 300 dpi is rejected as
  pixelated; above 600 dpi may be downsampled.
- **File size:** <= 10 MB. TIFF: LZW compression, layers flattened.
- **Fonts:** Arial, Times, or Symbol only, 8-12 pt. For EPS, embed fonts or convert to outlines.
- **No text inside the image file** for the caption, title, or figure number — those live
  in `manuscript.tex` only.

## Before upload

Run each file through PLOS's figure checker, NAAS
(<https://journals.plos.org/plosone/s/figures#loc-tools-for-figure-preparation>).
This could not be done from this environment — it is a manual step.
Do **not** run Supporting Information figures through NAAS; they have relaxed requirements.
