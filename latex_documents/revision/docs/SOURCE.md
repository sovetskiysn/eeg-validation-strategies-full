# Provenance of the official PLOS assets

Downloaded 2026-08-05 (UTC). Checksums re-verified against the files in this repository
on 2026-08-08 — all three match.

| Asset | Source | Version / SHA-256 |
| --- | --- | --- |
| PLOS LaTeX package | https://journals.plos.org/plosone/s/file?id=1457%2FPLOS_latex_template.zip | Template v3.8, April 2026; archive `ea3a8a0fdbac77f95de47639541b09ef1583e059ace6783367490af9fa0b9a60` |
| `plos2025.bst` | Extracted unchanged from the package above | `8470ef5e547189a61d8eef2cfd7a65fbb84b77b881ccb42ed6815da86a04c490` |
| `PLOS_Affiliations_Formatting_Guidelines.pdf` | https://journals.plos.org/plosone/s/file?id=3fac%2FPLOS+Affiliations+Formatting+Guidelines.pdf | `110ff7db84f6b270d49ce8324bbcb9d46c79c01ceeffa5266dea10d930e0079c` |
| `PLOS_Manuscript_Body_Formatting_Guidelines.pdf` | https://journals.plos.org/plosone/s/file?id=9cba%2FPLOS+Manuscript+Body+Formatting+Guidelines.pdf | `9790279b49143233c4275a3bb8d83b0dba17179264f6393c9a9ea2055f01e38d` |
| PLOS ONE LaTeX instructions | https://journals.plos.org/plosone/s/latex | accessed 2026-08-05 |
| PLOS ONE Submission Guidelines | https://journals.plos.org/plosone/s/submission-guidelines | accessed 2026-08-05, re-checked 2026-08-08 |
| PLOS ONE Figure Guidelines | https://journals.plos.org/plosone/s/figures | checked 2026-08-08 |
| PLOS ONE Supporting Information | https://journals.plos.org/plosone/s/supporting-information | checked 2026-08-08 |
| PLOS Materials and Software Sharing (code) | https://journals.plos.org/plosone/s/materials-and-software-sharing | checked 2026-08-08 |

The unmodified template package is kept in `../../resources/templates/PLOS_latex_template/`.
`plos2025.bst` in `../manuscript_clean/` and `../initial_submission/` is byte-identical to it
(all three match `8470ef5e…`, re-verified 2026-08-10).

The manuscript's own bibliography is `../manuscript_clean/refs.bib`. The PLOS sample bibliography
(`plos_bibtex_sample.bib`, `cfecbdb00bc4955a0f9b36ab40ee99c293f3c5034ad49da1afc741b59792a3ac`)
is deliberately **not** part of the revision project — it contains only template placeholder
entries. It remains in `../../resources/templates/PLOS_latex_template/` and, as a whitespace-only
variant (`4143c26544478432aeb20427c526206bd0e635cccda29bdb49621ce6307971a2`, same 13 placeholder
entries), in `../initial_submission/`.
