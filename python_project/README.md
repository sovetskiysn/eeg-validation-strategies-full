# EEG validation strategies

Analysis code for the study *"Zero-shot generalization of EEG attention-state
decoding across increasingly stringent validation strategies on open datasets."*

Two public EEG datasets are harmonized into a shared 12-channel input space with
a common binary attention label mapping, and five decoders are evaluated under
within-dataset, cross-subject, cross-task and cross-dataset validation without
target-specific adaptation. Every figure and table in the article is rebuilt
from the run artifacts produced here.

## How to cite

If you use this code, or if it helped your work, please cite the article:

```bibtex
@article{sovet_zeroshot_eeg,
  author  = {Sovet, Sanzhar and Abdikenov, Beibit and Zholdasova, Manzura
             and Kamzanova, Altyngul and Mukushev, Medet and Kustubayeva, Almira},
  title   = {Zero-shot generalization of {EEG} attention-state decoding across
             increasingly stringent validation strategies on open datasets},
  journal = {PLOS ONE},
  year    = {<YEAR>},
  doi     = {<ARTICLE-DOI>}
}
```

Archived snapshot of this code (all versions):
<https://doi.org/10.5281/zenodo.22255472>

## Stack

Python 3.11, environment managed with [uv](https://docs.astral.sh/uv/).

* **Hydra** — composed configuration and multirun sweeps
* **scikit-learn / XGBoost / braindecode** — decoders and training
* **MNE-Python, MNE-BIDS, MNE-BIDS-Pipeline, mne-features** — EEG I/O,
  standardization, preprocessing and feature extraction
* **pandas / PyArrow** — run artifacts as Parquet; **Matplotlib / seaborn** — figures

## Structure

```text
configs/      Hydra config groups: preparation, features, pipeline,
              validation_strategy, and named sweep presets in experiments/
src/          the analysis logic: download, BIDS standardization, preparation,
              decoder pipelines, validation splitters, metrics and figures
scripts/      thin runners, one per stage
notebooks/    quality-control and exploratory checks
datasets/     raw downloads and BIDS output (not versioned)
results/      per-run artifacts produced by the sweeps
```

## Usage

```bash
uv sync            # create the locked environment
make download      # fetch both public datasets (Figshare + Kaggle)
make standardize   # convert them to BIDS
make experiment    # run the decoder x scenario sweep into results/
make analysis      # render article figures and tables from a finished sweep
```

`make analysis` takes the sweep to read and the output directory as environment
variables:

```bash
make analysis ANALYSIS_INPUT_DIR="results/<sweep dir>" ANALYSIS_OUTPUT_DIR="../analysis"
```

Neither dataset is redistributed here; `make download` fetches SAM-40 from
Figshare and the mental attention state dataset from Kaggle (Kaggle API
credentials required). Please cite the original datasets alongside this work.

## Run artifacts

Each leaf run directory is self-describing: `.hydra/` holds the fully resolved
configuration, `windows.parquet` the window index and true labels,
`folds.parquet` the fold assignments and per-window predictions,
`importances.parquet` the per-fold feature importances where available. All
reported metrics are recomputed from these files, so tables and figures can be
rebuilt without rerunning the experiments. The global seed is set in
`configs/config.yaml`.

## License

MIT. See `LICENSE`.
