"""Compose and run one EEG experiment with Hydra."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import sklearn
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import cross_validate

# LogisticRegressionCV's inner group-aware split (configs/pipeline/3_estimator/
# logistic_regression.yaml) needs `groups` routed to it from the outer
# `cross_validate` call below; sklearn only does this when routing is enabled.
sklearn.set_config(enable_metadata_routing=True)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the composed scientific condition and persist what cannot be recomputed."""
    # =============================================================================
    # Step 0: CUDA execution mode
    # =============================================================================
    # CUDA models still use CPU threads for dispatch and tensor preparation.
    # This is independent of the CPU workers that extract classical features.
    torch.set_num_threads(int(cfg.torch_num_threads))
    # EEG windows have one fixed geometry, so cuDNN can benchmark once and use
    # the fastest kernel thereafter.  TF32 uses H200 Tensor Cores for float32
    # matrix operations; it trades bitwise reproducibility for throughput.
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # =============================================================================
    # Stage 1: artifacts and validation protocol
    # =============================================================================
    epochs, groups, cv = instantiate(
        cfg.validation_strategy.protocol,
        dataset=cfg.dataset,
        preparation=cfg.preparation,
        seed=int(cfg.seed),
    )

    # =============================================================================
    # Stage 2: NumPy samples, provenance and estimator
    # =============================================================================
    X = epochs.get_data(copy=False).astype(np.float32, copy=False)
    y = epochs.events[:, -1]
    metadata = epochs.metadata.copy()
    class_names = dict(sorted((int(code), name) for name, code in epochs.event_id.items()))
    # The fitted extractor is held in a name of its own because it is what knows
    # the feature names the coefficients below are indexed by.
    transform = instantiate(cfg.features.transform)
    X = transform.fit_transform(X)

    estimator = instantiate(cfg.pipeline)
    # `scoring=None` takes the estimator's own `.score()`, which is accuracy for
    # every model here. It is a liveness check on the run and nothing more: every
    # reported metric is recomputed by `run_analysis.py` from the tables below,
    # so a metric list in the run config would be a second source of truth.
    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        params={"groups": groups},
        scoring=None,
        return_train_score=False,
        return_indices=True,
        return_estimator=True,
        error_score="raise",
    )
    indices = results["indices"]
    estimators = results["estimator"]

    # =============================================================================
    # Results: window identity, grain `window_index`
    # =============================================================================
    # The only place the identity of a window and its true label live, and the
    # only place windows that no fold ever tests appear at all: cross-dataset and
    # cross-task train on source windows that are in no test set.
    windows = (
        metadata.rename(columns={"label": "y_true"})
        .reset_index(drop=True)
        .rename_axis("window_index")
        .reset_index()
    )
    windows["y_true_name"] = windows["y_true"].map(class_names)

    # =============================================================================
    # Results: fold membership and model output, grain `(fold, window_index)`
    # =============================================================================
    # Train rows carry no prediction: a fold's own training windows are not an
    # independent estimate of anything. They are here because the composition of
    # training is the other thing a rerun could not recover -- every class count,
    # the majority class and therefore the whole majority baseline come from it,
    # and joining it against `windows` is what makes the splits auditable.
    folds = pd.concat(
        [
            pd.DataFrame({"fold": fold, "window_index": train_idx, "part": "train"})
            for fold, train_idx in enumerate(indices["train"])
        ]
        + [
            pd.DataFrame(
                {
                    "fold": fold,
                    "window_index": test_idx,
                    "part": "test",
                    "y_pred": estimator.predict(X[test_idx]),
                    # `decision_function` and not a probability, for models
                    # that expose one but not the other; inventing a
                    # probability is not allowed.
                    "score": (
                        estimator.predict_proba(X[test_idx])[:, 1]
                        if hasattr(estimator, "predict_proba")
                        else estimator.decision_function(X[test_idx])
                    ),
                }
            )
            for fold, (test_idx, estimator) in enumerate(
                zip(indices["test"], estimators, strict=True)
            )
        ],
        ignore_index=True,
    ).astype({"y_pred": "Int64"})

    # =============================================================================
    # Results: fitted model contents, grain `(fold, feature_index)`
    # =============================================================================
    # The only place what a fold learned survives the run, and it costs nothing:
    # `cross_validate` already returns these estimators. Read off the last
    # pipeline step, because pipelines forward methods but not fitted
    # attributes -- `coef_` on the pipeline itself is always absent.
    # LogisticRegression fills `coef`, xgboost fills `feature_importance`,
    # and the networks fill neither, and that is not an error.
    # `LogisticRegressionCV`/`GridSearchCV` wrap the search around the fitted
    # model rather than being it: unwrap to the winning estimator first, or
    # `coef_`/`feature_importances_` below would silently read as absent.
    feature_names = getattr(transform, "get_feature_names_out", None)
    feature_names = None if feature_names is None else feature_names()
    importance_frames = []
    for fold, estimator in enumerate(estimators):
        model = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        model = getattr(model, "best_estimator_", model)
        coef = getattr(model, "coef_", None)
        feature_importance = getattr(model, "feature_importances_", None)
        if coef is None and feature_importance is None:
            continue
        importance_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "feature_index": np.arange(
                        np.size(coef) if coef is not None else np.size(feature_importance)
                    ),
                    "feature": feature_names,
                    "coef": np.nan if coef is None else np.ravel(coef),
                    "feature_importance": (
                        np.nan if feature_importance is None else feature_importance
                    ),
                }
            )
        )
    importances = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(
            {
                "fold": pd.Series(dtype="int64"),
                "feature_index": pd.Series(dtype="int64"),
                "feature": pd.Series(dtype="object"),
                "coef": pd.Series(dtype="float64"),
                "feature_importance": pd.Series(dtype="float64"),
            }
        )
    )

    # =============================================================================
    # Results: one directory per transfer direction
    # =============================================================================
    # A transfer run holds one source and several targets so the decoder is
    # fitted once per source fold instead of once per direction. That saving is
    # an execution detail and must not reach the analysis: every direction is
    # written out as the self-contained result it would have been on its own,
    # under the same three-table contract. Source train rows and the fitted model
    # contents therefore repeat between neighbouring directions -- they are the
    # same fact about the same fold, and they are small next to the EEG windows.
    output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    result_dirs = []
    if "target_index" not in windows.columns:
        windows.to_parquet(output_dir / "windows.parquet", index=False)
        folds.to_parquet(output_dir / "folds.parquet", index=False)
        importances.to_parquet(output_dir / "importances.parquet", index=False)
        result_dirs.append(output_dir)
    else:
        source_rows = windows["target_index"].isna()
        for index, target in enumerate(cfg.dataset.targets):
            kept = windows.index[source_rows | (windows["target_index"] == index)]
            # `window_index` is renumbered inside each direction so the analysis
            # keeps joining and checking uniqueness exactly as it does for a
            # single-target run. Source windows come first in the concatenation,
            # so they keep the numbers they already had.
            local = pd.Series(np.arange(len(kept)), index=windows.loc[kept, "window_index"])
            target_windows = windows.loc[kept].drop(columns="target_index").assign(
                window_index=np.arange(len(kept))
            )
            target_folds = folds[folds["window_index"].isin(local.index)].copy()
            target_folds["window_index"] = target_folds["window_index"].map(local)

            # The directory name is derived from the resolved target recipe, so it
            # is not a hand-written run label: it says which composition was
            # tested without opening anything.
            excluded = sorted(target.exclude_conditions)
            target_dir = output_dir / "targets" / (
                f"{target.name}__" + ("-".join(("ex", *excluded)) if excluded else "full")
            )
            target_dir.mkdir(parents=True)
            # A projection of this one direction, under the same field paths the
            # analysis reads from a saved Hydra config. Reading it needs no
            # knowledge of how many directions shared the execution job, and the
            # decoder name is here because the leaf has to stand on its own.
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "validation_strategy": {"name": str(cfg.validation_strategy.name)},
                        "dataset": {
                            "source": OmegaConf.to_container(cfg.dataset.source, resolve=True),
                            "target": OmegaConf.to_container(target, resolve=True),
                        },
                        "pipeline_components": {
                            "model": {"name": str(cfg.pipeline_components.model.name)}
                        },
                    }
                ),
                target_dir / "scenario.yaml",
            )
            target_windows.to_parquet(target_dir / "windows.parquet", index=False)
            target_folds.to_parquet(target_dir / "folds.parquet", index=False)
            importances.to_parquet(target_dir / "importances.parquet", index=False)
            result_dirs.append(target_dir)

    # `test_score` is accuracy over every target of a fold at once. It is the
    # liveness check and nothing else; every reported metric is recomputed by
    # `run_analysis.py` from the tables above. Logged rather than printed so it
    # lands in Hydra's own per-job log file, not only on the console.
    log.info(
        f"Completed {HydraConfig.get().job.name}: {len(result_dirs)} result(s) under "
        f"{output_dir} (mean accuracy {results['test_score'].mean():.3f})"
    )


if __name__ == "__main__":
    main()
