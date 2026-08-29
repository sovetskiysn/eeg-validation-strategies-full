"""Pipeline components that need train-fold data and labels together."""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from skorch.callbacks import Callback
from xgboost import XGBClassifier


class BalancedClassWeight(Callback):
    """Weight the loss by inverse class frequency of the fold being trained on.

    This is `class_weight="balanced"` for the deep decoders. skorch has no such
    parameter and `criterion__weight` is a fixed tensor, but the imbalance here
    is a property of the composed cohort rather than a constant: `distinguishing`
    is ~1:1, and SAM40 is ~1:1, ~2:1 or ~3:1 depending on how many high-demand
    tasks the recipe selects alongside `relax`. A number written into
    the estimator recipe would therefore be wrong for most scenarios, so it is
    read off the fold instead, at the one point where skorch exposes it.

    A class rather than a function because `on_train_begin` is skorch's contract
    for reaching a fold before its first optimiser step; there is no callable
    form of it.
    """

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        """Set the criterion weights from the labels this fold will train on."""
        # skorch passes y=None when it is handed a Dataset instead of arrays.
        # The runner always passes arrays, so this is a loud stop rather than a
        # silent fallback to unweighted training, which would look like a
        # successful run with the treatment quietly missing.
        if y is None:
            raise ValueError(
                "BalancedClassWeight needs the fold labels, and skorch received none. "
                "It supports estimators fitted on arrays, not on a Dataset."
            )
        y = np.asarray(y)
        # The same n / (k * n_c) formula sklearn applies for
        # `class_weight="balanced"`, so the classical and deep decoders are
        # rebalanced by one rule rather than by two similar ones.
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        net.criterion_.weight = torch.as_tensor(
            weights, dtype=torch.float32, device=net.device
        )


class BalancedXGBClassifier(XGBClassifier):
    """`class_weight="balanced"` for XGBoost, read off the fold being fitted.

    Same treatment as `class_weight` on the logistic regression and as
    `BalancedClassWeight` on the deep decoders, so all five decoders are
    rebalanced by one rule rather than by three similar ones.

    XGBoost exposes neither. Of the 40 sklearn parameters on `XGBClassifier`
    only `scale_pos_weight` covers imbalance, and that is a single scalar for
    the binary case, so it can follow neither a cohort whose ratio is ~1:1,
    ~2:1 or ~3:1 depending on the selected SAM40 tasks, nor a third class. The
    other route, `sample_weight` on `fit`, cannot be routed in from outside:
    `imblearn.Pipeline` resamples X and y but not the fit params travelling
    alongside them, which is the same structural fact recorded in
    logistic_regression.yaml about `LogisticRegressionCV`.

    Neither limit binds here, because the weights are not external data: they
    are a function of the `y` this estimator is already handed. sklearn's
    `compute_sample_weight("balanced", y)` is the per-sample form of the very
    n / (k * n_c) formula behind `class_weight="balanced"`, so deriving it in
    `fit` needs no metadata routing and no resampling step.
    """

    def __init__(self, class_weight=None, **kwargs):
        # Stored under the parameter's own name, which is sklearn's contract
        # for `get_params`/`clone` -- `cross_validate` clones per fold.
        self.class_weight = class_weight
        super().__init__(**kwargs)

    def get_xgb_params(self):
        """Keep `class_weight` out of the params handed to the booster."""
        # XGBoost forwards get_params() into the C++ booster, which warns
        # `Parameters: { "class_weight" } are not used` for keys it does not
        # know. This one is consumed in Python, so it is dropped here.
        params = super().get_xgb_params()
        params.pop("class_weight", None)
        return params

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Derive the sample weights from this fold's labels, then fit."""
        # An explicitly passed sample_weight wins: the automatic weighting is
        # a default, not an override of a caller who asked for something else.
        if self.class_weight is not None and sample_weight is None:
            sample_weight = compute_sample_weight(self.class_weight, y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)
