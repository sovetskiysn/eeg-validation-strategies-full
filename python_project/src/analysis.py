"""Article-table calculations derived from completed experiment jobs."""

from __future__ import annotations

import colorsys
from collections import Counter
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from utils import LATEX_ARTIFACT_TEMPLATES_DIR


# Article-level order of scenario rows. A dataset side is identified by its
# saved dataset name and selected BIDS tasks.
DISTINGUISHING = ("distinguishing", ("attention",))
SAM40_FULL = ("sam40", ("arithmetic", "mirror", "relax", "stroop"))
SAM40_STROOP = ("sam40", ("relax", "stroop"))
SAM40_ARITHMETIC = ("sam40", ("arithmetic", "relax"))
SAM40_MIRROR = ("sam40", ("mirror", "relax"))
SAM40_STROOP_ARITHMETIC = ("sam40", ("arithmetic", "relax", "stroop"))
SAM40_STROOP_MIRROR = ("sam40", ("mirror", "relax", "stroop"))
SAM40_ARITHMETIC_MIRROR = ("sam40", ("arithmetic", "mirror", "relax"))

SCENARIO_ORDER = (
    ("baseline", DISTINGUISHING, DISTINGUISHING),
    ("baseline", SAM40_FULL, SAM40_FULL),
    ("baseline", SAM40_STROOP, SAM40_STROOP),
    ("baseline", SAM40_ARITHMETIC, SAM40_ARITHMETIC),
    ("baseline", SAM40_MIRROR, SAM40_MIRROR),
    ("cross_task", SAM40_ARITHMETIC_MIRROR, SAM40_STROOP),
    ("cross_task", SAM40_ARITHMETIC, SAM40_MIRROR),
    ("cross_task", SAM40_ARITHMETIC, SAM40_STROOP),
    ("cross_task", SAM40_ARITHMETIC, SAM40_STROOP_MIRROR),
    ("cross_task", SAM40_MIRROR, SAM40_ARITHMETIC),
    ("cross_task", SAM40_MIRROR, SAM40_STROOP),
    ("cross_task", SAM40_MIRROR, SAM40_STROOP_ARITHMETIC),
    ("cross_task", SAM40_STROOP_ARITHMETIC, SAM40_MIRROR),
    ("cross_task", SAM40_STROOP_MIRROR, SAM40_ARITHMETIC),
    ("cross_task", SAM40_STROOP, SAM40_ARITHMETIC),
    ("cross_task", SAM40_STROOP, SAM40_ARITHMETIC_MIRROR),
    ("cross_task", SAM40_STROOP, SAM40_MIRROR),
    ("cross_dataset", DISTINGUISHING, SAM40_FULL),
    ("cross_dataset", SAM40_FULL, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_STROOP),
    ("cross_dataset", SAM40_STROOP, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_ARITHMETIC),
    ("cross_dataset", SAM40_ARITHMETIC, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_MIRROR),
    ("cross_dataset", SAM40_MIRROR, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_STROOP_ARITHMETIC),
    ("cross_dataset", SAM40_STROOP_ARITHMETIC, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_STROOP_MIRROR),
    ("cross_dataset", SAM40_STROOP_MIRROR, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_ARITHMETIC_MIRROR),
    ("cross_dataset", SAM40_ARITHMETIC_MIRROR, DISTINGUISHING),
)


def discover_scenario_results(scenario_glob: str) -> list[Path]:
    """Return one directory per logical scenario result under a job glob.

    A transfer job holds one source and several targets, so it writes one
    self-contained result per direction under `targets/`. Fewer execution jobs
    must not mean fewer scientific directions, and this is the only place the two
    counts are reconciled: everything below reads an ordinary result directory
    and never learns how many directions shared a job.
    """
    results = []
    for path in sorted(glob(scenario_glob)):
        job_dir = Path(path)
        targets_dir = job_dir / "targets"
        results.extend(sorted(targets_dir.iterdir()) if targets_dir.is_dir() else [job_dir])
    return results

def read_scenario_result(job_dir: Path) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Return the description and the two tables of one logical scenario result.

    A transfer direction saved its own `scenario.yaml`; an ordinary run kept the
    Hydra config it executed under. Both describe one direction under the same
    field paths, so which of the two a result carries is decided here and nowhere
    else -- the callers must not learn how the result was produced.
    """
    config_path = job_dir / "scenario.yaml"
    if not config_path.exists():
        config_path = job_dir / ".hydra" / "config.yaml"
    required = [job_dir / name for name in ("windows.parquet", "folds.parquet")]
    missing = [path.name for path in required if not path.exists()]
    if not config_path.exists() or missing:
        absent = [str(config_path.relative_to(job_dir))] if not config_path.exists() else []
        raise FileNotFoundError(f"{job_dir} is incomplete: missing {absent + missing}.")

    windows, folds = (pd.read_parquet(path) for path in required)
    return OmegaConf.load(config_path), windows, folds


def scenario_key(cfg) -> tuple[object, ...]:
    """Return the (protocol, source, target) identity one result stands for.

    Non-transfer protocols name one dataset and are their own target, which is
    what keeps every result addressable by the same three-part key.
    """
    if "preparation" in cfg:
        source = cfg.preparation.source if "source" in cfg.preparation else cfg.preparation
        target = cfg.preparation.target if "target" in cfg.preparation else source
        sides = source, target
    elif "recipe" in cfg:
        # Transfer leaves written by the preceding config generation.
        sides = cfg.recipe.source, cfg.recipe.target
    else:
        raise ValueError("Scenario config has neither preparation nor legacy recipe fields.")
    return (str(cfg.validation_strategy.name), *(_side_key(side) for side in sides))


def _side_key(side) -> tuple[str, tuple[str, ...]]:
    """Return the scientific identity of one composed preparation side."""
    dataset_name = side.name
    pipeline = side.mne_bids_pipeline if "mne_bids_pipeline" in side else side
    task = pipeline.task
    tasks = (task,) if isinstance(task, str) else tuple(task)
    return (
        str(dataset_name),
        tuple(sorted(tasks)),
    )


def decoder_name(cfg) -> str:
    """Read a decoder name from either generation of saved run configuration."""
    if "pipeline" in cfg and "name" in cfg.pipeline:
        return str(cfg.pipeline.name)
    return str(cfg.pipeline_components.model.name)


def craft_main_table(scenario_glob: str) -> str:
    """Fill the scenario-table template from one complete scenario-run glob."""
    model_display_names = {
        "logistic_regression": "logistic regression",
        "xgboost": "XGBoost",
        "eegnet": "EEGNet",
        "shallownet": "ShallowFBCSPNet",
        "eegconformer": "EEGConformer",
    }
    job_dirs = discover_scenario_results(scenario_glob)
    if len(job_dirs) != len(SCENARIO_ORDER):
        raise ValueError(
            f"A main scenario table needs {len(SCENARIO_ORDER)} scenario results, "
            f"got {len(job_dirs)}."
        )

    metrics_by_scenario: dict[tuple[str, str, str], tuple[str, str, str, str, str]] = {}
    decoder_names = set()
    for job_dir in job_dirs:
        cfg, windows, folds = read_scenario_result(job_dir)
        decoder = decoder_name(cfg)
        if decoder not in model_display_names:
            raise ValueError(f"{job_dir}: unsupported decoder for article table: {decoder}.")
        decoder_names.add(decoder)
        scenario = scenario_key(cfg)
        if scenario in metrics_by_scenario:
            raise ValueError(f"Duplicate scenario for one decoder: {scenario}.")

        rows = folds.merge(windows, on="window_index", how="left", validate="many_to_one")
        if rows["y_true"].isna().any():
            raise ValueError(f"{job_dir}: folds.parquet references missing windows.")
        test = rows[rows["part"] == "test"].astype({"y_pred": "int64"})
        if not len(test):
            raise ValueError(f"{job_dir}: the job has no test predictions.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")

        class_ids = sorted(test["y_true"].unique())
        if len(class_ids) != 2:
            raise ValueError(f"{job_dir}: expected exactly two tested classes, got {class_ids}.")
        _, recall, _, _ = precision_recall_fscore_support(
            test["y_true"], test["y_pred"], labels=class_ids, zero_division=0
        )
        roc_auc = roc_auc_score(test["y_true"], test["score"])
        format_metric = lambda value: "-" if pd.isna(value) else f"{value:.2f}"
        metrics_by_scenario[scenario] = (
            format_metric(balanced_accuracy_score(test["y_true"], test["y_pred"])),
            format_metric(f1_score(test["y_true"], test["y_pred"], average="macro", zero_division=0)),
            format_metric(matthews_corrcoef(test["y_true"], test["y_pred"])),
            format_metric(roc_auc),
            f"{recall[0]:.2f}/{recall[1]:.2f}",
        )

    expected, actual = Counter(SCENARIO_ORDER), Counter(metrics_by_scenario.keys())
    if actual != expected:
        missing = list((expected - actual).elements())
        unexpected = list((actual - expected).elements())
        raise ValueError(f"Scenario set does not match the table template; missing={missing}, unexpected={unexpected}.")
    if len(decoder_names) != 1:
        raise ValueError(f"A main scenario table needs one decoder, got {sorted(decoder_names)}.")
    decoder = decoder_names.pop()

    template = (LATEX_ARTIFACT_TEMPLATES_DIR / "scenario_table_template.tex").read_text()
    template = template.replace("MODEL_DISPLAY_NAME", model_display_names[decoder]).replace(
        "MODEL_SLUG", decoder
    )
    lines = template.splitlines(keepends=True)
    metric_lines = []
    for index, line in enumerate(lines):
        cells = line.rstrip("\n").split("&")
        if len(cells) == 7 and all(cell.strip().removesuffix(r"\\").strip() == "-" for cell in cells[2:]):
            metric_lines.append(index)
    if len(metric_lines) != len(SCENARIO_ORDER):
        raise ValueError(
            f"{LATEX_ARTIFACT_TEMPLATES_DIR / 'scenario_table_template.tex'} has {len(metric_lines)} metric rows; "
            f"expected {len(SCENARIO_ORDER)}."
        )

    for index, scenario in zip(metric_lines, SCENARIO_ORDER, strict=True):
        cells = lines[index].rstrip("\n").split("&")
        lines[index] = f"{cells[0]}&{cells[1]}& " + " & ".join(metrics_by_scenario[scenario]) + r" \\" + "\n"
    return "".join(lines)


def craft_all_scenarios_absolute_accuracy_figure(scenario_glob: str) -> Figure:
    """Build one all-scenarios, subject-level accuracy figure for one decoder."""
    protocol_labels = (
        ("Baseline", 5),
        ("Cross-subject", 2),
        ("Cross-session", 1),
        ("Cross-task (Dataset B)", 12),
        ("Cross-dataset", 14),
    )
    validation_method_labels = (
        "Stratified K-fold(Full A)",
        "Stratified K-fold(Full B)",
        "Stratified K-fold(B: Stroop)",
        "Stratified K-fold(B: Arithmetic)",
        "Stratified K-fold(B: Mirror)",
        "Leave-one-subject-out(Full A)",
        "Leave-one-subject-out(Full B)",
        "Leave-one-session-out(Full A)",
        "Stroop → Arithmetic",
        "Stroop → Mirror",
        "Arithmetic → Stroop",
        "Arithmetic → Mirror",
        "Mirror → Stroop",
        "Mirror → Arithmetic",
        "Arithmetic+Mirror → Stroop",
        "Stroop+Mirror → Arithmetic",
        "Stroop+Arithmetic → Mirror",
        "Stroop → Arithmetic+Mirror",
        "Arithmetic → Stroop+Mirror",
        "Mirror → Stroop+Arithmetic",
        "Full A → Full B",
        "Full B → Full A",
        "Full A → Stroop",
        "Stroop → Full A",
        "Full A → Arithmetic",
        "Arithmetic → Full A",
        "Full A → Mirror",
        "Mirror → Full A",
        "Full A → Stroop+Arithmetic",
        "Stroop+Arithmetic → Full A",
        "Full A → Stroop+Mirror",
        "Stroop+Mirror → Full A",
        "Full A → Arithmetic+Mirror",
        "Arithmetic+Mirror → Full A",
    )
    if len(validation_method_labels) != len(SCENARIO_ORDER):
        raise RuntimeError("Figure validation-method labels must match the scenario order.")
    job_dirs = discover_scenario_results(scenario_glob)
    if len(job_dirs) != len(SCENARIO_ORDER):
        raise ValueError(
            f"An all-scenarios figure needs {len(SCENARIO_ORDER)} scenario results, "
            f"got {len(job_dirs)}."
        )

    scores_by_scenario: dict[tuple[object, ...], np.ndarray] = {}
    decoder_names = set()
    for job_dir in job_dirs:
        cfg, windows, folds = read_scenario_result(job_dir)
        decoder = decoder_name(cfg)
        if decoder not in {
            "logistic_regression",
            "xgboost",
            "eegnet",
            "shallownet",
            "eegconformer",
        }:
            raise ValueError(f"{job_dir}: unsupported decoder for article figure: {decoder}.")
        decoder_names.add(decoder)
        scenario = scenario_key(cfg)
        if scenario not in SCENARIO_ORDER:
            raise ValueError(f"Unexpected scenario in {job_dir}: {scenario}.")
        if scenario in scores_by_scenario:
            raise ValueError(f"Duplicate scenario for one decoder: {scenario}.")

        if windows["window_index"].duplicated().any():
            raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
        test = folds.loc[folds["part"] == "test"].merge(
            windows[["window_index", "subject_id", "y_true"]],
            on="window_index",
            how="left",
            validate="many_to_one",
        )
        if test.empty:
            raise ValueError(f"{job_dir}: the job has no test predictions.")
        if test[["subject_id", "y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"{job_dir}: test predictions reference missing or incomplete windows.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")

        subject_scores = []
        for subject_id, subject_test in test.groupby("subject_id", sort=True):
            if subject_test["y_true"].nunique() != 2:
                raise ValueError(f"{job_dir}: target subject {subject_id} does not contain two classes.")
            subject_scores.append(
                balanced_accuracy_score(subject_test["y_true"], subject_test["y_pred"])
            )
        scores_by_scenario[scenario] = np.asarray(subject_scores, dtype=float)

    if len(decoder_names) != 1:
        raise ValueError(f"An all-scenarios figure needs one decoder, got {sorted(decoder_names)}.")
    expected, actual = set(SCENARIO_ORDER), set(scores_by_scenario)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Scenario set does not match the article figure; missing={missing}, unexpected={unexpected}."
        )

    decoder_names.pop()
    means, lower_errors, upper_errors = [], [], []
    rng = np.random.default_rng(42)
    for protocol, source, target in SCENARIO_ORDER:
        scores = scores_by_scenario[(protocol, source, target)]
        mean = scores.mean()
        bootstrap_means = rng.choice(scores, size=(10_000, len(scores)), replace=True).mean(axis=1)
        low, high = np.quantile(bootstrap_means, (0.025, 0.975))
        means.append(mean)
        lower_errors.append(mean - low)
        upper_errors.append(high - mean)
    method_y_positions = []
    protocol_y_positions = []
    y_positions = []
    y_labels = []
    next_y_position = len(SCENARIO_ORDER) + len(protocol_labels) - 1
    validation_label_start = 0
    for protocol_label, row_count in protocol_labels:
        protocol_y_positions.append(next_y_position)
        y_positions.append(next_y_position)
        y_labels.append(protocol_label)
        next_y_position -= 1

        method_positions = list(range(next_y_position, next_y_position - row_count, -1))
        method_y_positions.extend(method_positions)
        y_positions.extend(method_positions)
        y_labels.extend(validation_method_labels[validation_label_start : validation_label_start + row_count])
        next_y_position -= row_count
        validation_label_start += row_count
    fig, ax = plt.subplots(figsize=(12.2, 14.0))
    ax.errorbar(
        means,
        method_y_positions,
        xerr=np.array((lower_errors, upper_errors)),
        fmt="o",
        color="#1F65AE",
        ecolor="#294E7A",
        markersize=5.5,
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )
    ax.axvline(0.5, color="#7C8AA0", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=10,
            transform=ax.get_xaxis_transform())
    ax.set_yticks(y_positions, ("",) * len(y_positions))
    ax.set_ylim(min(y_positions) - 0.8, max(y_positions) + 0.8)
    plotted_scores = np.concatenate(
        (
            np.asarray(means) - np.asarray(lower_errors),
            np.asarray(means) + np.asarray(upper_errors),
            np.array((0.5,)),
        )
    )
    score_span = np.ptp(plotted_scores)
    score_padding = max(0.01, score_span * 0.12)
    ax.set_xlim(
        max(0.0, plotted_scores.min() - score_padding),
        min(1.0, plotted_scores.max() + score_padding),
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color="#E0E6EF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    protocol_label_set = dict(protocol_labels)
    for y_position, label in zip(y_positions, y_labels, strict=True):
        is_protocol_label = label in protocol_label_set
        ax.text(
            -0.27,
            y_position,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontweight="bold" if is_protocol_label else "normal",
            fontsize=11 if is_protocol_label else 10,
        )
    for protocol_y_position in protocol_y_positions:
        ax.axhline(protocol_y_position, color="#7A7A7A", alpha=0.65, linewidth=1.3, zorder=1)

    fig.subplots_adjust(left=0.23, right=0.96, top=0.97, bottom=0.04)
    return fig


def craft_transfer_matrix_figure(scenario_glob: str) -> Figure:
    """Build source-on-row matrices for all directed transfer scenarios.

    A confusion matrix would describe predicted labels within one evaluation;
    it cannot show which data composition trained a decoder.  These matrices
    instead use source compositions as rows and target compositions as columns.
    The diagonal is the matching within-composition baseline where the sweep
    contains one; every other populated cell is a directed zero-shot transfer
    estimate. Gray cells are source--target compositions the sweep did not run.
    """
    dataset_a = ("distinguishing", ("drowsy",))
    dataset_b = ("sam40", ())
    stroop = ("sam40", ("arithmetic", "mirror"))
    arithmetic = ("sam40", ("mirror", "stroop"))
    mirror = ("sam40", ("arithmetic", "stroop"))
    stroop_arithmetic = ("sam40", ("mirror",))
    stroop_mirror = ("sam40", ("arithmetic",))
    arithmetic_mirror = ("sam40", ("stroop",))
    matrix_specs = (
        (
            "Cross-task transfer (Dataset B)",
            (
                ("Stroop", stroop),
                ("Arithmetic", arithmetic),
                ("Mirror", mirror),
                ("Stroop+\nArithmetic", stroop_arithmetic),
                ("Stroop+\nMirror", stroop_mirror),
                ("Arithmetic+\nMirror", arithmetic_mirror),
            ),
            "cross_task",
        ),
        (
            "Cross-dataset transfer",
            (
                ("Dataset A", dataset_a),
                ("Full B", dataset_b),
                ("Stroop", stroop),
                ("Arithmetic", arithmetic),
                ("Mirror", mirror),
                ("Stroop+\nArithmetic", stroop_arithmetic),
                ("Stroop+\nMirror", stroop_mirror),
                ("Arithmetic+\nMirror", arithmetic_mirror),
            ),
            "cross_dataset",
        ),
    )

    scores: dict[tuple[object, ...], float] = {}
    for job_dir in discover_scenario_results(scenario_glob):
        cfg, windows, folds = read_scenario_result(job_dir)
        scenario = scenario_key(cfg)
        if scenario in scores:
            raise ValueError(f"Duplicate scenario in transfer matrix: {scenario}.")
        if windows["window_index"].duplicated().any():
            raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
        test = folds.loc[folds["part"] == "test"].merge(
            windows[["window_index", "y_true"]],
            on="window_index",
            how="left",
            validate="many_to_one",
        )
        if test.empty or test[["y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"{job_dir}: incomplete test predictions for transfer matrix.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")
        if test["y_true"].nunique() != 2:
            raise ValueError(f"{job_dir}: expected exactly two tested classes.")
        scores[scenario] = balanced_accuracy_score(test["y_true"], test["y_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), layout="constrained")
    cmap = plt.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#F1F3F5")
    image = None
    for ax, (title, sides, transfer_protocol) in zip(axes, matrix_specs, strict=True):
        labels, compositions = zip(*sides, strict=True)
        values = np.full((len(compositions), len(compositions)), np.nan)
        for row, source in enumerate(compositions):
            for column, target in enumerate(compositions):
                protocol = "baseline" if row == column else transfer_protocol
                scenario = (protocol, source, target)
                if scenario in scores:
                    values[row, column] = scores[scenario]

        image = ax.imshow(values, cmap=cmap, vmin=0.4, vmax=0.7, aspect="equal")
        for row in range(len(compositions)):
            for column in range(len(compositions)):
                baseline = row == column
                label = (
                    f"{values[row, column]:.2f}" + ("\nbaseline" if baseline else "")
                    if not np.isnan(values[row, column])
                    else "—"
                )
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8 if len(compositions) > 6 else 9,
                    fontweight="bold" if baseline and not np.isnan(values[row, column]) else "normal",
                    color="#6B7280" if np.isnan(values[row, column]) else "#111827",
                )
                if baseline and not np.isnan(values[row, column]):
                    ax.add_patch(
                        Rectangle(
                            (column - 0.5, row - 0.5), 1, 1,
                            fill=False,
                            edgecolor="#1E293B",
                            linewidth=2.2,
                        )
                    )
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Target (test)", fontweight="bold")
        ax.set_ylabel("Source (train)", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.83, pad=0.03)
    colorbar.set_label("Balanced accuracy", fontweight="bold")
    fig.suptitle(
        "Directed zero-shot transfer: diagonal cells are within-composition baselines when available",
        fontweight="bold",
    )
    fig.text(0.5, 0.01, "Gray cells were not evaluated in the sweep.", ha="center", color="#6B7280", fontsize=9)
    return fig


def craft_cross_subject_model_comparison_figure(analysis_input_dir: Path) -> Figure:
    """Build the separate Dataset A/B cross-subject decoder comparison figure."""
    decoder_specs = (
        ("logistic_regression", "Logistic Regression", "#4E79A7"),
        ("xgboost", "XGBoost", "#F28E2B"),
        ("eegnet", "EEGNet", "#E15759"),
        ("shallownet", "ShallowNet", "#76B7B2"),
        ("eegconformer", "EEGConformer", "#59A14F"),
    )
    expected_scenarios = {
        (decoder, dataset): recipe
        for decoder, _, _ in decoder_specs
        for dataset, recipe in (("distinguishing", DISTINGUISHING), ("sam40", SAM40_FULL))
    }
    subject_scores_by_scenario: dict[tuple[str, str], np.ndarray] = {}

    for job_dir in sorted(analysis_input_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        config_path = job_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        if str(cfg.validation_strategy.name) != "cross_subject":
            continue

        decoder = decoder_name(cfg)
        if decoder not in {name for name, _, _ in decoder_specs}:
            continue
        if "preparation" in cfg and "source" in cfg.preparation:
            dataset_recipe = cfg.preparation.source
        elif "preparation" in cfg and "name" in cfg.preparation:
            dataset_recipe = cfg.preparation
        else:
            raise ValueError(f"{job_dir}: config has no preparation identity.")
        dataset = str(dataset_recipe.name)
        scenario = (decoder, dataset)
        if scenario not in expected_scenarios:
            raise ValueError(f"Unexpected cross-subject scenario in {job_dir}: {scenario}.")
        if _side_key(dataset_recipe) != expected_scenarios[scenario]:
            raise ValueError(f"{job_dir}: unexpected dataset composition for {scenario}.")
        if scenario in subject_scores_by_scenario:
            raise ValueError(f"Duplicate cross-subject scenario: {scenario}.")

        windows_path = job_dir / "windows.parquet"
        folds_path = job_dir / "folds.parquet"
        missing = [path.name for path in (windows_path, folds_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{job_dir} is incomplete: missing {missing}.")

        windows = pd.read_parquet(windows_path)
        folds = pd.read_parquet(folds_path)
        if windows["window_index"].duplicated().any():
            raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
        test = folds.loc[folds["part"] == "test"].merge(
            windows[["window_index", "subject_id", "y_true"]],
            on="window_index",
            how="left",
            validate="many_to_one",
        )
        if test.empty:
            raise ValueError(f"{job_dir}: the job has no test predictions.")
        if test[["subject_id", "y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"{job_dir}: test predictions reference missing or incomplete windows.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")
        if (test.groupby("fold")["subject_id"].nunique() != 1).any():
            raise ValueError(f"{job_dir}: a cross-subject fold tests more than one target subject.")
        if (test.groupby("subject_id")["fold"].nunique() != 1).any():
            raise ValueError(f"{job_dir}: a target subject occurs in more than one test fold.")

        subject_scores = []
        for subject_id, subject_test in test.groupby("subject_id", sort=True):
            if subject_test["y_true"].nunique() != 2:
                raise ValueError(f"{job_dir}: target subject {subject_id} does not contain two classes.")
            subject_scores.append(
                balanced_accuracy_score(subject_test["y_true"], subject_test["y_pred"])
            )
        subject_scores_by_scenario[scenario] = np.asarray(subject_scores, dtype=float)

    expected = set(expected_scenarios)
    actual = set(subject_scores_by_scenario)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Cross-subject model comparison needs one complete job for each decoder and dataset; "
            f"missing={missing}, unexpected={unexpected}."
        )

    fig, ax = plt.subplots(figsize=(5.2, 3.5), layout="constrained")
    group_centres = np.array((0.0, 1.45))
    bar_width = 0.18
    offsets = (np.arange(len(decoder_specs)) - (len(decoder_specs) - 1) / 2) * (bar_width + 0.015)
    rng = np.random.default_rng(42)

    for decoder_index, (decoder, label, colour) in enumerate(decoder_specs):
        means, lower_errors, upper_errors = [], [], []
        for dataset in ("distinguishing", "sam40"):
            scores = subject_scores_by_scenario[(decoder, dataset)]
            mean = scores.mean()
            bootstrap_means = rng.choice(scores, size=(10_000, len(scores)), replace=True).mean(axis=1)
            low, high = np.quantile(bootstrap_means, (0.025, 0.975))
            means.append(mean)
            lower_errors.append(mean - low)
            upper_errors.append(high - mean)
        ax.bar(
            group_centres + offsets[decoder_index],
            means,
            width=bar_width,
            yerr=np.array((lower_errors, upper_errors)),
            label=label,
            color=colour,
            edgecolor="#24354F",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#24354F"},
        )

    ax.axhline(0.5, color="#7C8AA0", linestyle=(0, (3, 3)), linewidth=0.9, zorder=0)
    ax.text(group_centres[-1] + 0.58, 0.515, "chance", color="#62718A", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(group_centres, ("Dataset A", "Dataset B"), fontweight="bold")
    ax.set_ylabel("Balanced accuracy", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#D9E0EA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=7.5)
    return fig


def craft_scenario_by_decoder_slope_figure(scenario_glob: str) -> Figure:
    """Build the four-panel figure that reads every scenario across the decoders.

    The all-scenarios figure asks how far a scenario falls; this one asks whether
    that fall belongs to the data or to the model reading it. The five decoders
    become the horizontal steps and each scenario walks across them as one line,
    so a line that stays flat is a scenario whose difficulty no decoder undoes.
    Panels follow the protocol families, because comparing a baseline line with a
    cross-dataset line inside one frame is what makes both unreadable.
    """
    decoder_specs = (
        ("logistic_regression", "Logistic\nRegression"),
        ("xgboost", "XGBoost"),
        ("eegnet", "EEGNet"),
        ("shallownet", "ShallowNet"),
        ("eegconformer", "EEG\nConformer"),
    )
    scenario_labels = (
        "Stratified K-fold(Full A)",
        "Stratified K-fold(Full B)",
        "Stratified K-fold(B: Stroop)",
        "Stratified K-fold(B: Arithmetic)",
        "Stratified K-fold(B: Mirror)",
        "Leave-one-subject-out(Full A)",
        "Leave-one-subject-out(Full B)",
        "Leave-one-session-out(Full A)",
        "Stroop → Arithmetic",
        "Stroop → Mirror",
        "Arithmetic → Stroop",
        "Arithmetic → Mirror",
        "Mirror → Stroop",
        "Mirror → Arithmetic",
        "Arithmetic+Mirror → Stroop",
        "Stroop+Mirror → Arithmetic",
        "Stroop+Arithmetic → Mirror",
        "Stroop → Arithmetic+Mirror",
        "Arithmetic → Stroop+Mirror",
        "Mirror → Stroop+Arithmetic",
        "Full A → Full B",
        "Full B → Full A",
        "Full A → Stroop",
        "Stroop → Full A",
        "Full A → Arithmetic",
        "Arithmetic → Full A",
        "Full A → Mirror",
        "Mirror → Full A",
        "Full A → Stroop+Arithmetic",
        "Stroop+Arithmetic → Full A",
        "Full A → Stroop+Mirror",
        "Stroop+Mirror → Full A",
        "Full A → Arithmetic+Mirror",
        "Arithmetic+Mirror → Full A",
    )
    if len(scenario_labels) != len(SCENARIO_ORDER):
        raise RuntimeError("Slope-figure scenario labels must match the scenario order.")

    # Hue carries the grouping a panel is about, lightness separates the lines
    # inside one group: the same two channels the mockup used, in HLS.
    hue_angles = {"rust": 0.055, "olive": 0.270, "teal": 0.490, "cyan": 0.570, "violet": 0.790}
    tone = lambda hue, level: colorsys.hls_to_rgb(hue_angles[hue], 0.40 + 0.34 * level, 0.55)
    # (panel title, per-scenario (hue, lightness)) in scenario order
    panel_specs = (
        (
            "Baseline",
            (("rust", 0.34), ("olive", 0.72), ("teal", 0.22), ("cyan", 0.72), ("violet", 0.34)),
        ),
        (
            "Cross-subject, Cross-session",
            (("rust", 0.26), ("rust", 0.66), ("cyan", 0.66)),
        ),
        (
            "Cross-task (Dataset B; single/double-task transfers)",
            (
                ("cyan", 0.18), ("olive", 0.18), ("rust", 0.18), ("olive", 0.50),
                ("rust", 0.50), ("cyan", 0.50), ("rust", 0.80), ("cyan", 0.80),
                ("olive", 0.80), ("violet", 0.18), ("violet", 0.50), ("violet", 0.80),
            ),
        ),
        (
            "Cross-dataset",
            tuple(
                ("rust" if index % 2 == 0 else "cyan", (index // 2) / 6)
                for index in range(14)
            ),
        ),
    )
    panel_sizes = tuple(len(colours) for _, colours in panel_specs)
    if sum(panel_sizes) != len(SCENARIO_ORDER):
        raise RuntimeError("Slope-figure panels must cover every scenario exactly once.")

    job_dirs = discover_scenario_results(scenario_glob)
    expected_result_count = len(SCENARIO_ORDER) * len(decoder_specs)
    if len(job_dirs) != expected_result_count:
        raise ValueError(
            f"A scenario-by-decoder figure needs {expected_result_count} scenario results "
            f"({len(SCENARIO_ORDER)} scenarios x {len(decoder_specs)} decoders), got {len(job_dirs)}."
        )

    known_decoders = {decoder for decoder, _ in decoder_specs}
    scores: dict[tuple[str, tuple[object, ...]], float] = {}
    for job_dir in job_dirs:
        cfg, windows, folds = read_scenario_result(job_dir)
        decoder = decoder_name(cfg)
        if decoder not in known_decoders:
            raise ValueError(f"{job_dir}: unsupported decoder for article figure: {decoder}.")
        scenario = scenario_key(cfg)
        if scenario not in SCENARIO_ORDER:
            raise ValueError(f"Unexpected scenario in {job_dir}: {scenario}.")
        if (decoder, scenario) in scores:
            raise ValueError(f"Duplicate result for decoder {decoder} and scenario {scenario}.")

        if windows["window_index"].duplicated().any():
            raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
        test = folds.loc[folds["part"] == "test"].merge(
            windows[["window_index", "y_true"]],
            on="window_index",
            how="left",
            validate="many_to_one",
        )
        if test.empty:
            raise ValueError(f"{job_dir}: the job has no test predictions.")
        if test[["y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"{job_dir}: test predictions reference missing or incomplete windows.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")
        if test["y_true"].nunique() != 2:
            raise ValueError(f"{job_dir}: expected exactly two tested classes.")
        scores[(decoder, scenario)] = balanced_accuracy_score(test["y_true"], test["y_pred"])

    missing = [
        (decoder, scenario)
        for decoder, _ in decoder_specs
        for scenario in SCENARIO_ORDER
        if (decoder, scenario) not in scores
    ]
    if missing:
        raise ValueError(f"Scenario set does not match the article figure; missing={missing}.")

    curves = np.array(
        [[scores[(decoder, scenario)] for decoder, _ in decoder_specs] for scenario in SCENARIO_ORDER]
    )
    padding = max(0.01, np.ptp(curves) * 0.08)
    y_low = max(0.0, curves.min() - padding)
    y_high = min(1.0, curves.max() + padding)
    if not y_low < 0.5 < y_high:
        y_low, y_high = min(y_low, 0.48), max(y_high, 0.52)

    def stack(anchors: np.ndarray, gap: float) -> np.ndarray:
        """Spread label positions apart while keeping them next to their lines."""
        positions = anchors.astype(float).copy()
        for _ in range(200):
            moved = False
            for index in range(1, len(positions)):
                overlap = gap - (positions[index - 1] - positions[index])
                if overlap > 0:
                    positions[index - 1] += overlap / 2
                    positions[index] -= overlap / 2
                    moved = True
            positions -= max(0.0, positions[0] - (y_high - gap / 2))
            positions += max(0.0, (y_low + gap / 2) - positions[-1])
            if not moved:
                break
        return positions

    decoder_positions = np.arange(len(decoder_specs), dtype=float)
    label_x = len(decoder_specs) - 1 + 0.62
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 11.5), layout="constrained")
    scenario_start = 0
    for ax, (title, colours) in zip(axes.ravel(), panel_specs, strict=True):
        panel_start = scenario_start
        panel = range(panel_start, panel_start + len(colours))
        scenario_start += len(colours)

        ax.axhline(0.5, color="#7C8AA0", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
        ax.text(0.0, 0.5, "chance", color="#62718A", ha="left", va="bottom", fontsize=8)
        for scenario_index, (hue, level) in zip(panel, colours, strict=True):
            ax.plot(
                decoder_positions,
                curves[scenario_index],
                color=tone(hue, level),
                linewidth=1.7,
                solid_capstyle="round",
                zorder=3,
            )

        # The key stands where the lines end, so a line is read without a legend
        # lookup; ties are pulled apart only as far as the panel allows.
        order = sorted(panel, key=lambda index: -curves[index, -1])
        anchors = np.array([curves[index, -1] for index in order])
        label_positions = stack(anchors, gap=(y_high - y_low) / 26)
        for scenario_index, anchor, position in zip(order, anchors, label_positions, strict=True):
            colour = tone(*colours[scenario_index - panel_start])
            ax.plot(
                (len(decoder_specs) - 1, label_x - 0.36, label_x - 0.12),
                (anchor, position, position),
                color=colour,
                linewidth=0.9,
                alpha=0.55,
                clip_on=False,
                zorder=2,
            )
            ax.text(
                label_x,
                position,
                scenario_labels[scenario_index],
                color="#3D4450",
                ha="left",
                va="center",
                fontsize=8,
                clip_on=False,
            )

        ax.text(0.0, 1.075, f"{title}  ({len(colours)} scenarios)", transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")
        ax.set_xlim(-0.3, label_x + 0.05)
        ax.set_ylim(y_low, y_high)
        ax.set_xticks(decoder_positions, [label for _, label in decoder_specs], fontsize=9)
        ax.set_ylabel("Balanced accuracy", fontweight="bold", fontsize=9)
        ax.grid(axis="y", color="#EDEFF4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8, length=0)
    return fig
