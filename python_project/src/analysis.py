"""Article-table calculations derived from completed experiment jobs."""

from __future__ import annotations

from collections import Counter
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
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
    ("cross_subject", DISTINGUISHING, DISTINGUISHING),
    ("cross_subject", SAM40_FULL, SAM40_FULL),
    ("cross_session", DISTINGUISHING, DISTINGUISHING),
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
        # The sweep keeps preparation snapshots alongside execution jobs.  They
        # are intentionally not results and carry neither predictions nor a
        # scenario config, even when a broad two-level glob reaches inside them.
        if any(part.startswith("_") for part in job_dir.parts):
            continue
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
    sides = {
        "distinguishing": DISTINGUISHING,
        "sam40_all": SAM40_FULL,
        "sam40_stroop": SAM40_STROOP,
        "sam40_arithmetic": SAM40_ARITHMETIC,
        "sam40_mirror": SAM40_MIRROR,
        "sam40_stroop_arithmetic": SAM40_STROOP_ARITHMETIC,
        "sam40_stroop_mirror": SAM40_STROOP_MIRROR,
        "sam40_arithmetic_mirror": SAM40_ARITHMETIC_MIRROR,
    }
    try:
        return sides[str(side.name)]
    except KeyError as error:
        raise ValueError(f"Unknown preparation side: {side.name}.") from error


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


def craft_transfer_degradation_matrix_figure(scenario_glob: str) -> Figure:
    """Build the all-decoder transfer matrix with target-specific baselines."""
    sides = (
        ("Stroop", SAM40_STROOP),
        ("Arithmetic", SAM40_ARITHMETIC),
        ("Mirror", SAM40_MIRROR),
        ("Full A", DISTINGUISHING),
        ("Full B", SAM40_FULL),
    )
    decoder_specs = (
        ("logistic_regression", "Logistic\nregression"),
        ("xgboost", "XGBoost"),
        ("eegnet", "EEGNet"),
        ("shallownet", "ShallowNet"),
        ("eegconformer", "EEGConformer"),
    )
    expected_scenarios = {
        ("baseline", side, side) for _, side in sides
    } | {
        ("cross_task", source, target)
        for _, source in sides[:3]
        for _, target in sides[:3]
        if source != target
    } | {
        ("cross_dataset", source, target)
        for _, source in sides
        for _, target in sides
        if source[0] != target[0]
    }

    job_dirs = discover_scenario_results(scenario_glob)
    expected_result_count = len(SCENARIO_ORDER) * len(decoder_specs)
    if len(job_dirs) != expected_result_count:
        raise ValueError(
            f"A transfer-performance matrix needs {expected_result_count} scenario results, "
            f"got {len(job_dirs)}."
        )

    known_decoders = {decoder for decoder, _ in decoder_specs}
    scores: dict[tuple[str, tuple[object, ...]], float] = {}
    for job_dir in job_dirs:
        cfg, windows, folds = read_scenario_result(job_dir)
        decoder = decoder_name(cfg)
        if decoder not in known_decoders:
            raise ValueError(f"{job_dir}: unsupported decoder for transfer-performance matrix: {decoder}.")
        scenario = scenario_key(cfg)
        if (decoder, scenario) in scores:
            raise ValueError(f"Duplicate scenario in transfer-performance matrix: {decoder}, {scenario}.")
        if scenario not in SCENARIO_ORDER:
            raise ValueError(f"Unexpected scenario in {job_dir}: {scenario}.")
        if windows["window_index"].duplicated().any():
            raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
        test = folds.loc[folds["part"] == "test"].merge(
            windows[["window_index", "y_true"]],
            on="window_index",
            how="left",
            validate="many_to_one",
        )
        if test.empty or test[["y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"{job_dir}: incomplete test predictions for transfer-performance matrix.")
        if not test["window_index"].is_unique:
            raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")
        if test["y_true"].nunique() != 2:
            raise ValueError(f"{job_dir}: expected exactly two tested classes.")
        scores[(decoder, scenario)] = balanced_accuracy_score(test["y_true"], test["y_pred"])

    missing = [
        (decoder, scenario)
        for decoder, _ in decoder_specs
        for scenario in expected_scenarios
        if (decoder, scenario) not in scores
    ]
    if missing:
        raise ValueError(f"Transfer-performance matrix is missing scenarios: {missing}.")

    labels, compositions = zip(*sides, strict=True)
    model_names = tuple(label for _, label in decoder_specs)
    row_count = len(sides) * len(decoder_specs)
    matrix = np.full((row_count, len(sides)), np.nan)
    protocols = np.full(matrix.shape, "", dtype=object)
    for side_index, (_, source) in enumerate(sides):
        for decoder_index, (decoder, _) in enumerate(decoder_specs):
            row = side_index * len(decoder_specs) + decoder_index
            for column, target in enumerate(compositions):
                if source == target:
                    matrix[row, column] = scores[(decoder, ("baseline", target, target))]
                    protocols[row, column] = "baseline"
                else:
                    scenario = next(
                        (
                            (protocol, source, target)
                            for protocol in ("cross_task", "cross_dataset")
                            if (decoder, (protocol, source, target)) in scores
                        ),
                        None,
                    )
                    if scenario:
                        matrix[row, column] = scores[(decoder, scenario)]
                        protocols[row, column] = scenario[0]

    # Summary values live at the source level.  For every displayed direction,
    # average the five decoders first, then compare it with the equally averaged
    # target baseline; individual decoder variation belongs only in the matrix.
    summary_values = np.full((len(sides), 2), np.nan)
    for source_index, (_, source) in enumerate(sides):
        for summary_column, protocol in enumerate(("cross_task", "cross_dataset")):
            transfer_means, baseline_means = [], []
            for target in compositions:
                scenario = (protocol, source, target)
                if scenario not in expected_scenarios:
                    continue
                transfer_means.append(
                    np.mean([scores[(decoder, scenario)] for decoder, _ in decoder_specs])
                )
                baseline_means.append(
                    np.mean(
                        [scores[(decoder, ("baseline", target, target))] for decoder, _ in decoder_specs]
                    )
                )
            if transfer_means:
                summary_values[source_index, summary_column] = (
                    np.mean(transfer_means) - np.mean(baseline_means)
                ) * 100

    figure_width, figure_height = 13.6, 8.8
    matrix_left, matrix_bottom = 0.22, 0.12
    matrix_width, matrix_height = 0.40, 0.78
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes((matrix_left, matrix_bottom, matrix_width, matrix_height))
    colorbar_ax = fig.add_axes((0.645, matrix_bottom, 0.016, matrix_height))
    summary_ax = fig.add_axes((0.69, matrix_bottom, 0.14, matrix_height))
    cmap = plt.colormaps["Reds"]
    normalization = Normalize(vmin=0.0, vmax=1.0)

    for row in range(row_count):
        for column in range(len(sides)):
            value = matrix[row, column]
            if np.isnan(value):
                facecolor, text, text_colour = "#F2F4F6", "—", "#667085"
            elif row // len(decoder_specs) == column:
                facecolor, text, text_colour = "#FFFFFF", f"{value * 100:.1f}", "#263341"
            else:
                facecolor = cmap(normalization(value))
                text = f"{value * 100:.1f}"
                text_colour = "white" if value >= 0.62 else "#263341"
            ax.add_patch(Rectangle((column, row), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=0.9))
            ax.text(column + 0.5, row + 0.5, text, ha="center", va="center",
                    fontsize=9.5, fontweight="bold" if not np.isnan(value) else "normal", color=text_colour)
    ax.set_xlim(0, len(sides))
    ax.set_ylim(row_count, 0)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(len(sides)) + 0.5, labels)
    ax.set_yticks(np.arange(row_count) + 0.5, model_names * len(sides))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=8.5, pad=5)
    ax.set_xlabel("Target (test)", fontweight="bold", labelpad=15, fontsize=11)
    ax.xaxis.set_label_position("top")
    for source_index, label in enumerate(labels):
        centre = source_index * len(decoder_specs) + len(decoder_specs) / 2
        ax.text(-1.75, centre, label, ha="right", va="center", fontsize=11, fontweight="bold", clip_on=False)
        if source_index:
            ax.axhline(source_index * len(decoder_specs), color="#8C98A8", linewidth=1.2, zorder=4)
            divider = ax.hlines(
                source_index * len(decoder_specs),
                xmin=-2.5,
                xmax=0,
                color="#8C98A8",
                linewidth=1.2,
                zorder=4,
            )
            divider.set_clip_on(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#344054")
        spine.set_linewidth(1.1)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Balanced accuracy (%)", fontsize=8, labelpad=8)
    colorbar.set_ticks(np.linspace(0, 1, 5), labels=("0", "25", "50", "75", "100"))
    colorbar.ax.tick_params(labelsize=8)

    for row in range(len(sides)):
        for column in range(2):
            value = summary_values[row, column]
            facecolor = "#F2F4F6" if np.isnan(value) else "#FFFFFF"
            text = "—" if np.isnan(value) else f"{value:.1f}"
            y_position = row * len(decoder_specs)
            summary_ax.add_patch(Rectangle((column, y_position), 1, len(decoder_specs), facecolor=facecolor, edgecolor="#D0D5DD", linewidth=0.9))
            summary_ax.text(column + 0.5, y_position + len(decoder_specs) / 2, text, ha="center", va="center", fontsize=12,
                            fontweight="bold" if not np.isnan(value) else "normal", color="#263341")
    summary_ax.set_xlim(0, 2)
    summary_ax.set_ylim(row_count, 0)
    summary_ax.set_xticks((0.5, 1.5), ("Mean Δ\ncross-task", "Mean Δ\ncross-dataset"))
    summary_ax.xaxis.tick_top()
    summary_ax.tick_params(axis="x", length=0, labelsize=10, pad=10)
    summary_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in summary_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#344054")
        spine.set_linewidth(1.1)

    return fig


def craft_all_scenarios_model_comparison_figure(scenario_glob: str) -> Figure:
    """Build one all-scenarios accuracy figure with every decoder and their mean."""
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
    decoder_specs = (
        ("logistic_regression", "Logistic Regression", "#1F77B4"),
        ("xgboost", "XGBoost", "#FF7F0E"),
        ("eegnet", "EEGNet", "#D62728"),
        ("shallownet", "ShallowNet", "#9467BD"),
        ("eegconformer", "EEGConformer", "#2CA02C"),
    )
    if len(validation_method_labels) != len(SCENARIO_ORDER):
        raise RuntimeError("Figure validation-method labels must match the scenario order.")

    job_dirs = discover_scenario_results(scenario_glob)
    expected_result_count = len(SCENARIO_ORDER) * len(decoder_specs)
    if len(job_dirs) != expected_result_count:
        raise ValueError(
            f"An all-models figure needs {expected_result_count} scenario results "
            f"({len(SCENARIO_ORDER)} scenarios x {len(decoder_specs)} decoders), got {len(job_dirs)}."
        )

    known_decoders = {decoder for decoder, _, _ in decoder_specs}
    scores: dict[tuple[str, tuple[object, ...]], np.ndarray] = {}
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
            subject_scores.append(balanced_accuracy_score(subject_test["y_true"], subject_test["y_pred"]))
        scores[(decoder, scenario)] = np.asarray(subject_scores, dtype=float)

    missing = [
        (decoder, scenario)
        for decoder, _, _ in decoder_specs
        for scenario in SCENARIO_ORDER
        if (decoder, scenario) not in scores
    ]
    if missing:
        raise ValueError(f"Scenario set does not match the article figure; missing={missing}.")

    model_means = np.array(
        [
            [scores[(decoder, scenario)].mean() for decoder, _, _ in decoder_specs]
            for scenario in SCENARIO_ORDER
        ]
    )
    average_means = model_means.mean(axis=1)
    lower_errors, upper_errors = [], []
    rng = np.random.default_rng(42)
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        bootstrap_means = np.stack(
            [
                rng.choice(scores[(decoder, scenario)], size=(10_000, len(scores[(decoder, scenario)])), replace=True)
                .mean(axis=1)
                for decoder, _, _ in decoder_specs
            ]
        ).mean(axis=0)
        low, high = np.quantile(bootstrap_means, (0.025, 0.975))
        mean = average_means[scenario_index]
        lower_errors.append(mean - low)
        upper_errors.append(high - mean)

    method_y_positions, protocol_y_positions, y_positions, y_labels = [], [], [], []
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

    fig, ax = plt.subplots(figsize=(12.2, 14.6))
    offsets = np.linspace(-0.22, 0.22, len(decoder_specs))
    for decoder_index, (_, label, colour) in enumerate(decoder_specs):
        ax.scatter(
            model_means[:, decoder_index],
            np.asarray(method_y_positions) + offsets[decoder_index],
            s=38,
            color=colour,
            alpha=0.65,
            edgecolor="#24354F",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
    ax.errorbar(
        average_means,
        method_y_positions,
        xerr=np.array((lower_errors, upper_errors)),
        fmt="o",
        color="#1F2937",
        ecolor="#1F2937",
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=0.9,
        elinewidth=1.15,
        capsize=3,
        label="Average",
        zorder=5,
    )
    ax.axvline(0.5, color="#7C8AA0", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=10,
            transform=ax.get_xaxis_transform())
    ax.set_yticks(y_positions, ("",) * len(y_positions))
    ax.set_ylim(min(y_positions) - 0.8, max(y_positions) + 0.8)
    plotted_scores = np.concatenate((model_means.ravel(), np.asarray(average_means) - lower_errors,
                                    np.asarray(average_means) + upper_errors, np.array((0.5,))))
    score_padding = max(0.01, np.ptp(plotted_scores) * 0.12)
    ax.set_xlim(max(0.0, plotted_scores.min() - score_padding), min(1.0, plotted_scores.max() + score_padding))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color="#E0E6EF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    protocol_label_set = dict(protocol_labels)
    for y_position, label in zip(y_positions, y_labels, strict=True):
        ax.text(-0.27, y_position, label, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontweight="bold" if label in protocol_label_set else "normal",
                fontsize=11 if label in protocol_label_set else 10)
    for protocol_y_position in protocol_y_positions:
        ax.axhline(protocol_y_position, color="#7A7A7A", alpha=0.65, linewidth=1.3, zorder=1)
    legend = ax.legend(
        ncols=6,
        loc="upper center",
        bbox_to_anchor=(0.37, 1.054),
        frameon=False,
        fontsize=11,
        markerscale=1.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    plt.setp(legend.get_texts(), fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.9385, bottom=0.02)
    return fig
