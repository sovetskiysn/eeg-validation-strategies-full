"""Article-table calculations derived from completed experiment jobs."""

from __future__ import annotations

from collections import Counter
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
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


# =============================================================================
# Article constants
# =============================================================================
# Article-level order of scenario rows. A dataset side is identified by its
# saved dataset name and selected BIDS tasks.
DISTINGUISHING = ("distinguishing", ("attention",))
SAM40_FULL = ("sam40", ("arithmetic", "mirror", "relax", "stroop"))
SAM40_STROOP = ("sam40", ("relax", "stroop"))
SAM40_ARITHMETIC = ("sam40", ("arithmetic", "relax"))
SAM40_MIRROR = ("sam40", ("mirror", "relax"))

# Every SAM-40 side the sweep prepared. Cross-task transfer is run in both
# directions between all of them, so naming them once here keeps the scenario
# order and the transfer figure from spelling the same twelve pairs out twice.
SAM40_SIDES = (SAM40_FULL, SAM40_STROOP, SAM40_ARITHMETIC, SAM40_MIRROR)

SCENARIO_ORDER = (
    ("baseline", DISTINGUISHING, DISTINGUISHING),
    ("baseline", SAM40_FULL, SAM40_FULL),
    ("baseline", SAM40_STROOP, SAM40_STROOP),
    ("baseline", SAM40_ARITHMETIC, SAM40_ARITHMETIC),
    ("baseline", SAM40_MIRROR, SAM40_MIRROR),
    # Cross-subject transfer is reported on the full composition of each dataset
    # only: a leave-one-subject-out run inside a single SAM-40 task would name
    # the participant shift after a scenario that also narrows the task set, so
    # the three single-task sides stay out of this protocol.
    ("cross_subject", DISTINGUISHING, DISTINGUISHING),
    ("cross_subject", SAM40_FULL, SAM40_FULL),
    ("cross_task", SAM40_FULL, SAM40_STROOP),
    ("cross_task", SAM40_FULL, SAM40_ARITHMETIC),
    ("cross_task", SAM40_FULL, SAM40_MIRROR),
    ("cross_task", SAM40_STROOP, SAM40_FULL),
    ("cross_task", SAM40_STROOP, SAM40_ARITHMETIC),
    ("cross_task", SAM40_STROOP, SAM40_MIRROR),
    ("cross_task", SAM40_ARITHMETIC, SAM40_FULL),
    ("cross_task", SAM40_ARITHMETIC, SAM40_STROOP),
    ("cross_task", SAM40_ARITHMETIC, SAM40_MIRROR),
    ("cross_task", SAM40_MIRROR, SAM40_FULL),
    ("cross_task", SAM40_MIRROR, SAM40_STROOP),
    ("cross_task", SAM40_MIRROR, SAM40_ARITHMETIC),
    ("cross_dataset", DISTINGUISHING, SAM40_FULL),
    ("cross_dataset", SAM40_FULL, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_STROOP),
    ("cross_dataset", SAM40_STROOP, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_ARITHMETIC),
    ("cross_dataset", SAM40_ARITHMETIC, DISTINGUISHING),
    ("cross_dataset", DISTINGUISHING, SAM40_MIRROR),
    ("cross_dataset", SAM40_MIRROR, DISTINGUISHING),
)

# Article-level order of the dataset-composition table's rows. A composition is
# named by the same side identity every scenario is keyed by, so the table and
# the scenarios describe the same five prepared window sets.
ARTICLE_COMPOSITIONS = (DISTINGUISHING, *SAM40_SIDES)

# One entry per article model, in the order every table and figure shows them.
# `full_name` spells the model out in a LaTeX caption; `short_name` labels the
# narrow matrix column and the figure legends, where the full one does not fit,
# and equals the full name wherever it already is short enough; `colour`
# identifies the model on every figure.
ARTICLE_DECODERS = {
    "logistic_regression": {
        "full_name": "Logistic Regression",
        "short_name": "Logistic reg",
        "colour": "#1F77B4",
    },
    "xgboost": {"full_name": "XGBoost", "short_name": "XGBoost", "colour": "#FF7F0E"},
    "eegnet": {"full_name": "EEGNet", "short_name": "EEGNet", "colour": "#D62728"},
    "shallownet": {
        "full_name": "ShallowFBCSPNet",
        "short_name": "ShallowNet",
        "colour": "#9467BD",
    },
    "eegconformer": {
        "full_name": "EEGConformer",
        "short_name": "EEGConformer",
        "colour": "#2CA02C",
    },
}


# =============================================================================
# Metric
# =============================================================================
# Every article artifact reports one metric, and it is calculated in exactly one
# place. Reporting another metric means writing a second function of this shape
# and pointing ARTICLE_METRIC at it: the tables, the figure axes and the colour
# bar name the metric after the function itself, so no artifact needs editing.


def balanced_accuracy(subject_test: pd.DataFrame) -> float:
    """Score the held-out windows of one target participant."""
    return balanced_accuracy_score(subject_test["y_true"], subject_test["y_pred"])


ARTICLE_METRIC = balanced_accuracy
ARTICLE_METRIC_LABEL = ARTICLE_METRIC.__name__.replace("_", " ").capitalize()


# =============================================================================
# Reading a finished sweep
# =============================================================================


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


def article_scenario_results(
    scenario_glob: str,
) -> list[tuple[Path, object, pd.DataFrame, pd.DataFrame]]:
    """Return the read results of every direction the article actually reports.

    A sweep holds every direction that was executed, and the article reports a
    subset of them: an already finished sweep still holds `cross_session` and the
    per-task cross-subject runs whose recipes were dropped afterwards, and they
    sit in `results/` next to the directions the tables and figures describe.
    Deciding here what belongs to the article keeps that fact in one place -- an
    artifact asks for the directions it reports and never learns that the sweep
    held more.
    """
    wanted = set(SCENARIO_ORDER)
    results = []
    for job_dir in discover_scenario_results(scenario_glob):
        cfg, windows, folds = read_scenario_result(job_dir)
        if scenario_key(cfg) in wanted:
            results.append((job_dir, cfg, windows, folds))
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

    known_sides = {
        "distinguishing": DISTINGUISHING,
        "sam40_all": SAM40_FULL,
        "sam40_stroop": SAM40_STROOP,
        "sam40_arithmetic": SAM40_ARITHMETIC,
        "sam40_mirror": SAM40_MIRROR,
    }
    identities = []
    for side in sides:
        if str(side.name) not in known_sides:
            raise ValueError(f"Unknown preparation side: {side.name}.")
        identities.append(known_sides[str(side.name)])
    return (str(cfg.validation_strategy.name), *identities)


def decoder_name(cfg) -> str:
    """Read a decoder name from either generation of saved run configuration."""
    if "pipeline" in cfg and "name" in cfg.pipeline:
        return str(cfg.pipeline.name)
    return str(cfg.pipeline_components.model.name)


def held_out_predictions(
    job_dir: Path, windows: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    """Return one validated held-out prediction per window for a logical result."""
    if windows["window_index"].duplicated().any():
        raise ValueError(f"{job_dir}: windows.parquet has duplicate window indices.")
    test = folds.loc[folds["part"] == "test"].merge(
        windows[["window_index", "subject_id", "y_true"]],
        on="window_index",
        how="left",
        validate="many_to_one",
    )
    required_columns = ["subject_id", "y_true", "y_pred", "score"]
    if test.empty or test[required_columns].isna().any().any():
        raise ValueError(f"{job_dir}: incomplete held-out predictions.")
    if not test["window_index"].is_unique:
        raise ValueError(f"{job_dir}: a test window occurs in more than one fold.")
    if test["y_true"].nunique() != 2:
        raise ValueError(f"{job_dir}: expected exactly two tested classes.")
    return test.astype({"y_true": "int64", "y_pred": "int64"})


def composition_windows(cfg, windows: pd.DataFrame, folds: pd.DataFrame) -> dict:
    """Return the prepared windows of every dataset composition one result holds.

    A non-transfer result was prepared from a single composition and its whole
    window table is that composition. A transfer result carries both sides in one
    table, and the folds are what tell them apart: the source was trained on and
    the target was tested on, so the two window-index sets never mix. This is the
    only place that knows a result can describe two compositions -- the callers
    ask for compositions and never learn which protocol produced them.
    """
    _, source, target = scenario_key(cfg)
    if source == target:
        return {source: windows}

    sides = {}
    for part, side in (("train", source), ("test", target)):
        part_windows = windows.loc[windows["window_index"].isin(folds.loc[folds["part"] == part, "window_index"])]
        if part_windows.empty:
            raise ValueError(f"A transfer result has no {part} windows for composition {side}.")
        sides[side] = part_windows
    return sides


def participant_scores(test: pd.DataFrame, job_dir: Path) -> pd.Series:
    """Score every target participant of one result before they are averaged.

    The participant is the unit of aggregation everywhere in the article, so the
    metric is never calculated on pooled windows: that would weight a participant
    by how many recordings they contributed.
    """
    class_ids = sorted(test["y_true"].unique())
    scores = {}
    for subject_id, subject_test in test.groupby("subject_id", sort=True):
        if sorted(subject_test["y_true"].unique()) != class_ids:
            raise ValueError(
                f"{job_dir}: target participant {subject_id} does not contain both classes."
            )
        scores[subject_id] = ARTICLE_METRIC(subject_test)
    return pd.Series(scores, name=ARTICLE_METRIC.__name__).rename_axis("subject_id")


def collect_participant_scores(
    scenario_glob: str, scenarios: tuple[tuple[object, ...], ...]
) -> dict[tuple[str, tuple[object, ...]], pd.Series]:
    """Score every article decoder on each named scenario of one finished sweep.

    A sweep holds every direction that was run, while an article figure compares
    a subset of them, so a result outside `scenarios` is skipped rather than
    counted. What has to hold is that each named scenario was measured by every
    article decoder, and naming the absent pairs is a stronger statement than any
    count of result directories.
    """
    wanted = set(scenarios)
    scores: dict[tuple[str, tuple[object, ...]], pd.Series] = {}
    for job_dir in discover_scenario_results(scenario_glob):
        cfg, windows, folds = read_scenario_result(job_dir)
        scenario, decoder = scenario_key(cfg), decoder_name(cfg)
        if scenario not in wanted or decoder not in ARTICLE_DECODERS:
            continue
        if (decoder, scenario) in scores:
            raise ValueError(f"Duplicate result for decoder {decoder} and scenario {scenario}.")
        scores[(decoder, scenario)] = participant_scores(
            held_out_predictions(job_dir, windows, folds), job_dir
        )

    missing = [
        (decoder, scenario)
        for decoder in ARTICLE_DECODERS
        for scenario in scenarios
        if (decoder, scenario) not in scores
    ]
    if missing:
        raise ValueError(f"The sweep is missing scenario results: {missing}.")
    return scores


def bootstrap_average_interval(
    decoder_scores: list[pd.Series], rng: np.random.Generator
) -> tuple[float, float]:
    """Return the 95% interval of the decoder-averaged metric of one scenario.

    Resampling the target participants once per draw keeps the decoders paired:
    they were measured on the very same participants, and drawing separately per
    decoder would count that shared variation as if it were independent.
    """
    participant_ids = decoder_scores[0].index
    for scores in decoder_scores[1:]:
        if not scores.index.equals(participant_ids):
            raise ValueError(
                "Target participants differ between decoders; paired bootstrap is not defined."
            )
    sampled_indices = rng.integers(0, len(participant_ids), size=(10_000, len(participant_ids)))
    bootstrap_means = np.stack(
        [scores.to_numpy()[sampled_indices].mean(axis=1) for scores in decoder_scores]
    ).mean(axis=0)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(low), float(high)


# =============================================================================
# Article artifacts
# =============================================================================
# One craft function per artifact of the article, plus the writer that renders
# all of them from one finished sweep. Everything above serves these.


def craft_scenario_metrics_table(scenario_glob: str) -> str:
    """Fill the scenario-table template from one complete scenario-run glob."""
    # =============================================================================
    # Step 1: score every scenario this decoder was measured on
    # =============================================================================
    # This is the only artifact that reports more than the article metric: the
    # supporting columns exist to characterise the same predictions from a second
    # angle, so they are calculated here rather than on every read of a result.
    metrics_by_scenario: dict[tuple[str, str, str], tuple[str, str, str, str, str]] = {}
    decoder_names = set()
    for job_dir, cfg, windows, folds in article_scenario_results(scenario_glob):
        decoder = decoder_name(cfg)
        if decoder not in ARTICLE_DECODERS:
            raise ValueError(f"{job_dir}: unsupported decoder for article table: {decoder}.")
        decoder_names.add(decoder)
        scenario = scenario_key(cfg)
        if scenario in metrics_by_scenario:
            raise ValueError(f"Duplicate scenario for one decoder: {scenario}.")

        test = held_out_predictions(job_dir, windows, folds)
        scores = participant_scores(test, job_dir)
        class_ids = sorted(test["y_true"].unique())
        supporting = []
        for _, subject_test in test.groupby("subject_id", sort=True):
            _, recall, _, _ = precision_recall_fscore_support(
                subject_test["y_true"], subject_test["y_pred"], labels=class_ids, zero_division=0
            )
            supporting.append(
                {
                    "macro_f1": f1_score(
                        subject_test["y_true"], subject_test["y_pred"], average="macro", zero_division=0
                    ),
                    "mcc": matthews_corrcoef(subject_test["y_true"], subject_test["y_pred"]),
                    "roc_auc": roc_auc_score(subject_test["y_true"], subject_test["score"]),
                    "low_recall": recall[0],
                    "high_recall": recall[1],
                }
            )
        supporting = pd.DataFrame(supporting)
        metrics_by_scenario[scenario] = (
            *(
                "-" if pd.isna(value) else f"{value:.2f}"
                for value in (
                    scores.mean(),
                    supporting["macro_f1"].mean(),
                    supporting["mcc"].mean(),
                    supporting["roc_auc"].mean(),
                )
            ),
            f"{supporting['low_recall'].mean():.2f}/{supporting['high_recall'].mean():.2f}",
        )

    # =============================================================================
    # Step 2: fill the template rows in the article's scenario order
    # =============================================================================

    # Directions outside the article were filtered out on the way in, so what is
    # left to check is that none of the reported ones is absent.
    missing = list((Counter(SCENARIO_ORDER) - Counter(metrics_by_scenario.keys())).elements())
    if missing:
        raise ValueError(f"The sweep is missing scenario results for this decoder: {missing}.")
    if len(decoder_names) != 1:
        raise ValueError(f"A main scenario table needs one decoder, got {sorted(decoder_names)}.")
    decoder = decoder_names.pop()

    template = (LATEX_ARTIFACT_TEMPLATES_DIR / "scenario_metrics_table_template.tex").read_text()
    template = template.replace("MODEL_DISPLAY_NAME", ARTICLE_DECODERS[decoder]["full_name"]).replace(
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
            f"{LATEX_ARTIFACT_TEMPLATES_DIR / 'scenario_metrics_table_template.tex'} has {len(metric_lines)} metric rows; "
            f"expected {len(SCENARIO_ORDER)}."
        )

    for index, scenario in zip(metric_lines, SCENARIO_ORDER, strict=True):
        cells = lines[index].rstrip("\n").split("&")
        lines[index] = f"{cells[0]}&{cells[1]}& " + " & ".join(metrics_by_scenario[scenario]) + r" \\" + "\n"
    return "".join(lines)


def craft_dataset_composition_table(scenario_glob: str) -> str:
    """Fill the dataset-composition table from one complete scenario-run glob.

    The table describes the prepared data rather than any model, so a composition
    is measured wherever the sweep touched it and every further sighting has to
    agree exactly. Prepared data that differs between the runs that shared it
    would make the table describe none of them, which is why a disagreement is an
    error here and not a choice between two numbers.
    """
    # =============================================================================
    # Step 1: measure every composition the sweep prepared
    # =============================================================================
    stats_by_composition: dict[tuple[object, ...], tuple[str, str, str, str]] = {}
    for job_dir, cfg, windows, folds in article_scenario_results(scenario_glob):
        for composition, prepared in composition_windows(cfg, windows, folds).items():
            counts = prepared["y_true_name"].value_counts()
            if set(counts.index) != {"low_attention", "high_attention"}:
                raise ValueError(f"{job_dir}: composition {composition} is not a two-class window set.")
            low, high = int(counts["low_attention"]), int(counts["high_attention"])
            total = low + high
            stats = (
                f"{prepared['subject_id'].nunique()}",
                f"{prepared['recording_unit'].nunique()}",
                f"{low:,} / {high:,}",
                f"{100 * low / total:.1f} / {100 * high / total:.1f}",
            )
            if stats_by_composition.setdefault(composition, stats) != stats:
                raise ValueError(
                    f"{job_dir}: composition {composition} was prepared differently elsewhere in the sweep; "
                    f"got {stats}, already measured {stats_by_composition[composition]}."
                )

    # =============================================================================
    # Step 2: fill the template rows in the article's composition order
    # =============================================================================
    expected, actual = set(ARTICLE_COMPOSITIONS), set(stats_by_composition)
    if actual != expected:
        raise ValueError(
            f"Composition set does not match the table template; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )

    template_path = LATEX_ARTIFACT_TEMPLATES_DIR / "dataset_compositions_table_template.tex"
    lines = template_path.read_text().splitlines(keepends=True)
    metric_lines = []
    for index, line in enumerate(lines):
        cells = line.rstrip("\n").split("&")
        if len(cells) == 6 and all(cell.strip().removesuffix(r"\\").strip() == "-" for cell in cells[2:]):
            metric_lines.append(index)
    if len(metric_lines) != len(ARTICLE_COMPOSITIONS):
        raise ValueError(
            f"{template_path} has {len(metric_lines)} measured rows; expected {len(ARTICLE_COMPOSITIONS)}."
        )

    for index, composition in zip(metric_lines, ARTICLE_COMPOSITIONS, strict=True):
        cells = lines[index].rstrip("\n").split("&")
        lines[index] = f"{cells[0]}&{cells[1]}& " + " & ".join(stats_by_composition[composition]) + r" \\" + "\n"
    return "".join(lines)


def craft_transfer_matrix_figure(scenario_glob: str) -> Figure:
    """Build the all-decoder transfer matrix with target-specific baselines."""
    sides = (
        ("Stroop", SAM40_STROOP),
        ("Arithmetic", SAM40_ARITHMETIC),
        ("Mirror", SAM40_MIRROR),
        ("Full A", DISTINGUISHING),
        ("Full B", SAM40_FULL),
    )
    decoders = tuple(ARTICLE_DECODERS)
    model_names = tuple(spec["short_name"] for spec in ARTICLE_DECODERS.values())
    # Cross-task transfer stays inside Dataset B and now includes its full
    # composition, so the displayed pairs follow SAM40_SIDES rather than a slice
    # of the row order.
    displayed_scenarios = {
        ("baseline", side, side) for _, side in sides
    } | {
        ("cross_task", source, target)
        for source in SAM40_SIDES
        for target in SAM40_SIDES
        if source != target
    } | {
        # Dataset A is shown only against the full Dataset B composition when it
        # is the source; its transfers into the single-task compositions stay
        # measured but out of the figure.
        ("cross_dataset", source, target)
        for _, source in sides
        for _, target in sides
        if source[0] != target[0] and (source != DISTINGUISHING or target == SAM40_FULL)
    }

    # Every cell of this matrix is a decoder average, so the participant-level
    # scores are collapsed to one number per direction right after reading them.
    scores = {
        key: participant_score.mean()
        for key, participant_score in collect_participant_scores(
            scenario_glob, SCENARIO_ORDER
        ).items()
    }

    labels, compositions = zip(*sides, strict=True)
    row_count = len(sides) * len(decoders)
    matrix = np.full((row_count, len(sides)), np.nan)
    for side_index, (_, source) in enumerate(sides):
        for decoder_index, decoder in enumerate(decoders):
            row = side_index * len(decoders) + decoder_index
            for column, target in enumerate(compositions):
                if source == target:
                    matrix[row, column] = scores[(decoder, ("baseline", target, target))]
                else:
                    scenario = next(
                        (
                            (protocol, source, target)
                            for protocol in ("cross_task", "cross_dataset")
                            if (protocol, source, target) in displayed_scenarios
                        ),
                        None,
                    )
                    if scenario:
                        matrix[row, column] = scores[(decoder, scenario)]

    # Summary values live at the source level. Each cell is the participant-level
    # metric; for every displayed direction, average the decoders first, then
    # compare that with the equally averaged target baseline.
    summary_values = np.full((len(sides), 2), np.nan)
    for source_index, (_, source) in enumerate(sides):
        for summary_column, protocol in enumerate(("cross_task", "cross_dataset")):
            transfer_means, baseline_means = [], []
            for target in compositions:
                scenario = (protocol, source, target)
                if scenario not in displayed_scenarios:
                    continue
                transfer_means.append(
                    np.mean([scores[(decoder, scenario)] for decoder in decoders])
                )
                baseline_means.append(
                    np.mean(
                        [scores[(decoder, ("baseline", target, target))] for decoder in decoders]
                    )
                )
            if transfer_means:
                summary_values[source_index, summary_column] = (
                    np.mean(transfer_means) - np.mean(baseline_means)
                ) * 100

    # The target columns carry a two-level header: the transfer protocol that
    # reaching each target from another source represents, above the target
    # itself. The groups are ranges over the column order set by `sides`.
    column_groups = (("Cross-task", 0, 3), ("Cross-dataset", 3, 5))
    figure_width, figure_height = 13.6, 8.8
    matrix_left, matrix_bottom = 0.05, 0.095
    matrix_width, matrix_height = 0.61, 0.84
    group_header_height = 0.8
    header_height = 1.45 + group_header_height
    source_left, decoder_left = -2.25, -1.25
    data_column_width = matrix_width / (len(sides) - source_left)
    data_left = matrix_left - source_left * data_column_width
    summary_left = 0.69
    separator_x = (matrix_left + matrix_width + summary_left) / 2
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes((matrix_left, matrix_bottom, matrix_width, matrix_height))
    colorbar_ax = fig.add_axes((data_left, 0.05, len(sides) * data_column_width, 0.016))
    summary_ax = fig.add_axes((summary_left, matrix_bottom, 2 * data_column_width, matrix_height))
    fig.add_artist(
        Line2D(
            (separator_x, separator_x),
            (0.0, 1.0),
            transform=fig.transFigure,
            color="#D0D5DD",
            linewidth=0.6,
        )
    )
    cmap = LinearSegmentedColormap.from_list(
        "transfer_accuracy",
        ("#FFF7E6", "#FFB17E", "#FF624D", "#C83279", "#5B1D8B", "#00002D"),
    )
    normalization = Normalize(vmin=0.0, vmax=1.0)
    border_colour = "#344054"
    grid_colour = (1.0, 1.0, 1.0, 0.3)

    for row in range(row_count):
        for column in range(len(sides)):
            value = matrix[row, column]
            if np.isnan(value):
                facecolor, text, text_colour = "#FFFFFF", "—", "#667085"
            elif row // len(decoders) == column:
                facecolor, text, text_colour = "#FFFFFF", f"{value * 100:.1f}", "#263341"
            else:
                facecolor = cmap(normalization(value))
                text = f"{value * 100:.1f}"
                text_colour = "white" if value >= 0.55 else "#263341"
            ax.add_patch(
                Rectangle(
                    (column, row), 1, 1, facecolor=facecolor,
                    edgecolor=grid_colour, linewidth=0.35,
                )
            )
            ax.text(column + 0.5, row + 0.5, text, ha="center", va="center",
                    fontsize=9.5, fontweight="bold" if not np.isnan(value) else "normal", color=text_colour)

    # The source and decoder labels are part of the table, so their group
    # borders share the exact row geometry of the performance cells.
    for source_index, label in enumerate(labels):
        y_position = source_index * len(decoders)
        ax.add_patch(
            Rectangle(
                (source_left, y_position), decoder_left - source_left,
                len(decoders), facecolor="#FFFFFF", edgecolor=border_colour,
                linewidth=0.9, clip_on=False,
            )
        )
        ax.add_patch(
            Rectangle(
                (decoder_left, y_position), -decoder_left,
                len(decoders), facecolor="#FFFFFF", edgecolor=border_colour,
                linewidth=0.9, clip_on=False,
            )
        )
        ax.text(
            (source_left + decoder_left) / 2,
            y_position + len(decoders) / 2,
            label, ha="center", va="center", fontsize=10, fontweight="bold",
            clip_on=False,
        )
        for decoder_index, model_name in enumerate(model_names):
            ax.text(
                -0.08, y_position + decoder_index + 0.5, model_name,
                ha="right", va="center", fontsize=10, fontweight="bold",
                clip_on=False,
            )
        ax.add_patch(
            Rectangle(
                (0, y_position), len(sides), len(decoders), fill=False,
                edgecolor=border_colour, linewidth=0.9, zorder=4,
            )
        )
        # The baseline block sits on the diagonal and is read against the transfer
        # cells beside it, so it gets a thin outline of its own to stand apart
        # from the group border it shares an edge with.
        ax.add_patch(
            Rectangle(
                (source_index, y_position), 1, len(decoders), fill=False,
                edgecolor=border_colour, linewidth=0.9, zorder=5,
            )
        )

    label_header_top = -header_height + group_header_height
    for column, label in enumerate(labels):
        ax.add_patch(
            Rectangle(
                (column, label_header_top), 1, -label_header_top, facecolor="#FFFFFF",
                edgecolor=border_colour, linewidth=0.9, clip_on=False,
            )
        )
        ax.text(
            column + 0.5, label_header_top / 2, label, ha="center", va="center",
            fontsize=10, fontweight="bold", clip_on=False,
        )
    for group_label, first_column, last_column in column_groups:
        ax.add_patch(
            Rectangle(
                (first_column, -header_height), last_column - first_column,
                group_header_height, facecolor="#FFFFFF", edgecolor=border_colour,
                linewidth=0.9, clip_on=False,
            )
        )
        ax.text(
            (first_column + last_column) / 2, -header_height + group_header_height / 2,
            group_label, ha="center", va="center", fontsize=10, fontweight="bold",
            clip_on=False,
        )
    ax.text(
        len(sides) / 2, -header_height - 0.42, "Target (test)", ha="center",
        va="center", fontsize=11, fontweight="bold", clip_on=False,
    )
    ax.text(
        source_left - 0.28, row_count / 2, "Source (train)", ha="center",
        va="center", rotation=90, fontsize=11, fontweight="bold", clip_on=False,
    )

    ax.set_xlim(source_left, len(sides))
    ax.set_ylim(row_count, -header_height)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks(np.linspace(0, 1, 5), labels=("0", "25", "50", "75", "100"))
    colorbar.set_label(f"{ARTICLE_METRIC_LABEL}, %", fontsize=9, fontweight="bold")
    colorbar.ax.tick_params(labelsize=8)

    for row in range(len(sides)):
        for column in range(2):
            value = summary_values[row, column]
            facecolor = "#FFFFFF"
            text = "—" if np.isnan(value) else f"{value:.1f}"
            y_position = row * len(decoders)
            summary_ax.add_patch(
                Rectangle(
                    (column, y_position), 1, len(decoders), facecolor=facecolor,
                    edgecolor=border_colour, linewidth=0.9,
                )
            )
            summary_ax.text(column + 0.5, y_position + len(decoders) / 2, text, ha="center", va="center", fontsize=12,
                            fontweight="bold" if not np.isnan(value) else "normal", color="#263341")
    for column, label in enumerate((
        "Baseline vs\ncross-task\nmean Δ",
        "Baseline vs\ncross-dataset\nmean Δ",
    )):
        summary_ax.add_patch(
            Rectangle(
                (column, -header_height), 1, header_height, facecolor="#FFFFFF",
                edgecolor=border_colour, linewidth=0.9, clip_on=False,
            )
        )
        summary_ax.text(
            column + 0.5, -header_height / 2, label, ha="center", va="center",
            fontsize=10, fontweight="bold", clip_on=False,
        )
    summary_ax.set_xlim(0, 2)
    summary_ax.set_ylim(row_count, -header_height)
    summary_ax.set_xticks([])
    summary_ax.set_yticks([])
    for spine in summary_ax.spines.values():
        spine.set_visible(False)

    return fig


def craft_scenario_accuracy_by_decoder_figure(scenario_glob: str) -> Figure:
    """Build one per-scenario accuracy figure with every decoder and their mean."""
    protocol_labels = (
        ("Baseline", 5),
        ("Cross-subject", 2),
        ("Cross-task (Dataset B)", 12),
        ("Cross-dataset", 8),
    )
    validation_method_labels = (
        "Stratified K-fold(Full A)",
        "Stratified K-fold(Full B)",
        "Stratified K-fold(B: Stroop)",
        "Stratified K-fold(B: Arithmetic)",
        "Stratified K-fold(B: Mirror)",
        "Leave-one-subject-out(Full A)",
        "Leave-one-subject-out(Full B)",
        "Full B → Stroop",
        "Full B → Arithmetic",
        "Full B → Mirror",
        "Stroop → Full B",
        "Stroop → Arithmetic",
        "Stroop → Mirror",
        "Arithmetic → Full B",
        "Arithmetic → Stroop",
        "Arithmetic → Mirror",
        "Mirror → Full B",
        "Mirror → Stroop",
        "Mirror → Arithmetic",
        "Full A → Full B",
        "Full B → Full A",
        "Full A → Stroop",
        "Stroop → Full A",
        "Full A → Arithmetic",
        "Arithmetic → Full A",
        "Full A → Mirror",
        "Mirror → Full A",
    )
    decoders = tuple(ARTICLE_DECODERS)
    if len(validation_method_labels) != len(SCENARIO_ORDER):
        raise RuntimeError("Figure validation-method labels must match the scenario order.")

    scores = collect_participant_scores(scenario_glob, SCENARIO_ORDER)
    model_means = np.array(
        [
            [scores[(decoder, scenario)].mean() for decoder in decoders]
            for scenario in SCENARIO_ORDER
        ]
    )
    average_means = model_means.mean(axis=1)
    lower_errors, upper_errors = [], []
    rng = np.random.default_rng(42)
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        low, high = bootstrap_average_interval(
            [scores[(decoder, scenario)] for decoder in decoders], rng
        )
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

    fig, ax = plt.subplots(figsize=(12.2, 12.9))
    offsets = np.linspace(-0.22, 0.22, len(decoders))
    for decoder_index, spec in enumerate(ARTICLE_DECODERS.values()):
        ax.scatter(
            model_means[:, decoder_index],
            np.asarray(method_y_positions) + offsets[decoder_index],
            s=38,
            color=spec["colour"],
            alpha=0.65,
            edgecolor="#24354F",
            linewidth=0.35,
            label=spec["short_name"],
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
    ax.set_xlabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")
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
    # This figure is saved with the margins it sets here rather than a tight
    # bounding box, so the bottom margin has to hold the metric label itself.
    fig.subplots_adjust(left=0.22, right=0.98, top=0.9385, bottom=0.048)
    return fig


def craft_baseline_vs_cross_subject_figure(scenario_glob: str) -> Figure:
    """Build the per-dataset baseline-to-cross-subject trajectory of every decoder."""
    panel_specs = (
        ("Mental attention-state dataset", DISTINGUISHING),
        ("SAM-40 dataset", SAM40_FULL),
    )
    protocol_specs = (("baseline", "Baseline"), ("cross_subject", "Cross-subject"))
    decoders = tuple(ARTICLE_DECODERS)
    # A panel column is one non-transfer scenario, which is its own target.
    panel_scenarios = tuple(
        tuple((protocol, side, side) for protocol, _ in protocol_specs)
        for _, side in panel_specs
    )

    # =============================================================================
    # Step 1: read the few scenarios this figure compares
    # =============================================================================
    # The sweep also holds every transfer direction, and none of them belongs on
    # a protocol axis, so only the scenarios of the two panels are asked for.
    scores = collect_participant_scores(
        scenario_glob,
        tuple(scenario for scenarios in panel_scenarios for scenario in scenarios),
    )

    # =============================================================================
    # Step 2: decoder means and the paired bootstrap interval of their average
    # =============================================================================
    # Resampling the participants once per draw keeps the five decoders paired:
    # they were measured on the very same target participants.
    rng = np.random.default_rng(42)
    panel_means, panel_averages, panel_errors = [], [], []
    for scenarios in panel_scenarios:
        means = np.array(
            [
                [scores[(decoder, scenario)].mean() for decoder in decoders]
                for scenario in scenarios
            ]
        )
        averages = means.mean(axis=1)
        lower_errors, upper_errors = [], []
        for scenario_index, scenario in enumerate(scenarios):
            low, high = bootstrap_average_interval(
                [scores[(decoder, scenario)] for decoder in decoders], rng
            )
            lower_errors.append(averages[scenario_index] - low)
            upper_errors.append(high - averages[scenario_index])
        panel_means.append(means)
        panel_averages.append(averages)
        panel_errors.append(np.array((lower_errors, upper_errors)))

    # =============================================================================
    # Step 3: draw both panels on one shared accuracy scale
    # =============================================================================
    # The accuracy range follows the data instead of a fixed window: an article
    # figure comparing protocols is read by the size of the gaps between them,
    # and a hardcoded scale would push a well-performing sweep into a corner.
    plotted_scores = np.concatenate(
        [means.ravel() for means in panel_means]
        + [averages - errors[0] for averages, errors in zip(panel_averages, panel_errors, strict=True)]
        + [averages + errors[1] for averages, errors in zip(panel_averages, panel_errors, strict=True)]
    )
    score_padding = np.ptp(plotted_scores) * 0.25

    x_positions = np.arange(len(protocol_specs))
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(11.4, 5.6), sharey=True)
    for panel_index, (ax, (title, _)) in enumerate(zip(axes, panel_specs, strict=True)):
        means, averages, errors = (
            panel_means[panel_index],
            panel_averages[panel_index],
            panel_errors[panel_index],
        )
        for decoder_index, spec in enumerate(ARTICLE_DECODERS.values()):
            ax.plot(
                x_positions,
                means[:, decoder_index],
                color=spec["colour"],
                linewidth=2.4,
                alpha=0.65,
                marker="o",
                markersize=7,
                markeredgecolor="#24354F",
                markeredgewidth=0.35,
                label=spec["short_name"] if panel_index == 0 else None,
                zorder=3,
            )
        # Only the connecting line of the average is translucent, so that it
        # summarises the five trajectories without painting over them; its point
        # and interval stay as solid as on the other article figures.
        ax.plot(x_positions, averages, color="#1F2937", alpha=0.45, linewidth=3.0, zorder=4)
        ax.errorbar(
            x_positions,
            averages,
            yerr=errors,
            fmt="o",
            color="#1F2937",
            ecolor="#1F2937",
            markerfacecolor=to_rgba("#1F2937", 0.6),
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.0,
            elinewidth=1.4,
            capsize=4,
            label="Average (95% bootstrap CI)" if panel_index == 0 else None,
            zorder=5,
        )
        ax.set_title(title, fontsize=15, fontweight="bold", color="#172B4D", loc="left", pad=12)
        ax.set_xticks(x_positions, [label for _, label in protocol_specs], fontsize=12,
                      fontweight="bold", color="#24354F")
        ax.set_xlim(-0.45, len(protocol_specs) - 0.55)
        ax.grid(axis="y", color="#E0E6EF", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=11)
    axes[0].set_ylim(plotted_scores.min() - score_padding, plotted_scores.max() + score_padding)
    axes[0].set_ylabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        ncols=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=11,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    plt.setp(legend.get_texts(), fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.145, wspace=0.07)
    return fig


def write_article_artifacts(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write all article tables and figures from one completed result sweep."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    source_png_dir = figures_dir / "source_png"
    source_svg_dir = figures_dir / "source_svg"
    for directory in (tables_dir, figures_dir, source_png_dir, source_svg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    written = []
    for decoder in ARTICLE_DECODERS:
        table_path = tables_dir / f"scenario_metrics_{decoder}.tex"
        table_path.write_text(craft_scenario_metrics_table(str(input_dir / decoder / "*")))
        written.append(table_path)

    scenario_glob = str(input_dir / "*" / "*")
    # Read across every decoder rather than one of them: the compositions are the
    # same prepared data for all five, so the whole sweep is the strongest place
    # to notice that they are not.
    composition_table_path = tables_dir / "dataset_compositions.tex"
    composition_table_path.write_text(craft_dataset_composition_table(scenario_glob))
    written.append(composition_table_path)

    # The label mapping states the class definition rather than a measured result,
    # so nothing is filled into it; it is copied so that every table the manuscript
    # inputs comes from the same place.
    label_mapping_path = tables_dir / "attention_label_mapping.tex"
    label_mapping_path.write_text(
        (LATEX_ARTIFACT_TEMPLATES_DIR / "attention_label_mapping_table_template.tex").read_text()
    )
    written.append(label_mapping_path)

    # Each figure carries its own \begin{figure} wrapper template, so the caption
    # and label travel with the image instead of living in the manuscript.
    figure_specs = (
        (
            "transfer_matrix",
            craft_transfer_matrix_figure,
            {"bbox_inches": "tight", "pad_inches": 0.03},
        ),
        (
            "baseline_vs_cross_subject",
            craft_baseline_vs_cross_subject_figure,
            {"bbox_inches": "tight", "pad_inches": 0.03},
        ),
        # The per-scenario accuracy figure places its own margins with subplots_adjust and
        # draws its row labels outside the axes, which a tight bounding box would
        # crop; it is saved with the margins it asked for.
        ("scenario_accuracy_by_decoder", craft_scenario_accuracy_by_decoder_figure, {}),
    )
    for name, craft_figure, savefig_kwargs in figure_specs:
        figure = craft_figure(scenario_glob)
        figure_path = source_png_dir / f"{name}.png"
        source_svg_path = source_svg_dir / f"{name}.svg"
        figure.savefig(figure_path, dpi=300, **savefig_kwargs)
        # Matplotlib pads SVG lines with trailing spaces, which the repository
        # whitespace check rejects, so the source is rewritten without them.
        figure.savefig(source_svg_path, format="svg", **savefig_kwargs)
        source_svg_path.write_text(
            "\n".join(line.rstrip() for line in source_svg_path.read_text().splitlines()) + "\n"
        )
        plt.close(figure)

        figure_template = (
            LATEX_ARTIFACT_TEMPLATES_DIR / f"{name}_figure_template.tex"
        ).read_text()
        figure_tex_path = figures_dir / f"{name}.tex"
        figure_tex_path.write_text(figure_template.replace("FIGURE_SLUG", name))
        written.extend((figure_tex_path, figure_path, source_svg_path))

    return written
