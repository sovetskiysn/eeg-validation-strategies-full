"""One transfer matrix with cross-task and cross-dataset column blocks.

Numbers are deterministic synthetic values for the visual mock-up only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


OUT = Path(__file__).parent
SOURCES = ("Stroop", "Arithmetic", "Mirror", "Full B", "Full A")
CROSS_TASK_TARGETS = ("Stroop", "Arithmetic", "Mirror", "Full B")
CROSS_DATASET_TARGETS = ("Stroop", "Arithmetic", "Mirror", "Full B", "Full A")
DECODERS = ("LogReg", "XGBoost", "EEGNet", "ShallowNet", "EEGConformer")
BASELINES = {"Full A": 0.68, "Full B": 0.71, "Stroop": 0.69, "Arithmetic": 0.70, "Mirror": 0.72}
CMAP = LinearSegmentedColormap.from_list(
    "transfer_accuracy", ("#FFF7E6", "#FFB17E", "#FF624D", "#C83279", "#5B1D8B", "#00002D")
)
NORM, BORDER, GRID = Normalize(0, 1), "#344054", (1.0, 1.0, 1.0, 0.3)


def scores(protocol: str, targets: tuple[str, ...]) -> np.ndarray:
    rng = np.random.default_rng(20260902 if protocol == "cross-task" else 20260903)
    matrix = np.full((len(SOURCES) * len(DECODERS), len(targets)), np.nan)
    shifts = (-0.026, 0.010, -0.018, 0.020, 0.006)
    for source_index, source in enumerate(SOURCES):
        for decoder_index, shift in enumerate(shifts):
            row = source_index * len(DECODERS) + decoder_index
            for column, target in enumerate(targets):
                if source == target:
                    matrix[row, column] = BASELINES[target] + shift
                elif protocol == "cross-task" and source != "Full A":
                    matrix[row, column] = BASELINES[target] - rng.uniform(0.035, 0.115) + shift
                elif protocol == "cross-dataset" and ((source == "Full A") != (target == "Full A")):
                    matrix[row, column] = BASELINES[target] - rng.uniform(0.105, 0.220) + shift
    return np.clip(matrix, 0, 1)


def mean_delta(block: np.ndarray, targets: tuple[str, ...], source_index: int) -> float:
    source_rows = block[source_index * len(DECODERS):(source_index + 1) * len(DECODERS)]
    contrasts = []
    for column, target in enumerate(targets):
        if SOURCES[source_index] == target or np.isnan(source_rows[:, column]).all():
            continue
        target_source_index = SOURCES.index(target)
        baseline = block[target_source_index * len(DECODERS):(target_source_index + 1) * len(DECODERS), column].mean()
        contrasts.extend(source_rows[:, column] - baseline)
    return np.mean(contrasts) * 100 if contrasts else np.nan


task = scores("cross-task", CROSS_TASK_TARGETS)
dataset = scores("cross-dataset", CROSS_DATASET_TARGETS)
matrix = np.concatenate((task, dataset), axis=1)
all_targets = CROSS_TASK_TARGETS + CROSS_DATASET_TARGETS
row_count, matrix_columns = matrix.shape
source_left, decoder_left = -2.50, -1.25
group_header_height, header_height = 1.30, 2.60
fig = plt.figure(figsize=(18.0, 8.8))
axis = fig.add_axes((0.045, 0.095, 0.54, 0.84))
data_column_width = 0.54 / (matrix_columns - source_left)
colorbar_axis = fig.add_axes((0.045 - source_left * data_column_width, 0.05, matrix_columns * data_column_width, 0.016))
summary_axis = fig.add_axes((0.62, 0.095, 0.13, 0.84))
fig.add_artist(plt.Line2D((0.6025, 0.6025), (0, 1), transform=fig.transFigure, color="#D0D5DD", linewidth=0.6))

for row in range(row_count):
    for column in range(matrix_columns):
        value = matrix[row, column]
        source = SOURCES[row // len(DECODERS)]
        target = all_targets[column]
        is_baseline = source == target
        if np.isnan(value):
            face, text, colour = "#FFFFFF", "—", "#667085"
        elif is_baseline:
            face, text, colour = "#EAECF0", f"{value * 100:.1f}", "#263341"
        else:
            face, text = CMAP(NORM(value)), f"{value * 100:.1f}"
            colour = "white" if value >= 0.55 else "#263341"
        axis.add_patch(Rectangle((column, row), 1, 1, facecolor=face, edgecolor=GRID, linewidth=0.35))
        axis.text(column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=8.4,
                  fontweight="bold" if not np.isnan(value) else "normal", color=colour)

for source_index, label in enumerate(SOURCES):
    y = source_index * len(DECODERS)
    axis.add_patch(Rectangle((source_left, y), decoder_left - source_left, len(DECODERS), facecolor="white", edgecolor=BORDER, linewidth=0.9, clip_on=False))
    axis.add_patch(Rectangle((decoder_left, y), -decoder_left, len(DECODERS), facecolor="white", edgecolor=BORDER, linewidth=0.9, clip_on=False))
    axis.text((source_left + decoder_left) / 2, y + len(DECODERS) / 2, label, ha="center", va="center", fontsize=9, fontweight="bold", clip_on=False)
    for decoder_index, decoder in enumerate(DECODERS):
        axis.text(-0.08, y + decoder_index + 0.5, decoder, ha="right", va="center", fontsize=9, fontweight="bold", clip_on=False)
    axis.add_patch(Rectangle((0, y), matrix_columns, len(DECODERS), fill=False, edgecolor=BORDER, linewidth=0.9, zorder=4))
    # Outline both baseline cells when a target appears in both column blocks.
    for column, target in enumerate(all_targets):
        if target == label:
            axis.add_patch(Rectangle((column, y), 1, len(DECODERS), fill=False, edgecolor=BORDER, linewidth=0.9, zorder=5))

label_header_top = -header_height + group_header_height
for column, label in enumerate(all_targets):
    axis.add_patch(Rectangle((column, label_header_top), 1, -label_header_top, facecolor="white", edgecolor=BORDER, linewidth=0.9, clip_on=False))
    axis.text(column + 0.5, label_header_top / 2, label, ha="center", va="center", fontsize=9, fontweight="bold", clip_on=False)
for label, start, end in (("Cross-task", 0, len(CROSS_TASK_TARGETS)), ("Cross-dataset", len(CROSS_TASK_TARGETS), matrix_columns)):
    axis.add_patch(Rectangle((start, -header_height), end - start, group_header_height, facecolor="white", edgecolor=BORDER, linewidth=0.9, clip_on=False))
    axis.text((start + end) / 2, -header_height + group_header_height / 2, label, ha="center", va="center", fontsize=9, fontweight="bold", clip_on=False)
axis.text(matrix_columns / 2, -header_height - 0.42, "Target (test)", ha="center", va="center", fontsize=11, fontweight="bold", clip_on=False)
axis.text(source_left - 0.28, row_count / 2, "Source (train)", ha="center", va="center", rotation=90, fontsize=11, fontweight="bold", clip_on=False)
axis.set(xlim=(source_left, matrix_columns), ylim=(row_count, -header_height)); axis.set_xticks([]); axis.set_yticks([])
for spine in axis.spines.values(): spine.set_visible(False)

for source_index in range(len(SOURCES)):
    y = source_index * len(DECODERS)
    for column, (block, targets) in enumerate(((task, CROSS_TASK_TARGETS), (dataset, CROSS_DATASET_TARGETS))):
        value = mean_delta(block, targets, source_index)
        summary_axis.add_patch(Rectangle((column, y), 1, len(DECODERS), facecolor="white", edgecolor=BORDER, linewidth=0.9))
        summary_axis.text(column + 0.5, y + len(DECODERS) / 2, "—" if np.isnan(value) else f"{value:.1f}", ha="center", va="center", fontsize=11, fontweight="bold", color="#263341")
for column, label in enumerate(("Baseline\nvs\ncross-task\nmean diff", "Baseline\nvs\ncross-dataset\nmean diff")):
    summary_axis.add_patch(Rectangle((column, -header_height), 1, header_height, facecolor="white", edgecolor=BORDER, linewidth=0.9, clip_on=False))
    summary_axis.text(column + 0.5, -header_height / 2, label, ha="center", va="center", fontsize=9, fontweight="bold", clip_on=False)
summary_axis.set(xlim=(0, 2), ylim=(row_count, -header_height)); summary_axis.set_xticks([]); summary_axis.set_yticks([])
for spine in summary_axis.spines.values(): spine.set_visible(False)

colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=colorbar_axis, orientation="horizontal")
colourbar.set_ticks(np.linspace(0, 1, 5), labels=("0", "25", "50", "75", "100"))
colourbar.set_label("Balanced accuracy, %", fontsize=9, fontweight="bold")
colourbar.ax.tick_params(labelsize=8)
fig.savefig(OUT / "fig1_double_header_transfer_matrix_mockup.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig1_double_header_transfer_matrix_mockup.svg", bbox_inches="tight", facecolor="white")
