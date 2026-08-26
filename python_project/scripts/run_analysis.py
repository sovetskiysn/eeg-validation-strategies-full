"""Render article tables and figures from a completed sweep."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from analysis import (
    craft_all_scenarios_absolute_accuracy_figure,
    craft_cross_subject_model_comparison_figure,
    craft_main_table,
    craft_scenario_by_decoder_slope_figure,
    craft_transfer_matrix_figure,
)
from utils import PROJECT_ROOT, REPOSITORY_ROOT


ANALYSIS_INPUT_DIR = Path(
    os.environ.get(
        "ANALYZE_DIR",
        PROJECT_ROOT / "results" / "scenario_decoder (2026-08-24 | 15-13-34)",
    )
)
ANALYSIS_OUTPUT_DIR = REPOSITORY_ROOT / "analysis"
DECODERS = (
    "logistic_regression",
    "xgboost",
    "eegnet",
    "shallownet",
    "eegconformer",
)


def main() -> None:
    """Render the selected article artifacts."""
    tables_dir = ANALYSIS_OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for decoder in DECODERS:
        scenario_glob = str(ANALYSIS_INPUT_DIR / f"+decoder={decoder},+scenario=*")
        table_path = tables_dir / f"scenario_table_{decoder}.tex"
        table_path.write_text(craft_main_table(scenario_glob))
        print(f"Wrote {table_path}")

    figures_dir = ANALYSIS_OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure = craft_cross_subject_model_comparison_figure(ANALYSIS_INPUT_DIR)
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"cross_subject_model_comparison.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_transfer_matrix_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=logistic_regression,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"transfer_matrix_logistic_regression.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_scenario_by_decoder_slope_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=*,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"scenario_by_decoder_slope.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_absolute_accuracy_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=logistic_regression,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"all_scenarios_absolute_accuracy_logistic_regression.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_absolute_accuracy_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=xgboost,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"all_scenarios_absolute_accuracy_xgboost.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_absolute_accuracy_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=eegnet,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"all_scenarios_absolute_accuracy_eegnet.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_absolute_accuracy_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=shallownet,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"all_scenarios_absolute_accuracy_shallownet.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_absolute_accuracy_figure(
        str(ANALYSIS_INPUT_DIR / "+decoder=eegconformer,+scenario=*")
    )
    for suffix in ("svg", "png"):
        figure_path = figures_dir / f"all_scenarios_absolute_accuracy_eegconformer.{suffix}"
        figure.savefig(figure_path, dpi=300)
        print(f"Wrote {figure_path}")
    plt.close(figure)

if __name__ == "__main__":
    main()
