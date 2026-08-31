"""Render article tables and figures from a completed sweep."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from analysis import (
    craft_all_scenarios_model_comparison_figure,
    craft_main_table,
    craft_transfer_degradation_matrix_figure,
)
from utils import PROJECT_ROOT, REPOSITORY_ROOT


ANALYSIS_INPUT_DIR = Path(
    os.environ.get(
        "ANALYZE_DIR",
        PROJECT_ROOT / "results" / "scenario_decoder (2026-08-25 % 12-36-43)",
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
        scenario_glob = str(ANALYSIS_INPUT_DIR / decoder / "*")
        table_path = tables_dir / f"result_table_{decoder}.tex"
        table_path.write_text(craft_main_table(scenario_glob))
        print(f"Wrote {table_path}")

    figures_dir = ANALYSIS_OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure = craft_transfer_degradation_matrix_figure(
        str(ANALYSIS_INPUT_DIR / "*" / "*")
    )
    figure_path = figures_dir / "transfer_degradation_matrix.png"
    figure.savefig(figure_path, dpi=300)
    print(f"Wrote {figure_path}")
    plt.close(figure)

    figure = craft_all_scenarios_model_comparison_figure(
        str(ANALYSIS_INPUT_DIR / "*" / "*")
    )
    figure_path = figures_dir / "all_scenarios_model_comparison.png"
    figure.savefig(figure_path, dpi=300)
    print(f"Wrote {figure_path}")
    plt.close(figure)

if __name__ == "__main__":
    main()
