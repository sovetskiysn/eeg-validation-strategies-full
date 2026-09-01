"""Render article tables and figures from a completed sweep."""

from __future__ import annotations

import os
from pathlib import Path

from analysis import write_article_artifacts
from utils import PROJECT_ROOT, REPOSITORY_ROOT


ANALYSIS_INPUT_DIR = Path(
    os.environ.get(
        "ANALYSIS_INPUT_DIR",
        PROJECT_ROOT / "results" / "scenario_decoder_maybe_new (2026-09-01 % 08-54-09)",
    )
)
ANALYSIS_OUTPUT_DIR = Path(
    os.environ.get(
        "ANALYSIS_OUTPUT_DIR",
        REPOSITORY_ROOT / "analysis_3",
    )
)


def main() -> None:
    """Render the selected article artifacts."""
    for artifact_path in write_article_artifacts(ANALYSIS_INPUT_DIR, ANALYSIS_OUTPUT_DIR):
        print(f"Wrote {artifact_path}")

if __name__ == "__main__":
    main()
