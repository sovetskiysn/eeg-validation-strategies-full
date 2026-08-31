"""Render article tables and figures from a completed sweep."""

from __future__ import annotations

import os
from pathlib import Path

from analysis import write_article_artifacts
from utils import PROJECT_ROOT, REPOSITORY_ROOT


ANALYSIS_INPUT_DIR = Path(
    os.environ.get(
        "ANALYZE_DIR",
        PROJECT_ROOT / "results" / "scenario_decoder (2026-08-25 % 12-36-43)",
    )
)
ANALYSIS_OUTPUT_DIR = Path(
    os.environ.get(
        "ANALYSIS_OUTPUT_DIR",
        REPOSITORY_ROOT / "analysis",
    )
)


def main() -> None:
    """Render the selected article artifacts."""
    for artifact_path in write_article_artifacts(ANALYSIS_INPUT_DIR, ANALYSIS_OUTPUT_DIR):
        print(f"Wrote {artifact_path}")

if __name__ == "__main__":
    main()
