"""Render article tables and figures from a completed sweep."""

from __future__ import annotations

import os
from pathlib import Path

from analysis import write_article_artifacts
from utils import PROJECT_ROOT


ANALYSIS_INPUT_DIR = Path(
    os.environ.get("ANALYSIS_INPUT_DIR")
    or max((PROJECT_ROOT / "results").glob("* (????-??-?? % ??-??-??)*"), key=lambda d: d.name.split("(")[-1])
)
ANALYSIS_OUTPUT_DIR = Path(
    os.environ.get("ANALYSIS_OUTPUT_DIR")
    or PROJECT_ROOT / "analysis"
)


def main() -> None:
    """Render the selected article artifacts."""
    print(f"Reading {ANALYSIS_INPUT_DIR}")
    for artifact_path in write_article_artifacts(ANALYSIS_INPUT_DIR, ANALYSIS_OUTPUT_DIR):
        print(f"Wrote {artifact_path}")

if __name__ == "__main__":
    main()
