"""Fixed filesystem locations shared by the project's scripts and stages."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARCHIVE_DIR = PROJECT_ROOT / "datasets" / "archive"
BIDS_DIR = PROJECT_ROOT / "datasets" / "bids"
LATEX_ARTIFACT_TEMPLATES_DIR = PROJECT_ROOT / "src" / "latex_artifact_templates"

SAM40_SOURCE_DIR = ARCHIVE_DIR / "SAM 40 dataset"
DISTINGUISHING_SOURCE_DIR = ARCHIVE_DIR / "Distinguishing"
