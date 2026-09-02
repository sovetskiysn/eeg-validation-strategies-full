.PHONY: download standardize
.PHONY: experiment
.PHONY: analysis
.PHONY: plos-submission
.PHONY: full-pipeline
.PHONY: clean-results clean-archive clean-bids clean-all

.ONESHELL:
.SHELLFLAGS := -ec

# Select one named root preset from configs/experiments/.  Edit only the
# `+experiments=...` choice below when switching the reproducible sweep.
# Each preset switches Hydra into MULTIRUN itself, so no -m flag here.
experiment:
	uv run python scripts/run_experiment.py +experiments=scenario_decoder

# The directory is an address of existing output, not an experiment override.
# Both ends have a default: the newest sweep in results/ is read, and analysis/
# next to it is written.
#   make analysis
#   make analysis ANALYSIS_INPUT_DIR="results/scenario_decoder (...)" ANALYSIS_OUTPUT_DIR="../analysis"
# Make exports the input and output paths to the analysis runner.
analysis:
	uv run python scripts/run_analysis.py

# Repackages an already rendered manuscript into a PLOS submission project.
# The manuscript side addresses both ends:
#   make plos-submission PLOS_SOURCE_DIR=... PLOS_OUTPUT_DIR=...
plos-submission:
	uv run python scripts/prepare_plos_submission.py

download:
	uv run python scripts/run_download.py

standardize:
	uv run python scripts/run_standardization.py

full-pipeline: clean-results download standardize experiment

clean-results:
	find results -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-archive:
	find datasets/archive -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-bids:
	find datasets/bids -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-all: clean-results clean-archive clean-bids
