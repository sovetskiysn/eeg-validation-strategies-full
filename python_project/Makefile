.PHONY: download standardize ingestion
.PHONY: experiment
.PHONY: analysis
.PHONY: full-pipeline
.PHONY: hello-world
.PHONY: clean-results clean-pipeline-cache clean-archive clean-bids clean-all

.ONESHELL:
.SHELLFLAGS := -ec

# Select one named root preset from configs/experiments/.  Edit only the
# `+experiments=...` choice below when switching the reproducible sweep.
# Each preset switches Hydra into MULTIRUN itself, so no -m flag here.
experiment:
	uv run python scripts/run_experiment.py +experiments=scenario_decoder

# The directory is an address of existing output, not an experiment override:
#   make analysis
#   make analysis ANALYZE_DIR="results/scenario_decoder (...)" ANALYSIS_OUTPUT_DIR="../analysis"
# Make exports the input and output paths to the analysis runner.
analysis:
	uv run python scripts/run_analysis.py

download:
	uv run python scripts/run_download.py

standardize:
	uv run python scripts/run_standardization.py

ingestion: download standardize

full-pipeline: clean-results ingestion experiment

hello-world:
	uv run python scripts/hello_world.py

clean-results:
	find results -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-pipeline-cache:
	find datasets/bids -type d -name _cache -prune -exec rm -rf -- {} +

clean-archive:
	find datasets/archive -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-bids:
	find datasets/bids -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-all: clean-results clean-pipeline-cache clean-archive clean-bids
