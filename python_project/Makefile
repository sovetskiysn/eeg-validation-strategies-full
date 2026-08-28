.PHONY: download standardize ingestion
.PHONY: sweep-scenario-decoder sweep-scenario-only
.PHONY: baseline-logistic-regression ica-ablation
.PHONY: analysis
.PHONY: full-pipeline
.PHONY: hello-world
.PHONY: clean-results clean-cache clean-archive clean-bids clean-all

.ONESHELL:
.SHELLFLAGS := -ec

# One target selects one recipe from the root of experiments/. MULTIRUN is
# enabled centrally in configs/config.yaml for every run. Makefile owns the
# result directory layout.

sweep-scenario-decoder:
	uv run python scripts/run_experiment.py --config-dir experiments --config-name scenario_decoder

baseline-logistic-regression:
	uv run python scripts/run_experiment.py --config-dir experiments --config-name baseline_logistic_regression

ica-ablation:
	uv run python scripts/run_experiment.py --config-dir experiments --config-name ica_ablation

# The directory is an address of existing output, not an experiment override:
#   make analysis
#   make analysis ANALYZE_DIR="results/sweep-scenario-decoder (2026-08-17 % 06-25-13)"
# Make exports command-line variables to the recipe. Without one, analysis uses
# the newest top-level directory under results/.
analysis:
	uv run python scripts/run_analysis.py

download:
	uv run python scripts/run_download.py

standardize:
	uv run python scripts/run_standardization.py

ingestion: download standardize

full-pipeline: ingestion sweep-scenario-decoder

hello-world:
	uv run python scripts/hello_world.py

clean-results:
	find results -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-cache:
	find datasets/prepared_cache -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-archive:
	find datasets/archive -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-bids: clean-cache
	find datasets/bids -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf -- {} +

clean-all: clean-results clean-cache clean-archive clean-bids
