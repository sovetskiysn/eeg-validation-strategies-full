.PHONY: prepare-marked-manuscript prepare-plos-submission pull-artifacts clean

.ONESHELL:
.SHELLFLAGS := -ec

# Kept in lockstep with .vscode/settings.json so the tracked-changes source is
# generated with the same TeX Live release as the manuscript PDF.
TEXLIVE_IMAGE := texlive/texlive:TL2025-historic

# The only handwritten route to revision/manuscript_marked/: the submitted
# manuscript is the read-only baseline and manuscript_clean/ is the editable
# source.
# Double spacing is required by PLOS but may be commented out in
# manuscript_clean/ while working; restore it before latexdiff so both the
# clean source and the tracked-changes file go out double-spaced.
prepare-marked-manuscript:
	@sed -i -E '/\\usepackage\{setspace\}|\\doublespacing/s/^%[[:space:]]*//' \
		revision/manuscript_clean/manuscript.tex
	@grep -qxF '\\doublespacing' revision/manuscript_clean/manuscript.tex \
		|| { echo "\\doublespacing not found in revision/manuscript_clean/manuscript.tex" >&2; exit 2; }
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(CURDIR):/work" -w /work \
		$(TEXLIVE_IMAGE) \
		latexdiff --encoding=utf8 --append-safecmd="rev,revs" \
		initial_submission/main.tex revision/manuscript_clean/manuscript.tex \
		> revision/manuscript_marked/manuscript-track-changes.tex

# Render publication figures and tables directly into the working manuscript.
# Example: make pull-artifacts ANALYSIS_INPUT_DIR="/abs/path/to/results"
pull-artifacts:
	export ANALYSIS_OUTPUT_DIR="$(CURDIR)/revision/manuscript_clean"
	$(MAKE) -C ../python_project analysis

# Create a separate self-contained PLOS submission project. It contains the
# manuscript without embedded graphics and figures/Fig<N>.tif. The packaging
# script lives in python_project/scripts/ (not wired into its own Makefile:
# it belongs to the manuscript side, not the analysis pipeline) and only
# repackages the figures/source_tiff/*.tif that pull-artifacts already
# produced -- it does not convert or validate anything itself.
# Example: make prepare-plos-submission SOURCE=revision/manuscript_clean OUTPUT=revision/manuscript_clean_plos
prepare-plos-submission:
	@test -n "$(SOURCE)" && test -n "$(OUTPUT)" || { echo "Usage: make prepare-plos-submission SOURCE=path/to/project OUTPUT=path/to/submission-project" >&2; exit 2; }
	cd ../python_project && uv run python scripts/prepare_plos_submission.py "$(CURDIR)/$(SOURCE)" "$(CURDIR)/$(OUTPUT)"

# Remove LaTeX build artifacts that ended up next to the sources. LaTeX
# Workshop keeps them in a per-document .build/ (.vscode/settings.json), so
# these files only appear after a build run by hand -- pdflatex or latexmk
# invoked in the source directory without --aux-directory.
# Left alone on purpose: .build/ (the latexmk cache lives there and keeps
# rebuilds fast), the PDFs, and the .synctex.gz the PDF viewer needs.
clean:
	@find . -not -path '*/.build/*' -type f \( \
		-name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.fls' \
		-o -name '*.fdb_latexmk' -o -name '*.bbl' -o -name '*.blg' \
		-o -name '*.toc' -o -name '*.lof' -o -name '*.lot' \
		\) -print -delete
	@echo "Cleaned stray LaTeX artifacts (kept .build/, PDFs and .synctex.gz)"
