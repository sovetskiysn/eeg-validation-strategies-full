.PHONY: prepare-marked-manuscript prepare-plos-submission pull-artifacts clean

.ONESHELL:
.SHELLFLAGS := -ec

# The only handwritten route to manuscript_marked/: the baseline is read-only
# and manuscript_clean/ is the editable source. Double spacing is required by
# PLOS but may be commented out while working, so restore it before latexdiff.
prepare-marked-manuscript:
	BASELINE=initial_submission/main.tex
	MANUSCRIPT=revision/manuscript_clean/manuscript.tex
	MARKED=revision/manuscript_marked/manuscript-track-changes.tex
	# Kept in lockstep with .vscode/settings.json so the tracked-changes source
	# is generated with the same TeX Live release as the manuscript PDF.
	IMAGE=texlive/texlive:TL2025-historic
	sed -i -E 's/^%[[:space:]]*(\\usepackage\{setspace\}|\\doublespacing)[[:space:]]*$$/\1/' "$$MANUSCRIPT"
	grep -qxF '\\doublespacing' "$$MANUSCRIPT" || { echo "$$MANUSCRIPT: \\doublespacing not found" >&2; exit 2; }
	docker run --rm --user "$$(id -u):$$(id -g)" -v "$(CURDIR):/work" -w /work "$$IMAGE" \
		latexdiff --encoding=utf8 --append-safecmd="rev,revs" "$$BASELINE" "$$MANUSCRIPT" > "$$MARKED"

# Render publication figures and tables directly into the working manuscript.
# Example: make pull-artifacts ANALYSIS_INPUT_DIR="/abs/path/to/results"
pull-artifacts:
	export ANALYSIS_OUTPUT_DIR="$(CURDIR)/revision/manuscript_clean"
	$(MAKE) -C ../python_project analysis

# Self-contained submission project: manuscript without embedded graphics or
# inline supporting items, plus figures/Fig<N>.tif repackaged from the TIFFs
# pull-artifacts already rendered and supporting_information/S<N>_<Type>.tex
# lifted out of the Supporting information section, each a standalone document.
# The manuscript keeps only the PLOS legend paragraph for each of them. The
# .tex files are left uncompiled, like every other source here: their PDFs come
# from the editor's build-on-save. The output directory is overwritten wholesale.
# Example: make prepare-plos-submission OUTPUT=/tmp/plos
prepare-plos-submission:
	export PLOS_SOURCE_DIR="$$(realpath -m "$${SOURCE:-revision/manuscript_clean}")"
	export PLOS_OUTPUT_DIR="$$(realpath -m "$${OUTPUT:-revision/manuscript_clean_plos}")"
	$(MAKE) -C ../python_project plos-submission

# Stray LaTeX artifacts left next to the sources by a hand-run build. Kept on
# purpose: .build/ (latexmk cache), the PDFs and the .synctex.gz.
clean:
	EXTS='aux|log|out|fls|fdb_latexmk|bbl|blg|toc|lof|lot'
	find . -not -path '*/.build/*' -type f -regextype posix-extended -regex ".*\.($$EXTS)$$" -print -delete
