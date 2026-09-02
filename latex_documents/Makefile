.PHONY: prepare-marked-manuscript prepare-plos-submission pull-artifacts clean

.ONESHELL:
.SHELLFLAGS := -ec

# Kept in lockstep with .vscode/settings.json so the tracked-changes source is
# generated with the same TeX Live release as the manuscript PDF.
TEXLIVE_IMAGE := texlive/texlive:TL2025-historic

# Python is kept here because it implements this one Make target only.
define PLOS_PREPARE
from pathlib import Path
import re
import shutil
import sys
from PIL import Image

source, output = map(Path, sys.argv[1:3])
source, output = source.resolve(), output.resolve()
# PLOS ONE figure limits at 300 dpi: https://journals.plos.org/plosone/s/figures
PLOS_MIN_WIDTH_PX = 789
PLOS_MAX_WIDTH_PX = 2250
PLOS_MAX_HEIGHT_PX = 2625
# initial_submission/ uses main.tex, revision/manuscript_clean/ uses manuscript.tex.
main = next((source / name for name in ("main.tex", "manuscript.tex")
             if (source / name).is_file()), None)
if main is None:
    raise SystemExit(f"main.tex/manuscript.tex not found in: {source}")
if output in (source, source.parent):
    raise SystemExit("OUTPUT must be a separate directory")

input_pattern = re.compile(r"^(?P<indent>\s*)\\input\{(?P<path>[^}]+)}\s*(?:%.*)?\Z")
def expand_inputs(text, directory):
    expanded = []
    for line in text.splitlines(keepends=True):
        match = input_pattern.match(line) if not line.lstrip().startswith("%") else None
        if not match:
            expanded.append(line)
            continue
        included = directory / match.group("path")
        if not included.suffix:
            included = included.with_suffix(".tex")
        if not included.is_file():
            raise SystemExit(f"input file not found: {included}")
        expanded.extend(expand_inputs(included.read_text(encoding="utf-8"), included.parent))
    return expanded

text = "".join(expand_inputs(main.read_text(encoding="utf-8"), source))
pattern = re.compile(r"^(?P<indent>\s*)\\includegraphics(?:\[[^]]*])?\{(?P<path>[^}]+)}\s*(?:%.*)?\Z")
graphics, lines = [], []
for line in text.splitlines(keepends=True):
    match = pattern.match(line) if not line.lstrip().startswith("%") else None
    if match:
        graphic = source / match.group("path")
        if not graphic.is_file():
            raise SystemExit(f"figure not found: {graphic}")
        graphics.append(graphic)
        lines.append(f"{match.group('indent')}% Figure uploaded separately as Fig{len(graphics)}.tif\n")
    else:
        lines.append(line)
if not graphics:
    raise SystemExit(f"no active one-line \\includegraphics commands found in {main.name}")

if output.exists():
    shutil.rmtree(output)
figures = output / "figures"
figures.mkdir(parents=True)
(output / main.name).write_text("".join(lines), encoding="utf-8")
for name in ("refs.bib", "plos2025.bst"):
    if (source / name).is_file():
        shutil.copy2(source / name, output / name)
for number, graphic in enumerate(graphics, 1):
    with Image.open(graphic) as image:
        if image.mode not in {"1", "L", "RGB"}:
            image = image.convert("RGB")
        width, height = image.size
        image.save(figures / f"Fig{number}.tif", format="TIFF",
                   compression="tiff_lzw", dpi=(300, 300))
    if width < PLOS_MIN_WIDTH_PX or width > PLOS_MAX_WIDTH_PX or height > PLOS_MAX_HEIGHT_PX:
        raise SystemExit(
            f"{graphic.name} is {width}x{height}px; outside PLOS figure limits "
            f"({PLOS_MIN_WIDTH_PX}-{PLOS_MAX_WIDTH_PX}px wide, up to {PLOS_MAX_HEIGHT_PX}px tall)."
        )
print(f"PLOS submission project: {output}")
endef
export PLOS_PREPARE

# The only handwritten route to revision/manuscript_marked/: the submitted
# manuscript is the read-only baseline and manuscript_clean/ is the editable
# source.
# Double spacing is required by PLOS but may be commented out in
# manuscript_clean/ while working; restore it before latexdiff so both the
# clean source and the tracked-changes file go out double-spaced.
prepare-marked-manuscript:
	@sed -i -E 's/^%[[:space:]]*(\\usepackage\{setspace\}|\\doublespacing)[[:space:]]*$$/\1/' \
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
# manuscript without embedded graphics and figures/Fig<N>.tif.
# Example: make prepare-plos-submission SOURCE=revision/manuscript_clean OUTPUT=revision/manuscript_clean_plos
prepare-plos-submission:
	@test -n "$(SOURCE)" && test -n "$(OUTPUT)" || { echo "Usage: make prepare-plos-submission SOURCE=path/to/project OUTPUT=path/to/submission-project" >&2; exit 2; }
	@uv run --with pillow python -c "$$PLOS_PREPARE" "$(SOURCE)" "$(OUTPUT)"

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
