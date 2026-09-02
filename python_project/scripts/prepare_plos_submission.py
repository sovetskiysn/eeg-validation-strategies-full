"""Package a self-contained PLOS ONE submission project from a manuscript source.

Expands \\input, strips \\includegraphics lines (replacing them with the PLOS
upload note), and copies each figure's pre-rendered TIFF (written by
write_article_artifacts in src/analysis.py) into figures/Fig<N>.tif in
\\includegraphics order.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

INPUT_PATTERN = re.compile(r"^(?P<indent>\s*)\\input\{(?P<path>[^}]+)}\s*(?:%.*)?\Z")
GRAPHICS_PATTERN = re.compile(r"^(?P<indent>\s*)\\includegraphics(?:\[[^]]*])?\{(?P<path>[^}]+)}\s*(?:%.*)?\Z")


def expand_inputs(text: str, directory: Path) -> list[str]:
    expanded = []
    for line in text.splitlines(keepends=True):
        match = INPUT_PATTERN.match(line) if not line.lstrip().startswith("%") else None
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


def main() -> None:
    """Build the submission project from sys.argv[1:3] = (SOURCE, OUTPUT)."""
    source, output = (Path(arg).resolve() for arg in sys.argv[1:3])
    # initial_submission/ uses main.tex, revision/manuscript_clean/ uses manuscript.tex.
    main_tex = next((source / name for name in ("main.tex", "manuscript.tex")
                      if (source / name).is_file()), None)
    if main_tex is None:
        raise SystemExit(f"main.tex/manuscript.tex not found in: {source}")
    if output in (source, source.parent):
        raise SystemExit("OUTPUT must be a separate directory")

    text = "".join(expand_inputs(main_tex.read_text(encoding="utf-8"), source))
    graphics, lines = [], []
    for line in text.splitlines(keepends=True):
        match = GRAPHICS_PATTERN.match(line) if not line.lstrip().startswith("%") else None
        if match:
            graphic = source / match.group("path")
            if not graphic.is_file():
                raise SystemExit(f"figure not found: {graphic}")
            graphics.append(graphic)
            lines.append(f"{match.group('indent')}% Figure uploaded separately as Fig{len(graphics)}.tif\n")
        else:
            lines.append(line)
    if not graphics:
        raise SystemExit(f"no active one-line \\includegraphics commands found in {main_tex.name}")

    if output.exists():
        shutil.rmtree(output)
    figures = output / "figures"
    figures.mkdir(parents=True)
    (output / main_tex.name).write_text("".join(lines), encoding="utf-8")
    for name in ("refs.bib", "plos2025.bst"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    for number, graphic in enumerate(graphics, 1):
        # graphic is .../figures/source_png/<name>.png; its TIFF sibling was
        # already rendered and size-checked by write_article_artifacts.
        tiff = graphic.parent.parent / "source_tiff" / f"{graphic.stem}.tif"
        if not tiff.is_file():
            raise SystemExit(
                f"{tiff} not found; run `make pull-artifacts` to regenerate figures first."
            )
        shutil.copy2(tiff, figures / f"Fig{number}.tif")
    print(f"PLOS submission project: {output}")


if __name__ == "__main__":
    main()
