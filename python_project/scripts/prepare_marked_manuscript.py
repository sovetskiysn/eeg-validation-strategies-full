"""Generate and compile a marked LaTeX manuscript."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--image", default="texlive/texlive:TL2025-historic")
    args = parser.parse_args()

    baseline = args.baseline.resolve()
    manuscript = args.manuscript.resolve()
    output_pdf = args.output_pdf.resolve()
    project_root = Path(os.path.commonpath([baseline.parent, manuscript.parent]))
    manuscript_dir = manuscript.parent

    if not baseline.is_file():
        raise FileNotFoundError(f"Baseline not found: {baseline}")
    if not manuscript.is_file():
        raise FileNotFoundError(f"Manuscript not found: {manuscript}")

    # The manuscript may include generated floats via ``\input``, while
    # a submission-format baseline may have tables expanded inside a local
    # ``\begingroup`` wrapper. Exclude either representation before latexdiff
    # and restore the current floats afterwards, or a deleted wrapper can leave
    # table-formatting commands inside a latexdiff deletion macro.
    float_pattern = re.compile(
        r"(?:\\begingroup\s*.*?\\begin\{(table|figure)\}.*?\\end\{\1\}"
        r"\s*\\endgroup|\\begin\{(table|figure)\}.*?\\end\{\2\})",
        re.DOTALL,
    )

    with tempfile.TemporaryDirectory(prefix="marked-manuscript-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        old_text = baseline.read_text(encoding="utf-8")
        new_text = manuscript.read_text(encoding="utf-8")
        new_float_blocks = [match.group(0) for match in float_pattern.finditer(new_text)]

        old_without_floats = float_pattern.sub("", old_text)
        new_without_floats = float_pattern.sub("", new_text)
        old_path = temporary_path / "baseline.tex"
        new_path = temporary_path / "manuscript.tex"
        diff_path = temporary_path / "manuscript-track-changes.tex"
        old_path.write_text(old_without_floats, encoding="utf-8")
        new_path.write_text(new_without_floats, encoding="utf-8")

        with diff_path.open("w", encoding="utf-8") as diff_handle:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-v",
                    f"{project_root}:/work",
                    "-v",
                    f"{temporary_path}:/tmp/marked",
                    "-w",
                    "/work",
                    args.image,
                    "latexdiff",
                    "--encoding=utf8",
                    "--type=CFONT",
                    "--graphics-markup=none",
                    "--append-textcmd=rev,revs",
                    "/tmp/marked/baseline.tex",
                    "/tmp/marked/manuscript.tex",
                ],
                check=True,
                stdout=diff_handle,
            )

        diff_text = diff_path.read_text(encoding="utf-8")
        # When latexdiff replaces the complete argument of a text command such
        # as ``\rev{...}``, it can emit an internal ``\MBLOCKRIGHTBRACE``
        # marker as a commented deletion command. That leaves the surrounding
        # ``\DIFdel{`` without its closing brace and makes the marked
        # manuscript uncompilable. Preserve the deleted command argument and
        # close both the command and the deletion wrapper instead.
        diff_text = diff_text.replace(
            "}%DIFDELCMD < \\MBLOCKRIGHTBRACE\n%DIFDELCMD < %%%",
            "}}",
        )
        float_text = "\n\n".join(new_float_blocks)
        bibliography_marker = "\\bibliography{refs}"
        if bibliography_marker not in diff_text:
            raise RuntimeError("Could not find bibliography insertion point")
        diff_text = diff_text.replace(
            bibliography_marker,
            f"{float_text}\n\n{bibliography_marker}",
            1,
        )
        diff_path.write_text(diff_text, encoding="utf-8")

        build_path = temporary_path / "build"
        build_path.mkdir()
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{project_root}:/work",
                "-v",
                f"{temporary_path}:/tmp/marked",
                "-w",
                f"/work/{manuscript_dir.relative_to(project_root)}",
                args.image,
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-outdir=/tmp/marked/build",
                "/tmp/marked/manuscript-track-changes.tex",
            ],
            check=True,
        )

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_path / "manuscript-track-changes.pdf", output_pdf)


if __name__ == "__main__":
    main()
