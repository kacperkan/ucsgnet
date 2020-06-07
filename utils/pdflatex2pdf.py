"""The script converts product of the Inkscape latex conversion to a single pdf


It reads XML file from the draw.io, converts it to the Inkscape specific format
puts it into `standalone` environment of the latex, and then compiles to a
single pdf file.

Requirements:
- Inkscape >= 1.0
- svg file exported from draw.io with "Mathematical Typesetting" turned off.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from drawio2inkscape import parse

LATEX_TEMPLATE = """
\\documentclass[border={{{left}pt {bottom}pt {right}pt {up}pt}}]{{standalone}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}

\\begin{{document}}

\\input{{{filename}.pdf_tex}}
\\end{{document}}
"""

DOCS = """
Converts SVG style of the draw.io to the Inkscape readable format.

While the output of SVG from the draw.io is readable, it contains `foreignObject`
tags which are not interpretable for the Inkscape. This causes text truncates
and with the present of huge amounts of text, it can become quickly tedious
to convert each string seperatly.

The script copies text contents of the `foreignObject` to the nearest `text`
tag under the `switch` tag. It assumes that all texts are written in latex
in draw.io as `$$<text>$$` and the scripts converts it to `$<text>$`.
"""


def convert(in_path: str, left: int, bottom: int, right: int, up: int):
    out_path = Path(in_path).with_suffix(".pdf")
    in_path = Path(in_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy2(in_path.as_posix(), temp_dir)
        temp_file_path = os.path.join(temp_dir, in_path.name)
        parse(temp_file_path, temp_file_path)
        retcode = subprocess.call(
            [
                "inkscape",
                "-D",
                temp_file_path,
                "--export-type=pdf",
                "--export-latex",
            ]
        )
        assert retcode == 0, "Incorrect Inkscape parsing"
        rendered_pdf = "output.tex"

        latex_file_content = LATEX_TEMPLATE.format(
            left=left,
            bottom=bottom,
            right=right,
            up=up,
            filename=in_path.with_suffix("").name,
        )

        with open(os.path.join(temp_dir, rendered_pdf), "w") as f:
            f.write(latex_file_content)

        current_workdir = os.getcwd()
        os.chdir(temp_dir)
        retcode = subprocess.call(
            [
                "pdflatex",
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-recorder",
                rendered_pdf,
            ]
        )
        os.chdir(current_workdir)
        assert retcode == 0, "Incorrect PDF latex generation"
        shutil.copy2(os.path.join(temp_dir, "output.pdf"), out_path.as_posix())


def main():
    parser = argparse.ArgumentParser(description=DOCS)
    parser.add_argument("in_path", help="Input SVG file from the draw.io")

    parser.add_argument(
        "--left",
        help=(
            "Left cut for the standalone environment to remove borders in "
            "some cases "
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--bottom",
        help=(
            "Bottom cut for the standalone environment to remove borders in "
            "some cases "
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--right",
        help=(
            "Right cut for the standalone environment to remove borders in "
            "some cases "
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--up",
        help=(
            "Upper cut for the standalone environment to remove borders in "
            "some cases "
        ),
        default=0,
        type=int,
    )

    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
