"""Converts SVG style of the draw.io to the Inkscape readable format.

While the output of SVG from the draw.io is readable, it contains `foreignObject`
tags which are not interpretable for the Inkscape. This causes text truncates
and with the present of huge amounts of text, it can become quickly tedious
to convert each string seperatly.

The script copies text contents of the `foreignObject` to the nearest `text`
tag under the `switch` tag. It assumes that all texts are written in latex
in draw.io as `$$<text>$$` and the scripts converts it to `$<text>$`.
"""
import argparse
from xml.etree import ElementTree

NAMESPACE = "{http://www.w3.org/2000/svg}"
ElementTree.register_namespace("", NAMESPACE[1:-1])
ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")

DOCS = """
Converts SVG style of the draw.io to the Inkscape readable format.

While the output of SVG from the draw.io is readable, it contains `foreignObject`
tags which are not interpretable for the Inkscape. This causes text truncates
and with the present of huge amounts of text, it can become quickly tedious
to convert each string seperatly.

The script copies text contents of the `foreignObject` to the nearest `text`
tag under the `switch` tag. It assumes that all texts are written in latex

Note: Remember about disabling Math Typesetting before exporting to SVG.
"""


def parse(in_path: str, out_path: str):
    with open(in_path) as f:
        doc = ElementTree.parse(f)
    root = doc.getroot()
    for parent_switch in root.findall(f".//{NAMESPACE}foreignObject/.."):
        foreign_object = parent_switch.find(f".//{NAMESPACE}foreignObject")[0][
            0
        ][0]
        original_text = foreign_object.text
        modified_text = original_text.replace("$$", "$")
        text_object = parent_switch.find(f".//{NAMESPACE}text")
        text_object.text = modified_text

    with open(out_path, "wb") as f:
        f.write(ElementTree.tostring(root))


def main():
    parser = argparse.ArgumentParser(description=DOCS)
    parser.add_argument("in_path", help="Input SVG file from the draw.io")
    parser.add_argument("out_path", help="Output SVG file from the draw.io")

    args = parser.parse_args()
    parse(**vars(args))


if __name__ == "__main__":
    main()
