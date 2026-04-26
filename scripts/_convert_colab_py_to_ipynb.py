"""One-shot helper: convert notebooks/colab_cells_sft.py (Jupytext-style)
to notebooks/colab_cells_sft.ipynb so judges get a one-click Colab notebook.

Splits on `# %% [Cell N] -- description` markers; emits a markdown header
cell + a code cell for each section. Module docstring at the top of the
.py file becomes the lead markdown cell.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks", "colab_cells_sft.py")
DST = os.path.join(ROOT, "notebooks", "colab_cells_sft.ipynb")


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True) or [""],
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True) or [""],
    }


def main() -> int:
    with open(SRC) as f:
        text = f.read()

    # Pull the module docstring as the lead markdown cell.
    m = re.match(r'\s*"""(.*?)"""\s*', text, re.DOTALL)
    if m:
        intro = m.group(1).strip()
        text = text[m.end():]
    else:
        intro = ""

    # Split on cell markers: keep the marker text so we can use it as a header.
    parts = re.split(r'(?m)^# %% (\[Cell \d+\][^\n]*)\n', text)
    # parts: [preamble_after_docstring, header1, code1, header2, code2, ...]

    cells: list[dict] = []
    cells.append(_md_cell(
        f"# Protocol One — Rejection-Sampling SFT (Colab T4)\n\n{intro}"
    ))

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        code = parts[i + 1].strip()
        # header looks like "[Cell 1] -- Verify GPU"
        nice = header.replace("[", "").replace("]", "").replace(" -- ", " — ", 1)
        cells.append(_md_cell(f"## {nice}"))
        cells.append(_code_cell(code))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(DST, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"OK wrote {DST}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
