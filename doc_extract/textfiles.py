"""textfiles.py - tiny handlers for text-like inputs (.txt .md .csv .html) so NOTHING in the folder is ignored."""

from __future__ import annotations

import csv
import io
from pathlib import Path


def _read_text(path: str) -> str:
    """Read a text file trying the usual encodings (UTF-8 first, then Windows)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):  # latin-1 never fails, so this always returns
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    return raw.decode("utf-8", "replace")


def text_file_to_pages(path: str) -> tuple[dict[int, str], dict[int, str], list[dict]]:
    """Return (md_pages, txt_pages, elements) for a text-like file (always a single page)."""
    ext = Path(path).suffix.lower()
    text = _read_text(path)
    elements: list[dict] = []
    if ext in (".html", ".htm"):
        from markdownify import markdownify  # HTML -> markdown (keeps headings, lists, tables)
        from bs4 import BeautifulSoup
        md = markdownify(text, heading_style="ATX").strip()
        plain = BeautifulSoup(text, "html.parser").get_text("\n").strip()
    elif ext == ".csv":
        rows = list(csv.reader(io.StringIO(text)))
        width = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (width - len(r)) for r in rows]
        esc = lambda c: c.replace("|", "\\|").replace("\n", " ")
        md = "\n".join(["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"] +
                       ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]) if rows else ""
        plain = "\n".join("\t".join(r) for r in rows)
    else:  # .txt / .md / .markdown
        md, plain = text.strip(), text.strip()
    for block in [b.strip() for b in plain.split("\n\n") if b.strip()]:
        elements.append({"type": "paragraph", "page number": 1, "content": block})
    return {1: md + "\n"}, {1: plain + "\n"}, elements
