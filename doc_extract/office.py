"""office.py - Word / PowerPoint / Excel / OpenDocument files.

OpenDataLoader PDF only reads PDFs (its README says "Process Word/Excel/PPT: No").
So Office files are handled in two ways:

  ROUTE A (best, default): LibreOffice (headless) converts the file to PDF, then the normal PDF pipeline
                           runs. Result: real page numbers, headers, footers, tables, images, bounding boxes.
  ROUTE B (fallback)     : if LibreOffice is not installed, .docx files are read directly with python-docx
                           (headings, lists, tables, images, headers/footers; pages only at explicit breaks).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # run LibreOffice as an external program
from pathlib import Path

from .renderers import IMG_TOKEN

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# ROUTE A: LibreOffice
# --------------------------------------------------------------------------------------
def find_libreoffice(cfg) -> str | None:
    """Return the path of soffice(.exe) or None if LibreOffice is not installed."""
    if cfg.libreoffice_path and os.path.exists(cfg.libreoffice_path):  # explicit setting wins
        return cfg.libreoffice_path
    for name in ("soffice", "libreoffice"):  # on PATH (Linux / macOS / some Windows)
        found = shutil.which(name)
        if found:
            return found
    for cand in (r"C:\Program Files\LibreOffice\program\soffice.exe",  # default Windows locations
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                 "/Applications/LibreOffice.app/Contents/MacOS/soffice"):  # macOS
        if os.path.exists(cand):
            return cand
    return None


def convert_to_pdf(cfg, src: str, out_dir: Path, profile_root: Path) -> Path:
    """Convert one Office file to PDF with LibreOffice and return the PDF path.

    Each worker process uses its OWN LibreOffice profile folder, otherwise parallel runs block each other.
    """
    exe = find_libreoffice(cfg)
    if not exe:
        raise RuntimeError("LibreOffice not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = (profile_root / f"lo_profile_{os.getpid()}").resolve()  # per-process profile
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [exe, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--norestore",
           "--convert-to", "pdf:writer_pdf_Export" if Path(src).suffix.lower() in (".docx", ".doc", ".odt", ".rtf") else "pdf",
           "--outdir", str(out_dir), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.office_timeout_s)  # wait for completion
    pdf = out_dir / (Path(src).stem + ".pdf")  # LibreOffice names the result after the source
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError(f"LibreOffice failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
    # Give the PDF a name that is unique per source file (the folder name is "office_<hash>"). Otherwise
    # "report.docx" -> "report.pdf" would collide with a real "report.pdf" inside OpenDataLoader's output folder.
    unique = out_dir / f"{out_dir.name}.pdf"
    os.replace(pdf, unique)
    return unique


# --------------------------------------------------------------------------------------
# ROUTE B: python-docx
# --------------------------------------------------------------------------------------
_EXT_BY_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/bmp": "bmp",
                "image/tiff": "tiff", "image/x-emf": "emf", "image/x-wmf": "wmf", "image/svg+xml": "svg"}


def docx_to_elements(src: str, image_dir: Path) -> tuple[list[dict], int]:
    """Read a .docx directly. Returns (elements with 'page number', page count).

    Images are written to image_dir and referenced with the {{IMG}} token.
    """
    from docx import Document  # imported lazily: only needed on this route
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(src)
    elements: list[dict] = []
    page = 1  # current page (counted from explicit/rendered page breaks)
    img_count = 0
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    def add_header_footer(kind: str, pg: int) -> None:
        """Add header/footer text of the first/last section as elements."""
        sections = doc.sections
        sec = sections[0] if kind == "header" else sections[-1]
        part = sec.header if kind == "header" else sec.footer
        text = " ".join(p.text.strip() for p in part.paragraphs if p.text.strip())
        if text:
            elements.append({"type": kind, "page number": pg, "content": text})

    add_header_footer("header", 1)  # page header first

    def save_image(blip) -> str | None:
        """Extract an embedded picture; return its token path."""
        nonlocal img_count
        rid = blip.get(R + "embed")
        part = doc.part.related_parts.get(rid) if rid else None
        if part is None:
            return None
        img_count += 1
        ext = _EXT_BY_MIME.get(part.content_type, "bin")
        name = f"imageFile{img_count}.{ext}"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / name).write_bytes(part.blob)  # write the raw picture bytes
        return f"{IMG_TOKEN}/{name}"

    def runs_to_text(par: Paragraph) -> str:
        """Paragraph text with **bold** / *italic* kept; also emits image elements found inside it."""
        nonlocal page
        parts = []
        for run in par.runs:
            r_el = run._r
            if r_el.findall(f".//{W}lastRenderedPageBreak") or any(b.get(W + "type") == "page" for b in r_el.findall(f".//{W}br")):
                page += 1  # a page break occurred inside this paragraph
            for blip in r_el.iter(A + "blip"):  # pictures anchored in this run
                src_path = save_image(blip)
                if src_path:
                    elements.append({"type": "image", "page number": page, "source": src_path})
            t = run.text
            if not t.strip():
                parts.append(t)
                continue
            if run.bold and run.italic:
                t = f"***{t.strip()}*** "
            elif run.bold:
                t = f"**{t.strip()}** "
            elif run.italic:
                t = f"*{t.strip()}* "
            parts.append(t)
        return "".join(parts).replace("**  **", " ").strip()

    def table_element(tbl: Table) -> dict:
        """Convert a Word table, preserving column spans and vertical merges."""
        rows, origin = [], {}  # origin[col_index] -> cell dict that started a vertical merge
        for r_idx, tr in enumerate(tbl._tbl.tr_lst, start=1):
            cells, col = [], 1
            for tc in tr.tc_lst:
                span = tc.grid_span or 1
                vm = tc.vMerge  # None | "restart" | "continue"
                # cell text = all non-empty paragraphs inside the cell, joined by spaces
                text = " ".join("".join(p.itertext()).strip() for p in tc.iterchildren(W + "p")
                                if "".join(p.itertext()).strip())
                if vm == "continue" and col in origin:  # continuation of a cell above: grow its row span
                    origin[col]["row span"] += 1
                else:
                    cell = {"type": "table cell", "page number": page, "row number": r_idx, "column number": col,
                            "row span": 1, "column span": span,
                            "kids": [{"type": "paragraph", "page number": page, "content": text}] if text else []}
                    cells.append(cell)
                    origin[col] = cell
                col += span
            rows.append({"type": "table row", "row number": r_idx, "cells": cells})
        n_cols = max((sum(c["column span"] for c in r["cells"]) for r in rows), default=0)
        return {"type": "table", "page number": page, "number of rows": len(rows), "number of columns": n_cols, "rows": rows}

    pending_items: list[dict] = []  # consecutive list paragraphs are grouped into one list element
    pending_style = "bullets"

    def flush_list() -> None:
        nonlocal pending_items
        if pending_items:
            elements.append({"type": "list", "page number": pending_items[0]["page number"], "numbering style": pending_style,
                             "number of list items": len(pending_items), "list items": pending_items})
            pending_items = []

    for child in doc.element.body.iterchildren():  # paragraphs and tables in true document order
        if child.tag == W + "p":
            par = Paragraph(child, doc)
            if par.paragraph_format.page_break_before:
                page += 1
            text = runs_to_text(par)
            style = (par.style.name if par.style is not None else "") or ""
            if not text:
                continue
            is_list = child.pPr is not None and child.pPr.numPr is not None or style.lower().startswith("list")
            if is_list:  # bullet / numbered paragraph
                pending_style = "arabic numbers" if "number" in style.lower() else "bullets"
                pending_items.append({"type": "list item", "page number": page, "content": text, "kids": []})
                continue
            flush_list()
            low = style.lower()
            if low == "title":
                elements.append({"type": "heading", "heading level": 1, "page number": page, "content": text})
            elif low.startswith("heading"):
                digits = "".join(ch for ch in low if ch.isdigit())
                elements.append({"type": "heading", "heading level": min(6, int(digits) + 1) if digits else 2, "page number": page, "content": text})
            elif low == "caption":
                elements.append({"type": "caption", "page number": page, "content": text})
            else:
                elements.append({"type": "paragraph", "page number": page, "content": text})
        elif child.tag == W + "tbl":
            flush_list()
            elements.append(table_element(Table(child, doc)))
    flush_list()
    add_header_footer("footer", page)  # footer at the end of the last page
    return elements, page
