"""renderers.py - convert ODL-style element lists into Markdown / plain text, and split big outputs.

Used for pages that did NOT come from OpenDataLoader's own markdown (OCR pages, python-docx fallback,
plain-text inputs). OpenDataLoader pages keep ODL's own markdown/text.
"""

from __future__ import annotations

import html  # escape characters inside HTML tables
import re

IMG_TOKEN = "{{IMG}}"  # placeholder replaced by the real relative image folder when files are written


def _esc_cell(text: str) -> str:
    """Make text safe for a markdown pipe-table cell."""
    return (text or "").replace("|", "\\|").replace("\n", "<br>").strip()


def _cell_text(cell: dict) -> str:
    """Join the text of every kid paragraph inside a table cell."""
    parts = []
    for kid in cell.get("kids", []):
        if kid.get("content"):
            parts.append(str(kid["content"]))
        elif kid.get("type") == "list":  # nested list inside a cell
            parts.extend(i.get("content", "") for i in kid.get("list items", []))
    return " ".join(p for p in parts if p).strip()


def table_to_markdown(tab: dict) -> str:
    """Render a table element. Pipe table normally; HTML table when cells are merged (spans > 1)."""
    rows = tab.get("rows", [])
    n_cols = tab.get("number of columns", 0) or max((len(r["cells"]) for r in rows), default=0)
    has_span = any(c.get("row span", 1) > 1 or c.get("column span", 1) > 1 for r in rows for c in r["cells"])
    if has_span:  # markdown cannot express merged cells -> use HTML (allowed in .md)
        out = ["<table>"]
        for r in rows:
            out.append("<tr>")
            for c in r["cells"]:
                rs, cs = c.get("row span", 1), c.get("column span", 1)
                attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                out.append(f"<td{attrs}>{html.escape(_cell_text(c))}</td>")
            out.append("</tr>")
        out.append("</table>")
        return "\n".join(out)
    grid: list[list[str]] = []
    for r in rows:  # simple grid: one entry per column position
        line = [""] * max(n_cols, 1)
        for c in r["cells"]:
            idx = c.get("column number", 1) - 1
            if 0 <= idx < len(line):
                line[idx] = _esc_cell(_cell_text(c))
        grid.append(line)
    if not grid:
        return ""
    md = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join(["---"] * len(grid[0])) + "|"]  # first row = header
    md += ["| " + " | ".join(row) + " |" for row in grid[1:]]
    return "\n".join(md)


def elements_to_markdown(elements: list[dict], img_prefix: str = IMG_TOKEN) -> str:
    """Render elements in order to markdown."""
    blocks: list[str] = []
    for el in elements:
        t = el.get("type")
        if t == "heading":
            lvl = max(1, min(6, int(el.get("heading level", 2))))
            blocks.append("#" * lvl + " " + str(el.get("content", "")).strip())
        elif t in ("paragraph", "caption", "text block", "formula", "header", "footer"):
            txt = str(el.get("content", "")).strip()
            if not txt:
                continue
            if t == "formula":
                blocks.append(f"$$\n{txt}\n$$")  # LaTeX block
            elif t in ("header", "footer"):
                blocks.append(f"<!-- {t} -->\n*{txt}*")  # keep page furniture, visually de-emphasised
            elif t == "caption":
                blocks.append(f"*{txt}*")
            else:
                blocks.append(txt)
        elif t == "list":
            items = []
            for it in el.get("list items", []):
                c = str(it.get("content", "")).strip()
                c = re.sub(r"^\s*([•●▪◦·■□\-–—*]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+", "", c)  # drop the original bullet/number
                items.append(c)
            numbered = el.get("numbering style", "") != "bullets"
            first = re.match(r"^\s*\(?(\d{1,3})[.)]", str(el["list items"][0].get("content", ""))) if el.get("list items") else None
            start = int(first.group(1)) if first else 1  # keep the document's own first number (e.g. a list starting at 3)
            blocks.append("\n".join((f"{start + i}. " if numbered else "- ") + c for i, c in enumerate(items)))
        elif t == "table":
            md = table_to_markdown(el)
            if md:
                blocks.append(md)
        elif t == "image":
            src = el.get("source", "")
            blocks.append(f"![{el.get('description', 'image')}](<{src}>)" if src else "")  # <...> keeps ( ) and spaces in paths valid
        elif t == "qr code":
            blocks.append(f"> **QR code:** {el.get('content', '')}")
    return "\n\n".join(b for b in blocks if b) + ("\n" if blocks else "")


def elements_to_text(elements: list[dict]) -> str:
    """Render elements to plain text (tables become tab-separated rows)."""
    out: list[str] = []
    for el in elements:
        t = el.get("type")
        if t == "table":
            for r in el.get("rows", []):
                out.append("\t".join(_cell_text(c) for c in r["cells"]))
        elif t == "list":
            out.extend(i.get("content", "") for i in el.get("list items", []))
        elif t == "qr code":
            out.append(f"[QR code] {el.get('content', '')}")
        elif t == "image":
            continue
        elif el.get("content"):
            out.append(str(el["content"]))
        out.append("")  # blank line between blocks
    return "\n".join(out).strip() + "\n"


def page_marker(cfg, kind: str, n: int) -> str:
    """Page separator text for 'md' or 'txt' as set in the config."""
    return (cfg.page_marker_md if kind == "md" else cfg.page_marker_txt).format(n=n)


def join_pages(cfg, kind: str, pages: dict[int, str]) -> str:
    """Join {page_number: text} into one document with visible page separators."""
    chunks = []
    for n in sorted(pages):
        body = pages[n].strip("\n")
        if kind == "md":
            chunks.append(f"{page_marker(cfg, 'md', n)}\n\n{body}\n\n---\n")  # marker, content, horizontal rule
        else:
            chunks.append(f"{page_marker(cfg, 'txt', n)}\n\n{body}\n")
    return "\n".join(chunks)


def split_into_parts(cfg, kind: str, pages: dict[int, str]) -> list[tuple[str, str]]:
    """Split pages into parts of cfg.split_pages pages. Returns [(label 'p0001-0100', text)]."""
    nums = sorted(pages)
    parts = []
    for i in range(0, len(nums), cfg.split_pages):
        group = nums[i:i + cfg.split_pages]
        text = join_pages(cfg, kind, {n: pages[n] for n in group})
        parts.append((f"p{group[0]:04d}-{group[-1]:04d}", text))
    return parts
