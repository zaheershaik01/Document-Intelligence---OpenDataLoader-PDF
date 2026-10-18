"""ocr_engine.py - structure-aware OCR built on Tesseract + OpenCV.

What "structure aware" means here (a plain Tesseract call only returns a flat string):
  1. Pre-processing  : several image variants (grayscale, contrast, binarised, bright/dark text masks).
  2. Best-of-N       : the variant that yields the most RELIABLE words wins.
  3. Language/script : detected automatically (Latin, Arabic, Devanagari ...) so English pages do not
                       get polluted with foreign characters.
  4. Layout analysis : words -> lines -> paragraphs -> headings / lists (by relative text height).
  5. Tables          : ruled tables are found with OpenCV line detection, every cell is OCR-ed on its
                       own and rebuilt as rows/columns (with row/column spans).
  6. QR codes        : decoded with OpenCV.
  7. Output          : elements in the SAME schema as OpenDataLoader JSON (type, page number,
                       bounding box in PDF points [left, bottom, right, top], content ...).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import statistics  # median of heights / widths
from dataclasses import dataclass, field
from pathlib import Path

import cv2  # OpenCV: image processing
import numpy as np  # arrays
import pytesseract  # Python wrapper around the tesseract command
from PIL import Image, ImageOps  # Pillow: loading images, EXIF rotation
from pytesseract import Output

log = logging.getLogger(__name__)

# Tesseract "script name" (from OSD) -> language packs that read that script.
SCRIPT_TO_LANG = {
    "Latin": "eng", "Arabic": "ara", "Devanagari": "hin", "Cyrillic": "rus",
    "Han": "chi_sim", "Japanese": "jpn", "Hangul": "kor", "Bengali": "ben",
    "Tamil": "tam", "Telugu": "tel", "Gujarati": "guj", "Gurmukhi": "pan",
    "Kannada": "kan", "Malayalam": "mal", "Greek": "ell", "Hebrew": "heb", "Thai": "tha",
}

_BULLET_RE = re.compile(r"^\s*([•●▪◦·■□\-–—*]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")  # "• x", "1. x", "a) x"
_RTL_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")  # Hebrew/Arabic code points
_ALNUM_RE = re.compile(r"\w", re.UNICODE)  # at least one letter/digit


# --------------------------------------------------------------------------------------
# data containers
# --------------------------------------------------------------------------------------
@dataclass
class Word:
    """One recognised word with its box in BASE-image pixels."""
    text: str
    conf: float
    left: int
    top: int
    width: int
    height: int
    block: int
    par: int
    line: int


@dataclass
class PageOCR:
    """Everything OCR learned about one page / image."""
    elements: list = field(default_factory=list)  # ODL-style element dicts
    layout_text: str = ""  # fixed-width text that mimics the page layout
    mean_conf: float = 0.0  # mean word confidence 0-100
    n_words: int = 0  # number of kept words
    lang: str = ""  # Tesseract language string used
    variant: str = ""  # name of the winning preprocessing variant
    warnings: list = field(default_factory=list)  # human-readable problems


# --------------------------------------------------------------------------------------
# Tesseract set-up
# --------------------------------------------------------------------------------------
_installed_cache: list[str] | None = None  # languages installed (filled lazily)


def configure_tesseract(cfg) -> None:
    """Locate the tesseract program and tessdata folder (needed on Windows where PATH is rarely set)."""
    global _installed_cache
    _installed_cache = None  # forget cached language list
    cmd = cfg.tesseract_cmd or shutil.which("tesseract")  # explicit path > PATH
    if not cmd:  # try the usual Windows install locations
        for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(cand):
                cmd = cand
                break
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd  # tell pytesseract where it is
    if cfg.tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = str(cfg.tessdata_dir)  # custom language-data folder
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # 1 thread per Tesseract: we parallelise by process


def tesseract_available() -> bool:
    """True if the tesseract program can be started."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # not installed / wrong path
        return False


def installed_langs() -> list[str]:
    """Language packs installed for Tesseract (cached)."""
    global _installed_cache
    if _installed_cache is None:
        try:
            _installed_cache = [l for l in pytesseract.get_languages(config="") if l != "osd"]
        except Exception:
            _installed_cache = []
    return _installed_cache


def resolve_lang(requested: str, gray: np.ndarray | None, hint: str | None = None) -> str:
    """Pick the Tesseract language string.

    requested : "auto" or explicit such as "eng+hin"
    gray      : image used for script detection when requested == "auto"
    hint      : language already chosen for another page of the same document (skips detection)
    """
    have = set(installed_langs())  # what is installed
    if requested and requested != "auto":  # explicit choice: keep only installed packs
        wanted = [l for l in requested.split("+") if l in have]
        missing = [l for l in requested.split("+") if l not in have]
        if missing:
            log.warning("Tesseract language pack(s) not installed: %s (install them for better OCR)", ",".join(missing))
        return "+".join(wanted) if wanted else ("eng" if "eng" in have else "+".join(sorted(have)) or "eng")
    if hint:
        return hint
    default = "eng" if "eng" in have or not have else sorted(have)[0]
    if gray is None or len(have - {"eng"}) == 0:  # only English installed -> nothing to choose
        return default
    try:  # OSD = Tesseract "orientation and script detection"
        small = gray if max(gray.shape) <= 1600 else cv2.resize(gray, None, fx=1600 / max(gray.shape), fy=1600 / max(gray.shape))
        osd = pytesseract.image_to_osd(small, config="--psm 0", output_type=Output.DICT)
        script = osd.get("script", "Latin")  # e.g. "Arabic"
        lang = SCRIPT_TO_LANG.get(script, "eng")
        if lang in have and lang != "eng":
            cand = f"{lang}+eng" if "eng" in have else lang  # foreign script + English for numbers / Latin words
            # OSD is unreliable on decorative lettering: only accept it if the foreign pack really reads more text.
            m_foreign, m_default = _quick_mass(gray, cand), _quick_mass(gray, default)
            if m_foreign >= 6.0 and m_foreign >= 1.25 * m_default:
                return cand
            return default
        return default  # OSD says Latin (or an uninstalled script)
    except Exception:  # OSD fails on images with very little text
        pass
    # OSD unsure: score each installed non-English pack quickly and keep the best candidate.
    best, best_mass = default, _quick_mass(gray, default)
    for l in sorted(have - {"eng"}):
        cand = f"{l}+eng" if "eng" in have else l
        mass = _quick_mass(gray, cand)
        if mass >= 8.0 and mass > best_mass * 1.25:  # needs real evidence (>=8 confident words) AND a clear win
            best, best_mass = cand, mass
    return best


def _quick_mass(gray: np.ndarray, lang: str) -> float:
    """Reliable-word mass of a down-scaled copy (used only to compare languages)."""
    try:
        small = gray if max(gray.shape) <= 1200 else cv2.resize(gray, None, fx=1200 / max(gray.shape), fy=1200 / max(gray.shape))
        words = _tsv_words(small, lang, 3, 60, 35.0)
        return _mass(words)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------------------
# image loading + pre-processing
# --------------------------------------------------------------------------------------
def load_image_bgr(path: str) -> np.ndarray:
    """Read any image file as an OpenCV BGR array (fixes EXIF rotation, flattens transparency, first frame of TIFF/GIF)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # honour camera rotation flag
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))  # white background
            im = Image.alpha_composite(bg, im)  # remove transparency
        rgb = im.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)  # PIL is RGB, OpenCV is BGR


def pil_to_bgr(im: Image.Image) -> np.ndarray:
    """Convert a PIL image (e.g. a rendered PDF page) to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _estimate_skew(gray: np.ndarray) -> float:
    """Projection-profile skew estimate in degrees (+/-10). Returns 0 when unsure."""
    scale = 700 / max(gray.shape) if max(gray.shape) > 700 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1 else gray
    _, binar = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # text = white
    h, w = binar.shape
    best_angle, best_score = 0.0, -1.0
    base = None
    for angle in np.arange(-10, 10.01, 0.5):  # try small rotations
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(binar, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))  # sharp text lines => high variance of row sums
        if abs(angle) < 1e-9:
            base = score
        if score > best_score:
            best_angle, best_score = float(angle), score
    if base is None or best_score < base * 1.25:  # no clear gain -> do not rotate
        return 0.0
    if abs(best_angle) >= 9.9:  # hit the edge of the search range => probably a false signal (photo horizon etc.)
        return 0.0
    return best_angle


def _deskew(bgr: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate colour + gray images so text lines are horizontal (only if the skew is clear)."""
    angle = _estimate_skew(gray)  # degrees, 0 when unsure
    if abs(angle) < 0.5:  # tiny or unknown skew: leave untouched
        return bgr, gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)  # same matrix for both images
    rot = lambda im: cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot(bgr), rot(gray)


def _variants(g: np.ndarray, effort: str = "normal"):
    """Yield (name, image, psm) preprocessing candidates, cheapest first."""
    yield "gray", g, 3  # plain grayscale, automatic page segmentation
    yield "gray_block", g, 6  # same image but "one uniform text block" (good for notes/letters/short lists)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)  # local contrast boost
    yield "clahe", clahe, 3
    _, otsu = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", otsu, 3  # hard black/white
    for name, thr in (("bright92", max(float(np.percentile(g, 92)), 150.0)), ("bright200", 200.0), ("bright220", 220.0)):
        # light text on busy/dark photos: keep only the brightest pixels, shown as black text on white
        yield name, np.where(g >= thr, 0, 255).astype(np.uint8), 6
    thr = min(float(np.percentile(g, 8)), 110.0)  # dark text on busy/light backgrounds
    yield "dark", np.where(g <= thr, 0, 255).astype(np.uint8), 6


def _tsv_words(img: np.ndarray, lang: str, psm: int, timeout: int, min_conf: float) -> list[Word]:
    """Run Tesseract once and return the words that pass the confidence filter."""
    d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm} -c preserve_interword_spaces=1",
                                  output_type=Output.DICT, timeout=timeout)
    words: list[Word] = []
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            continue
        if not txt or conf < min_conf or not _ALNUM_RE.search(txt):  # drop empties / pure punctuation noise
            continue
        words.append(Word(txt, conf, int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i]),
                          int(d["block_num"][i]), int(d["par_num"][i]), int(d["line_num"][i])))
    return words


def _mass(words: list[Word]) -> float:
    """'Reliable word mass': sum of confidences/100 - rewards MANY CONFIDENT words, punishes noise."""
    return sum(w.conf / 100.0 for w in words if w.conf >= 60 and (len(w.text) > 1 or w.text.isalnum()))


# --------------------------------------------------------------------------------------
# tables (ruled tables via OpenCV)
# --------------------------------------------------------------------------------------
def _cluster(values: list[int], tol: int) -> list[int]:
    """Merge nearby coordinates (within tol px) into representative edges."""
    out: list[list[int]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(sum(g) / len(g)) for g in out]


def _nearest(edges: list[int], v: int) -> int:
    """Index of the edge closest to v."""
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def detect_tables(gray: np.ndarray) -> list[dict]:
    """Find ruled tables. Returns [{'bbox':(x,y,w,h), 'cells':[(x,y,w,h),...]}] in pixel coordinates."""
    h, w = gray.shape
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)  # local contrast
    _, otsu_bin = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # global
    if (otsu_bin > 0).mean() > 0.45:  # more than 45% "ink" => photo / dark page, no ruled table to find
        return []
    binar = cv2.bitwise_and(adaptive, otsu_bin)  # a real ruling line is dark both locally and globally
    hor = cv2.morphologyEx(binar, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 25), 1)))
    ver = cv2.morphologyEx(binar, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 25))))
    grid = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((3, 3), np.uint8))  # long straight lines only
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 80 or bh < 30 or bw * bh < 0.01 * w * h:  # too small to be a table
            continue
        roi = cv2.copyMakeBorder(grid[y:y + bh, x:x + bw], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)  # pad: outside = 1 region
        n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(roi), connectivity=4)  # cells = enclosed white areas
        cells = []
        for i in range(1, n):
            cx, cy, cw, ch, area = stats[i]
            if cx <= 4 and cy <= 4 or (cx + cw >= roi.shape[1] - 4 and cy + ch >= roi.shape[0] - 4 and cw > 0.9 * bw):
                continue  # the outside region touching the padded border
            if cw < 12 or ch < 10 or cw >= 0.98 * bw and ch >= 0.98 * bh:
                continue
            if cx <= 0 or cy <= 0 or cx + cw >= roi.shape[1] or cy + ch >= roi.shape[0]:
                continue  # touches border => outside region
            cells.append((x + cx - 4, y + cy - 4, cw, ch))
        if not cells:
            continue
        # --- validation: photos and drawings also contain long edges, a REAL table has a closed frame ---
        frame = grid[y:y + bh, x:x + bw]  # line mask of this candidate
        band = 4  # thickness (px) of the strip checked along each side
        sides = [frame[:band, :], frame[-band:, :], frame[:, :band], frame[:, -band:]]  # top, bottom, left, right
        frame_ok = all((side > 0).any(axis=0 if i < 2 else 1).mean() >= 0.6 for i, side in enumerate(sides))
        tiled = sum(cw * ch for (_, _, cw, ch) in cells) / float(bw * bh)  # share of the box covered by cells
        if frame_ok and tiled >= 0.5 and len(cells) <= 2500:
            tables.append({"bbox": (x, y, bw, bh), "cells": cells})
    return tables


def _ocr_cell(crop: np.ndarray, lang: str, cfg) -> str:
    """OCR one table cell. Single-line mode (psm 7) on a 2x enlarged, binarised crop was the most accurate in tests;
    tall cells may hold several lines, so both psm 7 and psm 6 are tried and the more confident result wins."""
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # enlarge small print
    _, big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # clean black/white
    big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)  # white margin helps Tesseract
    modes = (7,) if crop.shape[0] <= 90 else (7, 6)  # tall cell => maybe multi-line
    best_txt, best_mass = "", -1.0
    for psm in modes:
        ws = _tsv_words(big, lang, psm, cfg.ocr_timeout_s, 0.0)  # keep every word: cells are short, filter later by mass
        ws = [w for w in ws if w.conf >= cfg.ocr_min_word_conf]
        m = sum(w.conf for w in ws)
        if m > best_mass:
            best_txt, best_mass = " ".join(w.text for w in ws), m
    if not best_txt and 7 in modes:  # nothing found in line mode: try block mode once
        ws = [w for w in _tsv_words(big, lang, 6, cfg.ocr_timeout_s, cfg.ocr_min_word_conf)]
        best_txt = " ".join(w.text for w in ws)
    return best_txt


def _build_table(gray: np.ndarray, tab: dict, lang: str, cfg, scale: float, page_h_pt: float, page_no: int) -> dict:
    """OCR each cell of one detected table and return an ODL-style 'table' element."""
    cells = tab["cells"]
    tol = max(6, int(0.35 * statistics.median([min(c[2], c[3]) for c in cells])))  # merge tolerance
    col_edges = _cluster([c[0] for c in cells] + [c[0] + c[2] for c in cells], tol)
    row_edges = _cluster([c[1] for c in cells] + [c[1] + c[3] for c in cells], tol)
    n_rows, n_cols = max(1, len(row_edges) - 1), max(1, len(col_edges) - 1)
    placed = {}  # (row, col) -> cell info
    for (cx, cy, cw, ch) in cells:
        c0, c1 = _nearest(col_edges, cx), _nearest(col_edges, cx + cw)
        r0, r1 = _nearest(row_edges, cy), _nearest(row_edges, cy + ch)
        if c1 <= c0 or r1 <= r0:
            continue
        # OCR the cell: crop with a small inset so grid lines are not read as characters
        crop = gray[max(0, cy + 3):cy + ch - 3, max(0, cx + 3):cx + cw - 3]
        text = ""
        if crop.size and (crop < 140).mean() > 0.004:  # skip empty cells quickly
            text = _ocr_cell(crop, lang, cfg)
        placed[(r0, c0)] = {"r0": r0, "c0": c0, "rs": r1 - r0, "cs": c1 - c0, "text": text, "box": (cx, cy, cw, ch)}
    rows = []
    for r in range(n_rows):  # build the ODL row/cell hierarchy
        row_cells = []
        for c in range(n_cols):
            cell = placed.get((r, c))
            if not cell:
                continue
            x, y, cw, ch = cell["box"]
            row_cells.append({
                "type": "table cell", "page number": page_no,
                "bounding box": _bbox_pt(x, y, x + cw, y + ch, scale, page_h_pt),
                "row number": r + 1, "column number": c + 1, "row span": cell["rs"], "column span": cell["cs"],
                "kids": [{"type": "paragraph", "page number": page_no, "content": cell["text"]}] if cell["text"] else [],
            })
        rows.append({"type": "table row", "row number": r + 1, "cells": row_cells})
    x, y, bw, bh = tab["bbox"]
    return {"type": "table", "page number": page_no, "bounding box": _bbox_pt(x, y, x + bw, y + bh, scale, page_h_pt),
            "number of rows": n_rows, "number of columns": n_cols, "rows": rows, "extracted by": "ocr", "_top": y}


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def _bbox_pt(x0: float, y0: float, x1: float, y1: float, scale: float, page_h_pt: float) -> list:
    """Pixel box (top-left origin) -> PDF-point box [left, bottom, right, top] (bottom-left origin)."""
    return [round(x0 / scale, 3), round(page_h_pt - y1 / scale, 3), round(x1 / scale, 3), round(page_h_pt - y0 / scale, 3)]


def _inside(word_box: tuple, region: tuple, pad: int = 2) -> bool:
    """True when the centre of word_box (x,y,w,h) lies inside region (x,y,w,h)."""
    cx, cy = word_box[0] + word_box[2] / 2, word_box[1] + word_box[3] / 2
    return region[0] - pad <= cx <= region[0] + region[2] + pad and region[1] - pad <= cy <= region[1] + region[3] + pad


# --------------------------------------------------------------------------------------
# words -> lines -> paragraphs/headings/lists
# --------------------------------------------------------------------------------------
def _group_lines(words: list[Word]) -> list[dict]:
    """Group words that share Tesseract's (block, paragraph, line) ids. Order = Tesseract reading order."""
    lines: dict[tuple, dict] = {}
    for w in words:
        key = (w.block, w.par, w.line)
        ln = lines.setdefault(key, {"key": key, "words": []})
        ln["words"].append(w)
    out = []
    for ln in lines.values():
        ws = ln["words"]
        x0, y0 = min(w.left for w in ws), min(w.top for w in ws)
        x1, y1 = max(w.left + w.width for w in ws), max(w.top + w.height for w in ws)
        ln.update(text=" ".join(w.text for w in ws), box=(x0, y0, x1, y1),
                  h=statistics.median([w.height for w in ws]), conf=sum(w.conf for w in ws) / len(ws),
                  block=ln["key"][0], par=ln["key"][1])
        out.append(ln)
    return out


def _lines_to_elements(lines: list[dict], scale: float, page_h_pt: float, page_no: int) -> list[dict]:
    """Turn lines into heading / paragraph / list elements (ODL schema)."""
    if not lines:
        return []
    page_h = statistics.median([l["h"] for l in lines])  # typical body text height on this page
    # Split each Tesseract paragraph into runs of "list lines" and "normal lines".
    groups: dict[tuple, list] = {}
    for l in lines:
        groups.setdefault((l["block"], l["par"]), []).append(l)
    elements: list[dict] = []
    for glines in groups.values():
        runs: list[tuple[bool, list]] = []  # (is_list, lines)
        for l in glines:
            is_item = bool(_BULLET_RE.match(l["text"]))
            if runs and runs[-1][0] == is_item:
                runs[-1][1].append(l)
            elif runs and not is_item and runs[-1][0]:
                runs[-1][1].append(l)  # a wrapped continuation line belongs to the previous list item
            else:
                runs.append((is_item, [l]))
        for is_list, rl in runs:
            x0 = min(l["box"][0] for l in rl); y0 = min(l["box"][1] for l in rl)
            x1 = max(l["box"][2] for l in rl); y1 = max(l["box"][3] for l in rl)
            bbox = _bbox_pt(x0, y0, x1, y1, scale, page_h_pt)
            conf = round(sum(l["conf"] for l in rl) / len(rl), 1)
            if is_list:  # build list items: a new item starts at each bullet line
                items: list[dict] = []
                for l in rl:
                    if _BULLET_RE.match(l["text"]) or not items:
                        items.append({"type": "list item", "page number": page_no, "content": l["text"],
                                      "bounding box": _bbox_pt(*l["box"], scale, page_h_pt), "kids": []})
                    else:
                        items[-1]["content"] += " " + l["text"]  # wrapped line
                numbered = bool(re.match(r"^\s*\(?\d", items[0]["content"]))
                elements.append({"type": "list", "page number": page_no, "bounding box": bbox,
                                 "numbering style": "arabic numbers" if numbered else "bullets",
                                 "number of list items": len(items), "list items": items,
                                 "extracted by": "ocr", "confidence": conf, "_top": y0})
                continue
            text = " ".join(l["text"] for l in rl)  # re-flow wrapped lines into one paragraph
            ph = statistics.median([l["h"] for l in rl])
            ratio = ph / page_h if page_h else 1.0
            etype = "paragraph"
            el = {"type": "paragraph", "page number": page_no, "bounding box": bbox, "content": text,
                  "font size": round(ph / scale, 1), "extracted by": "ocr", "confidence": conf, "_top": y0}
            if ratio >= 1.4 and len(rl) <= 3 and len(text) < 160:  # clearly larger text, short => heading
                etype = "heading"
                el["heading level"] = 1 if ratio >= 2.0 else (2 if ratio >= 1.6 else 3)
            el["type"] = etype
            elements.append(el)
    # Tesseract often puts every list line into its own paragraph: join neighbouring lists of the same kind.
    merged: list[dict] = []
    for el in elements:
        prev = merged[-1] if merged else None
        if (el["type"] == "list" and prev and prev["type"] == "list"
                and prev["numbering style"] == el["numbering style"]):
            prev["list items"].extend(el["list items"])
            prev["number of list items"] = len(prev["list items"])
            prev["bounding box"] = [min(prev["bounding box"][0], el["bounding box"][0]), min(prev["bounding box"][1], el["bounding box"][1]),
                                    max(prev["bounding box"][2], el["bounding box"][2]), max(prev["bounding box"][3], el["bounding box"][3])]
        else:
            merged.append(el)
    return merged


# --------------------------------------------------------------------------------------
# layout-preserving plain text
# --------------------------------------------------------------------------------------
def render_layout_text(lines: list[dict], page_w_px: int) -> str:
    """Make fixed-width text where indentation and blank lines imitate the page (like `pdftotext -layout`)."""
    if not lines:
        return ""
    char_w = statistics.median([(l["box"][2] - l["box"][0]) / max(1, len(l["text"])) for l in lines]) or 10
    width_cols = min(170, max(40, int(page_w_px / char_w)))  # number of text columns
    line_h = statistics.median([l["h"] for l in lines]) or 20
    ordered = sorted(lines, key=lambda l: (round(l["box"][1] / max(line_h, 1)), l["box"][0]))  # top->bottom, left->right
    out, prev_bottom = [], None
    for l in ordered:
        if prev_bottom is not None and l["box"][1] - prev_bottom > 1.2 * line_h:  # big vertical gap => blank line
            out.append("")
        if _RTL_RE.search(l["text"]):  # right-to-left script: align to the right edge of the line
            end_col = int(l["box"][2] / char_w)
            out.append(l["text"].rjust(min(width_cols, max(len(l["text"]), end_col))))
        else:
            out.append(" " * int(l["box"][0] / char_w) + l["text"])  # left indent from x position
        prev_bottom = l["box"][3]
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# QR codes
# --------------------------------------------------------------------------------------
def detect_qr(bgr: np.ndarray, scale: float, page_h_pt: float, page_no: int) -> list[tuple[dict, tuple]]:
    """Decode QR codes. Returns [(element, (x,y,w,h) pixel box)]."""
    found: list[tuple[dict, tuple]] = []
    try:
        det = cv2.QRCodeDetector()
        ok, infos, points, _ = det.detectAndDecodeMulti(bgr)
        if not ok or points is None:
            return found
        for info, pts in zip(infos, points):
            if not info:  # detected but unreadable
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            el = {"type": "qr code", "page number": page_no, "content": info,
                  "bounding box": _bbox_pt(x0, y0, x1, y1, scale, page_h_pt), "extracted by": "ocr", "_top": y0}
            found.append((el, (x0, y0, x1 - x0, y1 - y0)))
    except Exception as exc:  # OpenCV can throw on odd images; QR is a bonus, never fatal
        log.debug("QR detection failed: %s", exc)
    return found


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------
def ocr_page(bgr: np.ndarray, cfg, page_no: int = 1, scale: float = 1.0, lang_hint: str | None = None) -> PageOCR:
    """OCR one page image.

    bgr       : page image (OpenCV BGR)
    scale     : pixels per PDF point (dpi/72 for rendered PDF pages, 1.0 for standalone images)
    lang_hint : language already chosen for this document (skips script detection)
    """
    res = PageOCR()
    h0, w0 = bgr.shape[:2]  # base size in pixels (all reported coordinates use this)
    page_h_pt = h0 / scale
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr, gray0 = _deskew(bgr, gray0)  # straighten once; all later coordinates refer to this image
    # Tesseract works best when letters are ~30px tall: enlarge small images, keep big ones as they are.
    up = min(3.0, max(1.0, 1800.0 / max(h0, w0)))
    g = cv2.resize(gray0, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1.01 else gray0
    lang = resolve_lang(cfg.ocr_langs, g, lang_hint)
    res.lang = lang

    # ---- best-of-N preprocessing ----
    best_words, best_name, best_mass = [], "none", -1.0
    inv_first = float(np.mean(g)) < 110  # dark background: also try the inverted image early
    candidates = list(_variants(g))
    if inv_first:
        candidates.insert(1, ("inverted", 255 - g, 3))
    for name, img, psm in candidates:
        try:
            words = _tsv_words(img, lang, psm, cfg.ocr_timeout_s, cfg.ocr_min_word_conf)
        except Exception as exc:  # timeout or Tesseract crash on this variant -> try the next one
            res.warnings.append(f"variant {name} failed: {exc}")
            continue
        m = _mass(words)
        if best_mass < 0 or m > best_mass * 1.10:  # a later (more aggressive) variant must be clearly better
            best_words, best_name, best_mass = words, name, m
        mean = sum(w.conf for w in words) / len(words) if words else 0
        if name in ("gray", "gray_block", "inverted") and len(words) >= 3 and mean >= 85:  # clean page: stop early (saves time)
            break
    res.variant = best_name

    # words are in 'g' coordinates (upscaled) -> convert to base-image pixels
    for w in best_words:  # undo the enlargement so boxes match the base image
        w.left, w.top = int(w.left / up), int(w.top / up)
        w.width, w.height = max(1, int(w.width / up)), max(1, int(w.height / up))

    # ---- tables & QR codes, then remove their words from the normal text flow ----
    extras: list[dict] = []
    regions: list[tuple] = []
    try:
        for tab in detect_tables(gray0):
            el = _build_table(gray0, tab, lang, cfg, scale, page_h_pt, page_no)
            extras.append(el)
            regions.append(tab["bbox"])
    except Exception as exc:
        res.warnings.append(f"table detection failed: {exc}")
    if cfg.detect_qr:
        for el, box in detect_qr(bgr, scale, page_h_pt, page_no):
            extras.append(el)
            regions.append(box)
    kept = [w for w in best_words if not any(_inside((w.left, w.top, w.width, w.height), r) for r in regions)]

    lines = _group_lines(kept)
    text_elements = _lines_to_elements(lines, scale, page_h_pt, page_no)
    elements = list(text_elements)  # keep Tesseract's reading order (handles columns)
    for ex in sorted(extras, key=lambda e: e.get("_top", 0)):  # put each table / QR at its vertical position
        idx = next((i for i, e in enumerate(elements) if e.get("_top", 0) > ex.get("_top", 0)), len(elements))
        elements.insert(idx, ex)
    res.elements = elements
    res.layout_text = render_layout_text(lines, w0)
    res.n_words = len(kept)
    res.mean_conf = round(sum(w.conf for w in kept) / len(kept), 1) if kept else 0.0
    if not kept and not extras:
        res.warnings.append("no text recognised")
    elif res.mean_conf < 60:
        res.warnings.append(f"low OCR confidence ({res.mean_conf}) - please review")
    return res
