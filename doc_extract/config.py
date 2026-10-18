"""config.py - every tunable setting lives in ONE dataclass so users never edit the code."""

from __future__ import annotations  # allows "list[str]" style hints on older Python versions

import json  # used to load an optional JSON config file
import os  # used to count CPU cores
from dataclasses import dataclass, field, fields, asdict  # dataclass helpers
from pathlib import Path  # object-oriented file paths (works on Windows and Linux)
from typing import Optional  # for "may be None" type hints


@dataclass
class Config:
    """All settings of the pipeline with safe defaults (a user can override any of them)."""

    # ---------------- folders ----------------
    input_dir: str = "input"  # folder that is scanned for documents (recursively)
    output_dir: str = "output"  # folder that receives md/ json/ txt/ images/ manifest.jsonl
    work_dir: Optional[str] = None  # scratch folder; default = <output_dir>/.work
    recursive: bool = True  # also scan sub-folders of input_dir

    # ---------------- OCR (Tesseract) ----------------
    ocr_mode: str = "auto"  # "auto" = OCR only pages that need it, "force" = OCR every page, "off" = never OCR
    ocr_langs: str = "auto"  # "auto" (detect script) or an explicit Tesseract string like "eng+hin"
    ocr_dpi: int = 300  # resolution used to render PDF pages before OCR
    ocr_min_word_conf: float = 35.0  # words below this Tesseract confidence (0-100) are dropped as noise
    ocr_timeout_s: int = 120  # hard limit for a single Tesseract call
    tesseract_cmd: Optional[str] = None  # full path to tesseract(.exe) if it is not on PATH
    tessdata_dir: Optional[str] = None  # folder holding *.traineddata if not default
    detect_qr: bool = True  # decode QR codes found in images / scanned pages
    scanned_text_threshold: int = 25  # a page with fewer native characters than this may be "scanned"
    scanned_image_coverage: float = 0.40  # ...and one image must cover at least this share of the page

    # ---------------- OpenDataLoader PDF ----------------
    include_header_footer: bool = True  # keep page headers/footers (ODL drops them by default)
    table_method: str = "cluster"  # "default" (ruled tables) or "cluster" (also borderless tables)
    use_struct_tree: bool = False  # trust PDF structure tags (only good for well-tagged PDFs)
    markdown_with_html: bool = True  # allow HTML tables for merged cells in markdown
    keep_line_breaks: bool = False  # keep original line breaks inside paragraphs
    pdf_password: Optional[str] = None  # password for encrypted PDFs
    odl_batch_size: int = 25  # PDFs per JVM launch (README: batch calls, JVM start-up is slow)
    big_pdf_pages: int = 300  # PDFs with more pages than this are processed alone
    big_pdf_mb: int = 60  # ...or bigger than this many MB
    java_heap_mb: Optional[int] = None  # total JVM memory budget in MB (default: 60% of RAM, split between JVMs)
    odl_jobs: int = 0  # OpenDataLoader JVMs running at the same time; 0 = automatic (CPU cores / 2, max 4)

    # ---------------- hybrid (AI) mode ----------------
    hybrid: str = "off"  # "off", "auto" (use if installed) or "on" (error if missing)
    hybrid_port: int = 5002  # port of the local hybrid server
    hybrid_url: Optional[str] = None  # use an already running server instead of starting one
    hybrid_force_ocr: bool = True  # server OCRs scanned pages itself
    hybrid_ocr_lang: str = "en"  # languages for the hybrid OCR, e.g. "en,hi"
    hybrid_formulas: bool = False  # LaTeX formulas (needs hybrid_mode full, slower)
    hybrid_pictures: bool = False  # AI descriptions of charts/pictures (slower)
    hybrid_heading_hierarchy: bool = True  # infer H1/H2/H3 depth

    # ---------------- Office files (docx, pptx, xlsx, ...) ----------------
    office_converter: str = "auto"  # "auto" (LibreOffice then python-docx), "libreoffice", "python"
    libreoffice_path: Optional[str] = None  # full path to soffice(.exe) if not auto-detected
    office_timeout_s: int = 300  # limit for one document conversion

    # ---------------- output ----------------
    split_md_over_mb: float = 2.0  # md/txt bigger than this are ALSO written as page-range parts
    split_pages: int = 100  # pages per part file
    txt_layout: bool = False  # digital PDFs: txt keeps column layout (pdfplumber) instead of reading order
    copy_source_image: bool = True  # image inputs: copy original next to extracted data
    page_marker_md: str = "<!-- Page {n} -->"  # marker written before every page in .md
    page_marker_txt: str = "======== Page {n} ========"  # marker written before every page in .txt

    # ---------------- run control ----------------
    workers: int = 0  # worker processes; 0 = (CPU cores - 1), minimum 1
    resume: bool = True  # skip files already finished (same size + modified time)
    force: bool = False  # ignore the manifest and redo everything
    keep_work: bool = False  # keep scratch files (debugging)
    log_level: str = "INFO"  # DEBUG, INFO, WARNING

    def resolved_workers(self) -> int:
        """Return the real number of worker processes to use."""
        if self.workers and self.workers > 0:  # user set it explicitly
            return self.workers
        return max(1, (os.cpu_count() or 2) - 1)  # leave one core for the OS / JVM

    def resolved_odl_jobs(self) -> int:
        """Number of concurrent OpenDataLoader JVMs."""
        if self.odl_jobs and self.odl_jobs > 0:
            return self.odl_jobs
        return max(1, min(4, (os.cpu_count() or 2) // 2))

    def resolved_work_dir(self) -> Path:
        """Return the scratch folder as a Path."""
        return Path(self.work_dir) if self.work_dir else Path(self.output_dir) / ".work"

    def to_dict(self) -> dict:
        """Plain dict (safe to pass to worker processes)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config from a dict, silently ignoring unknown keys."""
        known = {f.name for f in fields(cls)}  # names of valid settings
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        """Load settings from a JSON file such as config.example.json."""
        with open(path, "r", encoding="utf-8") as fh:  # open the file as UTF-8
            return cls.from_dict(json.load(fh))  # parse + convert
