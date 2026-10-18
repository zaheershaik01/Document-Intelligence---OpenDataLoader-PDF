"""cli.py - command-line interface. Every option maps to one field of Config."""

from __future__ import annotations

import argparse
import multiprocessing
import sys

from . import __version__
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    """Define all command-line options."""
    p = argparse.ArgumentParser(prog="negd-docextract", description="Extract text, tables, images and layout from PDF, Word, scans and images into .md / .json / .txt")
    p.add_argument("-i", "--input", default=None, help="input folder (default: ./input)")
    p.add_argument("-o", "--output", default=None, help="output folder (default: ./output)")
    p.add_argument("-c", "--config", default=None, help="JSON file with settings (see config.example.json)")
    p.add_argument("-w", "--workers", type=int, default=None, help="worker processes (default: CPU cores - 1)")
    p.add_argument("--ocr", choices=["auto", "force", "off"], default=None, help="auto = OCR scanned pages only (default)")
    p.add_argument("--lang", default=None, help='OCR language: "auto" or Tesseract codes such as "eng+hin"')
    p.add_argument("--hybrid", choices=["off", "auto", "on"], default=None, help="OpenDataLoader AI hybrid mode")
    p.add_argument("--formulas", action="store_true", help="hybrid: extract formulas as LaTeX (slower)")
    p.add_argument("--pictures", action="store_true", help="hybrid: AI description of charts/pictures (slower)")
    p.add_argument("--txt-layout", action="store_true", help="digital PDFs: keep column layout in .txt")
    p.add_argument("--batch-size", type=int, default=None, help="PDFs per OpenDataLoader JVM launch (default 25)")
    p.add_argument("--java-heap-mb", type=int, default=None, help="JVM memory for huge PDFs, e.g. 4096")
    p.add_argument("--odl-jobs", type=int, default=None, help="parallel OpenDataLoader JVMs (default: automatic)")
    p.add_argument("--password", default=None, help="password for encrypted PDFs")
    p.add_argument("--force", action="store_true", help="re-process files even if already done")
    p.add_argument("--keep-work", action="store_true", help="keep scratch files for debugging")
    p.add_argument("--no-recursive", action="store_true", help="do not scan sub-folders")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING"])
    p.add_argument("--version", action="version", version=__version__)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build Config, run the pipeline."""
    multiprocessing.freeze_support()  # needed for Windows .exe builds
    args = build_parser().parse_args(argv)
    cfg = Config.from_json_file(args.config) if args.config else Config()  # file first, flags override
    if args.input: cfg.input_dir = args.input
    if args.output: cfg.output_dir = args.output
    if args.workers is not None: cfg.workers = args.workers
    if args.ocr: cfg.ocr_mode = args.ocr
    if args.lang: cfg.ocr_langs = args.lang
    if args.hybrid: cfg.hybrid = args.hybrid
    if args.formulas: cfg.hybrid_formulas = True
    if args.pictures: cfg.hybrid_pictures = True
    if args.txt_layout: cfg.txt_layout = True
    if args.batch_size: cfg.odl_batch_size = args.batch_size
    if args.java_heap_mb: cfg.java_heap_mb = args.java_heap_mb
    if args.odl_jobs: cfg.odl_jobs = args.odl_jobs
    if args.password: cfg.pdf_password = args.password
    if args.force: cfg.force = True
    if args.keep_work: cfg.keep_work = True
    if args.no_recursive: cfg.recursive = False
    if args.log_level: cfg.log_level = args.log_level
    from .pipeline import Pipeline  # imported late so --help is instant
    return Pipeline(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
