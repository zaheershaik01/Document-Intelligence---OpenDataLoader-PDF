"""odl_runner.py - the only place that talks to OpenDataLoader PDF.

Rules taken from the OpenDataLoader README:
  * batch many files into ONE convert() call (each call starts a JVM, which is slow);
  * formats are requested together: format="markdown,json,text";
  * hybrid mode = a local AI server (`opendataloader-pdf-hybrid`) + hybrid="docling-fast" on the client.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket  # to check whether the hybrid server port is open
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Unique marker ODL puts at the start of every page; we split on it and later replace it by pretty markers.
PAGE_SENTINEL = "<!--ODL_PAGE %page-number%-->"
PAGE_SENTINEL_RE = r"<!--ODL_PAGE (\d+)-->"


def auto_heap_mb() -> int:
    """JVM heap to use when the user did not set one: 60% of physical RAM, between 1 GB and 8 GB.

    The JVM default is only 25% of RAM, which is NOT enough for big PDFs (a 1,153-page book crashed with it).
    -Xmx is a ceiling, not a reservation, so a larger value costs nothing for small files.
    """
    try:
        import psutil  # accurate on every OS
        total = psutil.virtual_memory().total
    except Exception:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # Linux / macOS
        except (ValueError, OSError, AttributeError):
            return 2048  # unknown: a reasonable default
    return int(min(8192, max(1024, total / 1024 / 1024 * 0.6)))


def run_odl(pdf_paths: list[str], out_dir: Path, cfg, hybrid_url: str | None = None) -> None:
    """Convert a list of PDFs to markdown+json+text inside out_dir with ONE JVM launch."""
    import opendataloader_pdf  # imported here so the module can be imported without Java installed

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_path=[str(p) for p in pdf_paths],  # list of files (README: batch in one call)
        output_dir=str(out_dir),
        format="markdown,json,text",  # the three deliverables
        quiet=True,  # no console spam
        include_header_footer=cfg.include_header_footer,  # keep headers/footers (default would drop them)
        markdown_page_separator=PAGE_SENTINEL,  # page boundaries for markdown
        text_page_separator=PAGE_SENTINEL,  # page boundaries for text
        table_method=cfg.table_method,  # "cluster" also finds borderless tables
        keep_line_breaks=cfg.keep_line_breaks,
        markdown_with_html=cfg.markdown_with_html,  # HTML tables for merged cells
        image_output="external",  # pictures written as files next to the output
        image_format="png",
    )
    if cfg.pdf_password:
        kwargs["password"] = cfg.pdf_password
    if cfg.use_struct_tree:
        kwargs["use_struct_tree"] = True  # NOTE: takes precedence over hybrid for tagged PDFs
    if hybrid_url:  # hybrid (AI) backend running locally
        kwargs["hybrid"] = "docling-fast"
        kwargs["hybrid_url"] = hybrid_url
        if cfg.hybrid_formulas or cfg.hybrid_pictures:
            kwargs["hybrid_mode"] = "full"  # README: formula/picture enrichment needs full mode
        kwargs["hybrid_fallback"] = True  # on backend error fall back to the Java engine instead of failing
    # NOTE: JVM memory is set through the JAVA_TOOL_OPTIONS environment variable by the pipeline
    # (once for all concurrent JVMs, see pipeline.Pipeline._odl_stage).
    opendataloader_pdf.convert(**kwargs)  # blocks until Java finishes


def java_available() -> bool:
    """True when `java` can be started (OpenDataLoader needs Java 11+)."""
    return shutil.which("java") is not None


class HybridServer:
    """Starts / stops the local `opendataloader-pdf-hybrid` AI server (needs pip install "opendataloader-pdf[hybrid]")."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None

    @staticmethod
    def executable() -> str | None:
        """Find the server program (installed next to python by pip)."""
        found = shutil.which("opendataloader-pdf-hybrid")
        if found:
            return found
        for name in ("opendataloader-pdf-hybrid", "opendataloader-pdf-hybrid.exe"):
            cand = Path(sys.executable).parent / name
            if cand.exists():
                return str(cand)
            cand = Path(sys.executable).parent / "Scripts" / name  # Windows virtualenv layout
            if cand.exists():
                return str(cand)
        return None

    def start(self) -> str | None:
        """Start (or attach to) the server. Returns its URL or None when hybrid should not be used."""
        cfg = self.cfg
        if cfg.hybrid == "off":
            return None
        if cfg.hybrid_url:  # user runs the server himself (maybe on another machine)
            self.url = cfg.hybrid_url
            return self.url
        exe = self.executable()
        if not exe:
            msg = 'Hybrid mode needs: pip install "opendataloader-pdf[hybrid]"'
            if cfg.hybrid == "on":
                raise RuntimeError(msg)
            log.warning("%s -> continuing without hybrid (Tesseract OCR is used for scanned pages)", msg)
            return None
        cmd = [exe, "--port", str(cfg.hybrid_port)]  # build the server command line
        if cfg.hybrid_force_ocr:
            cmd += ["--force-ocr", "--ocr-lang", cfg.hybrid_ocr_lang]
        if cfg.hybrid_formulas:
            cmd += ["--enrich-formula"]
        if cfg.hybrid_pictures:
            cmd += ["--enrich-picture-description"]
        if cfg.hybrid_heading_hierarchy:
            cmd += ["--heading-hierarchy"]
        log.info("Starting hybrid server: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 600  # model loading can take minutes the first time
        while time.time() < deadline:
            if self.proc.poll() is not None:  # server died during start-up
                self.proc = None
                break
            try:
                with socket.create_connection(("127.0.0.1", cfg.hybrid_port), timeout=1):
                    self.url = f"http://127.0.0.1:{cfg.hybrid_port}"  # port is open => ready
                    log.info("Hybrid server ready at %s", self.url)
                    return self.url
            except OSError:
                time.sleep(2)
        self.stop()
        if cfg.hybrid == "on":
            raise RuntimeError("Hybrid server did not start")
        log.warning("Hybrid server did not start -> continuing without hybrid")
        return None

    def stop(self) -> None:
        """Terminate the server process if we started it."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
