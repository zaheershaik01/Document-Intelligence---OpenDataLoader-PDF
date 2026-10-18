"""pipeline.py - the conductor. Scales from 1 file to 10,000+ files.

Design for volume
  * RESUMABLE   : manifest.jsonl is appended after every file; re-running skips finished files.
  * BATCHED     : PDFs go to OpenDataLoader in batches (one JVM start per batch, as the README advises);
                  huge PDFs run alone so one monster cannot break 24 others.
  * PARALLEL    : OCR, Office conversion, splicing and writing run in a pool of worker processes;
                  OCR is scheduled PAGE BY PAGE so one 800-page scan uses all cores.
  * ISOLATED    : an error in one file is recorded in the manifest; the run continues.
  * BOUNDED DISK/RAM : each batch uses a scratch folder that is deleted afterwards; pages are rendered one at a time.

Flow per batch of PDF-route files (pdf + office):
    Office -> PDF (pool)  ->  OpenDataLoader batch (JVM)  ->  plan (pool)  ->  OCR pages (pool)  ->  finalize+write (pool)
Images and text files skip OpenDataLoader and are handled completely inside the pool.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import __version__, ocr_engine, office, pdf_pipeline
from .config import Config
from .discovery import FileTask, Manifest, discover
from .odl_runner import HybridServer, auto_heap_mb, java_available, run_odl
from .renderers import IMG_TOKEN, elements_to_markdown, elements_to_text
from .textfiles import text_file_to_pages
from .utils import setup_logging, short_hash
from .writer import write_outputs

log = logging.getLogger(__name__)


# ======================================================================================
# WORKER-SIDE FUNCTIONS (top level so that worker processes can import and run them)
# ======================================================================================
def _init_worker(cfg_dict: dict) -> None:
    """Runs once in every worker process: logging + Tesseract location."""
    cfg = Config.from_dict(cfg_dict)
    setup_logging(cfg.log_level)
    ocr_engine.configure_tesseract(cfg)


def _record(task: dict, status: str, t0: float, **extra) -> dict:
    """Build one manifest line."""
    rec = {"source": task["rel"], "category": task["category"], "status": status, "size": task["size"],
           "mtime": task["mtime"], "seconds": round(time.time() - t0, 2), "version": __version__,
           "finished": datetime.now().isoformat(timespec="seconds")}
    rec.update(extra)
    return rec


def _status(warnings: list) -> str:
    """'ok' or 'ok_with_warnings'."""
    return "ok_with_warnings" if warnings else "ok"


def w_image(task: dict, cfg_dict: dict) -> dict:
    """Process an image file completely: OCR -> md/json/txt."""
    t0 = time.time()
    cfg = Config.from_dict(cfg_dict)
    try:
        bgr = ocr_engine.load_image_bgr(task["src"])
        h, w = bgr.shape[:2]
        res = ocr_engine.ocr_page(bgr, cfg, page_no=1, scale=1.0)  # standalone image: 1 pixel = 1 "point"
        els = res.elements
        pic = Path(task["src"])
        head = []
        if cfg.copy_source_image:  # link the original picture at the top of the markdown
            head = [{"type": "image", "page number": 1, "source": f"{IMG_TOKEN}/{pic.name}",
                     "bounding box": [0, 0, w, h], "description": "original image"}]
        kids = head + els
        has_extras = any(e.get("type") in ("table", "qr code") for e in els)
        md = {1: elements_to_markdown(kids)}
        txt = {1: elements_to_text(els) if has_extras else (res.layout_text or elements_to_text(els))}
        doc = {"file name": pic.name, "number of pages": 1, "page width": w, "page height": h, "kids": kids,
               "extraction": {"engine": "tesseract", "language": res.lang, "variant": res.variant,
                              "mean confidence": res.mean_conf, "note": "coordinates are image pixels (1 px = 1 pt)"}}
        paths = write_outputs(cfg, task["stem"], md, txt, doc, None, [pic] if cfg.copy_source_image else None)
        return _record(task, _status(res.warnings), t0, pages=1, ocr_pages=1, ocr_mean_conf=res.mean_conf,
                       engine="tesseract", warnings=res.warnings, outputs=paths)
    except Exception as exc:
        return _record(task, "failed", t0, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-800:])


def w_text(task: dict, cfg_dict: dict) -> dict:
    """Process .txt/.md/.csv/.html."""
    t0 = time.time()
    cfg = Config.from_dict(cfg_dict)
    try:
        md, txt, els = text_file_to_pages(task["src"])
        doc = {"file name": Path(task["src"]).name, "number of pages": 1, "kids": els, "extraction": {"engine": "text-passthrough"}}
        paths = write_outputs(cfg, task["stem"], md, txt, doc, None)
        return _record(task, "ok", t0, pages=1, engine="text", outputs=paths)
    except Exception as exc:
        return _record(task, "failed", t0, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-800:])


def w_office(task: dict, cfg_dict: dict, work_dir: str) -> dict:
    """Office file -> PDF via LibreOffice ({'kind':'pdf'}) or, for .docx without LibreOffice, direct python-docx ({'kind':'done'})."""
    t0 = time.time()
    cfg = Config.from_dict(cfg_dict)
    ext = Path(task["src"]).suffix.lower()
    err = ""
    if cfg.office_converter in ("auto", "libreoffice") and office.find_libreoffice(cfg):
        try:
            pdf = office.convert_to_pdf(cfg, task["src"], Path(work_dir) / f"office_{short_hash(task['rel'])}", Path(work_dir))
            return {"kind": "pdf", "pdf": str(pdf)}
        except Exception as exc:  # fall through to python-docx when possible
            err = str(exc)
    if ext == ".docx" and cfg.office_converter in ("auto", "python"):
        try:
            img_dir = Path(work_dir) / f"docx_img_{short_hash(task['rel'])}"
            els, n_pages = office.docx_to_elements(task["src"], img_dir)
            md, txt = {}, {}
            for p in range(1, n_pages + 1):
                page_els = [e for e in els if e.get("page number") == p]
                md[p], txt[p] = elements_to_markdown(page_els), elements_to_text(page_els) if page_els else ""
            doc = {"file name": Path(task["src"]).name, "number of pages": n_pages, "kids": els,
                   "extraction": {"engine": "python-docx", "note": "page numbers follow explicit/rendered page breaks only"}}
            paths = write_outputs(cfg, task["stem"], md, txt, doc, img_dir if img_dir.exists() else None)
            warn = ["LibreOffice not available: used python-docx (page layout approximate)" + (f" [{err}]" if err else "")]
            return {"kind": "done", "record": _record(task, "ok_with_warnings", t0, pages=n_pages, engine="python-docx", warnings=warn, outputs=paths)}
        except Exception as exc:
            return {"kind": "done", "record": _record(task, "failed", t0, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-800:])}
    msg = err or f"LibreOffice is required for {ext} files (install it from libreoffice.org or set libreoffice_path)"
    return {"kind": "done", "record": _record(task, "failed", t0, error=msg)}


def w_plan(pdf: str, odl_dir, odl_stem, cfg_dict: dict) -> dict:
    """Pool wrapper for pdf_pipeline.plan_pdf."""
    return pdf_pipeline.plan_pdf(pdf, odl_dir, odl_stem, cfg_dict)


def w_ocr(pdf: str, page: int, lang, cfg_dict: dict) -> dict:
    """Pool wrapper for pdf_pipeline.ocr_pdf_page."""
    return pdf_pipeline.ocr_pdf_page(pdf, page, lang, cfg_dict)


def w_finalize(plan: dict, ocr_results: dict, task: dict, cfg_dict: dict, engine_note: str) -> dict:
    """Splice ODL + OCR pages and write all outputs for one PDF."""
    t0 = time.time()
    cfg = Config.from_dict(cfg_dict)
    try:
        fin = pdf_pipeline.finalize_pdf(plan, ocr_results, Path(task["src"]).name, cfg_dict)
        paths = write_outputs(cfg, task["stem"], fin["md_pages"], fin["txt_pages"], fin["json"], fin["image_src_dir"])
        st = fin["stats"]
        return _record(task, _status(st["warnings"]), t0, pages=st["pages"], ocr_pages=st["ocr_pages"],
                       ocr_mean_conf=st["ocr_mean_conf"], engine=fin["json"]["extraction"]["engine"] + engine_note,
                       warnings=st["warnings"][:20], outputs=paths)
    except Exception as exc:
        return _record(task, "failed", t0, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-800:])


# ======================================================================================
# MAIN-PROCESS ORCHESTRATION
# ======================================================================================
@dataclass
class Item:
    """A PDF travelling through the batch stages."""
    task: FileTask
    pdf: str  # path handed to OpenDataLoader (original PDF or converted Office PDF)
    odl_stem: str  # file stem ODL uses for its output names
    odl_dir: Path | None = None  # where ODL wrote its output (None = ODL failed)


def _gather(pool: ProcessPoolExecutor, jobs: dict) -> dict:
    """Run {key: (fn, args...)} in the pool; return {key: result or Exception} as they finish."""
    futures = {pool.submit(v[0], *v[1:]): k for k, v in jobs.items()}
    out = {}
    for fut in as_completed(futures):
        key = futures[fut]
        try:
            out[key] = fut.result()
        except Exception as exc:  # a worker crash (e.g. out of memory) must not kill the run
            out[key] = exc
    return out


class Pipeline:
    """Runs a complete extraction job."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.work = cfg.resolved_work_dir()
        self.counts = {"ok": 0, "ok_with_warnings": 0, "failed": 0, "skipped": 0, "resumed": 0}
        self.done_n = 0
        self.total = 0
        self.t_start = time.time()
        self._lock = threading.Lock()  # protects counters + manifest (batches run in several threads)
        self._n_jobs = 1  # concurrent ODL JVMs (set in run())
        self._slots = threading.Semaphore(1)  # one permit per concurrent JVM
        self._slot_gate = threading.Lock()  # makes "take ALL permits" (big PDFs) atomic
        self._heap_small = 2048  # MB per JVM when several run together
        self._heap_big = 2048  # MB for a PDF that runs alone

    # ---------------- reporting ----------------
    def _commit(self, manifest: Manifest, rec: dict) -> None:
        """Save a record + update counters + log one progress line."""
        with self._lock:  # several batch threads may finish at the same time
            manifest.append(rec)
            self.counts[rec["status"] if rec["status"] in self.counts else "failed"] += 1
            self.done_n += 1
            el = time.time() - self.t_start
            eta = el / self.done_n * (self.total - self.done_n) if self.done_n else 0
            extra = f"  [{rec.get('error', '')[:90]}]" if rec["status"] == "failed" else ""
            log.info("[%d/%d] %-18s %s%s  (ETA %dm%02ds)", self.done_n, self.total, rec["status"], rec["source"], extra, eta // 60, eta % 60)

    # ---------------- run ----------------
    def run(self) -> int:
        """Execute the job. Returns a process exit code (0 = no failures)."""
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        setup_logging(cfg.log_level, self.out / "logs" / "run.log")
        log.info("NeGD DocExtract %s | input=%s output=%s workers=%d", __version__, cfg.input_dir, self.out, cfg.resolved_workers())
        tasks = discover(cfg)
        manifest = Manifest(self.out / "manifest.jsonl")
        todo: list[FileTask] = []
        for t in tasks:
            if t.category == "unsupported":
                self._commit_skip(manifest, t)
            elif cfg.resume and not cfg.force and manifest.is_done(t):
                self.counts["resumed"] += 1
            else:
                todo.append(t)
        self.total = len(todo)
        log.info("Found %d files: %d to process, %d already done, %d unsupported", len(tasks), len(todo), self.counts["resumed"], self.counts["skipped"])
        if not todo:
            return self._finish()
        if not ocr_engine.tesseract_available():
            ocr_engine.configure_tesseract(cfg)
        if not ocr_engine.tesseract_available():
            log.warning("Tesseract OCR not found: scans and images cannot be read. Install it (see README) or set tesseract_cmd.")
        have_java = java_available()
        pdf_route = [t for t in todo if t.category in ("pdf", "office")]
        if pdf_route and not have_java:
            log.warning("Java 11+ not found: PDFs will be processed with OCR only (lower quality). Install Java for best results.")

        cfg_dict = cfg.to_dict()
        self.work.mkdir(parents=True, exist_ok=True)
        server = HybridServer(cfg)
        pool = ProcessPoolExecutor(max_workers=cfg.resolved_workers(), initializer=_init_worker, initargs=(cfg_dict,))
        try:
            # ---- 1) images + text files: fully handled inside the pool ----
            simple = {t.rel: ((w_image if t.category == "image" else w_text), t.to_dict(), cfg_dict) for t in todo if t.category in ("image", "text")}
            if simple:
                futs = {pool.submit(*v): k for k, v in simple.items()}
                for fut in as_completed(futs):
                    try:
                        self._commit(manifest, fut.result())
                    except Exception as exc:
                        t = next(x for x in todo if x.rel == futs[fut])
                        self._commit(manifest, {"source": t.rel, "category": t.category, "status": "failed", "size": t.size, "mtime": t.mtime, "error": f"worker crashed: {exc}"})
            # ---- 2) PDFs and Office files in batches ----
            if pdf_route:
                hybrid_url = server.start() if (cfg.hybrid != "off" and have_java) else None
                self._n_jobs = cfg.resolved_odl_jobs() if have_java else 1
                self._slots = threading.Semaphore(self._n_jobs)
                budget = int(cfg.java_heap_mb) if cfg.java_heap_mb else auto_heap_mb()  # total JVM memory
                self._heap_small = max(1024, budget // self._n_jobs)  # each concurrent JVM gets its share
                self._heap_big = budget  # a huge PDF runs alone and may use everything
                log.info("OpenDataLoader: %d JVM(s) in parallel, %d MB each (big PDFs: %d MB)", self._n_jobs, self._heap_small, self._heap_big)
                old_opts = os.environ.get("JAVA_TOOL_OPTIONS")
                os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{self._heap_small}m"  # inherited by every JVM we start
                try:
                    # each batch runs in a thread (its ODL step blocks on a JVM); heavy work still goes to the process pool
                    with ThreadPoolExecutor(max_workers=self._n_jobs) as tp:
                        futs = [tp.submit(self._run_batch, batch, big, pool, manifest, cfg_dict, hybrid_url, have_java)
                                for batch, big in self._batches(pdf_route)]
                        for f in as_completed(futs):
                            try:
                                f.result()
                            except Exception as exc:  # a bug inside one batch must not stop the others
                                log.error("batch crashed: %s\n%s", exc, traceback.format_exc())
                finally:
                    if old_opts is None:
                        os.environ.pop("JAVA_TOOL_OPTIONS", None)
                    else:
                        os.environ["JAVA_TOOL_OPTIONS"] = old_opts
        except KeyboardInterrupt:
            log.warning("Interrupted. Finished files are saved - run the same command again to continue.")
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            server.stop()
            if not cfg.keep_work:
                shutil.rmtree(self.work, ignore_errors=True)
        return self._finish()

    def _commit_skip(self, manifest: Manifest, t: FileTask) -> None:
        """Record an unsupported file (so users can see it was seen, not forgotten)."""
        manifest.append({"source": t.rel, "category": "unsupported", "status": "skipped", "size": t.size, "mtime": t.mtime,
                         "error": "unsupported file type"})
        self.counts["skipped"] += 1

    def _finish(self) -> int:
        """Write summary.json and return the exit code."""
        el = time.time() - self.t_start
        summary = {"counts": self.counts, "seconds": round(el, 1), "version": __version__, "output": str(self.out)}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log.info("DONE in %.0fs | ok=%d warnings=%d failed=%d skipped=%d resumed=%d",
                 el, self.counts["ok"], self.counts["ok_with_warnings"], self.counts["failed"], self.counts["skipped"], self.counts["resumed"])
        if self.counts["failed"]:
            log.warning("Some files failed - see manifest.jsonl (status 'failed') for the reason.")
        return 1 if self.counts["failed"] else 0

    # ---------------- batching ----------------
    def _batches(self, tasks: list[FileTask]):
        """Yield lists of FileTasks. Big PDFs go alone; stems inside a batch must be unique (ODL names outputs by stem)."""
        cfg = self.cfg
        cur: list[FileTask] = []
        stems: set[str] = set()
        for t in tasks:
            stem = Path(t.src).stem.lower() if t.category == "pdf" else f"office_{short_hash(t.rel)}"
            big = False
            if t.category == "pdf":
                big = t.size > cfg.big_pdf_mb * 1024 * 1024
                if not big:
                    try:
                        import pypdfium2 as pdfium
                        d = pdfium.PdfDocument(t.src, password=cfg.pdf_password)
                        big = len(d) > cfg.big_pdf_pages
                        d.close()
                    except Exception:
                        pass  # corrupt / encrypted: the real error is reported later
            if big:
                if cur:
                    yield cur, False
                    cur, stems = [], set()
                yield [t], True  # processed alone with ALL the JVM memory
                continue
            if stem in stems or len(cur) >= cfg.odl_batch_size:
                yield cur, False
                cur, stems = [], set()
            cur.append(t)
            stems.add(stem)
        if cur:
            yield cur, False

    # ---------------- one batch ----------------
    def _run_batch(self, batch: list[FileTask], big: bool, pool, manifest: Manifest, cfg_dict: dict, hybrid_url, have_java: bool) -> None:
        """Run all stages for one batch of PDF-route files."""
        cfg = self.cfg
        bdir = self.work / f"batch_{short_hash(batch[0].rel + str(time.time()))}"
        bdir.mkdir(parents=True, exist_ok=True)
        tm = {}  # seconds spent in every stage (logged at the end of the batch)
        t_mark = time.time()

        def lap(name: str) -> None:
            """Record the time since the previous lap under 'name'."""
            nonlocal t_mark
            tm[name] = round(time.time() - t_mark, 1)
            t_mark = time.time()
        try:
            # --- stage A: Office -> PDF (parallel) ---
            items: list[Item] = []
            batch = self._prevalidate(batch, manifest)  # corrupt / password-protected PDFs are rejected here, cheaply
            office_jobs = {t.rel: (w_office, t.to_dict(), cfg_dict, str(bdir)) for t in batch if t.category == "office"}
            office_res = _gather(pool, office_jobs) if office_jobs else {}
            for t in batch:
                if t.category == "pdf":
                    items.append(Item(t, t.src, Path(t.src).stem))
                    continue
                res = office_res.get(t.rel)
                if isinstance(res, Exception):
                    self._commit(manifest, {"source": t.rel, "category": t.category, "status": "failed", "size": t.size, "mtime": t.mtime, "error": f"worker crashed: {res}"})
                elif res["kind"] == "done":
                    self._commit(manifest, res["record"])
                else:
                    items.append(Item(t, res["pdf"], Path(res["pdf"]).stem))
            lap("prep")
            if not items:
                return
            # --- stage B: OpenDataLoader, ONE JVM for the whole batch ---
            if have_java:
                self._odl_stage(items, bdir / "odl", hybrid_url, big)
            lap("odl")
            # --- stage C: plan (which pages need OCR?) ---
            plans = _gather(pool, {i.task.rel: (w_plan, i.pdf, str(i.odl_dir) if i.odl_dir else None, i.odl_stem if i.odl_dir else None, cfg_dict) for i in items})
            by_rel = {i.task.rel: i for i in items}
            ok_plans = {}
            for rel, plan in plans.items():
                if isinstance(plan, Exception):  # e.g. encrypted / corrupt PDF
                    t = by_rel[rel].task
                    self._commit(manifest, {"source": rel, "category": t.category, "status": "failed", "size": t.size, "mtime": t.mtime,
                                            "error": f"cannot open PDF: {type(plan).__name__}: {plan}"})
                else:
                    ok_plans[rel] = plan
            lap("plan")
            # --- stage D: OCR, page by page across ALL workers ---
            ocr_jobs = {(rel, p): (w_ocr, pl["pdf"], p, pl["lang"], cfg_dict) for rel, pl in ok_plans.items() for p in pl["ocr_pages"]}
            ocr_out = _gather(pool, ocr_jobs) if ocr_jobs else {}
            per_doc: dict[str, dict] = {rel: {} for rel in ok_plans}
            doc_errors: dict[str, list] = {rel: [] for rel in ok_plans}
            for (rel, p), res in ocr_out.items():
                if isinstance(res, Exception):
                    doc_errors[rel].append(f"page {p}: OCR failed ({type(res).__name__}: {res})")
                else:
                    per_doc[rel][p] = res
            lap("ocr")
            # --- stage E: splice + write (parallel) ---
            fin_jobs = {}
            for rel, pl in ok_plans.items():
                pl["notes"] = pl.get("notes", []) + doc_errors[rel]
                note = " + hybrid" if hybrid_url else ""
                fin_jobs[rel] = (w_finalize, pl, per_doc[rel], by_rel[rel].task.to_dict(), cfg_dict, note)
            for rel, rec in _gather(pool, fin_jobs).items():
                if isinstance(rec, Exception):
                    t = by_rel[rel].task
                    rec = {"source": rel, "category": t.category, "status": "failed", "size": t.size, "mtime": t.mtime, "error": f"worker crashed: {rec}"}
                self._commit(manifest, rec)
            lap("finalize")
            log.info("batch of %d files: %s", len(items), " | ".join(f"{k} {v}s" for k, v in tm.items()))
        finally:
            if not cfg.keep_work:
                shutil.rmtree(bdir, ignore_errors=True)  # free disk space immediately

    def _prevalidate(self, batch: list[FileTask], manifest: Manifest) -> list[FileTask]:
        """Open every PDF with pdfium (milliseconds). Bad files fail immediately with a clear reason, so they never
        make a whole OpenDataLoader batch fail (which would cost one extra JVM launch per file in the batch)."""
        import pypdfium2 as pdfium
        good = []
        for t in batch:
            if t.category != "pdf":
                good.append(t)
                continue
            try:
                pdfium.PdfDocument(t.src, password=self.cfg.pdf_password).close()
                good.append(t)
            except Exception as exc:
                reason = str(exc)
                hint = " - use --password <pw> for protected PDFs" if "assword" in reason else ""
                self._commit(manifest, {"source": t.rel, "category": t.category, "status": "failed", "size": t.size,
                                        "mtime": t.mtime, "error": f"cannot open PDF: {reason}{hint}"})
        return good

    def _odl_stage(self, items: list[Item], odl_out: Path, hybrid_url, big: bool = False) -> None:
        """Run OpenDataLoader on the batch (one JVM). If the batch call fails, retry file by file to isolate the culprit.

        Concurrency: a normal batch takes ONE slot; a big PDF takes ALL slots (so it runs alone with the full heap).
        """
        cfg = self.cfg
        need = self._n_jobs if big else 1
        with self._slot_gate:  # acquire several permits atomically (avoids two big batches deadlocking each other)
            for _ in range(need):
                self._slots.acquire()
        saved = os.environ.get("JAVA_TOOL_OPTIONS")
        if big:  # nobody else runs now, so the environment can be changed safely
            os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{self._heap_big}m"
        try:
            try:
                run_odl([i.pdf for i in items], odl_out, cfg, hybrid_url)
                for i in items:
                    i.odl_dir = odl_out if (odl_out / f"{i.odl_stem}.json").exists() else None  # missing = ODL skipped this file
            except Exception as exc:
                log.warning("OpenDataLoader batch failed (%s: %s) - retrying files one by one", type(exc).__name__, str(exc)[:150])
                for n, i in enumerate(items):
                    sub = odl_out / f"solo_{n}"
                    try:
                        run_odl([i.pdf], sub, cfg, hybrid_url)
                        i.odl_dir = sub if (sub / f"{i.odl_stem}.json").exists() else None
                    except Exception as exc2:
                        log.warning("OpenDataLoader failed for %s: %s", i.task.rel, str(exc2)[:150])
                        i.odl_dir = None  # plan_pdf will fall back to degraded mode
        finally:
            if big:
                if saved is None:
                    os.environ.pop("JAVA_TOOL_OPTIONS", None)
                else:
                    os.environ["JAVA_TOOL_OPTIONS"] = saved
            for _ in range(need):
                self._slots.release()
