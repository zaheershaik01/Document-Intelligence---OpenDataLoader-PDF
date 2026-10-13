# Document Extraction

One module that turns **PDF, Word (docx/doc/odt/rtf), PowerPoint, Excel, scanned PDFs, images (png/jpg/tif/...) and text/html/csv**
into three faithful outputs: **`.md`**, **`.json`**, **`.txt`** — with pages, headers, footers, tables, images and
reading order preserved. Built on [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf)
(follows its README rules) plus Tesseract OCR, OpenCV and LibreOffice for everything OpenDataLoader does not read itself.

## 1. Install (once per computer)

| Needed                       | Why                                   | How (Windows)                                                                                       |
| ---------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Python 3.10+                 | runs the module                       | python.org                                                                                          |
| **Java 11+**           | OpenDataLoader runs on the JVM        | adoptium.net → check with`java -version`                                                         |
| **Tesseract OCR**      | reads scans and images                | UBM-Mannheim installer; tick extra languages (Hindi, Arabic, ...)                                   |
| LibreOffice*(recommended)* | Word/PPT/Excel → PDF with real pages | libreoffice.org (without it`.docx` still works through python-docx, `.doc/.pptx/.xlsx` need it) |

```bash
pip install -r requirements.txt
# optional AI mode (scanned pages, complex tables, formulas):  pip install "opendataloader-pdf[hybrid]"
```

## 2. Use

```bash
python main.py                              # reads ./input  ->  writes ./output
python main.py -i D:\scans -o D:\results    # any folders (sub-folders are scanned too)
python main.py --lang eng+hin               # OCR languages (default: auto-detect script)
python main.py --hybrid auto --formulas     # use OpenDataLoader AI hybrid mode if installed
python main.py -w 8                         # 8 worker processes (default: CPU cores - 1)
python main.py                              # run again after a crash / Ctrl+C: finished files are skipped
python main.py --force                      # redo everything
```

Every option can also be stored in a JSON file: `python main.py -c config.json` (see `config.example.json`, all
fields are documented in `negd_docextract/config.py`).

## 3. Output

```
output/
  md/    <name>.md            markdown: <!-- Page N --> markers, headings, tables, image links
  json/  <name>.json          OpenDataLoader schema: type, page number, bounding box [l,b,r,t] in pt, content ...
  txt/   <name>.txt           plain text, "======== Page N ========" markers
  images/<name>/imageFileN.png  pictures / diagrams extracted from that document
  md/<name>.parts/  txt/<name>.parts/   ONLY for big documents: page-range files of 100 pages
  manifest.jsonl              one line per input: status, pages, OCR pages, confidence, warnings, error
  summary.json  logs/run.log
```

Sub-folders of the input folder are mirrored inside `md/`, `json/`, `txt/`, `images/`.

**Opening huge results:** a 1,000-page book gives a 4 MB `.md` with 80,000 lines. VS Code's *Markdown Editor* freezes on
that ("window is not responding"). Open the `.parts/` files instead, or right-click → *Open With → Text Editor*.

## 4. How it works

```
input/ ──discover──► classify each file by type ─────────────────────────────────────────────┐
                                                                                              │
 image  ───────────► structure-aware OCR (ocr_engine) ───────────────────────────────────────┤
 txt/md/csv/html ──► text handler ───────────────────────────────────────────────────────────┤
 Office ──LibreOffice──► PDF ┐  (fallback: python-docx)                                       │
 PDF ────────────────────────┴─► OpenDataLoader (one JVM per batch of 25 PDFs)               │
                                   └► per page: digital? keep it │ scanned? render + OCR + replace
                                                                                              ▼
                                                              writer: md/ json/ txt/ images/ + manifest
```

| File                                    | Responsibility                                                                      |
| --------------------------------------- | ----------------------------------------------------------------------------------- |
| `main.py`, `negd_docextract/cli.py` | command line →`Config` → `Pipeline`                                           |
| `config.py`                           | every setting in one dataclass                                                      |
| `discovery.py`                        | find files, classify, unique output names,**manifest = resume**               |
| `pipeline.py`                         | batching, process pool, page-level OCR scheduling, error isolation, progress/ETA    |
| `odl_runner.py`                       | the only code that calls OpenDataLoader; optional**hybrid server** start/stop |
| `pdf_pipeline.py`                     | per-page triage (digital vs scanned), OCR scheduling, splicing ODL + OCR pages      |
| `ocr_engine.py`                       | Tesseract + OpenCV: preprocessing, language detection, tables, QR, layout           |
| `office.py`                           | LibreOffice conversion and python-docx fallback                                     |
| `renderers.py`                        | elements → markdown / text, page markers, splitting large outputs                  |
| `writer.py`                           | output folder layout, image moving, link rewriting                                  |
| `textfiles.py`                        | txt / md / csv / html                                                               |
| `tests/check_outputs.py`              | automatic quality gate for a finished run                                           |

## 5. Which framework does what

| Framework                                                   | Used for                                                                                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **OpenDataLoader PDF** (Java, `opendataloader-pdf`) | digital PDFs: reading order (XY-Cut++), headings, lists, tables, images, headers/footers, bounding boxes → md/json/text |
| OpenDataLoader**hybrid** (Docling models, optional)   | AI layout model, built-in OCR, LaTeX formulas, borderless/complex tables, picture descriptions                           |
| **Tesseract** (`pytesseract`)                       | OCR of scanned pages and image files, 100+ languages incl. Arabic/Hindi                                                  |
| **OpenCV**                                            | deskew, contrast/threshold variants, ruled-table detection, QR decoding                                                  |
| **pypdfium2**                                         | page triage (text characters, image coverage) and page rendering for OCR                                                 |
| **LibreOffice**                                       | Word/PowerPoint/Excel/ODF → PDF (so they get true pages/headers/footers)                                                |
| **python-docx**                                       | direct`.docx` reading when LibreOffice is missing                                                                      |
| **pdfplumber**                                        | optional column-preserving`.txt` for digital PDFs                                                                      |
| `concurrent.futures`                                      | multi-process parallelism                                                                                                |

## 6. Scaling to 10,000+ files

* **Resumable**: `manifest.jsonl` is written after every file; re-running skips finished files (same size + modified time).
* **Batched JVM**: 25 PDFs per OpenDataLoader launch (`odl_batch_size`); PDFs over 300 pages / 60 MB run alone; a failing batch is retried file by file.
* **Parallel**: OCR runs page-by-page across all worker processes; one Tesseract thread per process.
* **Isolated**: a corrupt file is recorded as `failed` with the reason; the run continues.
* **Bounded**: scratch files are deleted batch by batch; pages are rendered one at a time.
* Run on a server with `-w <cores>` and give huge PDFs more memory with `--java-heap-mb 4096`.

## 7. Known limits (honest list)

* OCR quality depends on the scan. Photographed pages, stamps and decorative/handwritten fonts give lower accuracy
  (see `ocr_mean_conf` and `warnings` in the manifest; < 60 is flagged for review).
* Tables inside **scanned** pages are rebuilt only when they have ruling lines. Borderless tables in scans need hybrid mode.
* Formulas become LaTeX only in hybrid mode (`--hybrid on --formulas`). Tesseract reads them as plain characters.
* Diagrams/charts are exported as image files and linked; their *content* is described only with `--pictures` (hybrid).
* Without LibreOffice, `.docx` page numbers follow explicit page breaks only.
