"""writer.py - writes the final deliverables in the layout required by the OpenDataLoader project style:

    output/
      md/<name>.md            markdown (page markers, tables, image links)
      json/<name>.json        structured elements with bounding boxes (OpenDataLoader schema)
      txt/<name>.txt          plain text
      images/<name>/...       extracted pictures (one sub-folder per document)
      manifest.jsonl          one status line per input file
    Large documents additionally get  md/<name>.parts/  and  txt/<name>.parts/  (page-range files).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .renderers import IMG_TOKEN, join_pages, split_into_parts
from .utils import atomic_write_text, rel_posix


def _prepare_json(doc: dict, img_rel_root: str) -> dict:
    """Finalize JSON: real image paths (relative to the output folder) and an id on every element."""
    counter = [0]

    def visit(node):
        if isinstance(node, dict):
            node.pop("_top", None)  # private helper key from the OCR engine
            src = node.get("source")
            if isinstance(src, str) and src.startswith(IMG_TOKEN):
                node["source"] = src.replace(IMG_TOKEN, img_rel_root, 1)  # -> images/<name>/imageFile1.png
            if "type" in node and "id" not in node:
                counter[0] += 1
                node["id"] = 100000 + counter[0]  # high range avoids clashes with ODL ids
            for v in list(node.values()):
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
    visit(doc)
    return doc


def write_outputs(cfg, stem: str, md_pages: dict[int, str], txt_pages: dict[int, str], doc: dict,
                  image_src_dir: Path | None, extra_images: list[Path] | None = None) -> dict:
    """Write md/json/txt (+ parts) and move images. Returns a dict of produced paths (relative to output_dir)."""
    out = Path(cfg.output_dir)
    md_path, json_path, txt_path = out / "md" / f"{stem}.md", out / "json" / f"{stem}.json", out / "txt" / f"{stem}.txt"
    img_dir = out / "images" / stem
    n_images = 0
    # ---- images: move ODL's image folder to images/<name>/ ----
    if image_src_dir and Path(image_src_dir).is_dir() and any(Path(image_src_dir).iterdir()):
        if img_dir.exists():
            shutil.rmtree(img_dir)  # replace results of a previous run
        img_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_src_dir), str(img_dir))
    for extra in extra_images or []:  # e.g. the original picture of an image input
        img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extra, img_dir / extra.name)
    if img_dir.exists():
        n_images = sum(1 for _ in img_dir.iterdir())

    rel_img_root = f"images/{stem}"  # location relative to the output folder (used inside JSON)

    def md_text_for(location: Path) -> str:
        """Join pages and point image links from 'location' (a folder) to the images folder."""
        prefix = rel_posix(img_dir, location)  # e.g. ../images/name
        return join_pages(cfg, "md", md_pages).replace(IMG_TOKEN, prefix)

    atomic_write_text(md_path, md_text_for(md_path.parent))
    atomic_write_text(txt_path, join_pages(cfg, "txt", txt_pages))
    atomic_write_text(json_path, json.dumps(_prepare_json(doc, rel_img_root), ensure_ascii=False, indent=1))

    # ---- big documents: ALSO write page-range parts (huge files freeze editors such as VS Code) ----
    limit = cfg.split_md_over_mb * 1024 * 1024
    parts: list[str] = []
    if md_path.stat().st_size > limit and len(md_pages) > cfg.split_pages:
        parts_dir = md_path.parent / f"{md_path.name[:-3]}.parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        for label, text in split_into_parts(cfg, "md", md_pages):
            pf = parts_dir / f"{Path(stem).name}_{label}.md"
            atomic_write_text(pf, text.replace(IMG_TOKEN, rel_posix(img_dir, parts_dir)))
            parts.append(pf.relative_to(out).as_posix())
    if txt_path.stat().st_size > limit and len(txt_pages) > cfg.split_pages:
        parts_dir = txt_path.parent / f"{txt_path.name[:-4]}.parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        for label, text in split_into_parts(cfg, "txt", txt_pages):
            pf = parts_dir / f"{Path(stem).name}_{label}.txt"
            atomic_write_text(pf, text)
            parts.append(pf.relative_to(out).as_posix())
    return {"md": md_path.relative_to(out).as_posix(), "json": json_path.relative_to(out).as_posix(),
            "txt": txt_path.relative_to(out).as_posix(), "images": rel_img_root if n_images else None,
            "n_images": n_images, "parts": parts}
