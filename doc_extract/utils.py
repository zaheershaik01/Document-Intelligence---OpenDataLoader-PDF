"""utils.py - small helpers shared by all modules (logging, safe names, atomic writes)."""

from __future__ import annotations

import hashlib  # short hashes for long file names
import logging  # standard logging
import os  # file replace
import re  # regular expressions
import tempfile  # temporary files for atomic writes
from pathlib import Path


def setup_logging(level: str, log_file: Path | None = None) -> None:
    """Configure console (and optional file) logging once at program start."""
    handlers = [logging.StreamHandler()]  # always log to the console
    if log_file is not None:  # optionally also log to a file
        log_file.parent.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(  # apply the configuration
        level=getattr(logging, level.upper(), logging.INFO),  # text level -> number
        format="%(asctime)s %(levelname)-7s %(message)s",  # one line per message
        datefmt="%H:%M:%S",  # short time stamp
        handlers=handlers,
        force=True,  # replace handlers installed earlier (needed in worker processes)
    )


def short_hash(text: str, n: int = 8) -> str:
    """Return the first n hex digits of the SHA-1 of text (used to keep names unique)."""
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:n]


def safe_stem(stem: str, max_len: int = 90) -> str:
    """Make a file stem safe for every OS: strip bad characters and cap the length."""
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", stem).strip(" .")  # Windows-illegal chars
    if not cleaned:  # nothing left -> use a placeholder
        cleaned = "file"
    if len(cleaned) > max_len:  # very long -> shorten but stay unique with a hash
        cleaned = cleaned[: max_len - 9] + "_" + short_hash(stem)
    return cleaned


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so that a crash never leaves a half-written file (write temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)  # create folders on demand
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # temp file in the same folder
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:  # UTF-8, Unix newlines
            fh.write(text)  # write everything
        os.chmod(tmp, 0o644)  # mkstemp creates 0600 files; make them normally readable
        os.replace(tmp, path)  # atomic rename over the target
    except BaseException:
        if os.path.exists(tmp):  # clean up the temp file on failure
            os.remove(tmp)
        raise


def rel_posix(path: Path, start: Path) -> str:
    """Relative path from start to path with forward slashes (valid in markdown links everywhere)."""
    return Path(os.path.relpath(path, start)).as_posix()
