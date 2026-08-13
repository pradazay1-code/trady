"""Build a searchable knowledge base from the extracted book text.

Creates knowledge/kb.sqlite with an FTS5 index over ~2000-character passages so
the agent can cite book passages at decision time instead of guessing.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

PASSAGE_CHARS = 2000
OVERLAP_CHARS = 300

# Page-header / print-artifact noise that survives EPUB extraction.
NOISE = re.compile(
    r"""^(
        JWBK\d+.*|
        \d+\s+Part\s+[IVX]+:.*|
        Chapter\s+\d+:\s+.*\s+\d+|
        \s*\d+\s*|
        [A-Z\s]{4,}\d*
    )$""",
    re.VERBOSE,
)


def clean(text: str) -> str:
    keep = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            keep.append("")
            continue
        if NOISE.match(s) and len(s) < 90:
            continue
        keep.append(s)
    out = "\n".join(keep)
    return re.sub(r"\n{3,}", "\n\n", out)


def chapter_of(text: str, pos: int) -> str:
    """Nearest preceding '## Chapter N' style heading."""
    head = text.rfind("\n## ", 0, pos)
    if head == -1:
        return ""
    line = text[head + 4 : text.find("\n", head + 4)]
    return line.strip()[:120]


def passages(text: str):
    step = PASSAGE_CHARS - OVERLAP_CHARS
    for start in range(0, len(text), step):
        chunk = text[start : start + PASSAGE_CHARS]
        if len(chunk.strip()) < 200:
            continue
        yield start, chunk


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "knowledge/extracted")
    db_path = Path(sys.argv[2] if len(sys.argv) > 2 else "knowledge/kb.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE book (
            id INTEGER PRIMARY KEY,
            slug TEXT UNIQUE,
            title TEXT,
            words INTEGER
        );
        CREATE VIRTUAL TABLE passage USING fts5(
            book_slug UNINDEXED,
            chapter,
            body,
            offset_ UNINDEXED,
            tokenize = 'porter unicode61'
        );
        """
    )

    manifest = json.loads((src / "_manifest.json").read_text())
    meta = {m["slug"]: m for m in manifest}

    total = 0
    for txt in sorted(src.glob("*.txt")):
        raw = txt.read_text(encoding="utf-8")
        body = clean(raw)
        slug = txt.stem
        info = meta.get(slug, {})
        con.execute(
            "INSERT INTO book (slug, title, words) VALUES (?,?,?)",
            (slug, info.get("title", slug), len(body.split())),
        )
        rows = [
            (slug, chapter_of(body, off), chunk, off)
            for off, chunk in passages(body)
        ]
        con.executemany(
            "INSERT INTO passage (book_slug, chapter, body, offset_) VALUES (?,?,?,?)",
            rows,
        )
        total += len(rows)
        print(f"  {slug}: {len(rows)} passages")

    con.commit()
    con.execute("INSERT INTO passage(passage) VALUES('optimize')")
    con.commit()
    con.close()
    print(f"\nknowledge base: {total} passages -> {db_path}")


if __name__ == "__main__":
    main()
