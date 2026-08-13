"""Extract EPUB books to clean, chapter-structured text in spine (reading) order.

EPUB is a zip of XHTML. We read META-INF/container.xml -> the OPF package file ->
the <spine> to get true reading order, then flatten each document to text while
keeping headings, list items and table rows as separate lines.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

BLOCK_TAGS = {
    "p", "div", "li", "tr", "td", "th", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "caption", "figcaption",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _ns(tag: str) -> str:
    return tag.split("}")[-1]


def opf_path(z: zipfile.ZipFile) -> str:
    root = ET.fromstring(z.read("META-INF/container.xml"))
    for el in root.iter():
        if _ns(el.tag) == "rootfile":
            return el.attrib["full-path"]
    raise RuntimeError("no rootfile in container.xml")


def spine_documents(z: zipfile.ZipFile) -> tuple[list[str], dict]:
    """Return (ordered hrefs, metadata dict)."""
    opf = opf_path(z)
    base = str(Path(opf).parent)
    root = ET.fromstring(z.read(opf))

    meta: dict[str, str] = {}
    manifest: dict[str, str] = {}
    spine: list[str] = []

    for el in root.iter():
        tag = _ns(el.tag)
        if tag in {"title", "creator", "date", "publisher", "identifier"}:
            if el.text and tag not in meta:
                meta[tag] = el.text.strip()
        elif tag == "item":
            manifest[el.attrib["id"]] = el.attrib["href"]
        elif tag == "itemref":
            spine.append(el.attrib["idref"])

    hrefs = []
    for idref in spine:
        href = manifest.get(idref)
        if not href:
            continue
        full = f"{base}/{href}" if base and base != "." else href
        # normalise ../ segments
        parts: list[str] = []
        for seg in full.split("/"):
            if seg == "..":
                if parts:
                    parts.pop()
            elif seg not in {".", ""}:
                parts.append(seg)
        hrefs.append("/".join(parts))
    return hrefs, meta


def doc_to_lines(html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "head"]):
        bad.decompose()

    lines: list[str] = []
    for el in soup.find_all(BLOCK_TAGS):
        # skip containers whose text is fully covered by nested block children
        if el.name == "div" and el.find(BLOCK_TAGS):
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in HEADING_TAGS:
            lines.append("")
            lines.append(f"{'#' * int(el.name[1])} {text}")
            lines.append("")
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name in {"td", "th"}:
            lines.append(f"| {text}")
        else:
            lines.append(text)
    return lines


def dedupe_consecutive(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if out and ln == out[-1] and ln.strip():
            continue
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    return out


def extract(path: Path, outdir: Path) -> dict:
    z = zipfile.ZipFile(path)
    hrefs, meta = spine_documents(z)
    names = set(z.namelist())

    all_lines: list[str] = []
    docs = 0
    for href in hrefs:
        if href not in names:
            continue
        try:
            lines = doc_to_lines(z.read(href))
        except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the book
            print(f"  ! {href}: {exc}", file=sys.stderr)
            continue
        if lines:
            all_lines.extend(lines)
            all_lines.append("")
            docs += 1

    all_lines = dedupe_consecutive(all_lines)
    text = "\n".join(all_lines).strip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)

    slug = re.sub(r"^[0-9a-f]{8}-", "", path.stem)
    out = outdir / f"{slug}.txt"
    out.write_text(text, encoding="utf-8")

    info = {
        "slug": slug,
        "source_file": path.name,
        "title": meta.get("title", slug),
        "creator": meta.get("creator", ""),
        "date": meta.get("date", ""),
        "spine_docs": len(hrefs),
        "extracted_docs": docs,
        "chars": len(text),
        "words": len(text.split()),
        "lines": text.count("\n"),
        "output": str(out),
    }
    print(f"  {slug}: {info['words']:,} words, {docs} docs -> {out.name}")
    return info


def main() -> None:
    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for epub in sorted(src.glob("*.epub")):
        print(f"extracting {epub.name}")
        manifest.append(extract(epub, outdir))

    (outdir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["words"] for m in manifest)
    print(f"\ntotal: {len(manifest)} books, {total:,} words")


if __name__ == "__main__":
    main()
