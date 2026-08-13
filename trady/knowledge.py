"""Query the book knowledge base.

The five reference books are indexed into SQLite FTS5 (see tools/build_kb.py).
This module is how the agent — or you — consults them at decision time, so a
rule can always be traced back to the page it came from instead of to a hunch.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "knowledge" / "kb.sqlite"


@dataclass
class Passage:
    book: str
    chapter: str
    text: str
    score: float

    def cite(self) -> str:
        title = self.book.replace("_", " ")
        return f"{title}{' — ' + self.chapter if self.chapter else ''}"


def _fts_query(text: str, mode: str = "and") -> str:
    """Quote each token so punctuation and digits can't break FTS5 syntax."""
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", text) if t]
    if not tokens:
        return '""'
    quoted = [f'"{t}"' for t in tokens]
    return (" OR " if mode == "or" else " ").join(quoted)


class KnowledgeBase:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path or DEFAULT_DB)
        if not self.path.exists():
            raise FileNotFoundError(
                f"knowledge base not found at {self.path}. "
                "Build it with: python3 tools/build_kb.py"
            )

    def _query(self, expr: str, limit: int, book: str | None) -> list[Passage]:
        sql = (
            "SELECT book_slug, chapter, body, rank FROM passage "
            "WHERE passage MATCH ?"
        )
        params: list = [expr]
        if book:
            sql += " AND book_slug LIKE ?"
            params.append(f"%{book}%")
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(limit))

        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        return [
            Passage(r["book_slug"], r["chapter"] or "", r["body"], float(r["rank"]))
            for r in rows
        ]

    def search(self, query: str, limit: int = 5, book: str | None = None) -> list[Passage]:
        """All-terms search, falling back to any-term when that finds nothing."""
        hits = self._query(_fts_query(query, "and"), limit, book)
        if not hits:
            hits = self._query(_fts_query(query, "or"), limit, book)
        return hits

    def ask(self, query: str, limit: int = 3, width: int = 700) -> str:
        """Formatted answer with citations, for the CLI and reports."""
        hits = self.search(query, limit)
        if not hits:
            return f"No passages found for {query!r}."
        out = [f'Knowledge base — "{query}"', "=" * 62]
        for i, p in enumerate(hits, 1):
            body = " ".join(p.text.split())
            if len(body) > width:
                body = body[:width].rsplit(" ", 1)[0] + " ..."
            out += [f"\n[{i}] {p.cite()}", f"    {body}"]
        return "\n".join(out)

    def books(self) -> list[tuple[str, int]]:
        con = sqlite3.connect(self.path)
        try:
            return [
                (r[0], r[1])
                for r in con.execute("SELECT slug, words FROM book ORDER BY slug")
            ]
        finally:
            con.close()

    def stats(self) -> dict:
        con = sqlite3.connect(self.path)
        try:
            passages = con.execute("SELECT count(*) FROM passage").fetchone()[0]
            books = con.execute("SELECT count(*), sum(words) FROM book").fetchone()
        finally:
            con.close()
        return {"books": books[0], "words": books[1], "passages": passages}


# ---------------------------------------------------------------- lookup
# Topics the agent consults when explaining a decision or a rule.
TOPIC_QUERIES = {
    "pdt": "pattern day trader margin account rule 2520 four day trades",
    "position_sizing": "fixed fractional position size equity trade risk",
    "kelly": "Kelly criterion percentage winning trades ratio",
    "stops": "stop loss order limit losses discipline",
    "hammer": "hammer umbrella line lower shadow real body reversal",
    "engulfing": "engulfing pattern real body opposite colour reversal",
    "doji": "doji indecision open close same warning reversal",
    "morning_star": "morning star three candle bullish reversal gap",
    "evening_star": "evening star three candle bearish reversal gap",
    "dark_cloud": "dark cloud cover piercing pattern penetration halfway",
    "harami": "harami small real body inside prior long body",
    "support_resistance": "support resistance trendline channel breakout",
    "breakout": "breakout false breakout resistance support new trend",
    "gaps": "gap open price break trading session news",
    "volume": "volume confirms price trend demand on-balance volume",
    "expectancy": "expected return winning losing trades percentage",
    "ruin": "probability of ruin advantage number of trades",
    "overtrading": "overtrading commissions fewer trades money management",
    "mistakes": "common day trading mistakes losing trades emotion",
    "backtesting": "backtesting curve fitting over-optimization data mining",
    "journal": "trading diary record why trade lessons learned",
    "money_management": "money management techniques limit losses trade size",
}


def explain(topic: str, kb: KnowledgeBase | None = None, limit: int = 2) -> str:
    """Answer from the books on a known topic, or fall back to free search."""
    kb = kb or KnowledgeBase()
    return kb.ask(TOPIC_QUERIES.get(topic, topic), limit=limit)
