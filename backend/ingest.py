"""Stage 0: read the 10-K files, split them into chunks, embed them, store them in Qdrant.

Run with:  uv run python -m backend.ingest
Safe to run twice: chunk IDs are stable, and finished filings are skipped.
"""
import csv
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag import count_points, ensure_collection, get_store

DATA = Path(__file__).resolve().parent.parent / "data"
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)  # e.g. "## Item 1A. Risk Factors"
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """Same inputs always give the same ID, so a re-run overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def make_chunks(row: dict) -> tuple[list[Document], list[str]]:
    """Split one filing into chunks. Chunks never cross from one Item into the next."""
    ticker, year = row["ticker"], int(row["fiscal_year"])
    docs, ids = [], []
    for item in read_items(DATA / row["file"]):
        for n, piece in enumerate(splitter.split_text(item["text"])):
            docs.append(Document(
                page_content=piece,
                metadata={"company": ticker, "year": year, "item": item["code"],
                          "item_title": item["title"], "source": row["source_url"]},
            ))
            ids.append(chunk_id(ticker, year, item["code"], n))
    return docs, ids


def ingest() -> int:
    """Ingest all filings. Returns the total number of points in the collection."""
    ensure_collection()
    store = get_store()
    with open(DATA / "filings.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        ticker, year = row["ticker"], int(row["fiscal_year"])
        docs, ids = make_chunks(row)
        if count_points([ticker], [year]) == len(docs):
            print(f"{ticker} {year}: already complete ({len(docs)} chunks), skipped")
            continue
        store.add_documents(docs, ids=ids, batch_size=200)
        print(f"{ticker} {year}: stored {len(docs)} chunks")
    total = count_points()
    print(f"Total points in collection: {total}")
    return total


if __name__ == "__main__":
    ingest()
