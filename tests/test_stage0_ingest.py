"""Stage 0 tests. They use the real Qdrant and OpenRouter services."""
import csv

from backend.ingest import DATA, ingest, make_chunks
from backend.rag import count_points

EXPECTED_TOTAL = 10_229


def test_chunk_count_without_network():
    # Counting chunks from the files alone needs no services.
    with open(DATA / "filings.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert sum(len(make_chunks(r)[0]) for r in rows) == EXPECTED_TOTAL


def test_collection_has_exact_count_and_second_run_changes_nothing():
    first = ingest()   # first run may store chunks; later runs skip everything
    assert first == EXPECTED_TOTAL
    second = ingest()  # must skip every filing and add nothing
    assert second == EXPECTED_TOTAL


def test_each_filing_has_points_and_metadata():
    with open(DATA / "filings.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert count_points([row["ticker"]], [int(row["fiscal_year"])]) > 0
