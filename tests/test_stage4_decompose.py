"""Stage 4 tests: the decompose pipeline."""
import json
from pathlib import Path

from langchain_core.documents import Document

from backend.pipelines import MAX_PARTS, answer, merge_parts, run_decompose

QUESTIONS = json.loads((Path(__file__).parent.parent / "test_data" / "questions.json").read_text())


def test_q3_sources_include_tesla_2025_and_ford_2025():
    q = next(x for x in QUESTIONS["questions"] if x["id"] == 3)
    out = answer(q["question"], "decompose", [], [], "test-chat")
    pairs = {(s["company"], s["year"]) for s in out["sources"]}
    assert ("TSLA", 2025) in pairs and ("F", 2025) in pairs
    assert out["answer"].strip()


def test_at_most_four_sub_questions():
    parts = run_decompose("Compare risks, R&D, revenue, cybersecurity, tariffs and recalls of Tesla, GM, Ford and Rivian.")
    assert 1 <= len(parts) <= MAX_PARTS


def test_duplicate_chunks_are_kept_once():
    a = Document(page_content="same text", metadata={"company": "F", "year": 2025, "item": "1A"})
    b = Document(page_content="same text", metadata={"company": "F", "year": 2025, "item": "1A"})
    c = Document(page_content="other text", metadata={"company": "F", "year": 2025, "item": "1A"})
    assert len(merge_parts({"part_chunks": [a, b, c]})["chunks"]) == 2


def test_sidebar_conflict_still_stops_decompose():
    out = answer("What was GM's R&D expense in 2025?", "decompose", ["TSLA", "F"], [], "test-chat")
    assert "General Motors" in out["answer"]
    assert out["sources"] == []
