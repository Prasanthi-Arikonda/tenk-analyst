"""Stage 7 tests. They use FastAPI's TestClient, so no server and no LLM are needed."""
import logging

import pytest
from fastapi.testclient import TestClient

from backend import api

CODE = "test-code-123"
GOOD = {"question": "What was Ford's R&D expense in 2025?", "chat_id": "c1", "pipeline": "agent",
        "companies": [], "years": []}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", CODE)  # a test value; the real code is never used here
    return TestClient(api.app)  # no "with": the start-up check is tested separately below


def post(client, body=None, code=CODE, **changes):
    headers = {"X-Access-Code": code} if code is not None else {}
    return client.post("/ask", json={**(body or GOOD), **changes}, headers=headers)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_missing_and_wrong_code(client):
    for code in (None, "wrong"):
        r = post(client, code=code)
        assert r.status_code == 401 and r.json() == {"error": "Wrong or missing access code."}


def test_empty_question(client):
    for q in ("", "   "):
        r = post(client, question=q)
        assert r.status_code == 400 and r.json() == {"error": "The question is empty."}


def test_question_too_long(client):
    r = post(client, question="x" * 1001)
    assert r.status_code == 400
    assert r.json() == {"error": "The question is too long. Use at most 1,000 characters."}


def test_question_of_exactly_1000_characters_is_allowed(client, monkeypatch):
    monkeypatch.setattr(api, "answer", lambda *a: {"answer": "ok"})
    assert post(client, question="x" * 1000).status_code == 200


def test_unknown_pipeline(client):
    r = post(client, pipeline="magic")
    assert r.status_code == 400
    assert r.json() == {"error": "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."}


def test_ok_response_is_passed_through(client, monkeypatch):
    fake = {"answer": "hi", "pipeline": "agent", "chat_id": "c1", "tools_used": [], "sources": []}
    seen = {}

    def fake_answer(*args):
        seen["args"] = args
        return fake

    monkeypatch.setattr(api, "answer", fake_answer)
    r = post(client)
    assert r.status_code == 200 and r.json() == fake
    assert seen["args"] == (GOOD["question"], "agent", [], [], "c1")


def test_service_failure_gives_503_and_logs_only_the_type(client, monkeypatch, caplog):
    def boom(*args):
        raise ConnectionError("secret-key-sk-12345")

    monkeypatch.setattr(api, "answer", boom)
    with caplog.at_level(logging.ERROR):
        r = post(client)
    assert r.status_code == 503
    assert r.json() == {"error": "The answer service is not available right now. Try again in a minute."}
    assert "secret-key" not in r.text and "secret-key" not in caplog.text
    assert "ConnectionError" in caplog.text


def test_start_up_check_names_the_variable_not_the_value():
    good = {n: "v" for n in api.REQUIRED}
    assert api.check_env(good) is None
    missing = {**good, "QDRANT_URL": ""}
    assert api.check_env(missing) == "Missing environment variable: QDRANT_URL"
    bad = {**good, "OPENROUTER_API_KEY": "sk-abc\nsk-def", "LANGSMITH_API_KEY": "ls-1"}
    msg = api.check_env(bad)
    assert "OPENROUTER_API_KEY" in msg and "sk-abc" not in msg
    assert "LANGSMITH_API_KEY" in api.check_env({**good, "LANGSMITH_API_KEY": "a\nb"})
