"""Stage 7: the FastAPI back end.

Start it with:  uv run python -m uvicorn backend.api:app
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.pipelines import answer

load_dotenv()
log = logging.getLogger("tenk.api")

REQUIRED = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL",
            "SEC_USER_AGENT", "ACCESS_CODE"]
OPTIONAL = ["LANGSMITH_API_KEY"]  # tracing is optional, but a bad value would break it
PIPELINES = ["simple", "filters", "rerank", "decompose", "agent"]
MAX_QUESTION = 1000

ERR_CODE = "Wrong or missing access code."
ERR_EMPTY = "The question is empty."
ERR_LONG = "The question is too long. Use at most 1,000 characters."
ERR_PIPELINE = "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."
ERR_SERVICE = "The answer service is not available right now. Try again in a minute."


def check_env(env=os.environ) -> str | None:
    """Return a message naming the first bad variable, or None if all is well.
    The message has only the variable's NAME, never its value."""
    for name in REQUIRED:
        if not env.get(name):
            return f"Missing environment variable: {name}"
    for name in REQUIRED + OPTIONAL:
        value = env.get(name) or ""
        if "\n" in value or "\r" in value:
            return f"Environment variable {name} contains a line break"
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    problem = check_env()
    if problem:
        raise SystemExit(problem)  # stop at start-up, before any request
    yield


app = FastAPI(title="10-K Analyst", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    chat_id: str
    pipeline: str
    companies: list[str] = []
    years: list[int] = []


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def bad_request(request, exc):
    # Never return FastAPI's detailed validation text.
    return error(400, "The request is not valid.")


@app.get("/health")
def health():
    # Calls no paid service.
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest, x_access_code: str | None = Header(default=None)):
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected or not x_access_code or not secrets.compare_digest(x_access_code, expected):
        return error(401, ERR_CODE)
    if not req.question.strip():
        return error(400, ERR_EMPTY)
    if len(req.question) > MAX_QUESTION:
        return error(400, ERR_LONG)
    if req.pipeline not in PIPELINES:
        return error(400, ERR_PIPELINE)
    try:
        return answer(req.question, req.pipeline, req.companies, req.years, req.chat_id)
    except Exception as e:
        # Log only the error TYPE: the message could contain a key or a connection string.
        log.error("Request failed: %s", type(e).__name__)
        return error(503, ERR_SERVICE)
