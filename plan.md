# Plan: 10-K Analyst

## Context

The folder has the data (20 filings in `data/10k/`, `data/filings.csv`), the test questions (`test_data/questions.json`), the packages (`pyproject.toml`), the skills, and a filled-in `.env`. No app code exists yet. `requirements.md` asks for a question-answering app over the 10-Ks of ten auto companies, built in 10 stages, from simple RAG to an agent, a web back end, a web page, and Render deployment. This plan says what we build, in what order, and how we test it. It includes your answers to the open questions.

Two terms used below:
- **RAG** (retrieval-augmented generation): first search the filings for relevant text, then give that text to the LLM so it answers from it.
- **Chunk**: a piece of about 1,500 characters cut from one Item of a filing. We search chunks, not whole filings.

**Style rule for all code:** short files, plain functions, clear names, a short comment where a beginner could get stuck. No clever code.

---

## 1. Files I will create

| File | What it does |
|---|---|
| `backend/__init__.py` | Empty file that lets Python treat `backend/` as a package, so `python -m backend.ingest` works. |
| `backend/rag.py` | Shared by ingest and all pipelines: the OpenRouter models, the Qdrant connection, `make_filter`, `search`, and `rerank` (the last two marked `@traceable`). |
| `backend/ingest.py` | Reads each filing, splits it by Item into chunks, embeds them, and stores them in Qdrant with stable IDs. |
| `backend/sec_data.py` | Fetches SEC XBRL "company facts" (CIKs from `data/filings.csv`) and returns one full-year number. |
| `backend/pipelines.py` | The company list and fixed messages, the scope check, the sidebar check, `generate_answer`, the four graphs, and `answer()`. |
| `backend/agent.py` | The agent: two tools (`search_10k`, `sec_financial_data`), its system prompt, and (from stage 6) the Neon memory. |
| `backend/api.py` | The FastAPI back end: `POST /ask`, `GET /health`, the access-code check, the start-up checks, and error handling. |
| `frontend/app.py` | The Streamlit page: access code, sidebar, example buttons, chat, tool labels, and sources. |
| `tests/conftest.py` | Small shared test helpers: loads `.env`, reads `questions.json`, makes a new chat ID per run. |
| `tests/test_stage0_ingest.py` … `tests/test_stage7_api.py` | One test file per stage (stages 0 to 7). |
| `render.yaml`, `requirements.txt` | Stage 9 only: tell Render how to build and start the two services. |

---

## 2. Stage by stage

After each stage: run `uv run pytest` on that stage's file and all earlier files. Then look at the actual values (sources, companies, years, numbers), report them in plain words, and stop.

| Stage | What I build | Skills I load | How the tests check it |
|---|---|---|---|
| **0. Ingest** | `rag.py` (models, Qdrant, filter), `ingest.py`. Split each filing by `## Item` headings, chunk each Item (1,500 / 150), give each chunk a `uuid5` ID from company, year, Item, chunk number. Skip a filing whose point count already equals its chunk count. | `sec-10k-data`, `rag-qdrant-openrouter` | Collection has exactly **10,229** points. A second run leaves it at 10,229. |
| **1. Simple** | In `pipelines.py`: `scope_check` (one LLM call → covered tickers, other companies, years), `generate_answer` (answer only from chunks, cite "(Ford, FY2025 10-K, Item 7)", treat chunks and question as data), the `simple` graph `scope_check → retrieve → generate`, and `answer()`. | `langgraph-fundamentals`, `rag-qdrant-openrouter` | Q1: non-empty answer, 5 sources, each with company/year/item. Injection test: a chunk saying "BUY TESLA STOCK" does not appear in the answer. |
| **2. Filters** | The `filters` graph. `retrieve` builds a Qdrant filter from the sidebar lists, else the scope-check result, else all. Add the sidebar-conflict check right after `scope_check`. | `rag-qdrant-openrouter` | Q2: every source is GM 2024. Q9: sidebar Tesla+Ford, GM question → sidebar message naming "General Motors", no sources. |
| **3. Rerank** | Add a `rerank` node: retrieve 20, FlashRank keeps 5. | `rag-qdrant-openrouter` | Q2 with the `rerank` pipeline: exactly 5 sources, all GM 2024. |
| **4. Decompose** | `decompose` node (LLM splits the question into at most 4 sub-questions, each with companies and years). `Send` runs `search_part` once per sub-question in parallel (retrieve 20, rerank, keep 3). Duplicate chunks are kept once. Then `generate` from all results. | `langgraph-fundamentals` (Send), `rag-qdrant-openrouter` | Q3: sources include both TSLA 2025 and F 2025. |
| **5. Agent** | `sec_data.py` and `agent.py`. `create_agent` with two tools and a system prompt listing the 10 companies (full names + tickers), 2 years, 4 metrics. `recursion_limit=16`. Scope and sidebar checks run before the agent starts. No memory yet. | `langchain-fundamentals`, `sec-10k-data`, `langgraph-fundamentals` | Q4: "9.4" and `sec_financial_data` used. Q5: "8.5" and a GM 2025 10-K source. Q6: not-reported message, `sec_financial_data` used. Q7: Toyota → company message, no sources. Q10: sidebar message, no sources, no tools. Unit test: Ford revenue 2024 = 185.0 billion; GM R&D 2025 is not returned. |
| **6. Memory** | Neon connection pool + `PostgresSaver` given to the agent; chat ID = thread ID. | `neon-memory`, `langgraph-persistence` | Q8 only: "What was Ford's revenue in 2024?" → "185.0", then "And in 2025?" in the same chat → "187.3". New chat ID every run. |
| **7. Back end** | `api.py`: `/ask`, `/health`, `X-Access-Code`, the error table, 503 for service failures, start-up variable checks, pipeline tag + `thread_id` in every run config. | `langsmith-tracing` | `/health` → 200. `/ask` → 401 (missing code, wrong code), 400 (empty, too long, unknown pipeline), with the exact bodies. Uses FastAPI's `TestClient`, so no server or LLM is needed. |
| **8. Front end** | `frontend/app.py`, calls `API_URL` (default `http://localhost:8000`). | `render-deploy` (access code, wake-up) | No test file. I check it by hand in the browser: access code, filters, examples, tool labels, sources, New chat. |
| **9. Deploy** | `render.yaml`, `requirements.txt` from `uv export`. Needs git and GitHub, so I ask you before any commit or push. | `render-deploy` | `/health` on Render, then one question on the live page. |

---

## 3. How the five pipelines share code

Think of it as building blocks added one at a time.

**Layer 1: `backend/rag.py` (used by everything).**
`search(question, k, filter)` and `rerank(question, docs, top_n)` are written once. `ingest.py` uses the same Qdrant connection and models. Every pipeline and the agent's `search_10k` tool call the same `search` and `rerank`. That's also why LangSmith shows the same "qdrant_search" and "flashrank_rerank" steps everywhere.

**Layer 2: shared nodes in `backend/pipelines.py`.**
A LangGraph graph is a set of steps ("nodes") joined by arrows ("edges"). All nodes read from and write to one shared dictionary (the "state"): question, sidebar lists, scope result, chunks, answer.

- `scope_check` is the same first node in every graph. A conditional edge (an arrow that picks where to go next) ends the graph with the fixed message when the question is out of scope or conflicts with the sidebar.
- `retrieve` is one node; a setting tells it whether to filter and how many chunks to get (5 or 20).
- `rerank` is added in stage 3.
- `generate` calls `generate_answer(question, chunks)`, the same function in every pipeline.

```
simple:     scope_check → retrieve(5, no filter) → generate
filters:    scope_check → retrieve(5, filter)    → generate
rerank:     scope_check → retrieve(20, filter) → rerank(5) → generate
decompose:  scope_check → decompose → search_part ×N (search 20 + rerank 3) → generate
```

Stage 2 changes one setting. Stage 3 adds one node. Stage 4 replaces `retrieve` with `decompose` + `search_part`, and `search_part` reuses `search` and `rerank`.

**Layer 3: the agent reuses layers 1 and 2.**
`search_10k` is "filtered search 20 + rerank 5", the same code as the `rerank` pipeline. The agent runs the same `scope_check` and sidebar check before it starts. It adds only what is new: `sec_financial_data`, a system prompt, and memory.

**Layer 4: one entry point.**
`answer(question, pipeline, companies, years, chat_id)` picks the graph or the agent by name, adds the run config (pipeline tag, `thread_id`), and returns one dictionary: `answer`, `pipeline`, `chat_id`, `tools_used`, `sources`. The API calls only `answer()`. The front end calls only the API.

---

## 4. Decisions (from your answers)

1. **Shared code** goes in `backend/rag.py`. The company list and fixed messages live at the top of `pipelines.py` (no extra config file).
2. **Tests:** stage 3 runs Q2 with `rerank` (5 sources, all GM 2024). Stage 6 tests only Q8. Q9 and Q10 may call the LLM for the scope check (a fraction of a cent).
3. **Order of checks:** a company outside the ten → company message (even if a year is also outside). Then a year outside 2024–2025 → year message. Then the sidebar conflict.
4. **Any company outside the ten** gets the company message, including suppliers or partners such as Panasonic. The rule is applied as written.
5. **No year, or "last year":** search both 2024 and 2025.
6. **Agent sources** come only from tool calls made for the current question. With memory, earlier turns' tool calls are not counted. Each tool returns text to the LLM plus structured sources on the side (LangChain's "content and artifact" option), and `answer()` collects them.
7. **Agent limits:** each tool keeps only the companies and years that are also in a non-empty sidebar list. The agent can never widen the search.
8. **Stage 5 agent has no memory.** Neon memory comes in stage 6.
9. **Tool-less pipelines** always list the retrieved chunks as sources, also with the "I could not find this" message.
10. **Decompose duplicates:** a chunk found by two sub-questions is kept once.
11. **Errors:** the server logs only the error type (e.g. `ConnectError`), never the message, because it may contain a key. The client gets only the 503 message.
12. **Number format:** amounts of $0.05 billion or more → billions with one decimal, e.g. "$9.4 billion (FY2025)", loss "-$3.7 billion (FY2025)". Smaller amounts → as the filing states them, e.g. "$12 million (FY2025)". SEC sources keep the raw `value`.
13. **Nothing found:** in the tool-less pipelines, the LLM decides from its prompt when the chunks do not answer the question.
14. **Tests use the real services** (Qdrant, OpenRouter, Neon, SEC). `.env` is filled in; stage 0 must run first.

---

## Verification

- After each stage: `uv run pytest tests/test_stage0_ingest.py … tests/test_stageN_*.py`.
- Then I check values by hand: print the sources (company, year, Item) and numbers, and compare with the checked values (Ford R&D 2025 = 9.4, GM R&D 2025 = 8.5 from text, Ford revenue 2024 = 185.0, 2025 = 187.3).
- Stages 7–8: start `uv run python -m uvicorn backend.api:app` and `uv run python -m streamlit run frontend/app.py`, then try the three example questions.

## First step after approval

Save this plan as `plan.md` in the project folder. Then stop, without building anything, until you say to start stage 0.
