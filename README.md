# Acme Analytics Assistant

A small AI assistant for a fictional business, **Acme Analytics**. It answers questions from
(1) internal **PDF documents** via RAG, (2) a relational **SQLite database** via natural-language-to-SQL,
and (3) **arithmetic** via a sandboxed calculator, and it can chain these (e.g. SQL → Calculator).
A **Coordinator** routes each question, validates every step, and returns a structured, auditable response.

> **Read this first: what has and has not been verified.**
> This project was built and tested in a sandbox with **no access to Hugging Face, OpenAI, or a Docker daemon**.
>
> | Area | Status |
> |---|---|
> | Unit/API tests (94, `pytest`) | **Run and passing.** They use a *scripted LLM* and a *hashing test embedder* (not e5). |
> | PDF extraction, cleaning, chunking, FAISS index/search, SQL validation + execution, Calculator, Coordinator chaining, FastAPI | **Real code paths, exercised by the tests.** |
> | `intfloat/e5-large-v2` embeddings | **Implemented per the e5 contract, never executed here** (weights unreachable). Optional integration tests skip. |
> | Real LLM provider (OpenAI-compatible) | **Implemented, never called against a live endpoint.** |
> | `RAG_MIN_SIMILARITY=0.80` | **Uncalibrated guess** for e5; calibrate on your corpus (see below). |
> | Docker image build / `docker compose up` | **Not built or run** (no daemon). `docker compose config` validates the syntax only. |
> | Degraded mode (no model/LLM) | **Run for real** with uvicorn: `/health` → 503, `/status` shows the error, `/ask` → structured 503. |

## Architecture

```
POST /ask ──► Coordinator ──► Planner (LLM → JSON Plan, validated; rule-based fallback)
                 │                   steps limited to: rag | database | calculator
                 ├─► RAGAgent ──► e5 query embedding → FAISS top-k → similarity gate → LLM answers from context only
                 ├─► DatabaseAgent ─► schema-grounded LLM → sqlglot validator → read-only SQLite (+authorizer, limits)
                 └─► CalculatorTool ◄─ variables bound from earlier DB step results (no eval)
                          └─► deterministic answer composition from real step outputs + per-step trace
```

| Component | Responsibility |
|---|---|
| `app/agents/coordinator.py` | Plan → validate → execute → compose. Enforces step/time budgets; stops a chain at the first failed or empty step so numbers are never invented. |
| `app/rag/` | `loader` (PyMuPDF) → `preprocess` → `chunker` (sentence-aware, never crosses a page) → `embedding` (e5) → `vector_store` (FAISS IP + JSON metadata, fingerprinted) → `agent`. |
| `app/database/` | `seed` (idempotent), `validator` (sqlglot), `connection` (read-only + authorizer + row/time limits), `agent`. |
| `app/tools/calculator.py` | AST whitelist evaluator: `+ - * / // % **`, `abs round min max sum avg mean sqrt floor ceil percent_of percent_change`, constants `pi`. |
| `app/llm/` | `LLMProvider` protocol, OpenAI-compatible provider (timeouts, bounded retry/backoff), test-only scripted provider. |
| `app/main.py`, `app/api/` | FastAPI app, Pydantic request/response models, error mapping. |

### Request flow
1. Input is whitespace-normalised and validated (non-empty, ≤ 1000 chars).
2. The planner LLM returns JSON; it is parsed into a `Plan` and checked: only allowed step kinds, ids `s1..sN`, ≤ `MAX_PLAN_STEPS`, and calculator bindings may only reference **earlier database steps**. The model never emits SQL or function names for execution. If the planner is unreachable or returns an invalid plan, a deterministic keyword scorer produces a conservative single-step plan (or declares the question out of scope).
3. Steps run in order. Database results flow into calculator variables via bindings (`completed_total = s1.completed_total`, first row, numeric only).
4. The final answer text is **assembled from the actual step outputs**, not re-narrated by an LLM, so a result is never presented unless it was produced.

### RAG
- PDFs → one record per page → cleaning (NFKC, de-hyphenation, page-number and repeated header/footer removal, near-empty pages dropped) → sentence-aware chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`) carrying `file`, `page`, `chunk_id`, `section`.
- Embeddings: **`intfloat/e5-large-v2` only**; `passage: ` prefix for chunks, `query: ` for questions, L2-normalised, so inner product = cosine. The model is loaded lazily once per process, CPU only.
- Index: FAISS `IndexFlatIP`, persisted under `INDEX_DIR` with a manifest whose fingerprint covers PDF bytes, model name and chunking parameters. Changing any of them marks the index stale and it is rebuilt (`scripts.build_index`, or automatically at startup).
- Answering: top-`RAG_TOP_K` chunks above `RAG_MIN_SIMILARITY` are passed to the LLM with a prompt that forbids outside knowledge and requires `INSUFFICIENT_EVIDENCE` when the context lacks the answer. **If no chunk passes the threshold, no LLM call is made** and the response is `insufficient_evidence`. Sources (file, page, chunk id, score, snippet) come from retrieval metadata, never from model text.

### SQL safety (defence in depth)
1. **Static validation** (sqlglot, SQLite dialect): exactly one statement; only `SELECT` (incl. `WITH … SELECT` and unions); no DML/DDL/PRAGMA/ATTACH/transactions; no `INTO`; no schema-qualified names; tables limited to `customers`, `orders` (CTE names allowed); functions limited to an allow-list.
2. **Read-only connection** (`mode=ro`) with a **SQLite authorizer** that only permits reads of the two tables and allow-listed functions — so even if the validator were bypassed, writes, `sqlite_master`, PRAGMA and ATTACH fail (tested directly).
3. **Limits**: `SQL_MAX_ROWS` (response flags `truncated`), `SQL_TIMEOUT_SECONDS` via a progress handler (a runaway recursive CTE is aborted in tests).
4. One repair attempt feeds the validation/SQLite error back to the model. Public errors are generic; details go to logs.

### Calculator
No `eval`/`exec`. Input is parsed with `ast`, then only whitelisted nodes are evaluated. Rejected: attribute access, subscripts, lambdas, strings, booleans, imports, unknown names, keyword args, huge exponents/expressions, non-finite or complex results, and division by zero (including `percent_of(x, 0)`).

## Setup

Requirements: Python 3.11+ (tested on 3.13), ~3 GB disk for torch + the e5 weights (~1.3 GB), ≥ 4 GB RAM. No GPU needed; first load and indexing on CPU can take minutes.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: avoids multi-GB CUDA wheels
pip install -r requirements-dev.txt
cp .env.example .env            # then set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL_NAME
```

**Embedding model.** On first use `sentence-transformers` downloads `intfloat/e5-large-v2` from Hugging Face into `HF_HOME`. For restricted networks, download it elsewhere (`huggingface-cli download intfloat/e5-large-v2 --local-dir /path/e5`) and set `EMBEDDING_MODEL_PATH=/path/e5`. A missing/unreachable model produces a clear error (`scripts.build_index` exits 2; the API starts degraded and `/health` returns 503). Fully offline operation is only possible if both the model and an LLM are local.

**LLM.** Any OpenAI-compatible `/chat/completions` server (OpenAI, vLLM, Ollama, LM Studio). Set `LLM_BASE_URL`, `LLM_API_KEY` (may be empty for local servers), `LLM_MODEL_NAME`. The planner uses `response_format: json_object`; servers lacking it may need a model that follows the JSON instruction anyway (invalid output falls back to rules).

### Data and index
```bash
python -m scripts.generate_documents   # (re)generate the 3 sample PDFs from scripts/document_sources.py
python -m scripts.seed_db              # create schema + load 40 customers / 251 orders (idempotent, deterministic)
python -m scripts.build_index          # embed chunks with e5 and write the FAISS index (--force to rebuild)
```

### Run
```bash
uvicorn app.main:app --port 8000
curl localhost:8000/health
curl localhost:8000/status
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the refund window for annual subscriptions?"}'
```

### Docker (untested here — see status table)
```bash
cp .env.example .env     # set the LLM variables
docker compose up --build
```
The image installs CPU torch, copies the code and sample PDFs, and **does not bake in model weights**. The entrypoint seeds the DB, builds the index only if missing/stale, then serves. Named volumes keep the DB/index (`acme_state`) and the e5 weights (`hf_cache`) across container removal. `/health` stays 503 until the index exists (first start downloads ~1.3 GB and embeds on CPU); the healthcheck has a 10-minute start period.

## API

`POST /ask` → `AskResponse` (`status`: `ok | insufficient_evidence | out_of_scope | needs_clarification | error`; `route`: `rag | database | calculation | hybrid | out_of_scope | clarify`). `GET /health` (200 ok / 503 degraded with per-check booleans). `GET /status` (models, index, paths, limits; never secrets).

HTTP mapping: invalid input → 422 `invalid_request`; LLM not configured/unavailable → 503; execution failure with a RAG step → 503, other failed steps → 422; internal errors → 500 with a generic body (stack traces only in logs).

### Example responses
Produced by `python -m scripts.generate_examples` using the **test harness** (scripted LLM + hashing test embedder). Sources, SQL, rows and calculator values are genuine outputs of the real code; **the prose of RAG answers is the scripted stand-in echoing retrieved text, and similarity scores are from the test embedder, not e5**. Abbreviated:

**RAG** — "What is the refund window for annual subscriptions?" → `route: rag`, source `refund_policy.pdf` page 1, chunk `refund_policy-p1-c0`, section "1. Refund eligibility" (30-day guarantee text).

**RAG** — "What discount do non-profit organizations receive?" → sources include `discount_policy.pdf` page 2 (20 percent flat discount).

**SQL (JOIN + GROUP BY)** — "Which 3 countries have the highest total completed order amount?"
```json
{"route": "database", "status": "ok",
 "sql": "SELECT c.country, ROUND(SUM(o.amount), 2) AS total_amount FROM orders AS o JOIN customers AS c ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.country ORDER BY total_amount DESC LIMIT 3",
 "result": {"columns": ["country", "total_amount"],
            "rows": [["Spain", 69517.69], ["Australia", 55721.49], ["United States", 54284.48]], "row_count": 3}}
```

**SQL (filter + aggregate)** — "How many pending orders are there?" → `SELECT COUNT(*) AS pending_orders FROM orders WHERE status = 'pending'` → `pending_orders = 38`.

**Hybrid SQL → Calculator** — "What is the total amount of completed orders, and what percentage of the total of all orders is that?"
steps: `s1 database` (`completed_total = 371401.78`) → `s2 database` (`all_total = 604712.61`) → `s3 calculator`
`completed_total / all_total * 100` with those bound values = **61.4179** (matches an independent reference computed in the test).

**Out of scope** — "Who won the football world cup in 2018?" → `status: out_of_scope`, no tools run.

## Tests
```bash
pytest                 # fast suite (94 tests): no network, no model download, no real LLM
pytest -m integration  # optional: real e5-large-v2 (skips if weights are unavailable)
```
Covers: calculator (valid/invalid/unsafe/zero-division), SQL validator (DELETE, DROP, multi-statement, unauthorized table, `sqlite_master`, PRAGMA, ATTACH, invalid SQL, `load_extension`), authorizer defence with validator bypassed and DB-unchanged check, row limit and timeout, seed idempotency + foreign keys, preprocessing/chunking invariants, three RAG retrievals with file/page assertions, off-topic → insufficient evidence with zero LLM calls, index reuse/staleness, DB agent aggregation and JOIN results compared with an independent SQL reference, repair attempt, SQL → Calculator chain vs. reference, planner fallback, API shape/validation/health/status/degraded mode.

**Test-suite caveat:** RAG tests assert that the evidence-bearing chunk is in the returned top-k with the right file/page (not a fixed rank), because the test embedder is lexical. They say nothing about e5's ranking quality.

## Calibrating `RAG_MIN_SIMILARITY`
e5 cosine scores are compressed (unrelated text often still scores ~0.7+). Run `pytest -m integration -s` to print in-domain vs. off-topic top scores, then set the threshold between them. The default `0.80` is an **untested starting point** and may wrongly reject valid questions or admit irrelevant ones.

## Limitations
- Quality depends on the LLM's SQL/planning ability; validation guarantees *safety*, not *correctness* of generated SQL.
- Planner fallback is intentionally crude (keyword scoring) and only covers single-step routes.
- SQLite/FAISS-flat are sized for a demo corpus; no auth, rate limiting, or multi-tenant isolation.
- Retries apply to LLM calls only; all database operations are read-only so retries cannot cause side effects.
- Logs record request id, route, steps, timings, and a question hash + length (not the raw question); API keys are never logged or returned.


