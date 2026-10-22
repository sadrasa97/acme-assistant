"""Coordinator: plan (LLM, structured, allow-listed) -> validate -> execute steps -> compose answer."""
from __future__ import annotations

import re
import time
import uuid

from pydantic import ValidationError

from app.agents.contracts import (
    AskResponse, CalculatorResult, DatabaseResult, Plan, PlanStep, RAGResult, Route, Status, StepTrace,
)
from app.core.config import Settings
from app.core.errors import AppError, PlanError, RequestTimeoutError
from app.core.logging import get_logger, log_event, question_fingerprint
from app.database.agent import DatabaseAgent
from app.llm.base import ChatMessage, LLMProvider, parse_json_object
from app.rag.agent import RAGAgent
from app.tools.calculator import CalculatorTool

log = get_logger("coordinator")

PLANNER_PROMPT = """You are the router of an assistant for the business "Acme Analytics". Choose how to answer.

Capabilities (the ONLY ones available):
- "rag": answers from internal PDF documents: refund/cancellation policy, discount & sales policy, customer support handbook (hours, response times, escalation).
- "database": answers from a SQL database with tables customers(id,name,country,created_at) and orders(id,customer_id,amount,status,order_date). Use for counts, sums, averages, rankings, lists of customers/orders.
- "calculator": arithmetic on numbers (percentages, ratios, averages, differences).

Return ONLY a JSON object:
{{"route": "rag|database|calculation|hybrid|out_of_scope|clarify",
  "reason": "<short>",
  "steps": [ {{"id":"s1","kind":"rag|database|calculator","question":"<self-contained sub-question>","expression":"<calculator only>","bindings":{{"<var>":"<step_id>.<column_alias>"}}}} ]}}

Rules:
- Max {max_steps} steps; ids are s1, s2, ...
- A calculator step uses variables in its expression; each variable MUST be bound to "<earlier_step_id>.<column name>" of a database step (first result row). Database sub-questions must request clearly named columns (aliases).
- If one database step can already compute the answer (e.g. a percentage via SQL), prefer that over a calculator step; use a calculator when the arithmetic combines separately retrieved values.
- Use "out_of_scope" (no steps) for questions unrelated to Acme's documents/data (general knowledge, chit-chat, requests to modify data).
- Use "clarify" (no steps) when the question is too vague to act on.
- Pure arithmetic with literal numbers: route "calculation" with one calculator step and no bindings.
- Never write SQL. Never invent step kinds."""


class Coordinator:
    def __init__(self, settings: Settings, llm: LLMProvider, rag: RAGAgent, db: DatabaseAgent, calc: CalculatorTool | None = None):
        self._s, self._llm, self._rag, self._db = settings, llm, rag, db
        self._calc = calc or CalculatorTool()

    # ------------------------------------------------------------ planning
    def plan(self, question: str) -> Plan:
        try:
            raw = self._llm.complete(
                [ChatMessage(role="system", content=PLANNER_PROMPT.format(max_steps=self._s.max_plan_steps)),
                 ChatMessage(role="user", content=question)],
                json_mode=True,
            )
            plan = self._validate_plan(Plan(**{**parse_json_object(raw), "source": "llm"}))
            return plan
        except (AppError, ValidationError, TypeError, ValueError) as exc:
            log.warning("planner unusable (%s); using rule-based fallback", getattr(exc, "detail", str(exc))[:200])
            return self._fallback_plan(question)

    def _validate_plan(self, plan: Plan) -> Plan:
        if plan.route in (Route.OUT_OF_SCOPE, Route.CLARIFY):
            return plan.model_copy(update={"steps": []})
        if not plan.steps:
            raise PlanError("plan has no steps")
        if len(plan.steps) > self._s.max_plan_steps or len(plan.steps) > self._s.max_tool_calls:
            raise PlanError("plan exceeds step budget")
        seen: dict[str, PlanStep] = {}
        for i, step in enumerate(plan.steps, 1):
            if step.id != f"s{i}" or step.id in seen:
                raise PlanError("step ids must be s1..sN in order")
            if step.kind in ("rag", "database") and not (step.question or "").strip():
                raise PlanError(f"{step.id}: missing question")
            if step.kind == "calculator":
                if not (step.expression or "").strip():
                    raise PlanError(f"{step.id}: missing expression")
                for var, ref in step.bindings.items():
                    m = re.fullmatch(r"(s[1-9])\.([A-Za-z_][A-Za-z0-9_]*)", ref)
                    if not m or m.group(1) not in seen or seen[m.group(1)].kind != "database":
                        raise PlanError(f"{step.id}: invalid binding {var}={ref!r}")
            seen[step.id] = step
        return plan

    def _fallback_plan(self, question: str) -> Plan:
        """Deterministic backstop used only when the LLM planner fails: keyword scoring over two domains."""
        q = question.lower()
        doc_terms = ("policy", "refund", "cancel", "discount", "coupon", "support", "escalat", "sla", "response time",
                     "handbook", "guarantee", "hours", "priority", "non-profit", "startup", "referral")
        data_terms = ("how many", "total", "sum", "average", "avg", "count", "customers", "orders", "revenue",
                      "amount", "top ", "highest", "lowest", "per country", "by country", "status")
        d, t = sum(k in q for k in doc_terms), sum(k in q for k in data_terms)
        if d == 0 and t == 0:
            return Plan(route=Route.OUT_OF_SCOPE, reason="no document or data intent detected", source="fallback")
        if t > d:
            return Plan(route=Route.DATABASE, source="fallback", reason="fallback: data intent",
                        steps=[PlanStep(id="s1", kind="database", question=question)])
        return Plan(route=Route.RAG, source="fallback", reason="fallback: document intent",
                    steps=[PlanStep(id="s1", kind="rag", question=question)])

    # ------------------------------------------------------------ execution
    def ask(self, question: str) -> AskResponse:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        deadline = started + self._s.request_timeout_seconds
        traces: list[StepTrace] = []
        resp = AskResponse(request_id=request_id, status=Status.ERROR, route=Route.OUT_OF_SCOPE, answer="")
        try:
            plan = self.plan(question)
            resp.route, resp.plan_source = plan.route, plan.source
            if plan.route == Route.OUT_OF_SCOPE:
                resp.status = Status.OUT_OF_SCOPE
                resp.answer = ("This assistant only answers questions about Acme Analytics' policy documents "
                               "and its customer/order data, so I can't help with that request.")
            elif plan.route == Route.CLARIFY:
                resp.status = Status.NEEDS_CLARIFICATION
                resp.answer = "Could you clarify your question? Please say whether you mean a company policy or customer/order data, and be specific."
            else:
                self._execute(plan, resp, traces, deadline)
        except RequestTimeoutError:
            resp.status, resp.answer = Status.ERROR, "The request exceeded its time budget and was stopped."
        except AppError as exc:
            log.error("request %s failed: %s", request_id, exc.detail)
            resp.status, resp.answer = Status.ERROR, exc.public_message
        except Exception:
            log.exception("request %s crashed", request_id)
            resp.status, resp.answer = Status.ERROR, "An internal error occurred."
        resp.steps = traces
        resp.duration_ms = round((time.monotonic() - started) * 1000, 1)
        log_event(
            log, "request", request_id=request_id, route=resp.route.value, plan_source=resp.plan_source,
            status=resp.status.value, duration_ms=resp.duration_ms, **question_fingerprint(question),
            steps=[{"id": t.step_id, "kind": t.kind, "status": t.status.value, "ms": t.duration_ms} for t in traces],
        )
        return resp

    def _execute(self, plan: Plan, resp: AskResponse, traces: list[StepTrace], deadline: float) -> None:
        results: dict[str, object] = {}
        for step in plan.steps:
            if time.monotonic() > deadline:
                raise RequestTimeoutError("deadline exceeded between steps")
            t0 = time.monotonic()
            trace = StepTrace(step_id=step.id, kind=step.kind, status=Status.OK, duration_ms=0)
            try:
                if step.kind == "rag":
                    trace.input = {"question": step.question}
                    r: RAGResult = self._rag.run(step.question)
                    results[step.id] = r
                    trace.status, trace.summary = r.status, f"{len(r.sources)} sources, top_score={r.top_score}"
                    trace.error = r.error
                elif step.kind == "database":
                    trace.input = {"question": step.question}
                    d: DatabaseResult = self._db.run(step.question)
                    results[step.id] = d
                    trace.status, trace.summary, trace.error = d.status, f"{d.row_count} rows", d.error
                else:
                    variables, err = self._resolve_bindings(step, results)
                    trace.input = {"expression": step.expression, "variables": variables, "bindings": step.bindings}
                    if err:
                        c = CalculatorResult(status=Status.ERROR, expression=step.expression or "", error=err)
                    else:
                        c = self._calc.run(step.expression or "", variables)
                    results[step.id] = c
                    trace.status, trace.summary, trace.error = c.status, f"result={c.result}", c.error
            finally:
                trace.duration_ms = round((time.monotonic() - t0) * 1000, 1)
                traces.append(trace)
            if trace.status != Status.OK:
                break  # never continue a chain on a failed/empty step: no invented numbers
        self._compose(plan, resp, results)

    @staticmethod
    def _resolve_bindings(step: PlanStep, results: dict) -> tuple[dict[str, float], str | None]:
        variables: dict[str, float] = {}
        for var, ref in step.bindings.items():
            sid, col = ref.split(".", 1)
            src = results.get(sid)
            if not isinstance(src, DatabaseResult) or src.status != Status.OK or not src.rows:
                return variables, f"step {sid} produced no data for {var}"
            cols = [c.lower() for c in src.columns]
            if col.lower() not in cols:
                return variables, f"column {col!r} not found in {sid} result (columns: {src.columns})"
            value = src.rows[0][cols.index(col.lower())]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return variables, f"value for {var} from {sid}.{col} is not numeric ({value!r})"
            variables[var] = float(value)
        return variables, None

    # ------------------------------------------------------------ composition
    def _compose(self, plan: Plan, resp: AskResponse, results: dict[str, object]) -> None:
        """Final text is assembled deterministically from real step outputs (no LLM re-narration),
        so nothing can be presented as executed that was not."""
        parts: list[str] = []
        status = Status.OK
        for step in plan.steps:
            r = results.get(step.id)
            if r is None:
                break
            if isinstance(r, RAGResult):
                resp.sources.extend(r.sources)
                parts.append(r.answer if r.status == Status.OK else (r.answer or f"Document lookup failed: {r.error}"))
                status = r.status if r.status != Status.OK else status
            elif isinstance(r, DatabaseResult):
                resp.sql, resp.result = r.sql, r
                if r.status == Status.OK:
                    parts.append(self._format_rows(r))
                else:
                    parts.append(f"Database lookup failed: {r.error}")
                    status = Status.ERROR if r.status == Status.ERROR else r.status
            elif isinstance(r, CalculatorResult):
                resp.calculation = r
                if r.status == Status.OK:
                    vars_txt = ", ".join(f"{k}={v:,.2f}" for k, v in r.inputs.items())
                    parts.append(f"Calculation: {r.expression}" + (f" with {vars_txt}" if vars_txt else "")
                                 + f" = {r.result:,.4f}".rstrip("0").rstrip("."))
                else:
                    parts.append(f"Calculation failed: {r.error}")
                    status = Status.ERROR
        resp.status = status
        resp.answer = "\n\n".join(p for p in parts if p).strip() or "No answer could be produced."

    @staticmethod
    def _format_rows(r: DatabaseResult) -> str:
        if not r.rows:
            return "The query returned no rows."
        if len(r.rows) == 1 and len(r.columns) <= 4:
            return "Result: " + ", ".join(f"{c} = {v}" for c, v in zip(r.columns, r.rows[0]))
        head = " | ".join(r.columns)
        body = "\n".join(" | ".join(str(v) for v in row) for row in r.rows[:10])
        more = f"\n... ({r.row_count} rows total)" if r.row_count > 10 else ""
        return f"{head}\n{body}{more}"
