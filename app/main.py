"""FastAPI entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.contracts import AskResponse, Status
from app.api.schemas import AskRequest, ErrorBody
from app.api.state import AppState, build_state
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import get_logger, setup_logging

log = get_logger("api")


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.acme = state or await run_in_threadpool(build_state, settings)
        yield

    app = FastAPI(title="Acme Analytics Assistant", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content=ErrorBody(error="invalid_request", message=msg).model_dump())

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        log.error("app error %s: %s", exc.code, exc.detail)
        return JSONResponse(status_code=exc.http_status, content=ErrorBody(error=exc.code, message=exc.public_message).model_dump())

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content=ErrorBody(error="internal_error", message="An internal error occurred.").model_dump())

    @app.post("/ask", response_model=AskResponse, responses={422: {"model": ErrorBody}, 503: {"model": ErrorBody}, 504: {"model": ErrorBody}})
    async def ask(body: AskRequest, request: Request):
        st: AppState = request.app.state.acme
        if not st.llm.is_configured():
            raise AppError("llm not configured", public_message="The language model is not configured.") from None
        resp: AskResponse = await run_in_threadpool(st.coordinator.ask, body.question)
        # Map execution failures to HTTP codes while keeping the structured body for debugging.
        if resp.status == Status.ERROR:
            code = 503 if any(t.kind == "rag" and t.status == Status.ERROR for t in resp.steps) else 502 if not resp.steps else 422
            return JSONResponse(status_code=code, content=resp.model_dump(mode="json"))
        return resp

    @app.get("/health")
    async def health(request: Request):
        st: AppState = request.app.state.acme
        ready = st.ready()
        checks = {
            "index": st.store is not None,
            "database": st.settings.sqlite_path.exists(),
            "llm_configured": st.llm.is_configured(),
        }
        return JSONResponse(status_code=200 if ready else 503, content={"status": "ok" if ready else "degraded", "checks": checks})

    @app.get("/status")
    async def status(request: Request):
        return request.app.state.acme.status()

    return app


app = create_app()
