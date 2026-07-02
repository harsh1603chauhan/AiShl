from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.logger import configure_logging


configure_logging()
app = FastAPI(title=settings.app_name)
app.include_router(health_router)
app.include_router(chat_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
	return JSONResponse(status_code=400, content={"detail": "Invalid request payload", "errors": exc.errors()})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
	return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
