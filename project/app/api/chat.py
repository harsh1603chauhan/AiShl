from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.agents.conversation_agent import ConversationAgent
from app.core.logger import get_logger
from app.models.request import ChatRequest


router = APIRouter(tags=["chat"])
logger = get_logger(__name__)
agent = ConversationAgent()


@router.post("/chat")
def chat(payload: ChatRequest) -> JSONResponse:
    start = time.perf_counter()
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        response = agent.respond(payload.messages)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("chat_response elapsed_ms=%s messages=%s", elapsed_ms, len(payload.messages))
        return JSONResponse(content=response.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process chat request") from exc
