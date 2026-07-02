from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
