from __future__ import annotations

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    test_type: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool
