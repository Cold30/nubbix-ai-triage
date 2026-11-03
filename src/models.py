
from pydantic import BaseModel, Field
from typing import Literal

class TriageResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0, le=1)
    category: Literal["billing", "technical", "account", "other"]
    actions: list[str]