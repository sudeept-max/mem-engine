from pydantic import BaseModel, Field
from typing import List, Optional, Literal

MemoryType = Literal["fact", "preference", "project", "event", "goal"]
MemoryStatus = Literal["ACTIVE", "REPLACED", "DELETED"]

class MemoryAddRequest(BaseModel):
    message: str = Field(..., description="Raw conversation message from user", example="I use Arch Linux.")

class MemoryRecallRequest(BaseModel):
    query: str = Field(..., description="Query to search memories for", example="What OS does the user use?")
    top_k: int = Field(default=5, ge=1, le=20)

class MemoryResponseItem(BaseModel):
    id: int
    text: str
    type: MemoryType
    score: float
    importance_score: float
    timestamp: str

class RecallResponse(BaseModel):
    memories: List[MemoryResponseItem]