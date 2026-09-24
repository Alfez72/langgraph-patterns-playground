from langgraph.graph import MessagesState
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator
import os

# =====================================================================
# WAY 1: TypedDict (Most common, lightweight, native dict format)
# =====================================================================
from typing import TypedDict


class state(TypedDict):
    topic: str
    summary: str
    score: int
    # Note: No runtime validation. Type hints are only for your IDE.


# =====================================================================
# WAY 2: Pydantic (Best for runtime type checking and data safety)
# =====================================================================
class state(BaseModel):
    topic: str
    score: int
    summary: str = ""  # Default value is empty string

    @field_validator("score")  # Validates the score field at runtime
    def score_positive(cls, v):  # v is the value, cls is the class
        if v < 0:
            raise ValueError("Score must be positive")
        return v


# =====================================================================
# WAY 3: Python Dataclass (Standard OOP object way, rarely used)
# =====================================================================
@dataclass
class State:
    topic: str = ""
    summary: str = ""
    # default_factory=list gives every new state its own fresh list in memory
    messages: list = field(default_factory=list)


# =====================================================================
# WAY 4: MessagesState (Built-in LangGraph class for chatbots)
# =====================================================================
class state(MessagesState):
    # 'messages' list is already built-in and automatically appends history.
    # Add your own extra tracking fields below:
    user_name: str
    language: str
