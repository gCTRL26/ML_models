from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkerName(str, Enum):
    FLIGHT = "flight_agent"
    POLICY = "policy_agent"
    BOOKING = "booking_agent"
    PROFILE = "profile_agent"


class WorkerTask(BaseModel):
    worker: WorkerName
    description: str = Field(
        description="Self-contained task description for one specialist"
    )
    priority: int = Field(default=2, ge=1, le=3)
    requires_confirmation: bool = False
    depends_on: list[str] = Field(default_factory=list)


class OrchestratorPlan(BaseModel):
    reasoning: str
    intent: Literal["search", "book", "change", "cancel", "policy", "profile", "mixed"]
    risk_level: RiskLevel = RiskLevel.LOW
    needs_critic: bool = False
    subtasks: list[WorkerTask]


class WorkerResult(BaseModel):
    worker: str
    status: Literal["success", "partial", "error"]
    result: str
    tools_used: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    structured_data: dict[str, Any] = Field(default_factory=dict)


class CriticVerdict(BaseModel):
    approved: bool
    score: float = Field(ge=0.0, le=10.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str


class BookingRequest(BaseModel):
    flight_id: str
    passenger_name: str
    email: str
    passport: str
    seat_preference: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v.lower().strip()

    @field_validator("passport")
    @classmethod
    def validate_passport(cls, v: str) -> str:
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) < 8:
            raise ValueError("Passport looks invalid")
        return v.strip()


class UpdateBookingRequest(BaseModel):
    booking_id: str
    new_flight_id: str
    new_date: str


class PassengerProfileUpdate(BaseModel):
    key: Literal[
        "name",
        "email",
        "passport",
        "seat_preference",
        "meal_preference",
        "home_airport",
        "loyalty_tier",
    ]
    value: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_profile: dict[str, Any]
    current_plan: dict[str, Any]
    worker_results: list[dict[str, Any]]
    final_answer: str
    critic: dict[str, Any]
    risk_level: str
    needs_critic: bool
    pending_action: dict[str, Any] | None
    resolved_user_query: str
    last_tool_output_clean: bool
    audit_log: list[dict[str, Any]]
