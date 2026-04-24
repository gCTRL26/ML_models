from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from airline_multiagent.schemas import AgentState


PII_PATTERNS = [
    (re.compile(r"\b\d{4}\s\d{6}\b"), "[PASSPORT]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[CARD]"),
    (re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"), "[PHONE]"),
]

INJECTION_PATTERNS = re.compile(
    r"\[SYSTEM[:\s]|ignore\s+|disregard\s+|new\s+instructions?|override\s+|you\s+are\s+now\s+|forget\s+|act\s+as\s+if",
    re.IGNORECASE,
)

OFF_TOPIC_PATTERNS = re.compile(
    r"\b(poem|recipe|stocks?|politics?|write code for malware|hack|exploit|joke)\b",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    for pattern, placeholder in PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def append_audit(state: AgentState, event: dict[str, Any]) -> dict[str, Any]:
    return {"audit_log": [*(state.get("audit_log", [])), event]}


def input_guard(state: AgentState) -> dict[str, Any]:
    messages = state["messages"]
    last_msg = messages[-1]
    content = getattr(last_msg, "content", str(last_msg))
    safe_content = mask_pii(content)

    if OFF_TOPIC_PATTERNS.search(content):
        blocked = AIMessage(
            content=(
                "I'm an airline support assistant. I can help with flights, bookings, baggage, policies, "
                "rebooking, cancellations and passenger profile questions."
            )
        )
        return {
            "messages": [blocked],
            "audit_log": [
                {
                    "type": "guard",
                    "name": "input_guard",
                    "decision": "blocked",
                    "message": safe_content,
                }
            ],
        }

    return {
        "audit_log": [
            {
                "type": "guard",
                "name": "input_guard",
                "decision": "passed",
                "message": safe_content,
            }
        ]
    }


def route_after_input_guard(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        return END
    return "orchestrator"


def tool_output_guard(state: AgentState) -> dict[str, Any]:
    messages = state["messages"]
    last_msg = messages[-1]
    if not isinstance(last_msg, ToolMessage):
        return {"last_tool_output_clean": True}

    cleaned = False
    content = last_msg.content
    try:
        payload = json.loads(content)
    except Exception:
        payload = None

    if last_msg.name == "search_flights" and isinstance(payload, list):
        kept = []
        for item in payload:
            fare_rules = item.get("fare_rules", "")
            if INJECTION_PATTERNS.search(fare_rules):
                cleaned = True
                continue
            kept.append(item)
        if cleaned:
            replacement = ToolMessage(
                content=json.dumps(kept, ensure_ascii=False),
                tool_call_id=last_msg.tool_call_id,
                name=last_msg.name,
                id=last_msg.id,
            )
            return {
                "messages": [replacement],
                "last_tool_output_clean": False,
                "audit_log": [
                    {
                        "type": "guard",
                        "name": "tool_output_guard",
                        "decision": "cleaned",
                        "tool": last_msg.name,
                    }
                ],
            }

    return {
        "last_tool_output_clean": True,
        "audit_log": [
            {
                "type": "guard",
                "name": "tool_output_guard",
                "decision": "passed",
                "tool": getattr(last_msg, "name", "unknown"),
            }
        ],
    }
