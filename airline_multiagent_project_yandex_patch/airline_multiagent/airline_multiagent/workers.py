from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from airline_multiagent.guards import tool_output_guard
from airline_multiagent.llm_factory import build_chat_llm
from airline_multiagent.schemas import WorkerResult


llm = build_chat_llm()

TOOL_GROUNDED_AGENTS = {
    "flight_agent",
    "booking_agent",
    "profile_agent",
    "policy_agent",
}

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
FLIGHT_ID_RE = re.compile(r"\b[A-Z]{2,3}\d{2,5}\b", re.IGNORECASE)
PASSENGER_NAME_RE = re.compile(r"\bfor\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b")


def create_react_agent(tools_list: list, system_prompt: str):
    llm_with_tools = llm.bind_tools(tools_list, parallel_tool_calls=False)
    prompt = (
        system_prompt
        + """
        You must ground domain facts in tool outputs.\n
        If the task needs factual airline data, bookings, policy, or profile information, call a tool.\n
        Do not claim that something exists, does not exist, or was found unless a tool result supports it.\n\n
    """
    )

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=prompt)] + messages
        response = llm_with_tools.invoke(messages)
        if response.tool_calls and not response.content:
            tool_info = ", ".join(tc["name"] for tc in response.tool_calls)
            thought = llm.invoke(
                messages
                + [
                    HumanMessage(
                        content=(
                            f"You chose to call: {tool_info}. In one concise sentence, explain why this is the right next step. "
                            "Reply with only the reasoning."
                        )
                    )
                ]
            )
            response.content = thought.content
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools_list))
    workflow.add_node("tool_output_guard", tool_output_guard)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "tool_output_guard")
    workflow.add_edge("tool_output_guard", "agent")
    return workflow.compile()


@dataclass
class SpecializedAgent:
    name: str
    tools_list: list
    system_prompt: str

    def __post_init__(self):
        self.graph = create_react_agent(self.tools_list, self.system_prompt)

    def process(self, task_description: str) -> WorkerResult:
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=task_description)]}
            )
            tools_used: list[str] = []
            for msg in result["messages"]:
                if getattr(msg, "tool_calls", None):
                    tools_used.extend(tc["name"] for tc in msg.tool_calls)
            final_message = result["messages"][-1]
            facts = _extract_facts(str(final_message.content))
            status = "success"
            final_result = str(final_message.content)

            if self.name in TOOL_GROUNDED_AGENTS and not tools_used:
                if self.name == "booking_agent" and _looks_like_booking_mutation(
                    task_description
                ):
                    missing = _missing_booking_fields(task_description)
                    if missing:
                        status = "partial"
                        final_result = (
                            "Booking action needs more required fields before calling book_flight: "
                            + ", ".join(missing)
                            + "."
                        )
                    else:
                        status = "error"
                        final_result = (
                            "Booking action looked executable, but no booking tool was called."
                        )
                else:
                    status = "partial"
                    final_result = (
                        "No supporting tool was called yet. Treat this as a clarification step, not a completed action."
                    )

            return WorkerResult(
                worker=self.name,
                status=status,
                result=final_result,
                tools_used=tools_used,
                facts=facts,
            )
        except (GraphInterrupt, GraphBubbleUp):
            raise
        except Exception as exc:
            return WorkerResult(
                worker=self.name, status="error", result=f"Error: {exc}", tools_used=[]
            )


def _extract_facts(text: str) -> list[str]:
    lines = [line.strip("-* ") for line in text.splitlines() if line.strip()]
    return lines[:5]


def _looks_like_booking_mutation(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in ("book", "reserve", "issue ticket", "buy ticket"))


def _missing_booking_fields(text: str) -> list[str]:
    missing: list[str] = []
    if not FLIGHT_ID_RE.search(text):
        missing.append("flight_id")
    if not EMAIL_RE.search(text):
        missing.append("email")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 8:
        missing.append("passport")
    if not PASSENGER_NAME_RE.search(text):
        missing.append("passenger_name")
    return missing
