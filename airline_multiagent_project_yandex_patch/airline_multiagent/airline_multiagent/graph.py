from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from airline_multiagent.config import SETTINGS
from airline_multiagent.guards import (
    input_guard,
    route_after_input_guard,
    tool_output_guard,
)
from airline_multiagent.memory_store import load_profile
from airline_multiagent.orchestrator import Orchestrator
from airline_multiagent.schemas import AgentState, WorkerName, WorkerResult
from airline_multiagent.tools import (
    ALL_TOOLS,
    BOOKING_TOOLS,
    FLIGHT_TOOLS,
    POLICY_TOOLS,
    PROFILE_TOOLS,
)
from airline_multiagent.workers import SpecializedAgent


memory = MemorySaver()


FLIGHT_AGENT = SpecializedAgent(
    name=WorkerName.FLIGHT.value,
    tools_list=FLIGHT_TOOLS,
    system_prompt=(
        "You are a flight search specialist.\n"
        "Use tools to search flights and compare concrete options.\n"
        "Mention prices, times, fare class, and best recommendation.\n"
        "Do not discuss policy unless it is returned by a tool.\n"
        "NEVER give any information, if you not sure about it on 100%"
    ),
)

POLICY_AGENT = SpecializedAgent(
    name=WorkerName.POLICY.value,
    tools_list=POLICY_TOOLS,
    system_prompt=(
        "You are an airline policy specialist.\n"
        "Use lookup_policy for baggage, refunds, cancellations, pets, and rebooking.\n"
        "If the user asks conversationally, use HyDE-style retrieval thinking: imagine the policy wording, then query the tool."
    ),
)

BOOKING_AGENT = SpecializedAgent(
    name=WorkerName.BOOKING.value,
    tools_list=BOOKING_TOOLS,
    system_prompt=(
        "You are a booking operations specialist.\n"
        "You can inspect bookings and, when clearly instructed, call booking mutation tools.\n"
        "Before any change or cancellation, ensure the booking_id and concrete target flight/date are known.\n"
        "NEVER invent booking data.\n"
        "You MUST use tools."
    ),
)

PROFILE_AGENT = SpecializedAgent(
    name=WorkerName.PROFILE.value,
    tools_list=PROFILE_TOOLS,
    system_prompt=(
        "You are a passenger profile specialist.\n"
        "Read and update persistent profile fields like name, email, passport, and preferences.\n"
        "Treat 'home_airport' as the passenger's usual departure city, not an airport code.\n"
        "Prefer saved profile data when the user asks to book or personalize a trip."
    ),
)

SPECIALISTS = {
    WorkerName.FLIGHT.value: FLIGHT_AGENT,
    WorkerName.POLICY.value: POLICY_AGENT,
    WorkerName.BOOKING.value: BOOKING_AGENT,
    WorkerName.PROFILE.value: PROFILE_AGENT,
}

orchestrator = Orchestrator(SPECIALISTS)


def hydrate_profile(state: AgentState) -> dict:
    profile = load_profile()
    return {"user_profile": profile}


def orchestrator_node(state: AgentState) -> dict:
    messages = state["messages"]

    recent_messages = messages[-8:]
    conversation_context_parts: list[str] = []
    latest_user_message = ""

    for msg in recent_messages:
        role = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue

        if role in {"human", "user"}:
            conversation_context_parts.append(f"User: {content}")
            latest_user_message = content
        elif role == "ai":
            conversation_context_parts.append(f"Assistant: {content}")

    conversation_context = "\n".join(conversation_context_parts).strip()
    if not latest_user_message:
        latest_user_message = str(getattr(messages[-1], "content", ""))

    profile = state.get("user_profile") or load_profile()

    planner_input = (
        "Use the recent conversation context to resolve references like "
        "'this', 'that one', follow-up answers, clarified dates, destinations, and profile values.\n\n"
        f"Recent conversation:\n{conversation_context}\n\n"
        f"Latest user request:\n{latest_user_message}"
    )

    plan = orchestrator.create_plan(planner_input, profile)
    results = orchestrator.execute_plan(plan)
    answer = orchestrator.synthesize(planner_input, plan, results)

    return {
        "current_plan": plan.model_dump(mode="json"),
        "worker_results": [r.model_dump(mode="json") for r in results],
        "risk_level": plan.risk_level.value,
        "needs_critic": plan.needs_critic,
        "final_answer": answer,
    }


def critic_node(state: AgentState) -> dict:
    messages = state["messages"]

    recent_messages = messages[-8:]
    conversation_context_parts: list[str] = []

    for msg in recent_messages:
        role = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue

        if role in {"human", "user"}:
            conversation_context_parts.append(f"User: {content}")
        elif role == "ai":
            conversation_context_parts.append(f"Assistant: {content}")

    conversation_context = "\n".join(conversation_context_parts).strip()
    if not conversation_context:
        conversation_context = str(getattr(messages[-1], "content", ""))

    worker_results = state.get("worker_results", [])

    results = [WorkerResult(**r) for r in worker_results]
    verdict = orchestrator.critique(
        conversation_context, state.get("final_answer", ""), results
    )

    final_answer = state.get("final_answer", "")
    if not verdict.approved:
        final_answer = orchestrator.revise_with_critic(
            conversation_context, final_answer, verdict, results
        )

    return {
        "critic": verdict.model_dump(mode="json"),
        "final_answer": final_answer,
    }


def route_after_orchestrator(state: AgentState) -> Literal["critic", "respond"]:
    if state.get("needs_critic"):
        return "critic"
    return "respond"


def respond_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=state.get("final_answer", "I could not prepare an answer.")
            )
        ]
    }


def build_multiagent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("hydrate_profile", hydrate_profile)
    builder.add_node("input_guard", input_guard)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("critic", critic_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "hydrate_profile")
    builder.add_edge("hydrate_profile", "input_guard")
    builder.add_conditional_edges("input_guard", route_after_input_guard)
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"critic": "critic", "respond": "respond"},
    )
    builder.add_edge("critic", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=memory)


app = build_multiagent_graph()
