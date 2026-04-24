from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from airline_multiagent.config import SETTINGS
from airline_multiagent.graph import app
from uuid import uuid4

THREAD = {
    "configurable": {"thread_id": f"{SETTINGS.default_thread_id}-{uuid4().hex[:8]}"}
}


def ask(text: str):
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=THREAD)
    print("\nAssistant:\n")
    print(result["messages"][-1].content)
    return result


def approve_pending(action: str = "approved"):
    result = app.invoke(Command(resume=action), config=THREAD)
    print("\nAssistant:\n")
    print(result["messages"][-1].content)
    return result


def interactive_chat():
    global THREAD

    print("Airline Multi-Agent Assistant")
    print(f"Session thread_id: {THREAD['configurable']['thread_id']}")
    print("Type your request. Commands: /approve, /reject <reason>, /new, /exit")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        if user_input.lower() == "/new":
            THREAD = {
                "configurable": {
                    "thread_id": f"{SETTINGS.default_thread_id}-{uuid4().hex[:8]}"
                }
            }
            print(f"Started new session: {THREAD['configurable']['thread_id']}")
            continue

        if user_input.lower().startswith("/approve"):
            approve_pending("approved")
            continue

        if user_input.lower().startswith("/reject"):
            reason = user_input[len("/reject") :].strip() or "rejected_by_user"
            approve_pending(reason)
            continue

        ask(user_input)
        state = app.get_state(THREAD)
        if state.next:
            interrupt_value = state.tasks[0].interrupts[0].value
            print("\nPending approval:\n")
            print(interrupt_value)
            print("\nUse /approve or /reject <reason>")


if __name__ == "__main__":
    interactive_chat()
