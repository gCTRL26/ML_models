from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from airline_multiagent.llm_factory import build_chat_llm
from airline_multiagent.schemas import CriticVerdict, OrchestratorPlan, WorkerResult


class Orchestrator:
    def __init__(self, specialist_agents: dict[str, object]):
        self.agents = specialist_agents
        self.planning_llm = build_chat_llm(temperature=0).with_structured_output(
            OrchestratorPlan
        )
        self.synthesis_llm = build_chat_llm(temperature=0)
        self.critic_llm = build_chat_llm(temperature=0).with_structured_output(
            CriticVerdict
        )

    def create_plan(
        self, user_query: str, user_profile: dict | None = None
    ) -> OrchestratorPlan:
        profile_hint = (
            f"Passenger profile: {user_profile}"
            if user_profile
            else "No profile available."
        )
        return self.planning_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the coordinator of an airline assistant team.\n"
                        "You have 4 specialists:\n"
                        "- flight_agent: search_flights, get_flight_details\n"
                        "- policy_agent: lookup_policy\n"
                        "- booking_agent: get_booking, update_booking, cancel_booking, book_flight\n"
                        "- profile_agent: get_passenger_profile, update_passenger_profile\n\n"
                        "Break the user's request into self-contained subtasks.\n"
                        "Every subtask must include explicit cities, dates, booking IDs, fare class, or profile fields when relevant.\n"
                        "Never assume one specialist can see another specialist's context.\n"
                        "Set risk_level=high for any request that may mutate bookings, cancel, or charge money.\n"
                        "Set needs_critic=true for multi-domain, risky, or ambiguous requests.\n"
                        "Current year: 2026.\n"
                        "Don't describe any booking or other info that you've got from worker, unless his result has status=success and includes tool usage"
                    )
                ),
                HumanMessage(content=f"User query: {user_query}\n\n{profile_hint}"),
            ]
        )

    def execute_plan(self, plan: OrchestratorPlan) -> list[WorkerResult]:
        results: list[WorkerResult] = []
        context_blocks: list[str] = []
        for task in sorted(plan.subtasks, key=lambda t: t.priority):
            agent = self.agents.get(task.worker.value)
            if agent is None:
                results.append(
                    WorkerResult(
                        worker=task.worker.value,
                        status="error",
                        result=f"Unknown agent: {task.worker.value}",
                    )
                )
                continue
            enriched_description = task.description
            if context_blocks:
                enriched_description += (
                    "\n\nContext from previous specialists:\n"
                    + "\n".join(context_blocks[-3:])
                )
            worker_result = agent.process(enriched_description)
            results.append(worker_result)
            context_blocks.append(
                f"[{worker_result.worker}] status={worker_result.status}; tools={', '.join(worker_result.tools_used) or 'none'}; result={worker_result.result}"
            )
        return results

    def synthesize(
        self, user_query: str, plan: OrchestratorPlan, results: list[WorkerResult]
    ) -> str:
        results_text = "\n\n".join(
            f'[{r.worker}] status={r.status}\nTools: {", ".join(r.tools_used) or "none"}\nResult:\n{r.result}'
            for r in results
        )
        response = self.synthesis_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the customer-facing response composer.\n"
                        "Use ONLY facts from the worker results.\n"
                        "If something is uncertain, say what is missing.\n"
                        "For booking/rebooking/cancellation actions, clearly separate information from action status.\n"
                        "Keep the answer concise but concrete."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Customer request: {user_query}\n\n"
                        f"Plan reasoning: {plan.reasoning}\n"
                        f"Risk level: {plan.risk_level.value}\n\n"
                        f"Worker results:\n{results_text}"
                    )
                ),
            ]
        )
        return str(response.content)

    def critique(
        self, user_query: str, proposed_answer: str, results: list[WorkerResult]
    ) -> CriticVerdict:
        context = "\n\n".join(f"[{r.worker}] {r.result}" for r in results)
        return self.critic_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a QA critic for an airline assistant.\n"
                        "Check completeness, correctness, safety, and specificity.\n"
                        "Approve only when the answer is well grounded in the worker results.\n"
                        "Do not invent issues unsupported by the evidence."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Customer request: {user_query}\n\nProposed answer:\n{proposed_answer}\n\n"
                        f"Worker evidence:\n{context}"
                    )
                ),
            ]
        )

    def revise_with_critic(
        self,
        user_query: str,
        proposed_answer: str,
        verdict: CriticVerdict,
        results: list[WorkerResult],
    ) -> str:
        evidence = "\n\n".join(f"[{r.worker}] {r.result}" for r in results)
        response = self.synthesis_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Revise the answer using the critic feedback. Use only grounded facts from the evidence. "
                        "Keep the revision customer-facing and concise."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Customer request: {user_query}\n\nOriginal answer:\n{proposed_answer}\n\n"
                        f"Critic issues: {verdict.issues}\nSuggestions: {verdict.suggestions}\n\nEvidence:\n{evidence}"
                    )
                ),
            ]
        )
        return str(response.content)
