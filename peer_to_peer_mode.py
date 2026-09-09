from collections.abc import Sequence
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


class TeamState(BaseModel):
    project_goal: str
    todo_tasks: Sequence[str]
    done_tasks: Sequence[str]
    status_updates: Sequence[str]
    is_finished: bool


AGENT_SKILLS = {
    "Product Agent": "Responsible for clarifying product requirements, designing MVP features, and producing product documentation, ensuring the product direction aligns with the project goals",
    "Development Agent": "Responsible for implementing the MVP based on the product documentation, solving technical issues, ensuring the features are functional, and delivering a testable product",
    "Operations Agent": "Responsible for designing a promotion strategy based on the MVP, writing promotional copy, carrying out initial promotion efforts, and attracting seed users",
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are the {agent_name} of a startup team. Your core skills are: {agent_skill}.
    The team has no supervisor, and all members are equal. You must independently decide whether to take action based on the overall project state. Follow these decision rules:
    1. First, check the "Status Updates": if there is a new status change that requires your skills to continue the work, you must proactively take action;
    2. Next, check the "Todo Tasks": if there is a pending task within your area of expertise, proactively claim and execute it;
    3. Finally, check the "Project Goal": if there are no pending tasks but the goal has not yet been completed, proactively propose a new task and execute it.

    ⚠️ Mandatory output format (must be followed strictly; all three sections are required, separated by ===, and must not be merged or omitted):
    Decision: Execute/Do Not Execute
    ===
    Reason: Provide the specific basis for your decision based on details from the global state; do not be vague
    ===
    Execution Content: If executing, describe exactly what you will do; if not executing, write exactly "None" and nothing else

    ⚠️ Format example (must follow this exact structure):
    Decision: Execute
    ===
    Reason: The status updates show that the product requirements have been completed, and the todo list contains a task to develop the MVP code, which falls within my development expertise
    ===
    Execution Content: Implement the core Python code for the AI agent tool MVP based on the product requirements document and complete local functional testing
    """,
        ),
        (
            "human",
            (
                "Global project state:\n"
                "Project goal: {project_goal}\n"
                "Todo tasks: {todo_tasks}\n"
                "Completed tasks: {done_tasks}\n"
                "Latest status updates: {status_updates}\n"
                "Is the project completed: {is_finished}"
            ),
        ),
    ]
)


def agent_node(agent_name: str, agent_skill: str):
    def node(state: TeamState) -> TeamState:
        chain = prompt | chat_llm | StrOutputParser()
        response = chain.invoke({"agent_name": agent_name, "agent_skill": agent_skill, **state.model_dump()})
        print("\n" + "=" * 20 + f"{agent_name} Original Response" + "=" * 20)
        print(response)
        parts = [part.strip() for part in response.split("===")]
        if len(parts) < 3:
            parts += [
                "Decision: Do Not Execute",
                "Reason: The model returned an invalid format; fallback decision applied",
                "Execution Content: None",
            ][len(parts) :]
        if len(parts) > 3:
            parts = parts[:3]
        try:
            decision_part, reason_part, action_part = parts
            decision = (
                decision_part.replace("Decision:", "").strip() if "Decision:" in decision_part else "Do Not Execute"
            )
            reason = (
                reason_part.replace("Reason:", "").strip()
                if "Reason:" in reason_part
                else "Invalid format; fallback to Do Not Execute"
            )
            action = (
                action_part.replace("Execution Content:", "").strip() if "Execution Content:" in action_part else "None"
            )
            if decision not in ["Execute", "Do Not Execute"]:
                decision = "Do Not Execute"
                reason = f"Invalid decision value ({decision}); fallback to Do Not Execute"
        except Exception as exc:
            decision = "Do Not Execute"
            reason = f"Format parsing failed: {str(exc)}; fallback to Do Not Execute"
            action = "None"
        print("\n" + "=" * 20 + f"{agent_name} Normalize Decision" + "=" * 20)
        print(f"Decison: {decision}")
        print(f"Reason: {reason}")
        print(f"Execution Content: {action}")

        if decision == "Execute" and action != "None":
            new_state = state.model_copy()
            new_state.done_tasks = list(new_state.done_tasks) + [action]
            new_state.todo_tasks = [
                task for task in new_state.todo_tasks if not any(key in task for key in action.split(":")[0].split(","))
            ]
            new_state.status_updates = list(new_state.status_updates) + [f"{agent_name}: {action}"]
            return new_state
        return state

    return node


product_agent = agent_node("Product Agent", AGENT_SKILLS["Product Agent"])
development_agent = agent_node("Development Agent", AGENT_SKILLS["Development Agent"])
operations_agent = agent_node("Operations Agent", AGENT_SKILLS["Operations Agent"])

builder = StateGraph(TeamState)
builder.add_node("product_agent", product_agent)
builder.add_node("development_agent", development_agent)
builder.add_node("operations_agent", operations_agent)


def should_continue(state: TeamState) -> Literal["product_agent", "development_agent", "operations_agent", "end"]:
    if state.is_finished or len(state.status_updates) > 3:
        return "end"
    return "product_agent"


builder.add_edge(START, "product_agent")
builder.add_edge("product_agent", "development_agent")
builder.add_edge("development_agent", "operations_agent")
builder.add_conditional_edges("operations_agent", should_continue, {"product_agent": "product_agent", "end": END})
graph = builder.compile()

if __name__ == "__main__":
    initial_state = TeamState(
        project_goal="Develop an MVP for an AI agent tool and carry out initial promotion to acquire seed users.",
        todo_tasks=[
            "Define the core requirements for the AI agent tool MVP",
            "Implement the core functionality of the MVP",
            "Write promotional copy for the MVP and publish an initial post on Xiaohongshu",
        ],
        done_tasks=[],
        status_updates=["Project launched: Begin developing and promoting the AI agent tool MVP"],
        is_finished=False,
    )
    print("\n" + "=" * 20 + "Startup Team Project Launch" + "=" * 20)
    print(f"Project goal: {initial_state.project_goal}")
    print(f"Initial todo: {initial_state.todo_tasks}")

    for step in graph.stream(initial_state):
        for node, state in step.items():
            print("\n" + "=" * 20 + f"{node} executed - State" + "=" * 20)
            print(f"✅ Done tasks: {state['done_tasks']}")
            print(f"📋 Todo tasks: {state['todo_tasks']}")
            print(f"📌 Status update: {state['status_updates'][-1]}")

    final_state = graph.invoke(initial_state)
    print("\n" + "=" * 20 + "Final" + "=" * 20)
    print(f"Project goal: {final_state['project_goal']}")
    print(f"✅ Done tasks: {final_state['done_tasks']}")
    print(f"📌 Status updates: {final_state['status_updates'][-1]}")
