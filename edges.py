from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph
from pydantic import BaseModel


class TaskState(BaseModel):
    user_query: str
    intent: str | None = None
    llm_answer: str | None = None
    tool_result: str | None = None
    final_answer: str | None = None
    progress: int = 0


class TaskStateUpdate(TypedDict):
    intent: NotRequired[str]
    llm_answer: NotRequired[str]
    progress: NotRequired[int]


def parse_intent(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "🔹 parse_intent" + "=" * 20)
    intent = "summarize" if "summarize" in state.user_query else "rewrite"
    return {"intent": intent, "progress": 30}


def summarize_node(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "🔹 summarize_node" + "=" * 20)
    return {"llm_answer": f"Summarize result: {state.user_query}", "progress": 60}


def rewrite_node(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "🔹 rewrite_node" + "=" * 20)
    return {"llm_answer": f"Rewrite result: {state.user_query}", "progress": 60}


def final_node(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "🔹 final_node" + "=" * 20)
    return {"progress": 100}


def route_by_intent(state: TaskState) -> str:
    if state.intent == "summarize":
        return "summarize_node"
    return "rewrite_node"


def loop_node(state: TaskState) -> TaskStateUpdate:
    progress = state.progress
    print("\n" + "=" * 20 + f"🔄 loop_node, progress = {progress}" + "=" * 20)
    return {"progress": progress + 30}


def loop_router(state: TaskState) -> str:
    if state.progress >= 100:
        return "final_node"
    return "loop_node"


builder = StateGraph(TaskState)
builder.add_node("parse_intent", parse_intent)
builder.add_node("summarize_node", summarize_node)
builder.add_node("rewrite_node", rewrite_node)
builder.add_node("loop_node", loop_node)
builder.add_node("final_node", final_node)

builder.set_entry_point("parse_intent")

builder.add_conditional_edges(
    "parse_intent", route_by_intent, {"summarize_node": "summarize_node", "rewrite_node": "rewrite_node"}
)

builder.add_edge("summarize_node", "loop_node")
builder.add_edge("rewrite_node", "loop_node")

builder.add_conditional_edges("loop_node", loop_router, {"loop_node": "loop_node", "final_node": "final_node"})

graph = builder.compile()

print("\n" + "=" * 20 + "Summarize" + "=" * 20)
print(graph.invoke(TaskState(user_query="Please help me summarize this sentence.")))

print("\n" + "=" * 20 + "Rewrite" + "=" * 20)
print(graph.invoke(TaskState(user_query="Please help me rewrite this sentence.")))
