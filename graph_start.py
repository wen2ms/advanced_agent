from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph
from pydantic import BaseModel


class TaskState(BaseModel):
    user_query: str
    tool_result: str | None = None
    final_answer: str | None = None
    progress: int = 0


class TaskStateUpdate(TypedDict):
    tool_result: NotRequired[str | None]
    final_answer: NotRequired[str | None]
    progress: NotRequired[int]


def parse_query(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "node1 parse_query input state" + "=" * 20)
    print(state)

    query = state.user_query
    update: TaskStateUpdate = {"tool_result": f"Resolved: {query}", "progress": 30}
    print("\n" + "=" * 20 + "node1 updated state" + "=" * 20)
    print(update)

    return update


def call_tool(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "node2 call_tool input state" + "=" * 20)
    print(state)

    result = f'Tool search result: Relevant knowledge about "{state.user_query}"'
    update: TaskStateUpdate = {"tool_result": result, "progress": 70}
    print("\n" + "=" * 20 + "node2 updated state" + "=" * 20)
    print(update)

    return update


def generate_answer(state: TaskState) -> TaskStateUpdate:
    print("\n" + "=" * 20 + "node3 generate_answer input state" + "=" * 20)
    print(state)

    answer = f"Final answer: -> {state.tool_result}"
    update: TaskStateUpdate = {"final_answer": answer, "progress": 100}
    print("\n" + "=" * 20 + "node3 updated state" + "=" * 20)
    print(update)

    return update


builder = StateGraph(TaskState)
builder.add_node("parse_query", parse_query)
builder.add_node("call_tool", call_tool)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("parse_query")
builder.add_edge("parse_query", "call_tool")
builder.add_edge("call_tool", "generate_answer")

graph = builder.compile()
init_state = TaskState(user_query="What is Langgraph?")
final_state = graph.invoke(init_state)

print("\n" + "=" * 20 + "Final state" + "=" * 20)
print(final_state)
