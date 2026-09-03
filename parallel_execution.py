import random
import time
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class TextProcessState(BaseModel):
    raw_text: str
    summary_text: str | None = None
    keyword_text: str | None = None
    has_sensitive: bool | None = None
    final_text: str | None = None


class StateUpdate(TypedDict):
    raw_text: NotRequired[str]
    summary_text: NotRequired[str]
    keyword_text: NotRequired[str]
    has_sensitive: NotRequired[bool]
    final_text: NotRequired[str]


def deduplicate_node(state: TextProcessState) -> StateUpdate:
    print("\n" + "=" * 20 + "[deduplicate_node] executing..." + "=" * 20)
    text = state.raw_text
    time.sleep(1)
    return {"raw_text": text}


def summary_node(state: TextProcessState) -> StateUpdate:
    print("\n" + "=" * 20 + "⚡ parallel summary_node executing..." + "=" * 20)
    time.sleep(random.uniform(1, 2))
    summary = "Summary: " + state.raw_text[:10]
    return {"summary_text": summary}


def keyword_node(state: TextProcessState) -> StateUpdate:
    print("\n" + "=" * 20 + "⚡ parallel keyword_node executing..." + "=" * 20)
    time.sleep(random.uniform(1, 2))
    keywords = "# ".join(state.raw_text.split(",")[:3])
    return {"keyword_text": keywords}


def sensitive_check_node(state: TextProcessState) -> StateUpdate:
    print("\n" + "=" * 20 + "[sensitive_check_node] executing..." + "=" * 20)
    print("summary_text = ", state.summary_text)
    print("keyword_text = ", state.keyword_text)
    sensitive = "violence" in state.raw_text
    return {
        "has_sensitive": sensitive,
        "final_text": f"Final | summary={state.summary_text} | keywords={state.keyword_text}",
    }


builder = StateGraph(TextProcessState)
builder.add_node("deduplicate_node", deduplicate_node)
builder.add_node("summary_node", summary_node)
builder.add_node("keyword_node", keyword_node)
builder.add_node("sensitive_check_node", sensitive_check_node)

builder.add_edge(START, "deduplicate_node")
builder.add_edge("deduplicate_node", "summary_node")
builder.add_edge("deduplicate_node", "keyword_node")

builder.add_edge("summary_node", "sensitive_check_node")
builder.add_edge("keyword_node", "sensitive_check_node")

builder.add_edge("sensitive_check_node", END)

graph = builder.compile()

if __name__ == "__main__":
    init_state = TextProcessState(
        raw_text="LangGraph is very powerful, it supports state management, dynamic branching, and parallel execution."
    )
    final_state = graph.invoke(init_state)
    print("\n" + "=" * 20 + "final_state" + "=" * 20)
    print(final_state)
