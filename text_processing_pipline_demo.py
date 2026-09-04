from pathlib import Path
from typing import NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


class TextProcessState(BaseModel):
    raw_text: str
    deduplicated_text: str | None = None
    summary_text: str | None = None
    has_sensitive: bool = False
    rewrite_count: int = 0
    quality_valid: bool = False
    final_output: str | None = None


class StateUpdate(TypedDict):
    summary_text: NotRequired[str]
    has_sensitive: NotRequired[bool]
    final_output: NotRequired[str]
    quality_valid: NotRequired[bool]
    rewrite_count: NotRequired[int]


def deduplicate_node(state: TextProcessState) -> TextProcessState:
    raw_text = state.raw_text
    lines = raw_text.split("\n")
    unique_lines: list[str] = []
    seen = set()
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            unique_lines.append(line)
    deduplicated_text = "\n".join(unique_lines)
    print("✅ Deduplication completed.")
    return state.model_copy(update={"deduplicated_text": deduplicated_text})


def summary_node(state: TextProcessState) -> StateUpdate:
    deduplicated_text = state.deduplicated_text
    if not deduplicated_text:
        return {"summary_text": "No valid text."}
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Please generate a concise summary of the following text in no more than 50 words, retaining the key information: \n{text}",
            )
        ]
    )
    chain = prompt | chat_llm
    summary = chain.invoke(input={"text": deduplicated_text}).text
    print("🤖 Summary completed.")
    return {"summary_text": summary}


def sensitive_check_node(state: TextProcessState) -> StateUpdate:
    summary = state.summary_text
    if summary is None:
        return {}
    sensitive_words = ["illegal", "non-compliant"]
    summary_lower = summary.lower()
    has_sensitive = any(word in summary_lower for word in sensitive_words)
    print(f"🔍 Sensitive check completed | has_sensitive={has_sensitive}")
    return {"has_sensitive": has_sensitive}


def quality_check_node(state: TextProcessState) -> StateUpdate:
    summary = state.summary_text
    if summary is None:
        return {}
    length_valid = 15 <= len(summary.split()) <= 50
    core_keywords = ["langgraph", "workflow"]
    summary_lower = summary.lower()
    info_valid = all(core_keyword in summary_lower for core_keyword in core_keywords)
    quality_valid = length_valid and info_valid
    print(
        f"📏 Quality check completed | length_valid={length_valid} | info_valid={info_valid} | quality_valid={quality_valid}"
    )
    return {"quality_valid": quality_valid}


def update_rewrite_count_node(state: TextProcessState) -> StateUpdate:
    rewrite_count = state.rewrite_count + 1
    print(f"🔢 Rewrite count -> {rewrite_count}")
    return {"rewrite_count": rewrite_count}


def rewrite_router(state: TextProcessState) -> str:
    quality_valid = state.quality_valid
    rewrite_count = state.rewrite_count
    print(f"🚦 [Router] quality={quality_valid}, rewrite_count={rewrite_count}")
    if quality_valid:
        return "to_output"
    if rewrite_count < 2:
        return "to_rewrite"
    return "to_force_output"


def output_node(state: TextProcessState) -> StateUpdate:
    summary = state.summary_text
    has_sensitive = state.has_sensitive
    if has_sensitive:
        final_output = "⚠️ Detected sensitive content. Unable to generate a summary."
    else:
        final_output = f"""
✅ Text processing completed.
Rewrite count: {state.rewrite_count}

[Summary]
{summary}

[Deduplicated text]
{state.deduplicated_text}
"""
    print("📤 Normal output.")
    return {"final_output": final_output}


def force_output_node(state: TextProcessState) -> StateUpdate:
    summary = state.summary_text
    if summary is None:
        summary = ""
    final_output = f"""
⚠️ Summary regeneration failed.
Rewrite count: {state.rewrite_count}
Length of summary: {len(summary)}

Forced output summary:
{summary}
"""
    print("📤 Forced output.")
    return {"final_output": final_output}


def build_linear_graph() -> CompiledStateGraph:
    builder = StateGraph(TextProcessState)
    builder.add_node("deduplicate_node", deduplicate_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("sensitive_check_node", sensitive_check_node)
    builder.add_node("quality_check_node", quality_check_node)
    builder.add_node("update_rewrite_count_node", update_rewrite_count_node)
    builder.add_node("output_node", output_node)
    builder.add_node("force_output_node", force_output_node)

    builder.add_edge(START, "deduplicate_node")
    builder.add_edge("deduplicate_node", "summary_node")
    builder.add_edge("summary_node", "sensitive_check_node")
    builder.add_edge("sensitive_check_node", "quality_check_node")

    builder.add_conditional_edges(
        "quality_check_node",
        rewrite_router,
        {"to_output": "output_node", "to_rewrite": "update_rewrite_count_node", "to_force_output": "force_output_node"},
    )
    builder.add_edge("update_rewrite_count_node", "summary_node")

    builder.add_edge("output_node", END)
    builder.add_edge("force_output_node", END)
    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_linear_graph()
    init_state = TextProcessState(
        raw_text="LangGraph is a stateful workflow framework in the LangChain ecosystem."
        " It supports graph-based modeling, state tracing, dynamic branching, and parallel execution,"
        " making it suitable for orchestrating complex AI tasks."
    )
    config: RunnableConfig = {"configurable": {"thread_id": "text_processing_pipline_demo"}}
    print("\n" + "=" * 20 + "🚀 Start workflow" + "=" * 20)
    final_state = graph.invoke(init_state, config)
    print("\n" + "=" * 20 + "🎯 Final" + "=" * 20)
    print(final_state)

    print("\n" + "=" * 20 + "History" + "=" * 20)
    history = list(graph.get_state_history(config))
    print("Number of state snapshots: ", len(history))
    for index, snapshot in enumerate(history, 1):
        print(f"Step {index} snapshot: ")
        print("State: ", snapshot.values)
        print("Next node: ", snapshot.next)
        print("-" * 20)

    image_data = graph.get_graph().draw_mermaid_png()
    image_path = Path("graph_image.png")
    with image_path.open("wb") as outfile:
        outfile.write(image_data)
    print(f"📊 Graph image saved: {image_path}")
