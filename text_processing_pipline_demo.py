from typing import NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


class TextProcessState(BaseModel):
    raw_text: str
    deduplicated_text: str | None = None
    summary_text: str | None = None
    has_sensitive: bool | None = None
    final_output: str | None = None


class StateUpdate(TypedDict):
    summary_text: NotRequired[str]
    has_sensitive: NotRequired[bool]
    final_output: NotRequired[str]


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
    sensitive_words = ["Illegal", "Non-compliant"]
    has_sensitive = any(word in summary for word in sensitive_words)
    print("🔍 Sensitive check completed: ", has_sensitive)
    return {"has_sensitive": has_sensitive}


def output_node(state: TextProcessState) -> StateUpdate:
    summary = state.summary_text
    has_sensitive = state.has_sensitive
    if has_sensitive:
        final_output = "⚠️ Detected sensitive content. Unable to generate a summary."
    else:
        final_output = f"""Text processing completed
[Summary]
{summary}

[Deduplicated text]
{state.deduplicated_text}
"""
    print("📤 Output completed.")
    return {"final_output": final_output}


def build_linear_graph() -> CompiledStateGraph:
    builder = StateGraph(TextProcessState)
    builder.add_node("deduplicate_node", deduplicate_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("sensitive_check_node", sensitive_check_node)
    builder.add_node("output_node", output_node)

    builder.add_edge(START, "deduplicate_node")
    builder.add_edge("deduplicate_node", "summary_node")
    builder.add_edge("summary_node", "sensitive_check_node")
    builder.add_edge("sensitive_check_node", "output_node")
    builder.add_edge("output_node", END)

    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    linear_graph = build_linear_graph()
    init_state = TextProcessState(
        raw_text="LangGraph is a workflow framework in the LangChain ecosystem."
        "\nLangGraph supports state management."
        "\nLangGraph is a workflow framework in the LangChain ecosystem."
        "\nIt supports dynamic branching and parallel execution."
    )
    config: RunnableConfig = {"configurable": {"thread_id": "text_processing_pipline_demo"}}
    final_state = linear_graph.invoke(init_state, config)
    print("\n" + "=" * 20 + "Final" + "=" * 20)
    print(final_state)

    print("\n" + "=" * 20 + "History" + "=" * 20)
    history = list(linear_graph.get_state_history(config))
    print("Number of state snapshots: ", len(history))
    for index, snapshot in enumerate(history, 1):
        print(f"Step {index} snapshot: ")
        print("State: ", snapshot.values)
        print("Next node: ", snapshot.next)
        print("-" * 20)
