from typing import NotRequired, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


output_parser = StrOutputParser()

MAX_RETRIES = 2


class NovelState(BaseModel):
    novel_name: str
    plot: str | None = None
    retry_count: int = 0
    review_result: str | None = None
    failed: bool = False


class StateUpdate(TypedDict):
    plot: NotRequired[str]
    review_result: NotRequired[str]
    retry_count: NotRequired[int]
    failed: NotRequired[bool]


plot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please write the plot of Chapter 1 of the novel "{novel_name}". Requirements:
1. Introduce the protagonist;
2. Establish the core conflict;
3. Approximately 200 words.""",
        )
    ]
)
plot_agent = plot_prompt | chat_llm | output_parser

review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please review whether the following novel plot meets the requirements. Review criteria:
1. Introduces the protagonist;
2. Establishes the core conflict;
3. Approximately 200 words.
4. Maintains a consistent language style throughout the plot.

Plot:
{plot}

⚠️ Note: 
- Return only 'pass' or 'retry'. Do not output anything else. Follow the instructions strictly!""",
        )
    ]
)
review_agent = review_prompt | chat_llm | output_parser


def plot_node(state: NovelState) -> StateUpdate:
    plot = plot_agent.invoke({"novel_name": state.novel_name})
    return {"plot": plot}


def review_node(state: NovelState) -> StateUpdate:
    review = review_agent.invoke({"plot": state.plot})
    review_stripped = review.strip().lower()
    retry_count = state.retry_count
    if review_stripped == "retry":
        retry_count += 1
        if retry_count >= MAX_RETRIES:
            return {"review_result": review_stripped, "retry_count": retry_count, "failed": True}
        return {"review_result": review_stripped, "retry_count": retry_count}
    else:
        return {"review_result": review_stripped}


def decide_next_node(state: NovelState) -> str:
    if state.failed:
        return END
    if state.review_result == "pass":
        return END
    return "plot"


builder = StateGraph(NovelState)
builder.add_node("plot", plot_node)
builder.add_node("review", review_node)

builder.add_edge(START, "plot")
builder.add_edge("plot", "review")

builder.add_conditional_edges("review", decide_next_node)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    initial_state = NovelState(novel_name="Interstellar Wanderings")
    thread_id = "novel_session"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 20 + "First Run (Interrupted After Plot Execution)" + "=" * 20)
    stream = graph.stream(initial_state, config=config)
    for step in stream:
        print(f"Current step: {step}")
        if "plot" in step:
            snapshot = graph.get_state(config)
            plot_state = snapshot.values
            print("🛑 Simulating program interruption (Ctrl+C scenario)")
            print(f"Version at interruption: Version {plot_state['retry_count']}")
            print(f"Plot content at interruption:\n{plot_state['plot']}")
            break

    print("\n" + "=" * 20 + "Second Run (Resumed from Checkpoint)" + "=" * 20)
    result = graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
    print("\n✅ Final Result After Resuming")
    print(f"Failed: {result['failed']}")
    print("\n" + "=" * 20 + f"Final Plot (Retried {result['retry_count']} Times)" + "=" * 20)
    print(result["plot"])
