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


class NovelState(BaseModel):
    novel_name: str
    protagonist: str
    plot: str | None = None


class StateUpdate(TypedDict):
    plot: NotRequired[str]


plot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please write the plot for the first chapter of the novel "{novel_name}", approximately 200 words.
The protagonist must be named: {protagonist}
""",
        )
    ]
)
plot_agent = plot_prompt | chat_llm | output_parser


def plot_node(state: NovelState) -> StateUpdate:
    plot = plot_agent.invoke({"novel_name": state.novel_name, "protagonist": state.protagonist})
    return {"plot": plot}


builder = StateGraph(NovelState)

builder.add_node("plot", plot_node)

builder.add_edge(START, "plot")
builder.add_edge("plot", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_after=["plot"])

if __name__ == "__main__":
    inital_state = NovelState(novel_name="Interstellar Wanderer", protagonist="Jack")
    thread_id = "state_management"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 20 + "First run: Generate the plot and interrupt." + "=" * 20)
    stream = graph.stream(inital_state, config)
    for step in stream:
        if "plot" in step:
            print("[Original Plot]\n")
            print(step["plot"]["plot"])
            print("⚠️ The workflow has been interrupted. You can now perform 【Dynamic State Editing】")

    print("\n" + "=" * 20 + "Dynamic State Editing: Manually modify the protagonist's name." + "=" * 20)
    new_name = input("Please enter the new protagonist's name: ").strip()

    checkpoint = checkpointer.get(config)
    if checkpoint is None:
        message = f"Checkpoint {thread_id} not found."
        raise RuntimeError(message)
    state = checkpoint["channel_values"]
    old_name = state["protagonist"]
    old_plot = state["plot"]
    new_plot = old_plot.replace(old_name, new_name)

    graph.update_state(config, values={"protagonist": new_name, "plot": new_plot})
    print("\n" + "=" * 20 + "✅ State updated. Continuing the workflow..." + "=" * 20)
    final_state = graph.invoke(None, config)
    print("\n" + "=" * 20 + "Final" + "=" * 20)
    print(final_state["plot"])
