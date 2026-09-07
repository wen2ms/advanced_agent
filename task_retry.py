from typing import Literal, NotRequired, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


output_parser = StrOutputParser()


class NovelState(BaseModel):
    novel_name: str
    plot: str | None = None
    retry_count: int = 0
    review_result: str | None = None


class StateUpdate(TypedDict):
    plot: NotRequired[str]
    review_result: NotRequired[str]


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

⚠️ Note: Return only 'pass' or 'retry'. Do not output anything else. Follow the instructions strictly!""",
        )
    ]
)
review_agent = review_prompt | chat_llm | output_parser


def plot_node(state: NovelState) -> StateUpdate:
    plot = plot_agent.invoke({"novel_name": state.novel_name})
    return {"plot": plot}


def review_node(state: NovelState) -> StateUpdate:
    review = review_agent.invoke({"plot": state.plot})
    return {"review_result": review.strip().lower()}


def decide_next_node(state: NovelState) -> Literal["end", "plot"]:
    if state.review_result == "pass":
        return "end"
    state.retry_count += 1
    return "plot"


builder = StateGraph(NovelState)
builder.add_node("plot", plot_node)
builder.add_node("review", review_node)

builder.add_edge(START, "plot")
builder.add_edge("plot", "review")

builder.add_conditional_edges("review", decide_next_node, {"plot": "plot", "end": END})

graph = builder.compile()

if __name__ == "__main__":
    initial_state = NovelState(novel_name="Interstellar Wanderings")
    result = graph.invoke(initial_state)
    print("\n" + "=" * 20 + f"Final Plot (Retried {result['retry_count']} Times)" + "=" * 20)
    print(result["plot"])
