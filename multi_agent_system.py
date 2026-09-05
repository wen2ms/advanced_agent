from pathlib import Path
from typing import NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


class AgentState(BaseModel):
    content: str | None = None
    error: str | None = None
    polished_content: str | None = None
    deduplicated_content: str | None = None


class StateUpdate(TypedDict):
    content: NotRequired[str]
    error: NotRequired[str]
    polished_content: NotRequired[str]
    deduplicated_content: NotRequired[str]


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Write a short article of about 150 words on "LangGraph Multi-Agent Systems,"
using clear and simple language suitable for beginners. No error correction or polishing is needed.""",
        )
    ]
)
write_agent = writer_prompt | chat_llm

corrector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please check the following short article and correct any technical errors about LangGraph,
such as interface names and functionality descriptions. Only output the corrected content without polishing:
{content}""",
        )
    ]
)
corrector_agent = corrector_prompt | chat_llm


polisher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please polish the following short article by adding one analogy that beginners can easily understand.
Make the language more fluent without changing the core content or technical accuracy:
{content}""",
        )
    ]
)

polisher_agent = polisher_prompt | chat_llm

duplicate_checker_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Check the following article for repeated sentences, redundant expressions, or unnecessarily repeated ideas.
Simulate a duplicate-content check and rewrite only the repetitive parts when necessary.
Preserve the original meaning, technical accuracy, and overall writing style.
Only output the final revised article:
{content}""",
        )
    ]
)
duplicate_checker_agent = duplicate_checker_prompt | chat_llm


def write_node(state: AgentState) -> StateUpdate:
    response = write_agent.invoke({})
    return {"content": response.text}


def correct_node(state: AgentState) -> StateUpdate:
    response = corrector_agent.invoke({"content": state.content})
    return {"content": response.text}


def polish_node(state: AgentState) -> StateUpdate:
    response = polisher_agent.invoke({"content": state.content})
    return {"polished_content": response.text}


def deduplicate_node(state: AgentState) -> StateUpdate:
    response = duplicate_checker_agent.invoke({"content": state.polished_content})
    return {"deduplicated_content": response.text}


workflow = StateGraph(AgentState)
workflow.add_node("writer", write_node)
workflow.add_node("corrector", correct_node)
workflow.add_node("polisher", polish_node)
workflow.add_node("deduplicator", deduplicate_node)

workflow.add_edge(START, "writer")
workflow.add_edge("writer", "corrector")
workflow.add_edge("corrector", "polisher")
workflow.add_edge("polisher", "deduplicator")
workflow.add_edge("deduplicator", END)

compiled_workflow = workflow.compile()
response = compiled_workflow.invoke(AgentState())
print("\n" + "=" * 20 + "Deduplicated content" + "=" * 20)
print(response["deduplicated_content"])

image_data = compiled_workflow.get_graph().draw_mermaid_png()
image_path = Path("multi_agent_graph.png")
image_path.write_bytes(image_data)
print(f"Graph image saved: {image_path}")
