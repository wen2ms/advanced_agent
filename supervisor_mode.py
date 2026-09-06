from operator import attrgetter
from typing import NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)


class TaskState(BaseModel):
    task: str
    research: str | None = None
    draft: str | None = None
    code: str | None = None
    math: str | None = None
    next_agent: str | None = None
    result: str | None = None
    round_count: int = 0
    supervisor_thoughts: str | None = None


class StateUpdate(TypedDict):
    research: NotRequired[str]
    draft: NotRequired[str]
    code: NotRequired[str]
    math: NotRequired[str]
    next_agent: NotRequired[str]
    result: NotRequired[str]
    round_count: NotRequired[int]
    supervisor_thoughts: NotRequired[str]


MAX_ROUNDS = 3

reasearch_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please research the background information for the following task and organize it into bullet points.
Output in Chinese:"
{task}""",
        )
    ]
)
reasearch_agent = reasearch_prompt | chat_llm

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Write a Chinese technical article or explanatory text based on the following information:
{research}""",
        )
    ]
)
writer_agent = writer_prompt | chat_llm

code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please generate Python example code based on the following task:
{task}""",
        )
    ]
)
code_agent = code_prompt | chat_llm

math_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please solve the following mathematical or logical problem and explain the process in detail:
{task}""",
        )
    ]
)
math_agent = math_prompt | chat_llm

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
You are the Supervisor agent in a multi-agent system.
You coordinate expert agents, but you do not perform tasks yourself.

Task:
{task}

Completed status:

- Research: {research_status}
- Writing: {writing_status}
- Coding: {coding_status}
- Math: {math_status}

Available agents:

- research_agent: Responsible for researching and organizing information
- writer_agent: Responsible for writing Chinese articles or explanatory texts
- code_agent: Responsible for writing Python code
- math_agent: Responsible for mathematical/logical calculations and reasoning

Constraints:

1. Do not select an agent that has already completed its task.
2. You must select an agent that is relevant to the task.
3. If all required tasks are completed, return "end".
4. Briefly explain the basis for your decision, then return the next agent name on the final line:
   research_agent / writer_agent / code_agent / math_agent / end

Please provide the complete response in Chinese.
""",
        )
    ]
)
supervisor_agent = supervisor_prompt | chat_llm


def supervisor_node(state: TaskState) -> StateUpdate:
    new_round = state.round_count + 1
    if new_round > MAX_ROUNDS:
        print(f"⚠️ Maximum number of rounds {MAX_ROUNDS} exceeded, triggering fallback -> end task")
        return {
            "round_count": new_round,
            "next_agent": "end",
            "supervisor_thoughts": "Maximum number of rounds exceeded",
        }
    result = supervisor_agent.invoke(
        {
            "task": state.task,
            "research_status": "Completed" if state.research else "Not completed",
            "writing_status": "Completed" if state.draft else "Not completed",
            "coding_status": "Completed" if state.code else "Not completed",
            "math_status": "Completed" if state.math else "Not completed",
        }
    )
    thoughts = result.text.strip()
    last_line = thoughts.splitlines()[-1]
    valid_agents = ("research_agent", "writer_agent", "code_agent", "math_agent", "end")
    next_agent = next((agent for agent in valid_agents if agent in last_line), "end")
    print(f"🧠 Supervisor thoughts:\n{thoughts}\n")
    print(f"🧠 Supervisor routing -> {next_agent} (Round {new_round})")
    return {"round_count": new_round, "next_agent": next_agent, "supervisor_thoughts": thoughts}


def research_node(state: TaskState) -> StateUpdate:
    print("\n" + "=" * 20 + "Reasearch Agent executing..." + "=" * 20)
    try:
        response = reasearch_agent.invoke({"task": state.task})
        result = response.text.strip()
    except Exception as exc:
        result = f"Research failed: {str(exc)[:50]}"
    return {"research": result, "result": result}


def writer_node(state: TaskState) -> StateUpdate:
    print("\n" + "=" * 20 + "Writer Agent executing..." + "=" * 20)
    try:
        response = writer_agent.invoke({"research": state.research})
        result = response.text.strip()
    except Exception as exc:
        result = f"Write failed: {str(exc)[:50]}"
    return {"draft": result, "result": result}


def code_node(state: TaskState) -> StateUpdate:
    print("\n" + "=" * 20 + "Code Agent executing..." + "=" * 20)
    try:
        response = code_agent.invoke({"task": state.task})
        result = response.text.strip()
    except Exception as exc:
        result = f"Research failed: {str(exc)[:50]}"
    return {"research": result, "result": result}


def math_node(state: TaskState) -> StateUpdate:
    print("\n" + "=" * 20 + "Math Agent executing..." + "=" * 20)
    try:
        response = math_agent.invoke({"task": state.task})
        result = response.text.strip()
    except Exception as exc:
        result = f"Research failed: {str(exc)[:50]}"
    return {"research": result, "result": result}


workflow = StateGraph(TaskState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("code_agent", code_node)
workflow.add_node("math_agent", math_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    attrgetter("next_agent"),
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "end": END,
    },
)

workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")

workflow_compiled = workflow.compile()

if __name__ == "__main__":
    tasks: list[str] = [
        "Write a Chinese article introducing multi-agent collaboration in LangGraph, aimed at beginners.",
    ]
    for task in tasks:
        print("\n" + "=" * 20 + task + "=" * 20)
        init_state = TaskState(task=task)
        result = workflow_compiled.invoke(init_state)
        print("\n" + "=" * 20 + "✅ Result" + "=" * 20)
        print(result["result"])
