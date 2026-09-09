import readline
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


class TaskState(BaseModel):
    agent_name: str
    task_type: str
    task: str | None = None
    task_result: str | None = None
    approval_status: str | None = None
    retry: bool = False


class StateUpdate(TypedDict):
    task: NotRequired[str]
    task_result: NotRequired[str]
    approval_status: NotRequired[str]
    retry: NotRequired[bool]


assign_task_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Please assign {agent_name} a {task_type} task. Make it specific, actionable, and no more than 50 words.",
        )
    ]
)
assign_task_agent = assign_task_prompt | chat_llm | output_parser

execute_task_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Please execute the following task: {task}. Provide the result clearly and concisely.",
        )
    ]
)
execute_task_agent = execute_task_prompt | chat_llm | output_parser


def assign_task_node(state: TaskState) -> StateUpdate:
    task = assign_task_agent.invoke({"agent_name": state.agent_name, "task_type": state.task_type})
    return {"task": task}


def execute_task_node(state: TaskState) -> StateUpdate:
    task_result = execute_task_agent.invoke({"task": state.task})
    print(f"✅ Task execution result:\n{task_result}")
    return {"task_result": task_result}


def approve_task_node(state: TaskState) -> StateUpdate:
    print("\n" + "=" * 20 + "📝 [Task approval node]" + "=" * 20)
    if state.retry:
        return {"approval_status": "rejected", "retry": False}
    return {"approval_status": "passed"}


def approve_router(state: TaskState) -> str:
    if state.approval_status == "rejected":
        return "execute_task"
    return END


builder = StateGraph(TaskState)
builder.add_node("assign_task", assign_task_node)
builder.add_node("execute_task", execute_task_node)
builder.add_node("approve_task", approve_task_node)

builder.add_edge(START, "assign_task")
builder.add_edge("assign_task", "execute_task")
builder.add_edge("execute_task", "approve_task")

builder.add_conditional_edges("approve_task", approve_router)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["approve_task"])

if __name__ == "__main__":
    initial_state = TaskState(agent_name="Plot Design Agent", task_type="Novel Chapter Plot Writing")
    thread_id = "multi_agent_task_approval"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 20 + "First run: Task Assignment and Execution" + "=" * 20)
    stream = graph.stream(initial_state, config=config)
    for step in stream:
        if "assign_task" in step:
            print(f"📝 Assigned task:\n{step['assign_task']['task']}")

    while True:
        print(
            "⚠️ The approval node has paused. Please perform manual approval (enter 'approve' to continue; any other input will reject the task)."
        )
        user_input = input("Please enter the approval instruction:")
        if user_input == "approve":
            print("\n" + "=" * 20 + "Approval granted. Continue executing the approval node." + "=" * 20)
            result = graph.invoke(None, config=config)
            print(f"✅ Workflow completed. Approval status: {result['approval_status']}")
            break
        print("\n" + "=" * 20 + "❌ Approval rejected. The task needs to be re-executed." + "=" * 20)
        graph.update_state(config, {"retry": True})
        result = graph.invoke(None, config=config)
        print(f"🔁 Workflow re-executed. Status: {result['approval_status']}")
