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


class EmailState(BaseModel):
    sender: str
    recipient: str
    email_type: str
    subject: str
    email_content: str | None = None
    send_status: str | None = None


class StateUpdate(TypedDict):
    email_content: NotRequired[str]
    send_status: NotRequired[str]


write_email_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Please write a {email_type} email as {sender} to {recipient},
with the subject "{subject}". Keep the content concise, formal, and properly formatted as an email.""",
        )
    ]
)
write_email_agent = write_email_prompt | chat_llm | output_parser


def write_email_node(state: EmailState) -> StateUpdate:
    email = write_email_agent.invoke(
        {"sender": state.sender, "recipient": state.recipient, "email_type": state.email_type, "subject": state.subject}
    )
    return {"email_content": email}


def send_email_node(state: EmailState) -> StateUpdate:
    print("📤 [Email Sent Successfully]")
    print(f"Recipient: {state.recipient}")
    print(f"Subject: {state.subject}")
    print(f"Content: {state.email_content}")
    return {"send_status": "success"}


builder = StateGraph(EmailState)

builder.add_node("write_email", write_email_node)
builder.add_node("send_email", send_email_node)

builder.add_edge(START, "write_email")
builder.add_edge("write_email", "send_email")
builder.add_edge("send_email", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["send_email"])

if __name__ == "__main__":
    print("\n" + "=" * 20 + "First Run: Generate Email and Wait for Human Approval" + "=" * 20)
    initial_state = EmailState(
        sender="Student Jack",
        recipient="teacher@xxx.edu.cn",
        email_type="leave_request",
        subject="Leave Request (1 Day)",
    )
    thread_id = "email_session"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    stream = graph.stream(initial_state, config=config)
    for step in stream:
        if "write_email" in step:
            print("✉️ Email content generated:")
            print(step["write_email"]["email_content"])
            print("⚠️ The system was interrupted before [Send Email]")
            print("Please enter: Confirm Send  →  Continue Execution")
    user_input = input("Please enter the authorization command: ").strip()

    if user_input == "Confirm Send":
        print("\n" + "=" * 20 + "Confirmed. Continuing to the send node" + "=" * 20)
        result = graph.invoke(None, config=config)
        print("✅ Workflow completed. Send status:", result["send_status"])
    else:
        print("❌ Sending canceled. Workflow terminated.")
