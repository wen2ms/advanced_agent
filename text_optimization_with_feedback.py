import readline
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

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


class InteractiveOptState(BaseModel):
    user_input: str
    optimized_text: str | None = None
    optimize_suggest: str | None = None
    user_feedback: str | None = None
    final_result: str | None = None


class StateUpdate(TypedDict):
    optimized_text: NotRequired[str]
    optimize_suggest: NotRequired[str]
    user_feedback: NotRequired[str]
    final_result: NotRequired[str]


def optimize_node(state: InteractiveOptState) -> StateUpdate:
    user_input = state.user_input
    user_feedback = state.user_feedback
    if not user_feedback:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    (
                        "Please refine the following text to improve fluency"
                        " and professionalism while strictly preserving the core information:\n{text}"
                        "\nAfter the refinement, add a separate line starting with [Reason for Optimization:]"
                        " and provide 1–2 concise reasons for the changes."
                    ),
                )
            ]
        )
        chain = prompt | chat_llm
        result = chain.invoke({"text": user_input}).text
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    (
                        "Refine the text based on the user's feedback while strictly preserving the core information:"
                        "\nOriginal text: {text}\nUser feedback: {feedback}\nAfter the refinement,"
                        " add a separate line starting with [Reason for Optimization:]"
                        " and provide 1–2 concise reasons for the changes."
                    ),
                )
            ]
        )
        chain = prompt | chat_llm
        result = chain.invoke({"text": user_input, "feedback": user_feedback}).text
    split_flag = "[Reason for Optimization:]"
    if split_flag in result:
        optimized_text, optimize_suggest = result.split(split_flag, 1)
    else:
        optimized_text = result
        optimize_suggest = "The AI did not provide a clear reason for the optimization. Re-optimization is recommended."
    return {"optimized_text": optimized_text.strip(), "optimize_suggest": optimize_suggest.strip()}


def feedback_node(state: InteractiveOptState) -> StateUpdate:
    print("\n" + "=" * 20 + "📝 AI Optimized text" + "=" * 20)
    print(state.optimized_text)
    print("\n💡 Optimized suggest: ")
    print(state.optimize_suggest)

    while True:
        action = input("Please select an action (confirm/modify/exit): ").strip()
        if action == "confirm" or action == "exit":
            return {"user_feedback": action}
        if action == "modify":
            detail = input("Please enter your specific revision feedback: ").strip()
            if detail:
                return {"user_feedback": detail}
            print("❌ Revision feedback cannot be empty. Please enter it again.\n")
        else:
            print("❌ please enter [confirm] [modify] or [exit]\n")


def feedback_router(state: InteractiveOptState) -> Literal["final_node", "exit_node", "optimize_node"]:
    feedback = state.user_feedback
    if feedback == "confirm":
        return "final_node"
    if feedback == "exit":
        return "exit_node"
    return "optimize_node"


def final_node(state: InteractiveOptState) -> StateUpdate:
    final_result = (
        "✅ [Text optimization completed] \n"
        f"\n📌 Final optimized text: \n{state.optimized_text}"
        f"\n💡 Optimize suggest: \n{state.optimize_suggest}"
    )
    return {"final_result": final_result}


def exit_node(state: InteractiveOptState) -> StateUpdate:
    final_result = (
        "🔚 [Text Optimization Process Terminated]\nYou chose to exit. No final optimized result was generated."
    )
    return {"final_result": final_result}


def build_interactive_graph() -> CompiledStateGraph:
    builder = StateGraph(InteractiveOptState)
    builder.add_node("optimize_node", optimize_node)
    builder.add_node("feedback_node", feedback_node)
    builder.add_node("final_node", final_node)
    builder.add_node("exit_node", exit_node)

    builder.add_edge(START, "optimize_node")
    builder.add_edge("optimize_node", "feedback_node")
    builder.add_conditional_edges("feedback_node", feedback_router)
    builder.add_edge("final_node", END)
    builder.add_edge("exit_node", END)

    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_interactive_graph()
    print("🔧 Multi-turn interactive text optimization tool has started...\n")
    print("\n" + "=" * 20 + "Enter the text to be optimized" + "=" * 20)
    while True:
        user_input_text = input("Please enter the sentence you want the AI to optimize: ").strip()
        if user_input_text:
            break
        print("❌ Sentence cannot be empty. Please enter it again.\n")
    initial_state = InteractiveOptState(user_input=user_input_text)
    print("\n🚀 Your sentence has been received. Starting the first round of AI optimization...")
    config: RunnableConfig = {"configurable": {"thread_id": "text_optimization_with_feedback"}}
    final_state = graph.invoke(initial_state, config)
    print("\n" + "=" * 20 + "Final" + "=" * 20)
    print(final_state["final_result"])

    history = list(graph.get_state_history(config))
    print("Number of state snapshots: ", len(history))

    image_data = graph.get_graph().draw_mermaid_png()
    image_path = Path("interactive_optimize_graph.png")
    image_path.write_bytes(image_data)
    print(f"📊 Graph image saved: {image_path}")
