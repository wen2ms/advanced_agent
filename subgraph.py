from typing import NotRequired, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)

output_parser = StrOutputParser()


class CorrectionSubgraphState(BaseModel):
    homework_content: str
    completion: str | None = None
    accuracy: str | None = None
    score: int = 0


class SubgraphStateUpdate(TypedDict):
    completion: NotRequired[str]
    accuracy: NotRequired[str]
    score: NotRequired[int]


completion_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'You are a homework completion checker. Output only "Completed" or "Incomplete" with no additional text!',
        ),
        (
            "user",
            "Homework content: {homework_content}. Determine whether it is completed (it is considered completed if it contains specific content and is not blank).",
        ),
    ]
)
completion_check_agent = completion_check_prompt | chat_llm | output_parser

accuracy_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 'You are a homework accuracy checker. Output only "Accuracy: XX%" with no additional text!'),
        (
            "user",
            "Homework content: {homework_content}. Assume it is a math calculation problem and reasonably estimate the accuracy.",
        ),
    ]
)
accuracy_check_agent = accuracy_check_prompt | chat_llm | output_parser

score_calc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a score calculation teacher. Output only an integer from 0 to 100 with no additional text!",
        ),
        (
            "user",
            "Completion: {completion}, Accuracy: {accuracy}. Scoring rule: completed work receives a base score of 60 points, and every 10% of accuracy adds 4 points; incomplete work receives 0 points.",
        ),
    ]
)
score_calc_agent = score_calc_prompt | chat_llm | output_parser


def check_completion_node(state: CorrectionSubgraphState) -> SubgraphStateUpdate:
    print("🔍 Subgraph Execution - Checking Homework Completion")
    completion = completion_check_agent.invoke({"homework_content": state.homework_content})
    return {"completion": completion}


def check_accuracy_node(state: CorrectionSubgraphState) -> SubgraphStateUpdate:
    print("🔍 Subgraph Execution - Checking Homework Accuracy")
    accuracy = accuracy_check_agent.invoke({"homework_content": state.homework_content})
    return {"accuracy": accuracy}


def calc_score_node(state: CorrectionSubgraphState) -> SubgraphStateUpdate:
    print("🔍 Subgraph Execution - Calculating Homework Score")
    completion = state.completion or "Incomplete"
    accuracy = state.accuracy or "Accuracy: 0%"
    try:
        score = int(score_calc_agent.invoke({"completion": completion, "accuracy": accuracy}))
    except:
        score = 0
    return {"score": score}


correction_subgraph = StateGraph(CorrectionSubgraphState)
correction_subgraph.add_node("check_completion", check_completion_node)
correction_subgraph.add_node("check_accuracy", check_accuracy_node)
correction_subgraph.add_node("calc_score", calc_score_node)

correction_subgraph.add_edge(START, "check_completion")
correction_subgraph.add_edge("check_completion", "check_accuracy")
correction_subgraph.add_edge("check_accuracy", "calc_score")
correction_subgraph.add_edge("calc_score", END)

compiled_correction_subgraph = correction_subgraph.compile()


class HomeworkMainState(BaseModel):
    homework_content: str
    correction_result: dict | None = None
    feedback: str | None = None


class MainStateUpdate(TypedDict):
    correction_result: NotRequired[dict | None]
    feedback: NotRequired[str | None]


feedback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a homeroom teacher. Based on the grading results, write 1-2 friendly sentences of feedback for the student, with a tone appropriate to the score.",
        ),
        (
            "user",
            "Homework content: {homework_content}\nGrading results: Completion: {completion}, Accuracy: {accuracy}, Score: {score}\nGenerate feedback:",
        ),
    ]
)
feedback_agent = feedback_prompt | chat_llm | output_parser


def receive_homework_node(state: HomeworkMainState) -> MainStateUpdate:
    print("\n" + "=" * 20 + f"📥 Main Graph Execution - Receiving Student Homework: {state.homework_content}" + "" * 20)
    return {}


def correction_subgraph_node(state: HomeworkMainState) -> MainStateUpdate:
    print("\n" + "=" * 20 + "📤 Main Graph Execution - Invoking Homework Grading Subgraph" + "=" * 20)
    subgraph_input = CorrectionSubgraphState(homework_content=state.homework_content)
    subgraph_output = compiled_correction_subgraph.invoke(subgraph_input)
    print(
        f"✅ Main Graph Received Subgraph Grading Results: Completion {subgraph_output['completion']}, Accuracy {subgraph_output['accuracy']}, Score {subgraph_output['score']}"
    )
    return {"correction_result": subgraph_output}


def generate_feedback_node(state: HomeworkMainState) -> MainStateUpdate:
    print("\n" + "=" * 20 + "📝 Main Graph Execution - Generating Student Grading Feedback" + "=" * 20)
    if not state.correction_result:
        return {"feedback": "Homework grading failed; unable to generate feedback!"}
    correction = state.correction_result
    homework = state.homework_content
    feedback = feedback_agent.invoke(
        {
            "homework_content": homework,
            "completion": correction["completion"] or "Incomplete",
            "accuracy": correction["accuracy"] or "Accuracy: 0%",
            "score": correction["score"] or 0,
        }
    )
    return {"feedback": feedback}


main_graph = StateGraph(HomeworkMainState)
main_graph.add_node("receive_homework", receive_homework_node)
main_graph.add_node("correction_subgraph", correction_subgraph_node)
main_graph.add_node("generate_feedback", generate_feedback_node)

main_graph.add_edge(START, "receive_homework")
main_graph.add_edge("receive_homework", "correction_subgraph")
main_graph.add_edge("correction_subgraph", "generate_feedback")
main_graph.add_edge("generate_feedback", END)
compiled_main_graph = main_graph.compile()

if __name__ == "__main__":
    print("\n" + "=" * 20 + "Test 1: Excellent homework (high completion + high accuracy)" + "=" * 20)
    test1_state = HomeworkMainState(homework_content="2 + 3 = 5, 4 + 6 = 10, 7 + 8 = 15, 9 + 11 = 20")
    test1_result = compiled_main_graph.invoke(test1_state)
    print("\n" + "=" * 20 + f"🎉 Final Result - Student Feedback: {test1_result['feedback']}" + "=" * 20)

    print("\n" + "=" * 20 + "Test 2: Unsatisfactory homework (incomplete + low accuracy)" + "=" * 20)
    test2_state = HomeworkMainState(homework_content="2 + 3 = 6, 4 + 6 = (blank), 7 + 8 = (blank)")
    test2_result = compiled_main_graph.invoke(test2_state)
    print("\n" + "=" * 20 + f"🎉 Final Result - Student Feedback: {test2_result['feedback']}" + "=" * 20)
