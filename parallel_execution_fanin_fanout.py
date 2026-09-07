from pathlib import Path
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


class CandidateState(BaseModel):
    resume: str
    job_requirements: str
    skills: str
    interview_feedback: str
    resume_info: str | None = None
    skill_match: str | None = None
    interview_summary: str | None = None
    personality_analysis: str | None = None
    summary: str | None = None


class StateUpdate(TypedDict):
    resume_info: NotRequired[str]
    skill_match: NotRequired[str]
    interview_summary: NotRequired[str]
    personality_analysis: NotRequired[str]
    summary: NotRequired[str]


resume_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Please read the following candidate resume and extract the key information (name, education, work experience, and skills):\n{resume}",
        )
    ]
)
resume_agent = resume_prompt | chat_llm | output_parser

skill_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            (
                "Based on the job requirements: {job_requirements}, please analyze how well the candidate's skills match and provide a match score (0-10):\n"
                "Candidate skills: {skills}"
            ),
        )
    ]
)
skill_agent = skill_prompt | chat_llm | output_parser

interview_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Please summarize the candidate's strengths and potential areas for improvement based on the following interview feedback. Keep it concise:\n{interview_feedback}",
        )
    ]
)
interview_agent = interview_prompt | chat_llm | output_parser

personality_prompt = ChatPromptTemplate.from_messages(
    [
        "human",
        (
            "Based on the following interview feedback, analyze the candidate's "
            "likely personality traits, communication style, teamwork tendencies, "
            "and potential leadership characteristics. "
            "Only make conclusions supported by the provided feedback and keep it concise:\n"
            "{interview_feedback}"
        ),
    ]
)
personality_agent = personality_prompt | chat_llm | output_parser

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            (
                "Please integrate the following candidate information and generate a complete hiring recommendation report (within 150 words):\n"
                "Key resume information: {resume_info}\n"
                "Skill match analysis: {skill_match}\n"
                "Interview summary: {interview_summary}"
                "Personality analysis: {personality_analysis}"
            ),
        )
    ]
)
summary_agent = summary_prompt | chat_llm | output_parser


def resume_node(state: CandidateState) -> StateUpdate:
    resume = resume_agent.invoke({"resume": state.resume})
    print("\n" + "=" * 20 + f"[Fan out] Resume info: {resume}" + "=" * 20)
    return {"resume_info": resume}


def skill_node(state: CandidateState) -> StateUpdate:
    skill_match = skill_agent.invoke({"job_requirements": state.job_requirements, "skills": state.skills})
    print("\n" + "=" * 20 + f"[Fan out] Skill match: {skill_match}" + "=" * 20)
    return {"skill_match": skill_match}


def interview_node(state: CandidateState) -> StateUpdate:
    interview_summary = interview_agent.invoke({"interview_feedback": state.interview_feedback})
    print("\n" + "=" * 20 + f"[Fan out] Interview summary: {interview_summary}" + "=" * 20)
    return {"interview_summary": interview_summary}


def personality_node(state: CandidateState) -> StateUpdate:
    personality = personality_agent.invoke({"interview_feedback": state.interview_feedback})
    print("\n" + "=" * 20 + f"[Fan out] Personality analysis: {personality}" + "=" * 20)
    return {"personality_analysis": personality}


def summary_node(state: CandidateState) -> StateUpdate:
    summary = summary_agent.invoke(
        {
            "resume_info": state.resume_info or "Failed to extract resume information",
            "skill_match": state.skill_match or "Skill match analysis not available",
            "interview_summary": state.interview_summary or "Interview summary not generated",
            "personality_analysis": state.personality_analysis or "Personality analysis not available",
        }
    )
    print("\n" + "=" * 20 + f"[Fan in] Hiring Recommendation Report: {summary}" + "=" * 20)
    return {"summary": summary}


builder = StateGraph(CandidateState)
builder.add_node("resume_info", resume_node)
builder.add_node("skill_match", skill_node)
builder.add_node("interview_summary", interview_node)
builder.add_node("personality_analysis", personality_node)
builder.add_node("summary", summary_node)

builder.add_edge(START, "resume_info")
builder.add_edge(START, "skill_match")
builder.add_edge(START, "interview_summary")
builder.add_edge(START, "personality_analysis")

builder.add_edge(["resume_info", "skill_match", "interview_summary", "personality_analysis"], "summary")
builder.add_edge("summary", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = CandidateState(
        resume="Jack, master's degree, 5 years of software development experience, proficient in Python, C++, and SQL.",
        job_requirements="Proficient in Python and data analysis, with strong teamwork skills.",
        skills="Python, C++, SQL, Data Analysis",
        interview_feedback="Communicates clearly and demonstrates strong logical thinking, but has relatively limited team management experience.",
    )
    result = graph.invoke(initial_state)
    print("\n" + "=" * 20 + "Final" + "=" * 20)
    print(result["summary"])

    image_data = graph.get_graph().draw_mermaid_png()
    image_path = Path("parallel_execution_fanin_fanout.png")
    image_path.write_bytes(image_data)
    print(f"Graph image saved: {image_path}")
