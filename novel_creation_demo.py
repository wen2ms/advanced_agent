import readline
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class Character(BaseModel):
    name: str
    description: str


class Chapter(BaseModel):
    chapter_name: str
    chapter_description: str


class NovelCreationState(BaseModel):
    user_requirement: str | None = None
    novel_title: str | None = None
    main_characters: list[Character] | None = None
    plot_overview: str | None = None
    is_setting_confirmed: bool = False
    is_outline_confirmed: bool = False
    novel_outline: str | None = None
    chapter_structure: list[Chapter] | None = None
    complete_novel: str | None = None
    current_stage: str | None = None
    chapter_generated_count: int = 0


class StateUpdate(TypedDict):
    user_requirement: NotRequired[str]
    novel_title: NotRequired[str]
    main_characters: NotRequired[list[Character]]
    plot_overview: NotRequired[str]
    is_setting_confirmed: NotRequired[bool]
    is_outline_confirmed: NotRequired[bool]
    novel_outline: NotRequired[str]
    chapter_structure: NotRequired[list[Chapter]]
    complete_novel: NotRequired[str]

    current_stage: NotRequired[str]
    chapter_generated_count: NotRequired[int]


class BasicSettingOutput(BaseModel):
    novel_title: str
    main_characters: list[Character]
    plot_overview: str


class OutlineOutput(BaseModel):
    chapter_structure: list[Chapter]
    novel_outline: str


chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-plus"
)
basic_setting_llm = chat_llm.with_structured_output(BasicSettingOutput, method="json_schema")
outline_llm = chat_llm.with_structured_output(OutlineOutput, method="json_schema")
output_parser = StrOutputParser()


def print_process_progress(current_stage: str, detail: str = "") -> None:
    stage_map = {
        "Requirement Collection": "1/4",
        "Setting Generation": "2/4",
        "Outline Generation": "3/4",
        "Novel Generation": "4/4",
    }
    progress = stage_map.get(current_stage, "Unknown Stage")
    print(f"\n🔄 [Overall Progress {progress}] - {current_stage} {detail}")


def print_chapter_progress(generated: int, total: int) -> None:
    percentage = (generated / total) * 100 if total > 0 else 0
    print(f"\n📖 [Chapter Progress] Completed {generated}/{total} chapters ({percentage:.1f}%)")


def user_input_node(state: NovelCreationState) -> StateUpdate:
    print_process_progress("Requirement Collection", "(Starting)")
    user_input = input(
        "Please enter your novel-writing requirements (example: science fiction, the protagonist is a computer science college student, include an AI-related plot twist, and keep it concise): "
    ).strip()
    print_process_progress("Requirement Collection", "(Completed) ✅")
    return {"user_requirement": user_input, "current_stage": "Requirement Collection"}


generate_basic_setting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please generate the basic setting for a novel based on the user's requirements. The output must include:

1. Novel title: Provide 1 candidate titles that are concise and appealing.
2. Main characters: At least 3 characters, in the format "Name: Personality description".
3. Plot overview: 100-200 words, clearly describing the overall direction of the story.

User requirements: {user_requirement}
""",
        )
    ]
)
generate_basic_setting_agent = generate_basic_setting_prompt | basic_setting_llm


def generate_basic_setting_node(state: NovelCreationState) -> StateUpdate:
    print_process_progress("Setting Generation", "(Start generating the title/characters/plot)")
    setting = cast(
        "BasicSettingOutput", generate_basic_setting_agent.invoke({"user_requirement": state.user_requirement})
    )
    print("\n" + "=" * 20 + "Generated Novel Setting" + "=" * 20)
    print(f"\nTitle: {setting.novel_title}")
    print("\nMain Characters:")
    for character in setting.main_characters:
        print(f"- {character.name}: {character.description}")
    print(f"Plot Overview: {setting.plot_overview}")

    print_process_progress("Setting Generation", "(Completed) ✅")
    return {
        "novel_title": setting.novel_title,
        "main_characters": setting.main_characters,
        "plot_overview": setting.plot_overview,
        "current_stage": "Setting Generation",
    }


confirm_basic_setting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please update the basic novel setting based on the user's original requirements and modification requests:

Original requirements: {user_requirement}
Modification requests: {modify_content}
""",
        )
    ]
)
confirm_basic_setting_agent = confirm_basic_setting_prompt | basic_setting_llm


def confirm_basic_setting_node(state: NovelCreationState) -> StateUpdate:
    print("\n" + "=" * 20 + "⚠️ Human Review - Basic Setting Confirmation Stage" + "=" * 20)
    confirm = input(
        "Do you confirm the basic setting above? (Enter y to confirm, or enter n and provide the changes you would like to make): "
    ).lower()

    if confirm == "y":
        print("✅ Basic setting confirmed. Proceeding to the next stage!")
        return {"is_setting_confirmed": True}

    modify_content = input(
        "Please enter your modification requests (e.g., change character names / adjust the plot / replace the title): "
    )
    print("🔄 Updating the basic setting based on your requirements...")

    setting = cast(
        "BasicSettingOutput",
        confirm_basic_setting_agent.invoke(
            {"user_requirement": state.user_requirement, "modify_content": modify_content}
        ),
    )
    print("\n" + "=" * 20 + "Modified Basic Setting" + "=" * 20)
    print(f"\nTitle: {setting.novel_title}")
    print("\nMain Characters:")
    for character in setting.main_characters:
        print(f"- {character.name}: {character.description}")
    print(f"Plot Overview: {setting.plot_overview}")

    re_confirm = input("Do you confirm the modified setting? (y/n): ").lower()
    if re_confirm == "y":
        print("✅ Basic setting confirmed!")
        return {
            "is_setting_confirmed": True,
            "novel_title": setting.novel_title,
            "main_characters": setting.main_characters,
            "plot_overview": setting.plot_overview,
        }
    print("❌ Not confirmed. The basic setting will be regenerated.")
    return {"is_setting_confirmed": False}


generate_outline_chapter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please generate the following based on the confirmed basic novel setting:

1. Overall novel outline: 200-300 words, clearly describing the beginning, development, climax, and ending of the story
2. Chapter structure: At least 8 chapters, in the format "Chapter X: Chapter plot summary (1-2 sentences)", with logical continuity between chapters

Basic Setting:
Title: {novel_title}
Main Characters: {main_characters}
Plot Overview: {plot_overview}
""",
        )
    ]
)
generate_outline_chapter_agent = generate_outline_chapter_prompt | outline_llm


def generate_outline_chapter_node(state: NovelCreationState) -> StateUpdate:
    if not state.is_setting_confirmed or not state.main_characters:
        message = "❌ The basic setting has not been confirmed, so the outline cannot be generated!"
        raise RuntimeError(message)
    print_process_progress("Outline Generation", "(Start generating the outline/chapter structure)")
    character_str = "\n".join([f"{character.name}: {character.description}" for character in state.main_characters])
    outline = cast(
        "OutlineOutput",
        generate_outline_chapter_agent.invoke(
            {
                "novel_title": state.novel_title,
                "main_characters": character_str,
                "plot_overview": state.plot_overview,
            }
        ),
    )
    print("\n" + "=" * 20 + "Generated Novel Outline and Chapter Structure" + "=" * 20)
    print(f"\nOverall Outline: {outline.novel_outline}")
    print("\nChapter Structure:")
    for chapter in outline.chapter_structure:
        print(f"- {chapter.chapter_name}: {chapter.chapter_description}")
    print_process_progress("Outline Generation", "(Completed) ✅")
    return {
        "novel_outline": outline.novel_outline,
        "chapter_structure": outline.chapter_structure,
        "current_stage": "Outline Generation",
    }


confirm_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please update the novel outline and chapter structure based on the confirmed basic setting and the user's modification requests:

Basic Setting:
Title: {novel_title}
Main Characters: {main_characters}
Plot Overview: {plot_overview}
Modification Requests: {modify_content}
""",
        )
    ]
)
confirm_outline_agent = confirm_outline_prompt | outline_llm


def confirm_outline_chapter_node(state: NovelCreationState) -> StateUpdate:
    if not state.main_characters:
        message = "❌ The basic setting has not been confirmed, so the outline cannot be generated!"
        raise RuntimeError(message)

    print("\n" + "=" * 20 + "⚠️ Human Review - Outline and Chapter Structure Confirmation Stage" + "=" * 20)
    confirm = input(
        "Do you confirm the outline and chapter structure above? (Enter y to confirm, or enter n and provide the changes you would like to make): "
    ).lower()
    if confirm == "y":
        print("✅ The outline and chapter structure have been confirmed. Proceeding to the novel generation stage!")
        return {"is_outline_confirmed": True}
    modify_content = input(
        "Please enter your modification requests (e.g., adjust the chapter order / modify the plot of a specific chapter / add or remove chapters): "
    )
    print("🔄 Updating the outline and chapter structure based on your requirements...")
    character_str = "\n".join([f"{character.name}: {character.description}" for character in state.main_characters])
    outline = cast(
        "OutlineOutput",
        confirm_outline_agent.invoke(
            {
                "novel_title": state.novel_title,
                "main_characters": character_str,
                "plot_overview": state.plot_overview,
                "modify_content": modify_content,
            }
        ),
    )
    print("\n" + "=" * 20 + "Modified Outline and Chapter Structure" + "=" * 20)
    print(f"\nOverall Outline: {outline.novel_outline}")
    print("\nChapter Structure:")
    for chapter in outline.chapter_structure:
        print(f"- {chapter.chapter_name}: {chapter.chapter_description}")
    re_confirm = input("Do you confirm the modified outline and chapter structure? (y/n): ").lower()
    if re_confirm == "y":
        print("✅ The outline and chapter structure have been confirmed!")
        return {
            "is_outline_confirmed": True,
            "chapter_structure": outline.chapter_structure,
            "novel_outline": outline.novel_outline,
        }
    print("❌ Not confirmed. The outline will be regenerated.")
    return {"is_outline_confirmed": False}


generate_complete_novel_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
Please generate the main text for the specified chapter based on the novel's basic setting and overall outline. Requirements:

Strictly follow the plot summary of the current chapter, with rich details and a novelistic writing style.
Keep each character's personality consistent with the basic setting. Dialogue should feel natural, and actions and inner thoughts should fit the characters.
Begin the chapter with the chapter title, and end with a slight transition that sets up the next chapter.
Keep each chapter between 200 and 400 words, with fluent language and coherent storytelling.

Novel Basic Setting: {novel_basic_info}
Current Chapter: {chapter_name}
Plot Summary of This Chapter: {chapter_description}
Generated Chapters: {generated_chapter_num}/{total_chapter}

Output format: Directly output the generated chapter text without any additional explanation.
""",
        )
    ]
)
generate_complete_novel_agent = generate_complete_novel_prompt | chat_llm | output_parser


def generate_complete_novel_node(state: NovelCreationState) -> StateUpdate:
    if not state.is_outline_confirmed or not state.chapter_structure or not state.main_characters:
        message = "❌ The outline and chapter structure have not been confirmed, so the novel cannot be generated!"
        raise RuntimeError(message)
    print_process_progress("Novel Generation", "(Start generating the novel chapter by chapter)")

    chapter_generated_count = 0
    chapter_total = len(state.chapter_structure)
    print_chapter_progress(0, chapter_total)

    character_str = "\n".join([f"{character.name}: {character.description}" for character in state.main_characters])
    novel_basic_info = f"""
Novel Title: {state.novel_title}
Main Characters: {character_str}
Overall Outline: {state.novel_outline}
"""
    full_novel_content = f"# {state.novel_title}\n\n## Novel basic info\n{novel_basic_info}\n\n---------\n"
    for index, chapter in enumerate(state.chapter_structure, 1):
        chapter_name = chapter.chapter_name
        chapter_description = chapter.chapter_description
        print(f"\n🔨 [Generating] {chapter_name}...")

        chapter_content = generate_complete_novel_agent.invoke(
            {
                "novel_basic_info": novel_basic_info,
                "chapter_name": chapter_name,
                "chapter_description": chapter_description,
                "generated_chapter_num": index,
                "total_chapter": chapter_total,
            }
        ).strip()

        full_novel_content += f"\n{chapter_content}\n\n---------\n"
        chapter_generated_count = index
        print_chapter_progress(index, chapter_total)
        print("\n" + "=" * 20 + f"✅ [Generation Complete] {chapter_name}:\n{chapter_content}\n" + "=" * 20)

    full_novel_content += f"\n### Novel Completed (Total Chapters: {chapter_total} | Created Based on User Requirements: {state.user_requirement})"
    print_process_progress("Novel Generation", "(Completed) ✅")
    print(
        f"\n🎉 Chapter-by-chapter generation completed! The novel contains {chapter_total} chapters, with a total word count of at least 2,000 words."
    )
    return {
        "complete_novel": full_novel_content,
        "chapter_generated_count": chapter_generated_count,
        "current_stage": "Novel Generation",
    }


def build_novel_creation_graph() -> CompiledStateGraph:
    builder = StateGraph(NovelCreationState)

    builder.add_node("user_input", user_input_node)
    builder.add_node("generate_basic_setting", generate_basic_setting_node)
    builder.add_node("confirm_basic_setting", confirm_basic_setting_node)
    builder.add_node("generate_outline_chapter", generate_outline_chapter_node)
    builder.add_node("confirm_outline_chapter", confirm_outline_chapter_node)
    builder.add_node("generate_complete_novel", generate_complete_novel_node)

    builder.add_edge(START, "user_input")
    builder.add_edge("user_input", "generate_basic_setting")
    builder.add_edge("generate_basic_setting", "confirm_basic_setting")

    def setting_confirm_router(
        state: NovelCreationState,
    ) -> Literal["generate_outline_chapter", "generate_basic_setting"]:
        return "generate_outline_chapter" if state.is_setting_confirmed else "generate_basic_setting"

    builder.add_conditional_edges("confirm_basic_setting", setting_confirm_router)

    builder.add_edge("generate_outline_chapter", "confirm_outline_chapter")

    def outline_confirm_router(
        state: NovelCreationState,
    ) -> Literal["generate_complete_novel", "generate_outline_chapter"]:
        return "generate_complete_novel" if state.is_outline_confirmed else "generate_outline_chapter"

    builder.add_conditional_edges("confirm_outline_chapter", outline_confirm_router)

    builder.add_edge("generate_complete_novel", END)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer, interrupt_before=["confirm_basic_setting", "confirm_outline_chapter"]
    )
    return graph


if __name__ == "__main__":
    graph = build_novel_creation_graph()
    thread_id = "novel_creation_demo"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    initial_state = NovelCreationState()
    print("\n" + "=" * 20 + "🚀 Novel Creation Assistant Launched" + "=" * 20)
    print("=" * 40)

    graph.invoke(initial_state, config)
    while True:
        state_snapshot = graph.get_state(config)
        if not state_snapshot.next:
            print("\n🎉 All processes have been completed!")
            break
        target_node = state_snapshot.next[0]
        print("\n" + "=" * 20 + f"⏸️ The workflow is waiting for human intervention at node [{target_node}]" + "=" * 20)
        graph.invoke(None, config)

    final_state = graph.get_state(config).values
    complete_novel = final_state.get("complete_novel")
    if complete_novel:
        filename = Path("novel.md")
        filename.write_text(complete_novel, encoding="utf-8")
        print(f"📁 The complete novel has been saved to: {filename}")
    else:
        print("⚠️ The workflow was unable to generate the complete content.")
