import random
from collections import Counter
from typing import Literal, NotRequired, TypedDict, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class WordPair(BaseModel):
    civilian_word: str
    undercover_word: str


class SpeechData(BaseModel):
    speech: str
    reason: str


class VoteData(BaseModel):
    vote: str
    vote_reason: str


Role = Literal["civilian", "undercover"]


class GameState(BaseModel):
    civilian_word: str | None = None
    undercover_word: str | None = None
    role_assignment: dict[str, tuple[Role, str]] | None = None
    curr_speeches: dict[str, str] | None = None
    history_speeches: list[dict[str, str]] | None = None
    speech_reasoning: dict[str, str] | None = None
    votes: dict[str, str] | None = None
    vote_reasoning: dict[str, str] | None = None
    game_status: str = "running"
    winner: Role | None = None
    eliminated: list[str] | None = None
    round: int = 1


class StateUpdate(TypedDict):
    civilian_word: NotRequired[str]
    undercover_word: NotRequired[str]
    role_assignment: NotRequired[dict[str, tuple[Role, str]]]
    curr_speeches: NotRequired[dict[str, str]]
    history_speeches: NotRequired[list[dict[str, str]]]
    speech_reasoning: NotRequired[dict[str, str]]
    votes: NotRequired[dict[str, str]]
    vote_reasoning: NotRequired[dict[str, str]]
    game_status: NotRequired[str]
    winner: NotRequired[Role]
    eliminated: NotRequired[list[str]]
    round: NotRequired[int]


chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-plus"
)
word_pair_llm = chat_llm.with_structured_output(WordPair, method="json_schema")
speech_data_llm = chat_llm.with_structured_output(SpeechData, method="json_schema")
vote_llm = chat_llm.with_structured_output(VoteData, method="json_schema")

generate_words_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional prompt designer for the game "Who's the Undercover." Your task is to generate a high-quality pair of words.

Core requirements:
1. Word types: Everyday objects, foods, or scenarios (e.g., milk tea - juice, toothbrush - toothpaste). Avoid obscure or uncommon words.
2. Semantic relationship: The civilian word and the undercover word should be highly similar but differ in key characteristics, leaving enough room for strategic descriptions and deduction.
3. Difficulty: Suitable for a 4-player game. The difference should not be too obvious, but players should still be able to distinguish the words through their descriptions.
""",
        ),
        ("human", 'Generate a pair of words for "Who\'s the Undercover" that meets the requirements.'),
    ]
)
generate_words_agent = generate_words_prompt | word_pair_llm


def generate_words_node(state: GameState) -> StateUpdate:
    try:
        word_pair = cast("WordPair", generate_words_agent.invoke({}))
        civilian_word = word_pair.civilian_word
        undercover_word = word_pair.undercover_word
    except Exception:
        fallbacks_pairs = [
            ("milk tea", "juice"),
            ("toothbrush", "toothpaste"),
            ("rice", "noodles"),
            ("smartphone", "tablet"),
            ("basketball", "soccer"),
            ("coffee", "black tea"),
        ]
        civilian_word, undercover_word = random.choice(fallbacks_pairs)

    print(f"\n🎯 Word generation complete: Civilian word={civilian_word} | Undercover word={undercover_word}")
    return {"civilian_word": civilian_word, "undercover_word": undercover_word}


def assign_roles_node(state: GameState) -> StateUpdate:
    if not state.undercover_word or not state.civilian_word:
        message = "❌ The role words have not been generated!"
        raise RuntimeError(message)
    agents = ["agent1", "agent2", "agent3", "agent4"]
    role_assignment: dict[str, tuple[Role, str]] = {}
    undercover = random.choice(agents)
    for agent in agents:
        if agent == undercover:
            role_assignment[agent] = ("undercover", state.undercover_word)
        else:
            role_assignment[agent] = ("civilian", state.civilian_word)
    print("\n🎭 Role assignment complete:")
    for agent, (role, word) in role_assignment.items():
        print(f"   {agent}: {role} (Word: {word})")
    return {"role_assignment": role_assignment}


generate_speech_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced player of the game "Who's the Undercover." It is currently round {curr_round}, and you need to develop your strategy based on the previous speeches.

[Core Rules]

1. Speech requirements:

   * Length: Must be strictly between 10 and 100 Chinese characters, excluding punctuation. Do not truncate the output; directly generate a complete response that satisfies the length requirement.
   * Content: Describe the characteristics of the word, but never directly reveal the word itself. Adjust your strategy based on previous speeches and avoid repeating your own or other players' descriptions.
   * Style: Natural and conversational, with complete and fluent sentences and clear logic.
   * Completeness: Make sure the speech consists of complete sentences with complete meaning and is not cut off.

2. Role strategies:

   * Civilian: Describe the core characteristics to help other civilians identify the undercover player. Avoid repeating descriptions from previous rounds and pay attention to players whose statements are inconsistent.
   * Undercover: Imitate the civilians' speaking style and blur the key differences. Avoid contradicting your own statements from previous rounds while keeping your identity hidden.

{history_context}
""",
        ),
        (
            "human",
            """Your player ID: {agent}
Your role: {role}
Your word: {word}

Currently alive players: {curr_agents}
Eliminated players: {eliminated}

In the speech history, entries labeled "{agent}" are your own speeches.
Other player IDs refer to other players.
Eliminated players' previous speeches remain available as historical evidence,
but those players no longer speak or vote.""",
        ),
    ]
)
generate_speech_agent = generate_speech_prompt | speech_data_llm


def generate_speech_node(state: GameState) -> StateUpdate:
    if not state.role_assignment:
        message = "❌ Role assignment has not been completed!"
        raise RuntimeError(message)

    speeches = {}
    reasoning = {}
    curr_round = state.round
    history_context = ""
    if state.history_speeches:
        history_context = "[Previous Speech History]\n"
        for index, round_speeches in enumerate(state.history_speeches, 1):
            history_context += f"Round {index} speeches:\n"
            for agent, speech in round_speeches.items():
                history_context += f"- {agent}: {speech}\n"
        history_context += "\n"

    curr_agents = [agent for agent in state.role_assignment if not state.eliminated or agent not in state.eliminated]
    curr_agents_context = ", ".join(curr_agents)
    eliminated_context = ", ".join(state.eliminated or []) or "None"
    print(f"\n🗣 Round {curr_round} speech phase (recommended length: 10-100 characters):")
    for agent, (role, word) in state.role_assignment.items():
        if state.eliminated and agent in state.eliminated:
            continue
        try:
            speech_data = cast(
                "SpeechData",
                generate_speech_agent.invoke(
                    {
                        "agent": agent,
                        "role": role,
                        "word": word,
                        "history_context": history_context,
                        "curr_agents": curr_agents_context,
                        "eliminated": eliminated_context,
                        "curr_round": curr_round,
                    }
                ),
            )
            raw_speech = speech_data.speech
            raw_reason = speech_data.reason
            raw_speech_length = len(raw_speech)
            if raw_speech_length > 100:
                print(
                    f"⚠️ {agent} ({role})'s speech exceeded 100 characters (actual length: {len(raw_speech)} characters). The full content has been preserved."
                )
            elif raw_speech_length < 10:
                print(
                    f"⚠️ {agent} ({role})'s speech is fewer than 10 characters (actual length: {len(raw_speech)} characters). The full content has been preserved."
                )
        except Exception:
            if role == "civilian":
                raw_speech = f"Round {curr_round} speech: This is something commonly used in daily life and is used very frequently. Different brands may vary slightly in style, but their core function is the same. Almost every household has this kind of item, making it an essential everyday necessity."
                raw_reason = f"Civilian fallback speech: In round {curr_round}, avoid repeating descriptions from previous rounds and fully describe the item's core characteristics without truncating the content."
            else:
                raw_speech = f"Round {curr_round} speech: This is an item everyone is familiar with and is commonly used in many everyday situations. Its appearance and function are quite similar across different types, making them difficult to distinguish quickly. It can be seen everywhere in daily life, and almost everyone has used this kind of item."
                raw_reason = f"Undercover fallback speech: In round {curr_round}, disguise yourself as a civilian and give a complete but intentionally vague description of the item's characteristics to avoid revealing your identity. Do not truncate the content."
        speeches[agent] = raw_speech
        reasoning[agent] = raw_reason

        print(f"\n{agent} ({role})")
        print(f"    Speech: {raw_speech}")
        print(f"    Reason: {raw_reason}")
    history_speeches = list(state.history_speeches or [])
    history_speeches.append(speeches.copy())
    return {"curr_speeches": speeches, "speech_reasoning": reasoning, "history_speeches": history_speeches}


vote_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a rational player of the game "Who's the Undercover." You need to analyze the current round and previous speeches and cast your vote.

[Analysis Rules]

1. Voting criteria:

   * Compare each player's current and previous speeches to identify contradictions or unusual descriptions. Undercover players often contradict themselves across rounds.
   * Civilian: Focus on players whose descriptions are inconsistent across rounds or deviate from the characteristics of the word.
   * Undercover: Vote for a player who appears to be a civilian to avoid drawing suspicion to yourself, while keeping your voting rationale consistent.

{speech_context}""",
        ),
        (
            "user",
            """Current round: {curr_round}

Your player ID: {agent}
Your role: {role}
Your word: {word}

Currently alive players: {curr_agents}
Eliminated players: {eliminated}
Choose the player you want to vote for and explain your reason. Keep the reason within 50 characters.
""",
        ),
    ]
)
vote_agent = vote_prompt | vote_llm


def vote_node(state: GameState) -> StateUpdate:
    if not state.role_assignment:
        message = "❌ Role assignment has not been completed!"
        raise RuntimeError(message)
    if not state.curr_speeches:
        message = "❌ Current speeches have not been generated!"
        raise RuntimeError(message)

    votes: dict[str, str] = {}
    reasons: dict[str, str] = {}
    curr_agents = [agent for agent in state.role_assignment if not state.eliminated or agent not in state.eliminated]
    curr_agents_context = ", ".join(curr_agents)
    eliminated_context = ", ".join(state.eliminated or []) or "None"
    curr_round = state.round

    speech_context = f"[Round {curr_round} Speeches]\n"
    speech_context += "\n".join([f"{agent}: {speech}" for agent, speech in state.curr_speeches.items()])
    if state.history_speeches and len(state.history_speeches) > 1:
        speech_context += "[Previous Speech History]\n"
        for index, round_speeches in enumerate(state.history_speeches[:-1], 1):
            speech_context += f"Round {index} speeches:\n"
            for agent, speech in round_speeches.items():
                speech_context += f"- {agent}: {speech}\n"
        speech_context += "\n"

    print(f"\n🗳 Round {curr_round} voting phase:")
    for agent, (role, word) in state.role_assignment.items():
        if state.eliminated and agent in state.eliminated:
            continue
        candidates = [candidate for candidate in curr_agents if candidate != agent]
        try:
            vote_data = cast(
                "VoteData",
                vote_agent.invoke(
                    {
                        "agent": agent,
                        "role": role,
                        "word": word,
                        "speech_context": speech_context,
                        "curr_agents": curr_agents_context,
                        "eliminated": eliminated_context,
                        "curr_round": curr_round,
                    }
                ),
            )
            vote = vote_data.vote
            reason = vote_data.vote_reason
        except Exception:
            vote = random.choice(candidates)
            reason = (
                f"No valid analysis was available in round {curr_round}, so the vote was made using a random strategy."
            )

        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n{agent} ({role})")
        print(f"   Vote to: {vote}")
        print(f"   Reason: {reason}")

    return {"votes": votes, "vote_reasoning": reasons}


def judge_result_node(state: GameState) -> StateUpdate:
    if not state.votes:
        message = "❌ Vote has not been completed!"
        raise RuntimeError(message)
    if not state.role_assignment:
        message = "❌ Role assignment has not been completed!"
        raise RuntimeError(message)
    vote_count = Counter(state.votes.values())
    max_vote = max(vote_count.values())
    candidates = [agent for agent, count in vote_count.items() if count == max_vote]
    eliminated = []
    new_eliminated = random.choice(candidates)
    if state.eliminated:
        eliminated = state.eliminated.copy()
    eliminated.append(new_eliminated)
    role = state.role_assignment[new_eliminated][0]
    curr_round = state.round
    print(f"\n❌ Round {curr_round} elimination result: {new_eliminated} ({role})")
    remaining = [agent for agent in state.role_assignment if agent not in eliminated]
    civilian_count = sum(1 for agent in remaining if state.role_assignment[agent][0] == "civilian")
    undercover_count = sum(1 for agent in remaining if state.role_assignment[agent][0] == "undercover")
    if role == "undercover":
        game_status = "end"
        winner = "civilian"
        print("🎉 Civilians win!")
        return {"eliminated": eliminated, "game_status": game_status, "winner": winner}
    if civilian_count == 1 and undercover_count == 1:
        game_status = "end"
        winner = "undercover"
        print("🎉 Undercover win!")
        return {"eliminated": eliminated, "game_status": game_status, "winner": winner}
    game_status = "running"
    print(f"➡ The game continues. Entering round {curr_round + 1}.")
    return {"eliminated": eliminated, "game_status": game_status, "round": curr_round + 1}


def show_final_result_node(state: GameState) -> StateUpdate:
    print("\n" + "=" * 30)
    print("📜 Game Over · Summary")
    print(f"Winner: {state.winner}")
    print(f"Civilian word: {state.civilian_word} | Undercover word: {state.undercover_word}")
    print(f"Total round: {state.round}")
    print(f"Eliminated: {state.eliminated}")
    print("=" * 30)
    return {}


def build_game_graph() -> CompiledStateGraph:
    builder = StateGraph(GameState)
    builder.add_node("generate_words", generate_words_node)
    builder.add_node("assign_roles", assign_roles_node)
    builder.add_node("generate_speech", generate_speech_node)
    builder.add_node("vote", vote_node)
    builder.add_node("judge_result", judge_result_node)
    builder.add_node("show_final_result", show_final_result_node)

    builder.add_edge(START, "generate_words")
    builder.add_edge("generate_words", "assign_roles")
    builder.add_edge("assign_roles", "generate_speech")
    builder.add_edge("generate_speech", "vote")
    builder.add_edge("vote", "judge_result")

    def route(state: GameState) -> Literal["generate_speech", "show_final_result"]:
        return "generate_speech" if state.game_status == "running" else "show_final_result"

    builder.add_conditional_edges("judge_result", route)
    builder.add_edge("show_final_result", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_game_graph()
    print("=" * 50)
    print("🎮 Who's the Undercover · Multi-Agent Multi-Round Strategy Version Launched")
    print("=" * 50)
    graph.invoke(GameState())
