from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agents.base import HumanOperative, HumanSpymaster
from agents.llm_agent import LLMOperative, LLMSpymaster
from engine.board import CardColor, Team
from engine.game import Game, LogEntry, Phase
from llm.client import LLMClient
from words import WORD_POOL

st.set_page_config(page_title="Codenames LLM Mockup", layout="wide")

COLOR_HEX = {
    CardColor.RED: "#d9534f",
    CardColor.BLUE: "#428bca",
    CardColor.NEUTRAL: "#d9d0c1",
    CardColor.ASSASSIN: "#2b2b2b",
}
SEAT_KEYS = [
    (Team.RED, "spymaster"),
    (Team.RED, "operative"),
    (Team.BLUE, "spymaster"),
    (Team.BLUE, "operative"),
]


def build_agents(cfg: dict, client: LLMClient) -> dict:
    agents = {}
    for team, role in SEAT_KEYS:
        is_human = cfg[(team, role)]
        if role == "spymaster":
            agents[(team, role)] = HumanSpymaster(team) if is_human else LLMSpymaster(team, client)
        else:
            agents[(team, role)] = HumanOperative(team) if is_human else LLMOperative(team, client)
    return agents


def sidebar_config() -> dict:
    st.sidebar.header("Setup")
    base_url = st.sidebar.text_input("LLM base URL", value="http://localhost:11434/v1")
    model = st.sidebar.text_input("Model name", value="qwen2.5:3b")

    st.sidebar.subheader("Seats (checked = human)")
    cfg = {}
    for team, role in SEAT_KEYS:
        cfg[(team, role)] = st.sidebar.checkbox(f"{team.value} {role}", value=False, key=f"human_{team}_{role}")

    if st.sidebar.button("New game", type="primary"):
        st.session_state.game = Game.new(WORD_POOL, starting_team=Team.RED)
        st.session_state.client = LLMClient(base_url=base_url, model=model)
        st.session_state.agents = build_agents(cfg, st.session_state.client)
        st.session_state.reveal_all = False

    st.sidebar.divider()
    st.session_state.reveal_all = st.sidebar.checkbox(
        "Debug: reveal all colors", value=st.session_state.get("reveal_all", False)
    )
    return cfg


def render_board(game: Game):
    cards = game.board.cards
    for row in range(5):
        cols = st.columns(5)
        for col in range(5):
            card = cards[row * 5 + col]
            show_color = card.revealed or st.session_state.get("reveal_all", False)
            bg = COLOR_HEX[card.color] if show_color else "#f0f0f0"
            fg = "white" if show_color else "black"
            opacity = "0.45" if card.revealed else "1"
            cols[col].markdown(
                f"""<div style="background:{bg};color:{fg};opacity:{opacity};
                border-radius:8px;padding:14px 4px;text-align:center;
                font-weight:600;margin-bottom:6px;">{card.word}</div>""",
                unsafe_allow_html=True,
            )


def render_log(game: Game):
    st.subheader("Log")
    for entry in reversed(game.log[-15:]):
        st.text(f"[{entry.team.value} {entry.role}] {entry.text}")


def render_debug_view(client: LLMClient):
    st.subheader(f"Raw LLM trace — {len(client.trace)} calls this session (model: {client.model})")
    st.caption("Every request/response actually sent to/from the model, no JSON post-processing.")

    if not client.trace:
        st.info("No LLM calls yet - click Step or Auto-play in Game view first.")
        return

    for i, entry in reversed(list(enumerate(client.trace))):
        usage = entry["raw_response"].get("usage") or {}
        label = f"#{i} [{entry['tag']}] tokens: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
        with st.expander(label, expanded=(i == len(client.trace) - 1)):
            st.markdown("**Messages sent** (`messages=` argument to `chat.completions.create`)")
            st.json(entry["messages"])

            st.markdown("**Raw response object** (full `resp.model_dump()`)")
            st.json(entry["raw_response"])

            st.markdown("**Extracted `.choices[0].message.content`**")
            st.code(entry["text"])


def human_clue_form(game: Game, team: Team):
    st.info(f"Waiting on {team.value} spymaster (human) for a clue.")
    with st.form(key=f"clue_form_{len(game.log)}"):
        word = st.text_input("Clue word")
        count = st.number_input("Count", min_value=0, max_value=9, value=1)
        if st.form_submit_button("Give clue"):
            try:
                game.give_clue(team, word, int(count))
            except Exception as e:
                st.error(str(e))
            st.rerun()


def human_guess_form(game: Game, team: Team):
    st.info(f"Waiting on {team.value} operative (human) for a guess.")
    unrevealed = game.board.unrevealed_words()
    col1, col2 = st.columns(2)
    choice = col1.selectbox("Guess", options=["(pass)"] + unrevealed)
    if col2.button("Submit"):
        try:
            if choice == "(pass)":
                game.pass_turn(team)
            else:
                game.guess(team, choice)
        except Exception as e:
            st.error(str(e))
        st.rerun()


def step_llm_turn(game: Game, agents: dict) -> bool:
    """Advance one LLM-driven action. Returns True if it advanced, False if waiting on a human."""
    team = game.current_team
    if game.phase == Phase.CLUE:
        agent = agents[(team, "spymaster")]
        if isinstance(agent, HumanSpymaster):
            return False
        word, count = agent.get_clue(game)
        game.give_clue(team, word, count)
        _log_reasoning(game, team, "spymaster", agent.last_reasoning)
        return True
    if game.phase == Phase.GUESS:
        agent = agents[(team, "operative")]
        if isinstance(agent, HumanOperative):
            return False
        guess = agent.get_guess(game)
        if guess is None:
            game.pass_turn(team)
        else:
            game.guess(team, guess)
        _log_reasoning(game, team, "operative", agent.last_reasoning)
        return True
    return False


def _log_reasoning(game: Game, team: Team, role: str, reasoning: str | None) -> None:
    if reasoning:
        game.log.append(LogEntry(team, f"{role}-reasoning", str(reasoning)))


def main():
    st.title("Codenames: LLM Spymaster vs Operative")
    cfg = sidebar_config()
    st.sidebar.divider()
    debug_view = st.sidebar.toggle("🔍 Debug view (raw prompts/responses)", value=False)

    if "game" not in st.session_state:
        st.session_state.game = Game.new(WORD_POOL, starting_team=Team.RED)
        st.session_state.client = LLMClient()
        st.session_state.agents = build_agents(cfg, st.session_state.client)

    game: Game = st.session_state.game
    agents = st.session_state.agents

    if debug_view:
        render_debug_view(st.session_state.client)
        return

    left, right = st.columns([3, 2])
    with left:
        render_board(game)

        if game.phase == Phase.GAME_OVER:
            st.success(f"Game over! Winner: {game.winner.value}")
        else:
            st.write(f"**Current turn:** {game.current_team.value} — phase: {game.phase.value}")
            c1, c2 = st.columns(2)
            if c1.button("Step"):
                step_llm_turn(game, agents)
                st.rerun()
            if c2.button("Auto-play until human input needed"):
                for _ in range(50):
                    if game.phase == Phase.GAME_OVER:
                        break
                    if not step_llm_turn(game, agents):
                        break
                st.rerun()

            team = game.current_team
            if game.phase == Phase.CLUE and isinstance(agents[(team, "spymaster")], HumanSpymaster):
                human_clue_form(game, team)
            elif game.phase == Phase.GUESS and isinstance(agents[(team, "operative")], HumanOperative):
                human_guess_form(game, team)

    with right:
        render_log(game)


if __name__ == "__main__":
    main()
