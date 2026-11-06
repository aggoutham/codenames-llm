from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agents.base import HumanOperative, HumanSpymaster
from agents.llm_agent import LLMOperative, LLMSpymaster
from engine.board import CardColor, Team
from engine.game import Game, InvalidMove, LogEntry, Phase
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
    model = st.sidebar.text_input("Model name", value="qwen2.5:7b-instruct-q4_K_M")

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


def render_counts_banner(game: Game):
    """Big bold remaining-card counts by color, always visible regardless of whose turn it is."""
    counts = [
        ("RED", game.board.remaining_for(CardColor.RED), COLOR_HEX[CardColor.RED]),
        ("BLUE", game.board.remaining_for(CardColor.BLUE), COLOR_HEX[CardColor.BLUE]),
        ("NEUTRAL", game.board.remaining_for(CardColor.NEUTRAL), COLOR_HEX[CardColor.NEUTRAL]),
        ("ASSASSIN", game.board.remaining_for(CardColor.ASSASSIN), COLOR_HEX[CardColor.ASSASSIN]),
    ]
    cols = st.columns(4)
    for col, (label, count, color_hex) in zip(cols, counts):
        col.markdown(
            f"<div style='background:{color_hex};border-radius:10px;padding:10px 4px;"
            f"text-align:center;margin-bottom:10px;'>"
            f"<div style='font-size:2.2rem;font-weight:800;color:white;line-height:1.1;'>{count}</div>"
            f"<div style='font-size:0.85rem;font-weight:700;color:white;letter-spacing:0.05em;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_turn_banner(game: Game):
    """Big, bold, team-colored 'whose turn' + clue display - the log/debug view already
    has this info in small text, but it shouldn't take hunting to see it during play."""
    team = game.current_team
    color_hex = COLOR_HEX[CardColor.RED if team == Team.RED else CardColor.BLUE]
    role = "spymaster" if game.phase == Phase.CLUE else "operative"

    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:800;color:{color_hex};'>"
        f"{team.value} {role}'s turn</div>",
        unsafe_allow_html=True,
    )
    if game.phase == Phase.GUESS and game.clue:
        st.markdown(
            f"<div style='font-size:1.7rem;font-weight:700;margin-bottom:0.5rem;'>"
            f"Clue: {game.clue.word} {game.clue.count} "
            f"<span style='font-weight:400;font-size:1.1rem;'>"
            f"({game.clue.guesses_left} guess(es) left)</span></div>",
            unsafe_allow_html=True,
        )


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
    """Advance one LLM-driven action. Returns True if it advanced, False if waiting on a human
    or if this step turned out to be stale (see the step_lock in main() for why that can happen)."""
    team = game.current_team
    if game.phase == Phase.CLUE:
        agent = agents[(team, "spymaster")]
        if isinstance(agent, HumanSpymaster):
            return False
        word, count = agent.get_clue(game)
        try:
            game.give_clue(team, word, count)
        except InvalidMove:
            # Another concurrent run (e.g. a second click before this one finished)
            # already advanced the game underneath us - this move is stale, drop it.
            return False
        _log_reasoning(game, team, "spymaster", agent.last_reasoning)
        return True
    if game.phase == Phase.GUESS:
        agent = agents[(team, "operative")]
        if isinstance(agent, HumanOperative):
            return False
        guess = agent.get_guess(game)
        try:
            if guess is None:
                game.pass_turn(team)
            else:
                game.guess(team, guess)
        except InvalidMove:
            return False
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
    show_log = st.sidebar.toggle("📜 Show log", value=True)

    if "game" not in st.session_state:
        st.session_state.game = Game.new(WORD_POOL, starting_team=Team.RED)
        st.session_state.client = LLMClient()
        st.session_state.agents = build_agents(cfg, st.session_state.client)
        # Streamlit doesn't serialize concurrent script runs for one session - a second
        # click while a slow LLM call is still in flight can start a second run against
        # this same session_state. This lock makes Step/Auto-play mutually exclusive so
        # only one is ever actually mutating `game` at a time.
        st.session_state.step_lock = threading.Lock()

    game: Game = st.session_state.game
    agents = st.session_state.agents

    if debug_view:
        render_debug_view(st.session_state.client)
        return

    left, right = st.columns([3, 2]) if show_log else (st.container(), None)
    with left:
        render_counts_banner(game)

        if game.phase == Phase.GAME_OVER:
            st.success(f"Game over! Winner: {game.winner.value}")
        else:
            render_turn_banner(game)

        render_board(game)

        if game.phase != Phase.GAME_OVER:
            c1, c2 = st.columns(2)
            if c1.button("Step"):
                if st.session_state.step_lock.acquire(blocking=False):
                    try:
                        step_llm_turn(game, agents)
                    finally:
                        st.session_state.step_lock.release()
                else:
                    st.toast("Still processing the previous action - please wait.")
                st.rerun()
            if c2.button("Auto-play until human input needed"):
                if st.session_state.step_lock.acquire(blocking=False):
                    try:
                        for _ in range(50):
                            if game.phase == Phase.GAME_OVER:
                                break
                            if not step_llm_turn(game, agents):
                                break
                    finally:
                        st.session_state.step_lock.release()
                else:
                    st.toast("Still processing the previous action - please wait.")
                st.rerun()

            team = game.current_team
            if game.phase == Phase.CLUE and isinstance(agents[(team, "spymaster")], HumanSpymaster):
                human_clue_form(game, team)
            elif game.phase == Phase.GUESS and isinstance(agents[(team, "operative")], HumanOperative):
                human_guess_form(game, team)

    if show_log:
        with right:
            render_log(game)


if __name__ == "__main__":
    main()
