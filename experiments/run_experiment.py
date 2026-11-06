"""Headless, multi-environment experiment runner.

Plays N fully-autonomous AI-vs-AI Codenames games against one or more LLM
backends (local Ollama, an Azure GPU VM, a RunPod pod - anything that speaks
the OpenAI-compatible chat API) and writes structured per-step JSON, one file
per environment, so results from different environments/hardware can be
compared later.

Usage:
    python experiments/run_experiment.py --env local --games 3
    python experiments/run_experiment.py --env local azure runpod --games 15
    python experiments/run_experiment.py --games 15   # runs every configured env
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.llm_agent import LLMOperative, LLMSpymaster  # noqa: E402
from engine.board import Team  # noqa: E402
from engine.game import Game, Phase  # noqa: E402
from llm.client import LLMClient  # noqa: E402
from words import WORD_POOL  # noqa: E402

PRINT_LOCK = threading.Lock()


def log(env_name: str, msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{env_name}] {msg}", flush=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def play_one_game(client: LLMClient, game_id: int, starting_team: Team, max_steps: int, on_step=None) -> dict:
    game = Game.new(WORD_POOL, starting_team=starting_team)
    agents = {
        (Team.RED, "spymaster"): LLMSpymaster(Team.RED, client),
        (Team.RED, "operative"): LLMOperative(Team.RED, client),
        (Team.BLUE, "spymaster"): LLMSpymaster(Team.BLUE, client),
        (Team.BLUE, "operative"): LLMOperative(Team.BLUE, client),
    }

    steps = []
    game_start = time.perf_counter()
    step_index = 0

    while game.phase != Phase.GAME_OVER and step_index < max_steps:
        step_index += 1
        team = game.current_team
        started_at = iso_now()

        if game.phase == Phase.CLUE:
            agent = agents[(team, "spymaster")]
            word, count = agent.get_clue(game)
            game.give_clue(team, word, count)
            step = {
                "step_index": step_index,
                "team": team.value,
                "role": "spymaster",
                "started_at": started_at,
                "duration_sec": round(agent.last_duration_sec, 3),
                "attempts": agent.last_attempts,
                "tier": agent.last_tier,
                "prompt_tokens": agent.last_usage.get("prompt_tokens"),
                "completion_tokens": agent.last_usage.get("completion_tokens"),
                "reasoning": agent.last_reasoning,
                "clue_word": word,
                "clue_count": count,
                "guess_word": None,
                "guess_outcome": None,
            }
        else:
            agent = agents[(team, "operative")]
            guess = agent.get_guess(game)
            if guess is None:
                game.pass_turn(team)
                outcome = "pass"
            else:
                outcome = game.guess(team, guess).value
            step = {
                "step_index": step_index,
                "team": team.value,
                "role": "operative",
                "started_at": started_at,
                "duration_sec": round(agent.last_duration_sec, 3),
                "attempts": agent.last_attempts,
                "tier": None,
                "prompt_tokens": agent.last_usage.get("prompt_tokens"),
                "completion_tokens": agent.last_usage.get("completion_tokens"),
                "reasoning": agent.last_reasoning,
                "clue_word": None,
                "clue_count": None,
                "guess_word": guess,
                "guess_outcome": outcome,
            }

        steps.append(step)
        if on_step:
            on_step(step)

    timed_out = game.phase != Phase.GAME_OVER
    return {
        "game_id": game_id,
        "starting_team": starting_team.value,
        "outcome": "timeout" if timed_out else "completed",
        "winner": game.winner.value if game.winner else None,
        "total_steps": len(steps),
        "total_duration_sec": round(time.perf_counter() - game_start, 3),
        "steps": steps,
    }


def summarize(games: list[dict]) -> dict:
    completed = [g for g in games if g["outcome"] == "completed"]
    all_steps = [s for g in games for s in g["steps"]]
    durations = [s["duration_sec"] for s in all_steps]
    prompt_tokens = [s["prompt_tokens"] for s in all_steps if s["prompt_tokens"] is not None]
    completion_tokens = [s["completion_tokens"] for s in all_steps if s["completion_tokens"] is not None]
    retried_steps = sum(1 for s in all_steps if s["attempts"] > 1)
    spymaster_tiers = [s["tier"] for s in all_steps if s["role"] == "spymaster"]
    tier_counts = {t: spymaster_tiers.count(t) for t in ("holistic", "safe_single", "deterministic")}

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "games_played": len(games),
        "games_completed": len(completed),
        "games_timed_out": sum(1 for g in games if g["outcome"] == "timeout"),
        "games_errored": sum(1 for g in games if g["outcome"] == "error"),
        "red_wins": sum(1 for g in games if g.get("winner") == "RED"),
        "blue_wins": sum(1 for g in games if g.get("winner") == "BLUE"),
        "avg_steps_per_game": avg([g["total_steps"] for g in games]),
        "avg_duration_sec_per_game": avg([g["total_duration_sec"] for g in games]),
        "avg_duration_sec_per_step": avg(durations),
        "avg_prompt_tokens_per_call": avg(prompt_tokens),
        "avg_completion_tokens_per_call": avg(completion_tokens),
        "retry_rate": round(retried_steps / len(all_steps), 3) if all_steps else None,
        "spymaster_tier_counts": tier_counts,
        # share of clues that needed anything beyond the first-choice holistic tier
        "spymaster_fallback_rate": (
            round(1 - tier_counts["holistic"] / len(spymaster_tiers), 3) if spymaster_tiers else None
        ),
    }


def run_environment(name: str, cfg: dict, num_games: int, max_steps: int, out_dir: Path) -> None:
    if "REPLACE_WITH" in cfg.get("base_url", ""):
        log(name, f"SKIPPED - base_url still a placeholder ({cfg['base_url']}). Fill in experiments/environments.json first.")
        return

    client = LLMClient(base_url=cfg["base_url"], model=cfg["model"])
    log(name, f"connectivity check against {cfg['base_url']} (model={cfg['model']})...")
    try:
        client.chat([{"role": "user", "content": "Reply with the single word: ready"}], tag="healthcheck")
    except Exception as e:
        log(name, f"FATAL - could not reach this environment: {e}")
        return
    log(name, "connected. starting games.")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = iso_now()
    out_path = out_dir / f"{name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"

    def on_step(s):
        detail = f"{s['clue_word']} {s['clue_count']}" if s["role"] == "spymaster" else f"{s['guess_word']} -> {s['guess_outcome']}"
        log(name, f"  game {i} step {s['step_index']}: {s['team']} {s['role']} {detail} ({s['duration_sec']}s, {s['attempts']} attempt(s){', tier=' + s['tier'] if s['tier'] else ''})")

    games = []
    for i in range(1, num_games + 1):
        starting_team = Team.RED if i % 2 == 1 else Team.BLUE
        game_start = time.time()
        try:
            record = play_one_game(client, i, starting_team, max_steps, on_step=on_step)
        except Exception as e:
            log(name, f"game {i}/{num_games} ERRORED: {e}")
            record = {"game_id": i, "starting_team": starting_team.value, "outcome": "error",
                       "error": str(e), "winner": None, "total_steps": 0, "total_duration_sec": 0, "steps": []}
        else:
            elapsed = time.time() - game_start
            log(name, f"game {i}/{num_games}: {record['outcome']}, winner={record['winner']}, "
                       f"steps={record['total_steps']}, {elapsed:.1f}s")
        games.append(record)

        out_path.write_text(json.dumps({
            "environment": name,
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "started_at": run_started_at,
            "updated_at": iso_now(),
            "games": games,
            "summary": summarize(games),
        }, indent=2))

    log(name, f"done. results written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", nargs="+", default=None, help="environment names from environments.json (default: all)")
    parser.add_argument("--games", type=int, default=15, help="games to play per environment (default: 15)")
    parser.add_argument("--max-steps", type=int, default=80, help="safety cap on steps per game (default: 80)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "environments.json")
    args = parser.parse_args()

    all_envs = json.loads(args.config.read_text())
    selected = args.env or list(all_envs.keys())
    missing = [e for e in selected if e not in all_envs]
    if missing:
        parser.error(f"unknown environment(s) {missing}; available: {list(all_envs.keys())}")

    threads = []
    for name in selected:
        t = threading.Thread(
            target=run_environment,
            args=(name, all_envs[name], args.games, args.max_steps, args.out_dir),
            name=f"env-{name}",
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("\nAll environments finished.")


if __name__ == "__main__":
    main()
