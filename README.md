# Codenames LLM

I built this to learn how small, free, open-source LLMs can be put to work on a narrow, focused job — here, playing Codenames as both the **spymaster** (gives a one-word clue and a count) and the **operative** (guesses words from that clue). The game rules, the UI and a benchmarking harness all live in this repo; the models only ever have to produce a clue or a guess as JSON.

![Codenames LLM game UI: remaining-card counts, turn and clue banner, board, and move log](pictures/screenshot.png)

## Highlights

- **Full Codenames rules engine** — 5x5 board (9/8/7/1 split), turn state machine, assassin and win conditions. Pure Python with no LLM or UI code in it.
- **Any seat can be human or LLM.** I can play a seat myself, or let two teams of small models play each other end to end.
- **Spymaster that never skips a turn.** A three-tier clue strategy (below) guarantees a legal clue every turn, even from a weak model.
- **Compact-history prompts.** Each model call gets the game history as a short JSON log plus the *current* board state — not an ever-growing transcript.
- **Streamlit UI** with big remaining-card counts, a team-colored turn and clue banner, a move log, and a **debug view** showing the raw prompt, raw response and token counts of every model call.
- **OpenAI-compatible client**, so the same code talks to Ollama locally, Ollama on a rented GPU, or anything else that speaks that API — just change the `base_url`.
- **Headless benchmark harness** that plays batches of fully autonomous games and writes per-step timing, retry and token data to JSON.

## How it works

**Engine (`engine/`)** — board generation, turn order, guess resolution (correct card, opponent card, neutral, assassin) and win detection. Every log entry is timestamped.

**Agents (`agents/`)**
- *Spymaster* sees every card's true color and tries three tiers in order: (1) a holistic clue that can target several of its words, (2) if that keeps getting rejected, a narrowed ask for one easy, safe single-word clue with the opponent, neutral and assassin words listed explicitly, (3) a deterministic fixed fallback word with no LLM call. Clues are validated (single alphabetic word, no overlap with any board word) and retried with feedback.
- *Operative* sees only the unrevealed words and the clue, guesses one word at a time, and can stop early by answering `null`.
- *Prompts* are rebuilt fresh on every call from `[system, user]`, where the user turn carries a JSON history (`{"step", "team", "role", "action"}` per move) and the current state listing **only unrevealed words**. A revealed word can't leak back into a later prompt.
- Each agent records how long its last move took, how many attempts it needed, which tier produced the clue, its reasoning, and token usage.

**LLM client (`llm/client.py`)** — thin wrapper over the OpenAI chat API with JSON extraction and repair, plus a rolling trace of every raw request and response that the debug view displays.

**UI (`ui/app.py`)** — Step / Auto-play controls (guarded by a lock so a second click can't race a slow in-flight model call), per-seat human/LLM checkboxes, a log toggle, and a debug toggle.

## Run it locally

Requires Python 3.9+ and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

brew install ollama
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M

streamlit run ui/app.py
```

In the sidebar, tick any seat you want to play yourself (unticked = LLM) and click **New game**; seat changes take effect on the next New game. The default model is `qwen2.5:7b-instruct-q4_K_M`.

## Benchmark harness

`experiments/run_experiment.py` plays batches of AI-vs-AI games with no UI, against any environment defined in `experiments/environments.json` (a name, a `base_url` and a model). Because the client only needs an OpenAI-compatible endpoint, the same harness runs unchanged against local Ollama, a GPU VM, or a RunPod pod, so different hardware can be compared on identical games.

```bash
python experiments/run_experiment.py --env local --games 15
python experiments/run_experiment.py --env local runpod --games 15   # environments run concurrently
```

- Starting team alternates game to game so first-move advantage doesn't skew win rates.
- A step cap (default 80) ends a game as `timeout` instead of hanging if both sides degenerate into never revealing a card.
- Results are written incrementally to `results/<env>_<timestamp>.json`, so a crash never loses completed games.
- **Per step:** team, role, start time, duration, attempts, spymaster tier, prompt/completion tokens, reasoning, clue/guess and outcome.
- **Per environment summary:** games completed / timed out / errored, red vs. blue wins, average steps per game, average seconds per step and per game, average tokens per call, retry rate, and how often the spymaster needed its fallback tiers.
- Model calls use an explicit timeout and retry limit (180 s and 1 retry by default; override with `LLM_TIMEOUT_SEC` / `LLM_MAX_RETRIES`), so a hung remote host fails fast and silent retries can't distort timings.

**Pointing it at a remote GPU** — on the GPU machine:

```bash
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=4096 ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Then set that environment's `base_url` in `environments.json`: `http://<host>:11434/v1` for a VM, or `https://<POD_ID>-11434.proxy.runpod.net/v1` for a RunPod pod with HTTP port 11434 exposed. Pin the same `OLLAMA_CONTEXT_LENGTH` on every host so context size isn't a hidden variable between environments. Ollama has no authentication, so restrict access (firewall or security group) or shut the machine down when finished.

## What I learned

- **Model size decides whether the task is even possible.** A 3B model kept answering with words already on the board; the 7B model followed the "don't name a board word" rule with no prompt changes.
- **A growing chat transcript made the models worse.** Old board snapshots stayed in context and the model kept considering already-revealed words. A fresh prompt per call (compact history plus current state) removes that failure by construction.
- **A spymaster must never pass.** With a pass fallback, two weak spymasters could pass forever and the game never progressed — hence the three-tier clue strategy and the harness's step cap.
- **Blocking model calls and Streamlit don't mix without care.** A second click during a slow call started a concurrent run against the same game object; a per-session lock fixed it.
- **CPU inference is workable for turn-based play but slow.** On my CPU-only Intel MacBook a 7B model takes roughly 15–25 seconds per clue or guess — fine for playing, too slow for batch benchmarking, which is what the remote-GPU setup above is for.

## Project layout

```
engine/       game rules: board, turns, win conditions
agents/       LLM and human seats, prompts, spymaster tiers
llm/          OpenAI-compatible client with raw call trace
ui/           Streamlit app
experiments/  benchmark harness and environments.json
words.py      generic board word pool (not the official Codenames list)
```
