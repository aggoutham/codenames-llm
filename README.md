# Codenames LLM Mockup

Small open-source LLMs play spymaster and operative against each other (or against you).
Game rules and turn logic are pure Python; the LLM only ever has to produce a clue or a
guess as JSON.

## 1. Setup

```bash
cd /Users/aggoutham/projects/codenames
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install Ollama and pull a small model

```bash
brew install ollama
ollama serve &          # starts the local API on http://localhost:11434
ollama pull qwen2.5:3b  # ~2GB, fast enough on CPU for wiring/testing
# once wiring works, try a stronger model for real clue/guess quality:
# ollama pull qwen2.5:7b-instruct-q4_K_M
```

On your Intel MacBook this runs CPU-only (~3-8 tok/s for a 7B model) — fine for
turn-based play, slow for rapid prompt iteration. That's expected; see the Azure
section below for when to move inference to a GPU.

## 3. Run the game

```bash
streamlit run ui/app.py
```

In the sidebar: set the model name (must match what you `ollama pull`ed), check any
seat you want to play yourself (unchecked = LLM), then click **New game**. Use
**Step** to advance one action at a time, or **Auto-play** to run all-LLM turns until
it needs human input.

## 4. Moving to Azure GPU (later)

No code changes needed — the LLM client speaks the OpenAI-compatible chat API, which
both Ollama and vLLM implement identically. Once you stand up a `vllm serve <model>`
endpoint on an Azure GPU VM, just change the sidebar's **LLM base URL** to
`http://<vm-ip>:8000/v1` and the model name to match.

Recommended VM: `Standard_NC4as_T4_v3` (1x T4, 16GB VRAM) — a 7-8B 4-bit model only
needs ~6GB. On-demand is ~$0.53/hr, spot is ~$0.10-0.22/hr. **Deallocate the VM when
you're not actively testing** — Azure only bills compute while it's running.
