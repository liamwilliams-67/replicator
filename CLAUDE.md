# CLAUDE.md

Guidance for Claude Code working in this repo. Read [PLAN.md](./PLAN.md) for the
full design; this file is the quick operational map.

## What this is
**Replicator** — a Discord bot finetuned to talk like our friend group's server
(aggressive/mature, *not* a friendly assistant). Pipeline: scrape the server →
QLoRA-finetune `Qwen/Qwen3.5-0.8B-Base` → serve a quantized GGUF on a CPU box.

**Status:** planning. No code yet — PLAN.md and this file only.

## Architecture in one breath
`/init` scrapes messages → JSONL (collect) → QLoRA on a **4060** (train) →
merge + GGUF + quantize → `llama.cpp` on a **CPU-only box** (serve). The two
machines are different: train on GPU, serve on CPU.

## Locked-in decisions (don't relitigate without asking)
- **Serve on CPU** via `llama.cpp`/GGUF (4 threads, 24GB RAM). Not GPU.
- **One blended group voice**, not per-user impersonation.
- **Model decides whether to reply** by emitting a `REPLY`/`IGNORE` sentinel as
  its first token. `IGNORE` = 1 token then stop (cheap). **Always** reply when
  @mentioned or replied-to (runtime primes `REPLY:` to force it).
- **No output content filter** (raw mimicry). Don't add one unless asked.
- **Base** model, not Instruct — the finetune supplies the personality.

## Things that will bite you
- **Top risk:** `llama.cpp` may not support Qwen3.5-small's MoE/Gated-DeltaNet
  arch for GGUF yet. **De-risk early** (PLAN.md M3) before any long train run.
  Fallback model: `Qwen/Qwen3-0.6B-Base`.
- Pin `transformers`/`unsloth`/`trl` versions — the arch is new.
- `REPLY`/`IGNORE` and `[GIF:tag]` sentinels must survive finetune→merge→GGUF→
  quantize. Use literal strings / verified reserved tokens + a round-trip test.
- Discord **Message Content Intent** (privileged) must be on, or scraped content
  is empty.
- Keep inference context short (≤1–2k tokens) — CPU prefill is the cost.

## Conventions
- Python. Layout in PLAN.md §12: `bot/`, `data_pipeline/`, `training/`,
  `assets/`, `data/` (git-ignored), `models/` (git-ignored).
- Secrets in `.env` (git-ignored), never committed. Tunables in `config.yaml`.
- `data/`, `models/`, `.env` stay out of git (raw chat logs + weights).
- Match surrounding style; keep modules small and single-purpose.

## Commands
_None yet — fill in as the project is scaffolded (M0)._
- run bot: `TBD`
- scrape: `/init` (in Discord, admin-only)
- preprocess / finetune / export: `TBD`

## Git workflow
- Work on branch **`claude/beautiful-ptolemy-wdavzi`**; create it locally if missing.
- Clear, descriptive commits. Push with `git push -u origin <branch>`.
- **Do not open a PR unless explicitly asked.**
