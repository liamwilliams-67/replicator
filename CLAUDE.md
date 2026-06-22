# CLAUDE.md

Guidance for Claude Code working in this repo. Read [PLAN.md](./PLAN.md) for the
full design; this file is the quick operational map.

## What this is
**Replicator** — a Discord bot finetuned to talk like our friend group's server
(aggressive/mature, *not* a friendly assistant). Pipeline: scrape the server →
QLoRA-finetune `Qwen/Qwen3.5-0.8B-Base` → serve a quantized GGUF on a CPU box.
The model is a **vision-language model**, so it can also read images when asked.

**Status:** planning. No code yet — PLAN.md and this file only.

## Architecture in one breath
Scrape (offline script + live append) → JSONL (collect) → QLoRA on a **4060** (train) →
merge + GGUF + quantize → `llama.cpp` on a **CPU-only box** (serve), with a
**weekly retrain** loop. Train on GPU, serve on CPU — two different machines.

## Locked-in decisions (don't relitigate without asking)
- **Serve on CPU** via `llama.cpp`/GGUF (4 threads, 24GB RAM), **text-only by
  default**; vision is lazy/opt-in via an `mmproj` projector. Not GPU.
- **One blended group voice** (cohesive + human-sounding, not per-user, not a
  sanitized committee).
- **Model decides whether to reply** by emitting a `REPLY`/`IGNORE` sentinel as
  its first token. `IGNORE` = 1 token then stop (cheap). **Always** reply when
  @mentioned or replied-to (runtime primes `REPLY:` to force it).
- **`activity` dial (0–1)** is the master chattiness knob: `0` = only mentions/
  replies, `1` = nearly every message. Implemented as a threshold on the model's
  REPLY/IGNORE probability; a don't-interrupt backoff still avoids cutting into
  active human exchanges. Settable per-channel. **`/sleep`·`/wake`** hard-mutes
  (persisted).
- **Commands are open to everyone** (you're not a server admin) — only the heavy
  scrape is owner-gated / run offline.
- **No output content filter** (raw mimicry; 18+ in scope). Only hard line is
  illegal content. Don't add a filter unless asked.
- **Base** model, not Instruct — the finetune supplies the personality.
- **Tiered memory** (recent window + retrieval over all history + events digest +
  metadata header). The 262k window is NOT a per-reply budget — feed ~1.5–3k tok.
- **Weekly full retrain from base** (recency-weighted, eval-gated, hot-swap).

## Things that will bite you
- **Top risk:** `llama.cpp` may not support Qwen3.5's MoE/Gated-DeltaNet arch (or
  the VL `mmproj`) for GGUF yet. **De-risk early** (PLAN.md M3) before any long
  train run. Fallbacks: `Qwen/Qwen3-0.6B-Base` (+ `Qwen3-VL` for vision).
- **Never train on the bot's own output** — exclude bot messages as targets, or it
  mimics itself and degenerates over weekly retrains.
- The **weekly retrain needs an eval gate + rollback** — never auto-ship a model
  that regresses on the frozen holdout.
- Pin `transformers`/`unsloth`/`trl` versions — the arch is new.
- `REPLY`/`IGNORE` and `[GIF:tag]` sentinels must survive finetune→merge→GGUF→
  quantize, and not collide with user text. Use rare strings/reserved tokens +
  escaping + a round-trip test.
- **Debounce** message bursts (reply per finished turn, not per message).
- Decode/re-encode Discord **mentions, custom emoji, stickers** both directions.
- Discord **Message Content Intent** (privileged) must be on, or scraped content
  is empty. Scrape **threads + forum channels** too.
- Keep inference context short (≤~3k tokens) — CPU prefill is the cost. Vision
  adds seconds/image, so run it lazily (only when present + engaging).
- Watch **disk** (150GiB): prune old weekly GGUFs/checkpoints.

## Conventions
- Python. Layout in PLAN.md §14: `bot/`, `data_pipeline/`, `training/`,
  `scripts/`, `assets/`, `data/` (git-ignored), `models/` (git-ignored).
- Secrets in `.env` (git-ignored), never committed. Tunables in `config.yaml`.
- `data/`, `models/`, `.env` stay out of git (raw chat logs + weights).
- Match surrounding style; keep modules small and single-purpose.

## Commands
_None yet — fill in as the project is scaffolded (M0)._
- run bot: `TBD`
- scrape: offline `scripts/scrape.py` (owner-run) + live append; optional owner-gated `/init`
- chattiness: `/activity [0..1] [scope]` (anyone, per-channel or global)
- mute / unmute: `/sleep [scope] [duration]` · `/wake` (anyone)
- preprocess / finetune / export / weekly-retrain: `TBD`

## Git workflow
- Work on branch **`claude/beautiful-ptolemy-wdavzi`**; create it locally if missing.
- Clear, descriptive commits. Push with `git push -u origin <branch>`.
- **Do not open a PR unless explicitly asked.**
