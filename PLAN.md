# PLAN.md — Replicator

A Discord bot that talks like our friend group, powered by a LoRA-finetuned
`Qwen/Qwen3.5-0.8B-Base`. We scrape the server's own messages, finetune the
model to match the group's style (more aggressive/mature, less "helpful
assistant"), and serve it on a CPU-only box.

---

## 1. Goal & high-level shape

Three stages, two machines:

```
  ┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ 1. COLLECT          │     │ 2. FINETUNE (4060)   │     │ 3. SERVE (CPU box)     │
  │ /init scrapes the   │ ──▶ │ QLoRA on Qwen3.5-    │ ──▶ │ llama.cpp + GGUF       │
  │ server → JSONL      │     │ 0.8B-Base → adapter  │     │ discord.py bot         │
  └─────────────────────┘     └──────────────────────┘     └────────────────────────┘
```

- **Train** on the 4060 (8GB assumed) with QLoRA — cheap, fits easily for 0.8B.
- **Serve** on the 4-core / 24GB / 150GiB CPU box via `llama.cpp` (GGUF, 4-bit).
- The bot **decides for itself** whether to chime in on undirected messages,
  but **must** reply when @mentioned or replied-to.
- It can also respond with a **GIF/image** from a curated list.

---

## 2. Resolved decisions (from kickoff)

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| 1 | Inference hardware | **CPU-only box** (4060 is train-only) | Target `llama.cpp`/GGUF, 4 threads, small quant. ~10–30 t/s. |
| 2 | Persona | **Single blended group voice** | One personality; target = any member's next message. Simpler pipeline. |
| 3 | Reply decision | **Model decides via a sentinel token** | Model emits `REPLY`/`IGNORE` as its first token; `IGNORE` = 1 token, then stop. |
| 4 | Output moderation | **None (raw mimicry)** | No runtime content filter. See §11 for the (non-filtering) responsible-use notes. |

Base (not Instruct) model is the right call here: the instruct/RLHF variants are
aligned to be friendly and would actively fight the "aggressive/less friendly"
target. A base model finetuned purely on our chat data learns the group's voice
with no assistant baggage.

---

## 3. Stage 1 — Data collection (`/init`)

A privileged, admin-only slash command that scrapes the server's message history
into a training corpus.

**Discord requirements**
- **Message Content Intent** (privileged) enabled in the Developer Portal.
- Bot permissions: *Read Message History*, *View Channels*.
- Slash commands must ack within 3s → `/init` **defers** immediately and runs the
  scrape as a background task, posting progress via follow-ups.

**What we capture per message** (→ `data/raw/<guild>_<ts>.jsonl`, one JSON/line):
```json
{
  "message_id": "...", "channel_id": "...", "channel_name": "general",
  "author_id": "...", "author_name": "alice",
  "timestamp": "2026-06-22T05:54:00Z",
  "content": "yo what's up",
  "reply_to": "<message_id or null>",
  "attachments": [{"url": "...", "content_type": "image/gif"}],
  "is_bot": false, "edited": false
}
```

**Behavior**
- Iterate every readable text channel; paginate history oldest→newest.
- **Incremental**: persist the last-scraped `message_id` per channel
  (`data/state.json`) so re-running `/init` only fetches new messages.
- JSONL (not one big JSON) so dumps stream and survive interruption.
- Progress follow-ups: `scraped 12,400 messages across 9 channels…`.
- `discord.py` handles rate limits; just let it run.

**Deliverable:** `bot/init_scrape.py`, raw JSONL, per-channel cursor state.

---

## 4. Stage 2a — Preprocessing → training set

Turn raw messages into supervised examples. Two example *types* share one chat
template.

### Chat template
Plain-text, GGUF-friendly markers (no reliance on model-specific chat tokens):
```
### channel: general
alice: yo what's up
bob: nothing much, you?
alice: same, bored af
<<<DECIDE>>>
```
The **target/completion** is exactly one of:
- `IGNORE`  — nothing else; the bot stays quiet.
- `REPLY: <message text>` — may contain a media tag, e.g. `REPLY: [GIF:deadinside]`.

At inference we feed context + `<<<DECIDE>>>\n` and read the first token(s):
`IGNORE` → stop (cheap); `REPLY:` → keep generating the message.

> **Sentinel choice:** use literal `REPLY`/`IGNORE` strings (or two verified
> reserved special-token IDs) rather than freshly-added tokens, so they survive
> the finetune → merge → GGUF round-trip without embedding-resize headaches.
> A round-trip test is part of Stage 2c.

### Example construction
- **Context window:** last *K* messages or *T* tokens (target ≤1–2k tokens to
  keep CPU prefill cheap). Reset context at large time gaps.
- **Blended voice:** the target message's original author is dropped — the bot
  speaks as "the group," so every member's messages become training targets.
- **Decision labels (the core of feature #3):**
  - **REPLY** example: message *did* get a response — i.e. an actual Discord
    reply-reference, or a different user posted within ~2 min. Target = that
    response text.
  - **IGNORE** example: no response within the window (conversation died).
    Target = `IGNORE`. Sampled to balance against REPLY.
  - **Forced-reply** examples: when a message @mentions the bot or replies to it,
    always label REPLY (improves quality even though runtime bypasses the gate).
- **GIF/image action examples:** messages that were *just* an attachment/GIF map
  to `REPLY: [GIF:<tag>]`, where `<tag>` comes from the media manifest (§7). This
  teaches the bot to sometimes answer with a reaction image.

**Cleaning:** drop other bots' and system messages, dedupe, strip/normalize
mentions and custom emoji, optionally drop link-only spam (except GIF-action
data), collapse multi-message bursts from one author.

**Deliverable:** `data_pipeline/preprocess.py`, `data_pipeline/build_decision_labels.py`,
`data_pipeline/template.py`, → `data/processed/{train,val}.jsonl`.

---

## 5. Stage 2b — Finetuning (on the 4060)

- **Method:** **QLoRA** (4-bit base + LoRA adapter). 0.8B is tiny; this fits an
  8GB 4060 with room to spare. (Plain fp16 LoRA also fits; QLoRA is the safe default.)
- **Stack:** **Unsloth** (verified to support Qwen3.5; big speed/VRAM win on a
  4060) over PEFT + TRL `SFTTrainer`. Falls back to plain PEFT+bitsandbytes if
  Unsloth lags the arch.
- **Config (starting point):** LoRA `r=32, alpha=32, dropout=0.05` on attention +
  MLP/expert projections; gradient checkpointing; paged 8-bit AdamW; bf16/fp16;
  sequence packing; cosine LR ~2e-4; 1–3 epochs; effective batch via grad accum.
- **Loss masking:** compute loss only on the completion (the `IGNORE` /
  `REPLY: …` span), not the context.
- **Eval:** track val loss + eyeball sample generations each epoch (does it nail
  the group's tone? does it correctly IGNORE dead messages?).
- **Overfitting guard:** a 0.8B model on a smallish corpus repeats easily —
  dedupe hard, keep LoRA small, use dropout, early-stop on val loss; optionally
  mix a little generic chat data so it doesn't collapse.

> **Version pinning is mandatory.** Qwen3.5-small uses a newer MoE + Gated
> Delta-Net architecture; pin `transformers`/`unsloth`/`trl` to versions that
> support it. See §10 risk #1.

**Deliverable:** `training/finetune.py`, `training/configs/`, saved LoRA adapter.

---

## 6. Stage 2c — Export for CPU inference

1. **Merge** LoRA into the base → fp16 model.
2. **Convert** to GGUF (`llama.cpp/convert_hf_to_gguf.py`).
3. **Quantize** → `Q4_K_M` (best CPU speed/quality balance) — test `Q5_K_M`/`Q8_0`
   too; at 0.8B all are small (~0.5–1GB).
4. **Verify the round-trip:** confirm `REPLY`/`IGNORE` sentinels and `[GIF:tag]`
   tokens still produce the intended behavior after quantization.
5. **Benchmark t/s** on the actual CPU box (see §9).

> ⚠️ **Biggest risk lives here** — `llama.cpp` must support the Qwen3.5 MoE/
> Gated-DeltaNet arch at conversion time. If it doesn't yet, see §10 risk #1 and
> the **Qwen3-0.6B-Base fallback**.

**Deliverable:** `training/export_gguf.py`, a quantized `models/replicator.gguf`.

---

## 7. Stage 3 — Bot runtime (CPU box)

### Inference server
- `llama.cpp` (`llama-server`) or `llama-cpp-python`, `n_threads=4`, model resident
  in RAM (trivial at 24GB), context capped ~1–2k tokens, per-channel KV-cache reuse.
- Single-worker **generation queue** — 4 cores can't generate concurrently;
  serialize requests, show a typing indicator while working.

### `bot/discord_client.py` — `on_message` flow
```
on_message(msg):
  if msg.author is bot or other bot: return
  forced = bot was @mentioned OR msg replies to a bot message
  if not forced and not pass_consideration_filter(msg): return   # cost cap only
  ctx = build_context(channel, last K msgs)
  if forced:
      out = generate(ctx + "<<<DECIDE>>>\nREPLY:")   # prime REPLY → never ignores
  else:
      first = generate(ctx + "<<<DECIDE>>>\n", max_tokens=2)
      if first.startswith("IGNORE"): return           # 1-token cost, done
      out = continue_generation(...)                   # it chose REPLY
  send(parse(out))                                     # text, or media via §8
```

- **Forced replies** (mention / reply-to-bot) **bypass the gate** by priming the
  completion with `REPLY:`, so the model is made to answer.
- **Consideration pre-filter** is *cost control only*, not the decision: a coarse
  gate (channel recently active? cooldown elapsed? sampled rate) bounds how often
  we even wake the model on a busy server. The actual reply/ignore call is still
  the model's.
- **Cooldowns** per channel/user to prevent spam and bound CPU.

### Generation params
Config-driven: `temperature`, `top_p`, `repeat_penalty`, `max_tokens`
(short — Discord messages are small), stop strings (`\n<name>:` to avoid the
model writing other people's lines).

**Deliverable:** `bot/main.py`, `bot/discord_client.py`, `bot/inference.py`,
`bot/formatting.py`, `bot/config.py`.

---

## 8. GIF / image subsystem

- `assets/media/` holds the files; `assets/media_manifest.json` maps tags → path/URL
  + a short description:
  ```json
  { "deadinside": {"path": "assets/media/deadinside.gif", "desc": "dead inside reaction"},
    "based":      {"path": "assets/media/based.png",      "desc": "based stamp"} }
  ```
- Model emits `REPLY: [GIF:deadinside]`; `bot/media.py` intercepts the tag and
  posts the file/URL instead of text. Unknown tag → nearest-by-description or
  fall back to text.
- Tags from the manifest are injected into preprocessing so the finetune learns
  the available vocabulary (§4).

---

## 9. Tokens-per-second — addressing the concern

| Stage | Cost | Notes |
|-------|------|-------|
| Decision (`IGNORE`) | prefill + **1 token** | Short context → sub-second to ~1–2s. |
| Full reply | prefill + 10–40 tokens | At ~10–30 t/s on 4 cores → ~1–4s. Fine for chat. |

**Levers if it's too slow:** smaller quant (Q4_K_M), shorter context, KV-cache
reuse per channel, single-worker queue, cooldowns, the consideration pre-filter,
and — last resort — the smaller fallback model (§10). For a friend-group server,
a few seconds of latency on short replies is well within acceptable.

---

## 10. Risks & mitigations

1. **🔴 GGUF/`llama.cpp` support for Qwen3.5-small (MoE + Gated-DeltaNet).**
   This is the top threat to the entire CPU plan. *Mitigations, in order:*
   (a) pin/upgrade `llama.cpp` to a version that supports the arch;
   (b) **fallback model: `Qwen/Qwen3-0.6B-Base`** — known-good GGUF support, nearly
   the same size, same pipeline; (c) last resort, CPU inference via `transformers`
   (slower). *Validate conversion EARLY (a Stage-2c spike) before investing in a
   long finetune run.*
2. **Sentinel/media tokens surviving finetune → merge → GGUF → quantize.**
   Mitigation: literal-string sentinels or verified reserved tokens + a round-trip
   test (§6).
3. **Privileged Message Content Intent** must be enabled, or `/init` sees empty
   content. Verify before scraping.
4. **Data quality / overfitting** on a small corpus → repetition/collapse.
   Mitigation: dedupe, small LoRA, dropout, val early-stop, optional generic mix.
5. **Scrape volume / rate limits** on large/old servers. Mitigation: incremental
   cursor + JSONL streaming.
6. **4060 VRAM** assumed 8GB → QLoRA. Confirm at training time; 16GB allows
   plain LoRA / bigger batches.
7. **Persona drift** — if output isn't "aggressive/mature" enough, iterate on
   data selection/weighting, not just hyperparameters.

---

## 11. Responsible-use notes (not a content filter)

Moderation was set to *none* by design, so these are operational, not filtering:
- **Private server only.** This scrapes members' messages and reproduces their
  style; don't deploy to public/open servers. Tell members their messages are
  being collected — it's their data and Discord's ToS expects consent.
- **Secrets** (bot token) live in `.env`, never committed.
- `data/`, `models/`, and `.env` are git-ignored (raw chat logs and weights stay
  off GitHub).

---

## 12. Proposed repo layout (to scaffold next)

```
replicator/
  CLAUDE.md  PLAN.md  README.md
  pyproject.toml / requirements.txt   .env.example   config.example.yaml
  bot/        main.py discord_client.py init_scrape.py inference.py media.py formatting.py config.py
  data_pipeline/  preprocess.py build_decision_labels.py template.py
  training/   finetune.py export_gguf.py configs/
  data/       raw/ processed/ state.json        # git-ignored
  assets/     media/ media_manifest.json
  models/     replicator.gguf                    # git-ignored
  scripts/    tests/
```

---

## 13. Milestones

- **M0 — Scaffold:** repo skeleton, deps, config, `.gitignore`, bot connects, `/ping`.
- **M1 — Collect:** `/init` incremental scrape → raw JSONL + progress.
- **M2 — Preprocess:** raw → `{train,val}.jsonl` with decision sentinels + GIF tags.
- **M3 — Spike (de-risk):** convert *stock* Qwen3.5-0.8B-Base to GGUF on the CPU
  box to confirm `llama.cpp` support **before** the big train run (risk #1).
- **M4 — Finetune:** QLoRA on the 4060; eval loss + sample gens.
- **M5 — Export:** merge + GGUF + quantize; round-trip + t/s benchmark.
- **M6 — Serve:** CPU runtime; model-driven REPLY/IGNORE; forced reply on
  mention/reply-to.
- **M7 — Media:** GIF/image manifest + sending.
- **M8 — Harden:** cooldowns, queue, logging, systemd service, `.env`.
- **M9 — Iterate:** tune persona/data until it sounds like us.
