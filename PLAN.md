# PLAN.md — Replicator

A Discord bot that talks like our friend group, powered by a LoRA-finetuned
`Qwen/Qwen3.5-0.8B-Base`. We scrape the server's own messages, finetune the
model to match the group's style (more aggressive/mature, less "helpful
assistant"), and serve it on a CPU-only box. The model is a vision-language
model, so it can also *read* images when we ask it to.

---

## 1. Goal & high-level shape

Three stages, two machines:

```
  ┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ 1. COLLECT          │     │ 2. FINETUNE (4060)   │     │ 3. SERVE (CPU box)     │
  │ /init scrapes the   │ ──▶ │ QLoRA on Qwen3.5-    │ ──▶ │ llama.cpp + GGUF       │
  │ server → JSONL      │     │ 0.8B-Base → adapter  │     │ discord.py bot         │
  └─────────────────────┘     └──────────────────────┘     └────────────────────────┘
         backup weekly  ◀───────────── retrain loop ──────────────────┘
```

- **Train** on the 4060 (8GB assumed) with QLoRA — cheap; fits easily for 0.8B.
  We only finetune the **text/style** behavior; the vision tower is left as-is.
- **Serve** on the 4-core / 24GB / 150GiB CPU box via `llama.cpp` (GGUF, 4-bit),
  **text-only by default** for speed, with vision invoked **lazily** (§9).
- The bot **decides for itself** whether to chime in on undirected messages, but
  **must** reply when @mentioned or replied-to.
- It can **send** a GIF/image from a curated list, **read** incoming links/GIFs/
  images, remember far back via a **tiered memory** (§8), and stay current via a
  **weekly retrain** (§10).

---

## 2. Resolved decisions (from kickoff + follow-ups)

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| 1 | Inference hardware | **CPU-only box** (4060 is train-only) | `llama.cpp`/GGUF, 4 threads, small quant. ~10–30 t/s. |
| 2 | Persona | **Blended group voice** | One *cohesive, human-sounding* voice distilled from the group — not per-user impersonation, not a sanitized committee. |
| 3 | Reply decision | **Model decides via a sentinel token** | Emits `REPLY`/`IGNORE` as its first token; `IGNORE` = 1 token then stop. |
| 4 | Output moderation | **None (raw mimicry)** | No runtime filter; 18+ content is in scope (§13 for the one legal line). |
| 5 | Reading media | **Links + GIFs (title-first, vision fallback); images via lazy vision** | Model *is* a VLM; vision runs only when engaged. Video = metadata only. |
| 6 | Memory | **Tiered (recent + retrieval over all history + events digest + metadata)** | Covers "far back" without paying 262k-token prefill. |
| 7 | Freshness | **Weekly full retrain from base** | Backup → retrain → eval gate → hot-swap. Recency-weighted. |

Base (not Instruct) is the right call: instruct/RLHF variants are aligned to be
friendly and would fight the "aggressive/less friendly" target. A base model
finetuned purely on our chat data learns the group's voice with no assistant
baggage and minimal refusal behavior.

---

## 3. Stage 1 — Data collection (`/init`)

Privileged, admin-only slash command that scrapes the server into a corpus.

**Discord requirements**
- **Message Content Intent** (privileged) enabled in the Developer Portal, else
  scraped content is empty.
- Permissions: *Read Message History*, *View Channels*.
- Slash commands must ack within 3s → `/init` **defers** and scrapes in the
  background, posting progress follow-ups.

**Scope (easy to miss):** regular text channels **and threads + forum-channel
posts** (each thread is its own channel). Skip voice/stage unless they have text.

**What we capture** (→ `data/raw/<guild>_<ts>.jsonl`, one JSON/line):
```json
{
  "message_id": "...", "channel_id": "...", "channel_name": "general",
  "parent_channel": "<for threads>", "is_thread": false,
  "author_id": "...", "author_name": "alice", "is_bot": false,
  "timestamp": "2026-06-22T05:54:00Z",
  "content": "yo <@123> look at this",
  "reply_to": "<message_id or null>",
  "attachments": [{"url": "...", "content_type": "image/gif", "source": "tenor"}],
  "edited": false
}
```

**Behavior**
- Iterate every readable channel + thread; paginate oldest→newest.
- **Incremental:** persist last-scraped `message_id` per channel
  (`data/state.json`) so re-runs only fetch new messages — this also feeds the
  weekly backup (§10).
- **Exclude our own bot and other bots** from being training *targets* (see §4,
  feedback-loop hole). Tag them at capture so preprocessing can filter.
- JSONL streams and survives interruption; `discord.py` handles rate limits.

**Deliverable:** `bot/init_scrape.py`, raw JSONL, per-channel cursor state.

---

## 4. Stage 2a — Preprocessing → training set

Turn raw messages into supervised examples. Two example *types* share one template.

### Chat template
Plain-text, GGUF-friendly markers (no reliance on model-specific chat tokens):
```
### channel: general | 2026-06-22 | here: alice, bob
[memory] alice once said she hates pineapple pizza
alice: yo what's up
bob: nothing much, you?
alice: same, bored af
‹DECIDE›
```
The **target/completion** is exactly one of:
- `IGNORE` — nothing else; bot stays quiet.
- `REPLY: <message text>` — may contain a media tag, e.g. `REPLY: [GIF:deadinside]`.

At inference we feed context + `‹DECIDE›\n` and read the first token(s): `IGNORE`
→ stop (cheap); `REPLY:` → keep generating.

> **Sentinels & collisions (hole):** use rare/unambiguous markers (e.g. `‹DECIDE›`,
> guillemets/private-use chars) **and escape** any literal occurrences in user
> text, so a human typing "IGNORE" or "[GIF:x]" can't corrupt training or parsing.
> Prefer literal strings or verified reserved token IDs over freshly-added tokens
> so they survive finetune → merge → GGUF → quantize. A round-trip test is in §6.

### Example construction
- **Context window:** last *K* messages / ≤~1–2k tokens. Reset at large time gaps.
- **Blended voice:** the target message's original author is dropped — the bot
  speaks as "the group."
- **Decision labels (feature #3):**
  - **REPLY** — message *got* a response (actual reply-reference, or a different
    user posted within ~2 min). Target = that response.
  - **IGNORE** — no response within the window. Target = `IGNORE`. Balanced vs REPLY.
  - **Forced-reply** — message @mentions/relies-to the bot → always REPLY.
- **Media-send tokens:** attachment-only messages → `REPLY: [GIF:<tag>]` (tags
  from the manifest, §9) so the bot learns to answer with a reaction image.

### Cleaning (several holes live here)
- **Drop the bot's own + other bots' messages as targets** (prevents the model
  training on its own output and degenerating over weekly retrains). They may
  remain as *context* for realistic turn-taking.
- **Decode Discord encodings both directions:** `<@123>`→`@alice`, custom emoji
  `<:kek:456>`→`:kek:`, stickers → `[sticker:name]`. Re-encode on send (§7).
- Dedupe, drop system messages, collapse one author's burst, normalize whitespace.
- **Recency weighting hook:** tag examples by age so the weekly retrain can
  upweight recent messages (new slang surfaces faster) — see §10.

**Deliverable:** `data_pipeline/preprocess.py`, `build_decision_labels.py`,
`template.py` → `data/processed/{train,val}.jsonl` + a frozen `holdout.jsonl`.

---

## 5. Stage 2b — Finetuning (on the 4060)

- **Method:** **QLoRA** (4-bit base + LoRA). 0.8B fits an 8GB 4060 with room.
- **Stack:** **Unsloth** (supports Qwen3.5; big speed/VRAM win) over PEFT + TRL
  `SFTTrainer`; fall back to plain PEFT + bitsandbytes if Unsloth lags the arch.
- **We finetune text/style only** — leave the vision encoder/projector frozen.
- **Config (start):** LoRA `r=32, α=32, dropout=0.05` on attn + MLP/expert projs;
  grad checkpointing; paged 8-bit AdamW; bf16/fp16; packing; cosine LR ~2e-4;
  1–3 epochs; loss masked to the completion only.
- **Eval:** val loss + the **frozen holdout** + eyeball samples (tone? correct
  IGNOREs?). The holdout is the gate for weekly retrains (§10).
- **Overfitting guard:** a 0.8B model repeats easily — dedupe hard, small LoRA,
  dropout, early-stop, optional small generic-data mix.
- **Reproducible recipe** (pinned versions, fixed seed) so the weekly retrain is
  one command. **Pin `transformers`/`unsloth`/`trl`** — the arch is new (§12 #1).

**Deliverable:** `training/finetune.py`, `training/configs/`, LoRA adapter.

---

## 6. Stage 2c — Export for CPU inference

1. **Merge** LoRA → fp16.
2. **Convert** to GGUF (`llama.cpp/convert_hf_to_gguf.py`). If vision is enabled,
   also produce/obtain the **`mmproj`** (vision projector) GGUF.
3. **Quantize** → `Q4_K_M` (CPU sweet spot); test `Q5_K_M`/`Q8_0` (all small at 0.8B).
4. **Round-trip test:** confirm `REPLY`/`IGNORE` + `[GIF:tag]` still behave after
   quantization.
5. **Benchmark t/s** on the real CPU box (§11).

> ⚠️ **Top risk lives here** — `llama.cpp` must support the Qwen3.5 MoE/Gated-
> DeltaNet arch (and the VL `mmproj`) at conversion time. Validate with an **early
> spike on the stock model** (M3) *before* a long train run. Fallbacks: pin/upgrade
> `llama.cpp`; else **text model `Qwen/Qwen3-0.6B-Base`** + (if vision wanted) a
> proven `Qwen3-VL` GGUF for captioning; last resort `transformers` CPU.

**Deliverable:** `training/export_gguf.py`, `models/replicator.gguf` (+ `mmproj.gguf`).

---

## 7. Stage 3 — Bot runtime (CPU box)

### Inference server
- `llama.cpp` (`llama-server` / `llama-cpp-python`; `llama-mtmd-cli` path for
  vision), `n_threads=4`, model resident in RAM, context ~1.5–3k tokens (§8).
- **Single-worker generation queue** — 4 cores can't generate concurrently;
  serialize, **drop stale** requests (don't answer a question the convo moved past).

### `on_message` flow
```
on_message(msg):
  if msg.author is our bot or any bot: return
  forced = bot @mentioned OR msg replies to a bot message
  buffer msg; wait out a short DEBOUNCE so multi-message bursts form one turn
  if not forced and not pass_consideration_filter(channel): return   # cost cap only
  ctx = assemble_context(channel)        # §8: metadata + memory + recent (+ captions)
  if forced:
      out = generate(ctx + "‹DECIDE›\nREPLY:")     # prime REPLY → never ignores
  else:
      first = generate(ctx + "‹DECIDE›\n", max_tokens=2)
      if first.startswith("IGNORE"): return        # 1-token cost, done
      out = continue_generation(...)               # it chose REPLY
  send(render(out))                                 # text, or media via §9; re-encode mentions/emoji
```

- **Forced replies bypass the gate** by priming `REPLY:`.
- **Debounce (hole):** wait ~a few seconds of silence before treating a burst as a
  finished turn, so the bot doesn't reply mid-thought.
- **Consideration pre-filter** = cost control only (channel active? cooldown
  elapsed? sampled rate). The actual reply/ignore call stays the model's.
- **Cooldowns** per channel/user; **anti-spam** so chiming into many undirected
  messages doesn't look bot-like to Discord (§12 #8).

**Deliverable:** `bot/main.py`, `discord_client.py`, `inference.py`,
`formatting.py`, `memory.py`, `media.py`, `config.py`.

---

## 8. Memory & context subsystem

The 262k window is a **maximum, not a per-reply budget**: on 4 CPU cores, every
context token is prefilled before the first output token, and the KV-cache for
huge contexts eats many GB — feeding 200k tokens/reply would take minutes. So we
cover "far back" with **tiers**, only ~1.5–3k of which are fed each reply.

| Your tier | How it's actually done | ~Tokens/reply |
|-----------|------------------------|---------------|
| "all past messages, far back" | **Retrieval** — embed every message into a local vector store; per reply, pull the top-k most relevant old messages (in-jokes, callbacks). | ~300–800 |
| "important events" | **Events digest** — a compact, auto-extracted + curatable memory book ("the group trip to Z", "X & Y fell out in March"); retrieved or injected whole. | ~200–500 |
| "channel name, time, people" | **Metadata header** — always injected, one line. | ~50–150 |
| (immediate) | **Recent rolling window** — last N messages verbatim. | ~800–1500 |
| (when media present) | **Image/GIF captions** — added only when relevant (§9). | ~100–300/img |

Net effect: effectively unbounded recall, fixed small prefill, fast on CPU. The
big window mostly helps **training** (we can show long conversations). Embeddings
via a small local model (e.g. a compact `bge`/`Qwen3-Embedding`), index in
`sqlite`+`faiss`/`chroma`, rebuilt incrementally as the backup grows.

**Deliverable:** `bot/memory.py`, `data/memory.db`, `assets/events.md`.

---

## 9. Media — reading & sending

### Reading (incoming)
- **Links:** fetch title + OpenGraph description (oEmbed for YouTube/Tweets) →
  inject `[link: <title> — <desc>]`. *Security:* size limit + domain allowlist +
  sandbox (don't blindly fetch arbitrary user URLs).
- **GIFs:** **title-first** — Tenor/Giphy slug/title is cheap and usually the most
  meaningful signal for reaction GIFs. **Fallback:** for non-Tenor/uploaded GIFs
  (or if we want literal content) sample a representative **frame** and caption it
  with the vision path below. (A GIF is frames; a VLM sees stills — so we sample.)
- **Static images:** caption via the model's **own vision tower** — it *is* a VLM —
  served on CPU through `llama.cpp` `--mmproj` / `llama-mtmd-cli`. Run **lazily**:
  only when an image is present **and** the bot is engaging (REPLY / forced), never
  on every message. Each image adds vision-encoder prefill (~seconds on 4 cores);
  config flag to disable if too slow.
- **Video:** **metadata only** (title/description). True frame-level understanding
  is out of scope for this box.

> Gated on the same arch spike as §6: Qwen3-VL has proven CPU `mmproj` GGUF
> support; Qwen3.5-VL may lag. If so, fall back to title-only reading, or a
> separate proven `Qwen3-VL` GGUF purely for captioning.

### Sending (outgoing)
- `assets/media/` + `assets/media_manifest.json` map tags → path/URL + description.
- Model emits `REPLY: [GIF:deadinside]`; `bot/media.py` posts the file/URL.
  Unknown tag → nearest-by-description or fall back to text. Manifest tags are
  injected into preprocessing (§4) so the finetune learns the vocabulary.

---

## 10. Continuous freshness — weekly retrain loop

QLoRA on a 0.8B model is cheap (likely well under an hour on a 4060), so a weekly
refresh is very feasible. Automated pipeline (cron on the train box):

1. **Backup / pull new messages** — incremental `/init` cursor (§3) appends to the
   canonical corpus. Text is tiny (even ~1M msgs ≈ a few hundred MB JSONL).
2. **Rebuild** train/val with **recency weighting** so new slang/terms surface,
   and **excluding the bot's own output** (critical — else it mimics itself and
   drifts; §12 #1).
3. **Retrain from base** (not continued-training) for stability/reproducibility —
   avoids catastrophic forgetting/drift across weeks.
4. **Eval gate:** score against the **frozen holdout** + last-known-good. If it
   regresses, **keep last-good and alert** — never auto-ship a worse model.
5. **Export + hot-swap** GGUF on the CPU box; keep last-good for instant rollback.
6. **Same-week freshness:** brand-new terms are available via **retrieval/memory**
   (§8) immediately, even before the next retrain.

**Disk policy (hole):** weekly GGUFs (~0.5–1GB) + checkpoints accumulate on
150GiB. Keep last ~4 GGUFs + last-good; prune old checkpoints; backups are small.

**Deliverable:** `scripts/weekly_retrain.sh` (or a Makefile target) + cron unit.

---

## 11. Tokens-per-second — addressing the concern

| Path | Cost | Notes |
|------|------|-------|
| Decision (`IGNORE`) | prefill + **1 token** | Short context → sub-second to ~1–2s. |
| Full reply | prefill + 10–40 tokens | ~10–30 t/s on 4 cores → ~1–4s. Fine for chat. |
| With memory | + retrieval (ms) + a few hundred tokens prefill | Kept to the §8 budget. |
| With vision | + vision-encoder prefill per image (~seconds) | Lazy; only when engaged. |

**Levers if slow:** smaller quant, shorter window, KV-cache reuse per channel,
single-worker queue + drop-stale, cooldowns, consideration cap, disable vision,
or the smaller fallback model (§6, §12 #1).

---

## 12. Risks & holes

1. **🔴 `llama.cpp` support for Qwen3.5 (MoE + Gated-DeltaNet, and VL `mmproj`).**
   Top threat to the whole CPU plan. De-risk with the **M3 spike** before training.
   Fallbacks: pin/upgrade `llama.cpp`; `Qwen3-0.6B-Base` (+ `Qwen3-VL` for vision);
   `transformers` CPU.
2. **🔴 Feedback loop — training on its own output.** Its replies get re-scraped
   weekly → it mimics itself and degenerates. **Exclude bot messages as targets.**
3. **Weekly retrain regressing silently.** → frozen-holdout **eval gate +
   rollback** (§10), never auto-ship worse.
4. **Sentinel/tag collisions** with user text → rare markers + escaping (§4).
5. **Mention/emoji/sticker encoding** both directions (`<@id>`, `<:emoji:id>`) (§4,§7).
6. **Turn debounce** — reply per finished turn, not per message (§7).
7. **Disk creep** on 150GiB from weekly GGUFs/checkpoints → retention policy (§10).
8. **Spam/ban risk** — a bot answering many undirected msgs looks bot-like →
   cooldowns + consideration cap.
9. **No clean "sounds like us" metric** — human spot-check protocol + holdout
   next-message perplexity as a rough proxy.
10. **Vision cost/feasibility on CPU** — lazy + title-first + disable flag (§9,§11).
11. **Privileged Message Content Intent** must be on (§3).
12. **Overfitting** a 0.8B on a smallish corpus → dedupe, small LoRA, dropout,
    early-stop, generic mix (§5).
13. **URL-fetch security** (SSRF/malware) — allowlist + size limit + sandbox (§9).
14. **4060 VRAM** assumed 8GB → QLoRA; 16GB allows plain LoRA / bigger batch.

---

## 13. Responsible-use notes (not a content filter)

Moderation is *none* by design — operational notes only, not filtering:
- **Private adult server only.** It scrapes members' messages and reproduces their
  style; tell members their messages are collected (their data + Discord ToS).
  Don't deploy to public/open servers.
- The **only hard line** is genuinely **illegal** content (e.g. anything sexual
  involving minors) — a legal/ToS boundary independent of the bot's personality.
  Everything else (crude, aggressive, explicit adult banter among consenting
  adults) is in scope.
- Model runs **text-only by default**; vision is opt-in/lazy (§9).
- **Secrets** in `.env`; `data/`, `models/`, `.env` are git-ignored (raw logs +
  weights stay off GitHub).

---

## 14. Proposed repo layout (to scaffold next)

```
replicator/
  CLAUDE.md  PLAN.md  README.md
  pyproject.toml / requirements.txt   .env.example   config.example.yaml
  bot/        main.py discord_client.py init_scrape.py inference.py
              formatting.py memory.py media.py config.py
  data_pipeline/  preprocess.py build_decision_labels.py template.py
  training/   finetune.py export_gguf.py configs/
  scripts/    weekly_retrain.sh
  data/       raw/ processed/ memory.db state.json   # git-ignored
  assets/     media/ media_manifest.json events.md
  models/     replicator.gguf mmproj.gguf            # git-ignored
  tests/
```

---

## 15. Milestones

- **M0 — Scaffold:** skeleton, deps, config, `.gitignore`, bot connects, `/ping`.
- **M1 — Collect:** `/init` incremental scrape (incl. threads/forums) → JSONL.
- **M2 — Preprocess:** raw → `{train,val,holdout}.jsonl` with decision sentinels,
  media tags, mention/emoji decoding, bot-output exclusion.
- **M3 — Spike (de-risk):** convert *stock* Qwen3.5-0.8B-Base (and `mmproj`) to
  GGUF; confirm `llama.cpp` runs it on the CPU box **before** training (risk #1).
- **M4 — Finetune:** QLoRA on the 4060; eval vs holdout + samples.
- **M5 — Export:** merge + GGUF + quantize; round-trip + t/s benchmark.
- **M6 — Serve:** CPU runtime; model-driven REPLY/IGNORE; forced reply; debounce.
- **M7 — Memory:** retrieval + events digest + metadata header (§8).
- **M8 — Media:** sending manifest + reading (links/GIF titles; lazy vision).
- **M9 — Freshness:** weekly retrain loop + eval gate + hot-swap + disk policy.
- **M10 — Harden:** cooldowns, queue, logging, systemd, retention.
- **M11 — Iterate:** tune persona/data until it sounds like us.
