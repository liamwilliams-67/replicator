# PLAN.md — Replicator

A Discord bot that talks like our friend group, powered by a LoRA-finetuned
`Qwen/Qwen3.5-0.8B-Base`. We scrape the server's own messages, finetune the
model to match the group's style (more aggressive/mature, less "helpful
assistant"), and serve it on a CPU-only box. It reads images/GIFs/video-thumbnails
via a **separate** `Qwen3-VL` vision model (Qwen3.5-VL isn't supported in
llama.cpp yet — §9, §12 #2).

> **Serve-stack status (validated by a web-research pass, June 2026):** Qwen3.5
> text *converts and runs* in llama.cpp, but its Gated-DeltaNet arch currently
> **breaks prompt-cache reuse** (full re-prefill every turn) and its **vision
> `mmproj` is unsupported**. So Qwen3.5 stays primary but is **gated on the M3
> spike** (§6); the confirmed-working fallback is **`Qwen3-0.6B-Base` (text) +
> `Qwen3-VL-2B/4B` (vision)**.

---

## 1. Goal & high-level shape

Three stages, two machines:

```
  ┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ 1. COLLECT          │     │ 2. FINETUNE (4060)   │     │ 3. SERVE (CPU box)     │
  │ scrape + live append│ ──▶ │ QLoRA on Qwen3.5-    │ ──▶ │ llama.cpp + GGUF       │
  │ → JSONL             │     │ 0.8B-Base → adapter  │     │ discord.py bot         │
  └─────────────────────┘     └──────────────────────┘     └────────────────────────┘
         backup weekly  ◀───────────── retrain loop ──────────────────┘
```

- **Train** on the 4060 (8GB assumed) with QLoRA — cheap; the 0.8B is **dense**
  (~3GB in bf16-LoRA), so it fits easily. We finetune **text/style only**.
- **Serve** on the 4-core / 24GB / 150GiB CPU box via `llama.cpp` (GGUF, 4-bit).
  ⚠️ Qwen3.5's Gated-DeltaNet arch currently **defeats llama.cpp prompt-cache
  reuse** (full re-prefill every turn), so context stays **small (~1–3k tok)**
  unless the M3 spike proves otherwise — or we fall back to `Qwen3-0.6B-Base`
  (plain transformer → cache reuse works → big context). Vision is a **separate
  `Qwen3-VL` captioner**, invoked only when media is present (§9).
- The bot **decides for itself** whether to chime in on undirected messages, but
  **must** reply when @mentioned or replied-to.
- It can **send** a GIF/image from a curated list, **read** incoming links/GIFs/
  images/video-thumbnails, remember far back via a **tiered memory** (§8), and
  stay current via a **weekly retrain** (§10).

> **Dev workflow (local-first).** Until the project is complete, run *everything*
> on the **dev PC** (the 4060 box) — scrape, train, serve, and the live bot —
> using the **same `llama.cpp` runtime** as prod but GPU-accelerated (`-ngl`).
> The CPU serve box only enters at **final deployment**. Backend is config-driven
> (`n_gpu_layers`, `n_threads`, `n_ctx`) so PC-dev → CPU-prod is a config swap.
> ⚠️ GPU dev *hides* the CPU prefill/cache limits (§8, §11): keep a **small-context
> "prod profile"** and validate at the target context — and run the M3 cache test,
> whose verdict transfers — **before** deploying.

---

## 2. Resolved decisions (from kickoff + follow-ups + research)

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| 1 | Inference hardware | **CPU-only box** (4060 is train-only) | `llama.cpp`/GGUF, 4 threads. ~10 t/s OK, ≤~60s/reply. **No prompt-cache reuse on Qwen3.5 → keep context ~1–3k** (M3 verifies; `Qwen3-0.6B` fallback unlocks big context). |
| 2 | Persona | **Blended group voice** | One *cohesive, human-sounding* voice — not per-user impersonation, not a sanitized committee. |
| 3 | Reply decision | **Model decides via a sentinel token** | Emits `REPLY`/`IGNORE` as its first token; `IGNORE` = 1 token then stop. |
| 4 | Output moderation | **None (raw mimicry)** | No runtime filter; 18+ content is in scope (§13 for the one legal line). |
| 5 | Reading media | **Links→text; GIF→single frame; video→Discord thumbnail; image→vision** | Vision via a **separate `Qwen3-VL` captioner** (Qwen3.5-VL unsupported); ~15–30s/image, ≤1/decision; captioned at consideration. |
| 6 | Memory | **Tiered** (recent window + retrieval + events digest + metadata) | "Far back" via **retrieval**, not a big prefill; window kept small (no cache reuse, §8). |
| 7 | Freshness | **Weekly full retrain from base** | Backup → retrain → eval gate → hot-swap. Recency-weighted. |
| 8 | Interjection | **`activity` dial (0–1) + reserved + kill switch** | `0` = only mentions/replies; `1` = nearly every message. Backs off in active convos. `/activity`·`/sleep`·`/wake` runnable by **anyone**; setup is owner-gated. |
| 9 | Serve model | **Qwen3.5-0.8B primary, gated on M3 spike** | Fallback (confirmed-working): `Qwen3-0.6B-Base` + `Qwen3-VL-2B/4B`. Decide on real numbers at M3. |

Base (not Instruct) is the right call: instruct/RLHF variants are aligned to be
friendly and would fight the "aggressive/less friendly" target. A base model
finetuned purely on our chat data learns the group's voice with no assistant
baggage and minimal refusal behavior.

---

## 3. Stage 1 — Data collection (offline scrape + live append)

Two paths into the corpus: a one-time **offline backfill** (`scripts/scrape.py`,
run by *you* with the bot token) for all history, and **live append** by the
running bot for new messages (so the backup stays current without re-scraping). An
owner-gated `/init` slash command is *optional* if you'd rather kick a backfill
from inside Discord — but the offline script is the primary path (you're not a
server admin, and it keeps a heavy scrape out of random hands).

**Discord requirements** (confirmed for discord.py 2.7.x)
- **Message Content Intent** (privileged) enabled in the Developer Portal, else
  `content`/`embeds`/`attachments` come back empty. Self-enable under 100 servers.
- Permissions: *Read Message History*, *View Channels*.
- The **offline script** runs to completion with progress logged to the console —
  no 3s slash-command ack limit to fight. If the optional `/init` is used, it must
  **defer** within 3s and post progress as follow-ups.
- **Live append:** the running bot writes each new (non-bot) message to the corpus
  as it arrives, sharing the decode/clean logic with the offline script (§4) so the
  weekly backup (§10) stays current with no re-scrape.

**Scope (easy to miss):** regular text channels **and threads + forum-channel
posts** — `channel.history(limit=None)` + `ForumChannel.archived_threads()`. Skip
voice/stage unless they have text.

**What we capture** (→ `data/raw/<guild>_<ts>.jsonl`, one JSON/line):
```json
{
  "message_id": "...", "channel_id": "...", "channel_name": "general",
  "parent_channel": "<for threads>", "is_thread": false,
  "author_id": "...", "author_name": "alice", "is_bot": false,
  "timestamp": "2026-06-22T05:54:00Z",
  "content": "yo <@123> look at this",
  "reply_to": "<message_id or null>",
  "attachments": [{"url": "...", "content_type": "image/gif", "source": "tenor",
                   "thumbnail": "<discord proxy/thumbnail url>"}],
  "edited": false
}
```

**Behavior & rate-limit guardrails (important)**
- **Scrape channels SERIALLY, not fanned-out**, and honor `Retry-After`.
  discord.py auto-retries 429s — but **>10k 4xx/429 responses in 10 min = a
  1-hour API ban**, so log retry counts and throttle during a big backfill.
- **Incremental:** persist last-scraped `message_id` per channel
  (`data/state.json`) so re-runs only fetch new messages.
- **Exclude our own bot and other bots** from being training *targets* (§4).
- Capture attachment **thumbnails** (Discord proxy URLs) — needed for video
  reading (§9). JSONL streams and survives interruption.

**Deliverable:** `scripts/scrape.py` (backfill + incremental), live-append in the
bot, raw JSONL, per-channel cursor state.

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
- **Context window:** match the realistic *serve* budget (§8/§11) — i.e. **small
  (~1–3k tokens)**, not "as long as possible." Reset at large time gaps.
- **Blended voice:** the target message's original author is dropped — the bot
  speaks as "the group."
- **Decision labels (feature #3):**
  - **REPLY** — message *got* a response (actual reply-reference, or a different
    user posted within ~2 min). Target = that response.
  - **IGNORE** — no response within the window. Target = `IGNORE`. Balanced vs REPLY.
  - **Forced-reply** — message @mentions/replies-to the bot → always REPLY.
  - **Reservedness** — over-sample `IGNORE` for messages mid-exchange between two
    active humans (learn to stay *out* of flowing convos), so REPLY skews toward
    lulls, direct address, or group-wide prompts. Reinforces the runtime gate (§7).
- **Media examples:** for messages whose context includes an image/GIF/video, the
  context line carries the **caption** (`[image] <caption>` / `[gif] <caption>`),
  so the finetune learns to react to described media. Attachment-only messages the
  bot would answer with a reaction map to `REPLY: [GIF:<tag>]` (tags from §9).

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

- **Method:** **QLoRA** (4-bit base + LoRA). The 0.8B is **dense** (~3GB bf16-LoRA),
  so 8GB is comfortable. (bf16-LoRA is only *required* for the big MoE variants.)
- **Stack:** **Unsloth** (confirmed Qwen3.5 QLoRA support) over PEFT + TRL
  `SFTTrainer`.
- **We finetune text/style only** — there is no vision tower to train here (vision
  is the separate Qwen3-VL captioner, §9).
- **Config (Unsloth's Qwen3.5 recipe):** LoRA `r=16, α=16, dropout=0`, targets
  `q,k,v,o,gate,up,down_proj`; `use_gradient_checkpointing="unsloth"`; paged 8-bit
  AdamW; bf16/fp16; packing; cosine LR ~2e-4; 1–3 epochs; loss masked to the
  completion only. **Seq len 1–4k, NOT long** — Unsloth has a gradient-explosion
  (NaN) bug on Qwen3.5 at long seq (#4906), and the serve budget is small anyway
  (§8). The dense 0.8B does **not** need the MoE env-vars.
- **Eval:** val loss + the **frozen holdout** + eyeball samples (tone? correct
  IGNOREs?). The holdout is the gate for weekly retrains (§10).
- **Overfitting guard:** a 0.8B model repeats easily — dedupe hard, small LoRA,
  dropout, early-stop, optional small generic-data mix.
- **Reproducible recipe** (fixed seed) so the weekly retrain is one command.
  **Pins:** `trl==0.22.2`, `transformers` from git (latest at pin time), `unsloth`
  latest, `peft`/`bitsandbytes` latest-compatible. We LoRA the **standard
  projections only**; leave the recurrent Gated-DeltaNet internals to Unsloth
  defaults — don't hand-add exotic targets.

**Deliverable:** `training/finetune.py`, `training/configs/`, LoRA adapter.

---

## 6. Stage 2c — Export for CPU inference

1. **Merge** LoRA → fp16.
2. **Convert** to GGUF (`llama.cpp/convert_hf_to_gguf.py` — Qwen3.5 supported).
   For vision, obtain the **`Qwen3-VL` `mmproj` + model** GGUF (separate captioner;
   Qwen3.5-VL's own mmproj isn't supported, §9, §12 #2).
3. **Quantize** → `Q4_K_M` (CPU sweet spot); test `Q5_K_M`/`Q8_0` (all small at
   0.8B). Consider a **quantized KV cache**. ⚠️ The **MTP speculative GGUF can't be
   used with `--mmproj` or `--np>1`** (§12 #22) — pick speed *or* vision.
4. **Round-trip test:** confirm `REPLY`/`IGNORE` + `[GIF:tag]` still behave after
   quantization.
5. **Benchmark** with `llama-bench` on the real CPU box — this sets the (small)
   context budget (§8, §11).

> ⚠️ **The serve-stack go/no-go (M3).** Qwen3.5 text conversion/running is confirmed,
> so the spike must check **three** things on the actual CPU box:
> **(a)** convert+load stock Qwen3.5-0.8B-Base;
> **(b) multi-turn prompt-cache reuse actually works** — measure turn-2 prefill; if
> it re-prefills the whole prompt (the known Gated-DeltaNet bug, §12 #1), context
> stays tiny or we fall back;
> **(c)** end-to-end latency at the target context.
> **Re-check llama.cpp release notes for a recurrent-cache fix right before M3.**
> Fallback (confirmed-working): **`Qwen3-0.6B-Base` + `Qwen3-VL-2B/4B`**, which
> sidesteps both the cache and vision blockers.

**Deliverable:** `training/export_gguf.py`, `models/replicator.gguf` (+ Qwen3-VL
GGUF + `mmproj.gguf`).

---

## 7. Stage 3 — Bot runtime (CPU box)

### Inference server
- `llama.cpp` (`llama-server` / `llama-cpp-python`; separate `llama-mtmd-cli` +
  Qwen3-VL for vision), `n_threads=4`, model resident in RAM.
- **Context is full-prefilled every turn on Qwen3.5** (no prompt-cache reuse for
  its Gated-DeltaNet arch, §12 #1), so **keep the window small (~1–3k)** and lean
  on retrieval (§8). Persistent per-channel prompt-cache slots help **only** if a
  llama.cpp fix lands, or on the `Qwen3-0.6B` fallback.
- **Single-worker generation queue** — 4 cores can't generate concurrently;
  serialize, **drop stale** requests (don't answer a question the convo moved past).

### `on_message` flow
```
on_message(msg):
  if msg.author is our bot or any bot: return
  if asleep(channel/server): return                  # /sleep kill-switch (below)
  forced = bot @mentioned OR msg replies to a bot message
  buffer msg; wait out a short DEBOUNCE so multi-message bursts form one turn
  if not forced and not want_to_interject(channel): return   # reserved gate + cost cap
  if msg has media: caption it now (§9) so the decision can see it
  ctx = assemble_context(channel)        # §8: metadata + memory + recent (+ captions)
  if forced:
      out = generate(ctx + "‹DECIDE›\nREPLY:")     # prime REPLY → never ignores
  else:
      first = generate(ctx + "‹DECIDE›\n", max_tokens=2)
      if first.startswith("IGNORE"): return        # 1-token cost, done
      out = continue_generation(...)               # it chose REPLY
  send(render(out))                                 # text, or media via §9; re-encode mentions/emoji
```

- **Forced replies bypass the gate** by priming `REPLY:` (still answered even when
  reserved — but **not** when asleep).
- **Caption media before deciding** so the bot can react to a GIF/video/image
  posted with no text. Bounded by the gate + cooldowns.
- **Debounce (hole):** wait ~a few seconds of silence before treating a burst as a
  finished turn, so the bot doesn't reply mid-thought.
- **Activity dial (`activity` 0–1) — the master chattiness knob.** `0` = only
  forced (mentions/replies), and we **skip undirected messages entirely** (no model
  call → saves CPU). `1` = chime in on nearly every message. Implemented by reading
  the model's first-token **REPLY-vs-IGNORE probability** and replying when it
  clears a threshold set by `activity` — principled (still respects context), not a
  blind coin. Settable **globally or per-channel** — persisted per-channel overrides
  on top of a global default (e.g. `#general` 0.4, `#serious` 0.05); default low.
- **Don't-interrupt backoff:** on top of `activity`, lower the effective chance
  during **active 2+ human exchanges** (strong at low activity, minimal near `1`),
  plus a post-speak **cooldown** — so even when chatty it waits for lulls. The
  finetune is also trained to be reserved (§4).
- **Kill switch (`/sleep`, `/wake`):** `/sleep [scope] [duration]` mutes per-channel
  or server-wide (optional auto-wake); `/wake` resumes. **State persisted** across
  restarts. While asleep it ignores everything (including @mentions) except `/wake`.
- **Command permissions:** runtime commands (`/activity`, `/sleep`, `/wake`) are
  **open to everyone** — no server-admin needed (you aren't one). Only the heavy
  scrape is owner-gated / offline (§3). Add per-command **cooldowns** to blunt
  griefing (§12 #18).
- **Ignore stale messages on reconnect** (§12 #16) — after downtime, only react to
  live messages; never mass-reply to a backlog.
- **Cooldowns** per channel/user; **anti-spam** so chiming into many undirected
  messages doesn't look bot-like to Discord (§12 #8).

**Deliverable:** `bot/main.py`, `discord_client.py`, `inference.py`,
`formatting.py`, `memory.py`, `media.py`, `config.py`.

---

## 8. Memory & context subsystem

The 262k window is a **maximum, not a per-reply budget** — on CPU every context
token is prefilled before the first output token. ⚠️ **Reality check:** Qwen3.5's
Gated-DeltaNet arch currently **defeats llama.cpp prompt-cache reuse** — *every*
turn re-prefills the whole prompt (§12 #1), so a big window is **not** cheap (a
~15k convo can take minutes/turn). So on Qwen3.5 we **keep the per-turn context
small (~1–3k tokens)** and let **retrieval** (not window size) cover "far back."
The large-window design only unlocks if (a) the M3 spike shows cache reuse works,
or (b) we fall back to `Qwen3-0.6B-Base`. RAM isn't the limit — **time (prefill
every turn) is.**

| Your tier | How it's actually done | Notes |
|-----------|------------------------|-------|
| "all past messages, far back" | **Retrieval** — embed every message into a local vector store; per reply pull the top-k most relevant old messages (in-jokes, callbacks). | Carries the long-term recall. |
| "important events" | **Events digest** — compact, auto-extracted + curatable memory book ("the trip to Z", "X & Y fell out in March"). | Small; injected or retrieved. |
| "channel name, time, people" | **Metadata header** — one line, always injected. | Cheap. |
| (immediate) | **Recent rolling window** — last N messages verbatim, **small** (no cache reuse on Qwen3.5). | Set N from the M3 prefill benchmark. |
| (when media present) | **Image/GIF/video captions** — added at consideration (§9). | ~300–1000 img-tokens each. |

**Sizing:** because every turn is cold prefill on Qwen3.5, set the window from the
M3/M5 `llama-bench` numbers — the largest whose *full* prefill fits ≤60s (likely
**a few hundred to ~2k tokens**). Lean on retrieval, not window size. If we fall
back to Qwen3-0.6B (cache reuse works), the window can grow substantially.

> **Cache-ordering note (only if a cache-reuse fix lands / on Qwen3-0.6B):** put
> stable parts first (metadata, recent window), volatile retrieved memory last, so
> volatile bits don't invalidate the cached prefix. Moot while full re-prefill is
> forced (§12 #1).

Embeddings via **`bge-small-en-v1.5`** (~130 MB, <30 ms/embed on CPU; default) or
**`Qwen3-Embedding-0.6B-GGUF`** for higher quality. Index in `sqlite`+`faiss`/
`chroma`; **precompute message embeddings at scrape/retrain time, embed only the
live query** (§12 #20) so retrieval doesn't fight generation for the 4 cores.

**Deliverable:** `bot/memory.py`, `data/memory.db`, `assets/events.md`.

---

## 9. Media — reading & sending

### Reading (incoming) — captioned at consideration so the bot can react
- **Links (no crawler needed):** Discord already unfurls most links into
  `message.embeds` (title/description/thumbnail) — **read those first, zero
  fetches**. Only when an embed is missing, do a **single-page** metadata fetch
  (`<title>` + `og:*`/`description`; oEmbed for YouTube/Tweets) — an *unfurler*,
  not a recursive spider. Inject `[link: <title> — <desc>]`. *Security:* domain
  allowlist + size limit + timeout + block private IPs (SSRF) — don't blindly
  fetch arbitrary user URLs (§12 #14).
- **GIFs:** **always sample a single frame from the file** (default the *middle*
  frame — most representative; first frame is often a title card; configurable) →
  caption it with the vision path below. (A GIF is frames; a VLM sees stills.)
- **Static images:** caption via a **separate `Qwen3-VL` (2B/4B) GGUF** + its
  `mmproj` (Qwen3.5-VL's own vision isn't supported in llama.cpp, §12 #2), through
  `llama-mtmd-cli`. Captions are injected as text, so a different vision model is fine.
- **Video:** caption the **Discord thumbnail** (the preview image Discord attaches
  to a video/embed, via `attachment.proxy_url`/embed thumbnail) via the same vision
  path — no frame extraction from the video itself.
- Vision runs **only when media is present** and we're already considering the
  message. On CPU it's **~15–30s/image** (not seconds) + ~300–1000 image tokens, so
  **cap to 1 image/decision** and keep it opt-in via a **config flag**. Combined
  with full-prefill-every-turn (§12 #1), watch the ≤60s budget.

> **Confirmed:** Qwen3-VL has working CPU `mmproj` GGUF (since Oct 2025); this is
> the **default** captioner. Qwen3.5-VL is **not** supported (§12 #2). A second
> model means extra resident RAM or on-demand load latency (§12 #23).

### Sending (outgoing)
- `assets/media/` + `assets/media_manifest.json` map tags → path/URL + description.
- Model emits `REPLY: [GIF:deadinside]`; `bot/media.py` posts the file/URL (Tenor
  URL auto-embeds, or `discord.File` upload). Unknown tag → nearest-by-description
  or fall back to text. Manifest tags are injected into preprocessing (§4).

---

## 10. Continuous freshness — weekly retrain loop

QLoRA on a 0.8B model is cheap (likely well under an hour on a 4060), so a weekly
refresh is very feasible. Automated pipeline (cron on the train box):

1. **Backup / pull new messages** — live-append + the incremental scrape script
   (§3) keep the canonical corpus current. Text is tiny (even ~1M msgs ≈ a few
   hundred MB JSONL). Back it up off-box (§12 #19).
2. **Rebuild** train/val with **recency weighting** so new slang/terms surface,
   and **excluding the bot's own output** (critical — else it mimics itself and
   drifts; §12 #2).
3. **Retrain from base** (not continued-training) for stability/reproducibility —
   avoids catastrophic forgetting/drift across weeks.
4. **Eval gate:** score against the **frozen holdout** + last-known-good. If it
   regresses, **keep last-good and alert** — never auto-ship a worse model.
5. **Export + hot-swap** GGUF on the CPU box; keep last-good for instant rollback.
6. **Same-week freshness:** brand-new terms are available via **retrieval/memory**
   (§8) immediately, even before the next retrain.

**Disk policy (hole):** weekly GGUFs (~0.5–1GB) + checkpoints + the Qwen3-VL model
accumulate on 150GiB. Keep last ~4 GGUFs + last-good; prune old checkpoints.

**Deliverable:** `scripts/weekly_retrain.sh` (or a Makefile target) + cron unit.

---

## 11. Tokens-per-second & the context budget

Target: **~10 t/s generation is fine; total reply ≤~1 min.** The constraint is
**prefill**, because llama.cpp re-prefills the *whole* prompt every turn on Qwen3.5.

| Path | Cost | Notes |
|------|------|-------|
| Decision (`IGNORE`) | **full prefill** + 1 token | No cache reuse on Qwen3.5 → pay the whole context each time; keep it small. |
| Reply prefill | **full prefill** every turn | Sets the context cap — largest window whose full prefill fits ≤60s (M3/M5 `llama-bench`). |
| Generation | ~10–60 tokens at ~10 t/s | A few seconds; trivial vs prefill. |
| Vision | encoder + 300–1000 img tokens | **~15–30s/image** on CPU; cap to 1. |

**Why context is constrained here:** llama.cpp re-prefills the *full* prompt every
turn for Qwen3.5's recurrent arch (§12 #1), and that prefill — not generation — is
the cost. So context is capped by what fits in the budget *every* turn (likely a
few hundred to ~2k tokens). (Falling back to Qwen3-0.6B restores normal prefix
caching and a much larger window.)

**Levers:** keep the window small + lean on retrieval; quantized KV cache; disable
vision or cap to 1 image; trim retrieved snippets. The big unlock is the **Qwen3-
0.6B fallback** (cache reuse) or a verified llama.cpp recurrent-cache fix (§6, §12 #1).

> **Unverified:** exact prefill/gen t/s for a 0.8B at Q4 on 4 cores isn't published
> — **measure with `llama-bench` at M3/M5** and size context from that.

---

## 12. Risks & holes

1. **🔴 No prompt-cache reuse for Qwen3.5's Gated-DeltaNet arch in llama.cpp** —
   *every* turn re-prefills the full prompt (open issues #20225/#22384/#18497…),
   so a big context is minutes/turn. Top serve risk (conversion itself works).
   De-risk at **M3**; fallback **`Qwen3-0.6B-Base`** (cache reuse works); re-check
   release notes for a fix right before M3.
2. **🔴 Feedback loop — training on its own output.** Its replies get re-scraped
   weekly → it mimics itself and degenerates. **Exclude bot messages as targets.**
3. **Weekly retrain regressing silently.** → frozen-holdout **eval gate +
   rollback** (§10), never auto-ship worse.
4. **Sentinel/tag collisions** with user text → rare markers + escaping (§4).
5. **Mention/emoji/sticker encoding** both directions (`<@id>`, `<:emoji:id>`) (§4,§7).
6. **Turn debounce** — reply per finished turn, not per message (§7).
7. **Disk creep** on 150GiB from weekly GGUFs/checkpoints + Qwen3-VL → retention (§10).
8. **Spam / ban / interrupting** — a bot answering many undirected msgs looks
   bot-like *and* annoys people → the **reserved gate** + cooldowns + active-convo
   backoff (§7) are the main control; tune `activity` vs going fully mute. The
   `/sleep` kill switch is the hard override.
9. **No clean "sounds like us" metric** — human spot-check protocol + holdout
   next-message perplexity as a rough proxy.
10. **Vision cost on CPU** — **~15–30s/image** (not seconds) → cap to 1/decision,
    cache reuse, disable flag (§9, §11).
11. **Cold-prefill latency every turn** (consequence of #1) — size the window so
    the *full* prefill fits ≤60s; lean on retrieval (§8, §11).
12. **Privileged Message Content Intent** must be on (§3).
13. **Overfitting** a 0.8B on a smallish corpus → dedupe, small LoRA, dropout,
    early-stop, generic mix (§5).
14. **URL-fetch security** (SSRF/malware) — allowlist + size limit + sandbox (§9).
15. **4060 VRAM** assumed 8GB → QLoRA (the dense 0.8B is ~3GB; comfortable).
16. **Backlog on reconnect** — ignore stale messages; only react to live ones (§7).
17. **CPU saturation at high `activity`** — `activity≈1` + media + a busy channel
    can outrun the single CPU worker → auto-throttle, cap in-flight, drop-stale.
18. **Open commands → griefing** — anyone can mute/crank chattiness → per-command
    cooldowns; keep the heavy scrape owner-gated/offline (§3, §7).
19. **Corpus backup / data loss** — the scraped JSONL is irreplaceable and
    git-ignored → back it up off-box on a schedule.
20. **Embedding compute shares the 4 cores** — precompute at scrape/retrain; only
    embed the *query* live (§8).
21. **🔴 Qwen3.5-VL vision unsupported in llama.cpp** (unsupported ops; absent from
    `multimodal.md`). Vision uses a **separate `Qwen3-VL-2B/4B` captioner** (§9).
22. **MTP-GGUF ⊕ vision/`--np>1` are mutually exclusive** — the speculative-decode
    Qwen3.5 GGUF can't use `--mmproj`. Pick speed *or* vision (§6).
23. **Two-model serve footprint** — a separate Qwen3-VL captioner = extra resident
    RAM or on-demand load latency, not yet in the §8 RAM budget.
24. **discord.py auto-retries 429s silently** — can creep toward the 10k-4xx/10-min
    **1-hour ban** during a big backfill; log/throttle retry counts (§3).
25. **Long-seq training instability** (Unsloth #4906, NaN at long seq on Qwen3.5) —
    keep training seq len modest (§5); don't quietly revert it.

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
- Vision is a separate `Qwen3-VL` model, run only when media is present (§9).
- **Secrets** in `.env`; `data/`, `models/`, `.env` are git-ignored (raw logs +
  weights stay off GitHub).

---

## 14. Proposed repo layout (to scaffold next)

```
replicator/
  CLAUDE.md  PLAN.md  README.md
  pyproject.toml / requirements.txt   .env.example   config.example.yaml
  bot/        main.py discord_client.py inference.py
              formatting.py memory.py media.py config.py
  data_pipeline/  preprocess.py build_decision_labels.py template.py
  training/   finetune.py export_gguf.py configs/
  scripts/    scrape.py weekly_retrain.sh
  data/       raw/ processed/ memory.db state.json   # git-ignored
  assets/     media/ media_manifest.json events.md
  models/     replicator.gguf  qwen3-vl-*.gguf + mmproj.gguf   # git-ignored
  tests/
```

---

## 15. Milestones

- **M0 — Scaffold:** skeleton, deps, config, `.gitignore`, bot connects, `/ping`.
- **M1 — Collect:** offline `scripts/scrape.py` backfill + live append (incl.
  threads/forums + thumbnails, serial + rate-limit-safe) → JSONL; optional `/init`.
- **M2 — Preprocess:** raw → `{train,val,holdout}.jsonl` with decision sentinels,
  media captions/tags, mention/emoji decoding, bot-output exclusion.
- **M3 — Spike (serve go/no-go):** on the CPU box — (a) convert+load stock
  Qwen3.5-0.8B-Base; (b) **measure turn-2 prefill** (does prompt-cache reuse work?);
  (c) e2e latency at target context. **Decide Qwen3.5 vs the `Qwen3-0.6B-Base` +
  `Qwen3-VL` fallback before training** (§6, §12 #1).
- **M4 — Finetune:** QLoRA on the 4060 (pinned versions; **seq len 1–4k**, not
  long — #4906); eval vs holdout + samples.
- **M5 — Export & benchmark:** merge + GGUF + quantize; round-trip; **`llama-bench`
  prefill/gen t/s → set the (small) per-turn context budget** (§8, §11).
- **M6 — Serve:** CPU runtime; model-driven REPLY/IGNORE; forced reply; debounce;
  **`activity` dial (0–1) + don't-interrupt backoff**; **`/activity`·`/sleep`·`/wake`**
  (open to all); ignore-stale-on-reconnect; prompt-cache reuse only if M3/fallback enables it.
- **M7 — Memory:** retrieval (`bge-small`) + events digest + metadata header; size
  the window from M3/M5 (§8).
- **M8 — Media:** sending manifest + reading (links; GIF single-frame; video
  thumbnail; image via **Qwen3-VL captioner**), captioned at consideration.
- **M9 — Freshness:** weekly retrain loop + eval gate + hot-swap + disk policy.
- **M10 — Harden:** cooldowns, queue, logging, systemd, retention.
- **M11 — Iterate:** tune persona/data/context until it sounds like us.
