# M3 spike — serve-stack go/no-go

The single most important de-risking step before any long training run
(PLAN.md §6, milestone **M3**). It decides, at the **actual serving
quantization (Q4_K_M / Q8_0)**, whether we serve the primary
`Qwen3.5-0.8B-Base` or fall back to `Qwen3-0.6B-Base` (+ `Qwen3-VL`).

**You do not need the CPU serve box to run this — your dev PC is fine.** The
decisive cache-reuse verdict (b, below) is a property of the model architecture
+ llama.cpp, so it **transfers** from your PC to the CPU box. Only the latency
numbers (c) are machine-specific; re-confirm those on the real box right before
final deploy.

## The three questions it answers

| # | Question | How |
|---|----------|-----|
| a | Does the stock model **convert + load** as GGUF? | `run_spike.sh` (convert + quantize) |
| b | Does llama.cpp **reuse the prompt cache** across turns, or re-prefill the whole prompt every turn (the Gated-DeltaNet blocker, §12 #1)? | `cachetest` — **the decisive test** |
| c | What **prefill / generation t/s** do we get → what context budget fits ≤ N s? | `bench` (`llama-bench`) |

`report` fuses **(b)+(c)** into a context-budget number and a GO / GO-constrained
/ NO-GO verdict.

### Why (b) is the crux
On CPU the cost is *prefill*, not generation. If llama.cpp can reuse a cached
prefix, a big context is prefilled once and cheap thereafter → the large-context
design works. If Qwen3.5's recurrent arch forces a **full re-prefill every turn**,
context must stay tiny (a few hundred–~2k tok) or we fall back. The test sends a
~1.5k-token prefix, then the **same prefix + a short appended message**, and
compares the warm-turn prefill *time* to the cold turn:

- warm ≈ 0 → **REUSE_WORKS** (only new tokens prefilled)
- warm ≈ cold → **FULL_REPREFILL** (the blocker)

The fallback `Qwen3-0.6B-Base` (a plain transformer) is also tested as a
**control**: it *should* show reuse. If it doesn't, the harness itself is suspect
— the report flags this.

## Run it

```bash
# from the repo root, on your dev PC (or eventually the serve box):
bash spike/run_spike.sh
```

### Running on your dev PC (GPU optional)
Defaults are **CPU-only** (`NGL=0`, `BUILD_CUDA=OFF`) so the latency numbers
resemble a CPU box. To answer the cache-reuse question *fast* on your 4060, set
in `config.env`:

```bash
BUILD_CUDA=ON     # build llama.cpp with CUDA (needs the CUDA toolkit installed)
NGL=99            # offload all layers to the GPU
THREADS=<cores>   # your PC's core count
```

The **REUSE_WORKS / FULL_REPREFILL verdict is unchanged** by GPU vs CPU — only
the t/s numbers change. So a quick GPU run still tells you whether Qwen3.5 is
viable; do a final CPU-only latency pass (here or on the serve box) before deploy.

It will: build/locate llama.cpp → download the base models → convert → quantize
to Q4_K_M + Q8_0 → benchmark → run the cache-reuse test → write
`spike/results/REPORT.md`. Re-runs are **idempotent** (existing GGUFs/results are
reused).

Tune everything in [`config.env`](./config.env) — models, quants, thread count
(`THREADS=4`), prefill sizes, and the per-turn prefill `BUDGET_S` (default 30s,
under the ~60s total reply budget to leave room for generation/vision).

### Prerequisites
- `git`, `cmake`, a C++ toolchain (to build llama.cpp), `python3`.
- Network access to GitHub (llama.cpp) and Hugging Face (model weights).
- Disk for the f16 GGUFs + quants (a few GB; lands in `models/spike/`, git-ignored).
- If you already have llama.cpp built, set `LLAMA_CPP_DIR` to skip the build.

> **Verify the HF repo ids** in `config.env` before running (base, not Instruct).
> If `Qwen/Qwen3.5-0.8B-Base` isn't published under that exact name, update it.

## Reading the report
`spike/results/REPORT.md` ends with one of:
- ✅ **GO** — reuse works → proceed with Qwen3.5, large context viable.
- ⚠️ **GO, constrained** — re-prefills every turn but a usable context (≥~1k tok)
  fits the budget → proceed, cap context, lean on retrieval (§8).
- ❌ **NO-GO** — only a tiny context fits each turn → fall back to
  `Qwen3-0.6B-Base` + `Qwen3-VL`.

Raw per-run JSON (`bench__*.json`, `cache__*.json`, `report.json`) and the
`server__*.log` files are kept alongside for inspection.

## Offline self-check (no model needed)
```bash
python3 spike/spike.py selftest   # exercises the parsing + verdict logic
```

## Scope
This phase tests the **text** serve stack only (the real risk). Vision is a
separate, already-confirmed `Qwen3-VL` captioner (PLAN §9) and is benchmarked
later (M8). Don't add it here.
