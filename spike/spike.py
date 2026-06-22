#!/usr/bin/env python3
"""Replicator M3 serve-stack spike (go/no-go).

Answers the three questions from PLAN.md §6 on the *actual* CPU box, at the
*actual* serving quantization (Q4_K_M / Q8_0):

  (a) does the stock model convert + load as GGUF?   -> handled by run_spike.sh
  (b) does llama.cpp REUSE the prompt cache across    -> `cachetest`
      turns, or re-prefill the whole prompt every
      turn (the Gated-DeltaNet blocker, PLAN §12 #1)?
  (c) what prefill / generation t/s do we get, so     -> `bench`
      what per-turn context budget fits <= N seconds?

`report` fuses (b)+(c) into a context-budget recommendation and a
Qwen3.5-vs-fallback verdict.

Only `requests` is needed beyond the stdlib, and only for `cachetest`
(imported lazily). Run `python3 spike.py selftest` to exercise the parsing and
verdict logic with no model or server required.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REUSE_WORKS = "REUSE_WORKS"
FULL_REPREFILL = "FULL_REPREFILL"
AMBIGUOUS = "AMBIGUOUS"


# --------------------------------------------------------------------------- #
# small io helpers
# --------------------------------------------------------------------------- #
def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def run_capture(cmd: list[str]) -> str:
    """Run a command, return stdout; raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


# --------------------------------------------------------------------------- #
# pure logic (covered by `selftest`)
# --------------------------------------------------------------------------- #
def summarize_bench(rows: list[dict]) -> dict:
    """Reduce `llama-bench -o json` rows to {prefill_tps: {ctx: t/s}, gen_tps}.

    A prefill (prompt-processing) row has n_gen == 0; a generation row has
    n_prompt == 0. avg_ts is tokens/sec.
    """
    prefill: dict[int, float] = {}
    gen_tps = None
    for r in rows:
        n_p = int(r.get("n_prompt", 0) or 0)
        n_g = int(r.get("n_gen", 0) or 0)
        tps = float(r.get("avg_ts", 0.0) or 0.0)
        if n_p > 0 and n_g == 0:
            prefill[n_p] = tps
        elif n_g > 0 and n_p == 0:
            gen_tps = tps
    return {"prefill_tps": prefill, "gen_tps": gen_tps}


def verdict_cache_reuse(cold: dict, warm: dict) -> dict:
    """Decide whether the warm (cache-primed) turn reused the prefix.

    Primary signal is wall-clock prefill time (version-independent): if the
    warm turn's prefill takes ~as long as the cold turn, the model re-prefilled
    everything. Token counts corroborate.
    """
    cms = float(cold.get("prompt_ms") or 0.0)
    wms = float(warm.get("prompt_ms") or 0.0)
    cN = int(cold.get("prompt_n") or 0)
    wN = int(warm.get("prompt_n") or 0)
    time_ratio = (wms / cms) if cms > 0 else float("inf")
    tok_ratio = (wN / cN) if cN > 0 else float("inf")
    if time_ratio <= 0.25:
        verdict = REUSE_WORKS
    elif time_ratio >= 0.75:
        verdict = FULL_REPREFILL
    else:
        verdict = AMBIGUOUS
    return {
        "verdict": verdict,
        "time_ratio": time_ratio,
        "tok_ratio": tok_ratio,
        "cold_ms": cms,
        "warm_ms": wms,
        "cold_tok": cN,
        "warm_tok": wN,
    }


def recommend_context(prefill_tps: dict, reuse_works: bool, budget_s: float) -> dict:
    """Turn a prefill curve + the reuse verdict into a context budget.

    If reuse works, per-turn cost is only the *new* tokens, so context isn't
    prefill-bound. If it doesn't, we pay full prefill EVERY turn, so the
    context cap is the largest size whose full prefill fits the budget.
    """
    points = sorted((int(c), float(t)) for c, t in prefill_tps.items())
    times = [[c, (c / t if t > 0 else float("inf"))] for c, t in points]
    if reuse_works:
        return {
            "mode": "reuse",
            "note": ("cache reuse works -> context is NOT prefill-bound per turn; "
                     "only new tokens are prefilled each turn."),
            "cold_prefill_times": times,
        }
    under = [c for c, tm in times if tm <= budget_s]
    max_measured = max(under) if under else 0
    # rough continuous estimate using the t/s at the largest measured context
    # (conservative, since prefill t/s tends to fall as context grows).
    est = int(budget_s * points[-1][1]) if points else 0
    return {
        "mode": "full_reprefill",
        "budget_s": budget_s,
        "max_ctx_under_budget_measured": max_measured,
        "est_token_budget": est,
        "full_prefill_times": times,
    }


# --------------------------------------------------------------------------- #
# subcommand: bench
# --------------------------------------------------------------------------- #
def cmd_bench(a) -> None:
    cmd = [a.bench, "-m", a.model, "-t", str(a.threads), "-ngl", str(a.ngl),
           "-p", a.ctx, "-n", str(a.ngen), "-o", "json"]
    rows = json.loads(run_capture(cmd))
    summary = summarize_bench(rows)
    path = os.path.join(a.out, f"bench__{a.label}__{a.quant}.json")
    write_json(path, {"label": a.label, "quant": a.quant, "model": a.model,
                      "threads": a.threads, "summary": summary, "raw": rows})
    print(f"  prefill t/s: {summary['prefill_tps']}")
    print(f"  gen t/s:     {summary['gen_tps']}")
    print(f"  -> {path}")


# --------------------------------------------------------------------------- #
# subcommand: cachetest (the decisive measurement)
# --------------------------------------------------------------------------- #
def _wait_for_health(base_url: str, proc, timeout: float) -> None:
    import requests
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(
                f"llama-server exited early (code {proc.returncode}); "
                "see the server__*.log in the results dir")
        try:
            if requests.get(base_url + "/health", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise SystemExit("llama-server did not become healthy in time")


def _count_tokens(base_url: str, text: str) -> int:
    import requests
    r = requests.post(base_url + "/tokenize", json={"content": text}, timeout=120)
    r.raise_for_status()
    return len(r.json().get("tokens", []))


def _build_base_text(base_url: str, target_tokens: int) -> str:
    """Grow a realistic chat-like prefix until it reaches target_tokens."""
    chunk = ("alice: yo did you see the thing yesterday it was actually insane. "
             "bob: lmao yeah i still can't believe that happened. "
             "alice: anyway what's the plan for tonight then. ")
    text = ""
    while _count_tokens(base_url, text) < target_tokens:
        text += chunk
    return text


def _completion_prefill(base_url: str, prompt: str) -> dict:
    """POST one /completion (n_predict=1) and return its prefill timings."""
    import requests
    r = requests.post(base_url + "/completion",
                      json={"prompt": prompt, "n_predict": 1,
                            "cache_prompt": True, "temperature": 0, "seed": 0},
                      timeout=900)
    r.raise_for_status()
    j = r.json()
    t = j.get("timings", {}) or {}
    return {"prompt_n": t.get("prompt_n", j.get("tokens_evaluated")),
            "prompt_ms": t.get("prompt_ms")}


def cmd_cachetest(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    base_url = f"http://127.0.0.1:{a.port}"
    log_path = os.path.join(a.out, f"server__{a.label}.log")
    server_log = open(log_path, "w")
    proc = subprocess.Popen(
        [a.server, "-m", a.model, "-t", str(a.threads), "-ngl", str(a.ngl),
         "-c", str(a.ctx), "--host", "127.0.0.1", "--port", str(a.port),
         "-np", "1"],
        stdout=server_log, stderr=subprocess.STDOUT)
    try:
        _wait_for_health(base_url, proc, timeout=600)
        base_text = _build_base_text(base_url, a.base_tokens)
        # turn 1: cold (cache empty) — prefill the whole prefix.
        cold = _completion_prefill(base_url, base_text)
        # turn 2: warm — same prefix + a short appended message. If the arch
        # supports prefix-cache reuse, only the appended tokens are prefilled.
        appended = base_text + "alice: ok but for real what do you all think.\nbob:"
        warm = _completion_prefill(base_url, appended)
        verdict = verdict_cache_reuse(cold, warm)
        path = os.path.join(a.out, f"cache__{a.label}.json")
        write_json(path, {"label": a.label, "model": a.model,
                          "base_tokens_target": a.base_tokens,
                          "cold": cold, "warm": warm, "verdict": verdict})
        print(f"  cold prefill: {cold['prompt_ms']:.0f} ms / {cold['prompt_n']} tok")
        print(f"  warm prefill: {warm['prompt_ms']:.0f} ms / {warm['prompt_n']} tok")
        print(f"  VERDICT: {verdict['verdict']} "
              f"(warm/cold time ratio {verdict['time_ratio']:.3f})")
        print(f"  -> {path}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        server_log.close()


# --------------------------------------------------------------------------- #
# subcommand: report
# --------------------------------------------------------------------------- #
def _fmt_prefill_line(ctx: int, tps: float, budget_s: float) -> str:
    secs = (ctx / tps) if tps > 0 else float("inf")
    flag = "" if secs <= budget_s else "  (> budget)"
    return f"      {ctx:>5} tok @ {tps:6.1f} t/s  ->  {secs:6.1f}s full prefill{flag}"


def cmd_report(a) -> None:
    bench: dict[str, dict[str, dict]] = {}
    cache: dict[str, dict] = {}
    for f in glob.glob(os.path.join(a.results, "bench__*.json")):
        d = read_json(f)
        bench.setdefault(d["label"], {})[d["quant"]] = d["summary"]
    for f in glob.glob(os.path.join(a.results, "cache__*.json")):
        d = read_json(f)
        cache[d["label"]] = d

    labels = sorted(set(bench) | set(cache))
    if not labels:
        raise SystemExit(f"no spike results found in {a.results}")

    out: list[str] = []
    out.append("# Replicator M3 spike — results\n")
    out.append(f"_Per-turn prefill budget: **{a.budget:g}s** "
               f"(threads as benchmarked). Q4_K_M = ship default; "
               f"Q8_0 = quality check._\n")

    for label in labels:
        out.append(f"\n## {label}\n")
        c = cache.get(label)
        reuse = bool(c and c["verdict"]["verdict"] == REUSE_WORKS)
        if c:
            v = c["verdict"]
            out.append(
                f"- **prompt-cache reuse: {v['verdict']}** "
                f"(warm/cold prefill time {v['time_ratio']:.3f}; "
                f"cold {v['cold_ms']:.0f}ms/{v['cold_tok']}tok, "
                f"warm {v['warm_ms']:.0f}ms/{v['warm_tok']}tok)")
        else:
            out.append("- prompt-cache reuse: (not measured)")

        for quant, summary in sorted(bench.get(label, {}).items()):
            out.append(f"- **{quant}** — gen {summary.get('gen_tps')} t/s; "
                       f"prefill:")
            for ctx, tps in sorted((int(k), v) for k, v in
                                   summary["prefill_tps"].items()):
                out.append(_fmt_prefill_line(ctx, tps, a.budget))
            rec = recommend_context(summary["prefill_tps"], reuse, a.budget)
            if rec["mode"] == "reuse":
                out.append("      => reuse works: context not prefill-bound "
                           "per turn (only new tokens prefill).")
            else:
                out.append(f"      => full re-prefill every turn: cap context "
                           f"~{rec['est_token_budget']} tok "
                           f"(<= {a.budget:g}s); largest measured under budget "
                           f"= {rec['max_ctx_under_budget_measured']} tok.")

    # ---- overall verdict ----
    primary = next((l for l in labels if "3.5" in l), None)
    fallback = next((l for l in labels if l != primary), None)
    out.append("\n## Verdict\n")
    if primary and cache.get(primary):
        pv = cache[primary]["verdict"]["verdict"]
        q4 = bench.get(primary, {}).get("Q4_K_M", {}).get("prefill_tps", {})
        rec = recommend_context(q4, pv == REUSE_WORKS, a.budget) if q4 else {}
        if pv == REUSE_WORKS:
            out.append(f"- ✅ **GO ({primary})** — prompt-cache reuse WORKS on "
                       "this box, so the large-context design is viable. "
                       "Proceed with Qwen3.5.")
        else:
            est = rec.get("est_token_budget", 0)
            if est >= 1024:
                out.append(f"- ⚠️ **GO, constrained ({primary})** — re-prefills "
                           f"every turn; cap context ~{est} tok (Q4) and lean "
                           "on retrieval (PLAN §8). Re-check llama.cpp for a "
                           "recurrent-cache fix.")
            else:
                out.append(f"- ❌ **NO-GO ({primary})** — full re-prefill every "
                           f"turn and only ~{est} tok fits under {a.budget:g}s. "
                           f"**Fall back to {fallback or 'Qwen3-0.6B-Base'} + "
                           "Qwen3-VL** (PLAN §6, §12 #1).")
    else:
        out.append("- Primary (Qwen3.5) cache result missing — cannot decide.")

    if fallback and cache.get(fallback):
        fv = cache[fallback]["verdict"]["verdict"]
        if fv != REUSE_WORKS:
            out.append(f"- ⚠️ **harness check:** control `{fallback}` did NOT "
                       f"show reuse ({fv}) — a plain transformer should. The "
                       "cache test may be misconfigured; treat results with care.")
        else:
            out.append(f"- control `{fallback}`: reuse works (harness validated).")

    text = "\n".join(out) + "\n"
    print("\n" + text)
    report_md = os.path.join(a.results, "REPORT.md")
    with open(report_md, "w") as fh:
        fh.write(text)
    write_json(os.path.join(a.results, "report.json"),
               {"bench": bench, "cache": cache, "budget_s": a.budget})
    print(f"-> {report_md}")


# --------------------------------------------------------------------------- #
# subcommand: selftest
# --------------------------------------------------------------------------- #
def cmd_selftest(_a) -> None:
    s = summarize_bench([
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 40.0},
        {"n_prompt": 2048, "n_gen": 0, "avg_ts": 25.0},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 11.0},
    ])
    assert s["prefill_tps"] == {512: 40.0, 2048: 25.0}, s
    assert s["gen_tps"] == 11.0, s

    assert verdict_cache_reuse({"prompt_ms": 5000, "prompt_n": 1500},
                               {"prompt_ms": 120, "prompt_n": 40}
                               )["verdict"] == REUSE_WORKS
    assert verdict_cache_reuse({"prompt_ms": 5000, "prompt_n": 1500},
                               {"prompt_ms": 5100, "prompt_n": 1540}
                               )["verdict"] == FULL_REPREFILL
    assert verdict_cache_reuse({"prompt_ms": 5000, "prompt_n": 1500},
                               {"prompt_ms": 2500, "prompt_n": 760}
                               )["verdict"] == AMBIGUOUS

    r = recommend_context({512: 40.0, 1024: 32.0, 2048: 25.0, 3072: 18.0},
                          reuse_works=False, budget_s=30)
    # 512/40=12.8s ok; 1024/32=32s over; so largest under budget = 512.
    assert r["mode"] == "full_reprefill", r
    assert r["max_ctx_under_budget_measured"] == 512, r
    assert r["est_token_budget"] == int(30 * 18.0), r

    assert recommend_context({512: 40.0}, reuse_works=True,
                             budget_s=30)["mode"] == "reuse"
    print("selftest OK")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Replicator M3 serve-stack spike")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="llama-bench prefill/gen t/s")
    b.add_argument("--bench", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--label", required=True)
    b.add_argument("--quant", required=True)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--ngl", type=int, default=0, help="GPU layers (0 = CPU)")
    b.add_argument("--ctx", default="512,1024,2048,3072")
    b.add_argument("--ngen", type=int, default=128)
    b.add_argument("--out", default="spike/results")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("cachetest", help="measure prompt-cache reuse across turns")
    c.add_argument("--server", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--ngl", type=int, default=0, help="GPU layers (0 = CPU)")
    c.add_argument("--ctx", type=int, default=4096)
    c.add_argument("--base-tokens", type=int, default=1536, dest="base_tokens")
    c.add_argument("--port", type=int, default=8080)
    c.add_argument("--out", default="spike/results")
    c.set_defaults(func=cmd_cachetest)

    r = sub.add_parser("report", help="aggregate results into a go/no-go verdict")
    r.add_argument("--results", default="spike/results")
    r.add_argument("--budget", type=float, default=30.0)
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("selftest", help="exercise parsing/verdict logic offline")
    s.set_defaults(func=cmd_selftest)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
